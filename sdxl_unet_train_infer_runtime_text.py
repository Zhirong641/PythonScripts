# -*- coding: utf-8 -*-
import os, io, math, json, glob, argparse, random, hashlib
from typing import List, Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection, Blip2Processor, Blip2ForConditionalGeneration
from safetensors.torch import save_file as safetensors_save, load_file as safetensors_load

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv as pycsv
from collections import deque, OrderedDict
from pathlib import Path

# -------------------------
# SDXL 常量/辅助
# -------------------------
VARIANT_NAMES = ["tags_fwd", "tags_rev", "caption_nl", "caption_auth_nl"]
PROBS = np.array([0.4, 0.2, 0.1, 0.3], dtype=np.float64)

def make_image_transform(resolution=1024):
    def resize_pad(img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = resolution / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BICUBIC)
        from PIL import ImageOps
        pad_w = (resolution - new_w) // 2
        pad_h = (resolution - new_h) // 2
        pad = (pad_w, pad_h, resolution - new_w - pad_w, resolution - new_h - pad_h)
        img = ImageOps.expand(img, border=pad, fill=(0,0,0))
        return img
    return transforms.Compose([
        transforms.Lambda(resize_pad),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])

def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    try: torch.set_float32_matmul_precision("high")
    except: pass

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _split_clean_comma_list(s: str) -> List[str]:
    if not s: return []
    items = [x.strip() for x in s.replace("，", ",").split(",")]
    return [x for x in items if x]

def _join_with_comma(items: List[str]) -> str:
    return ", ".join(items)

def _build_variants_from_cap_author(caption_tags: str, caption_nl: str, author: str):
    tags = _split_clean_comma_list(caption_tags)
    auth = _split_clean_comma_list(author)
    tags_fwd = _join_with_comma(tags) if tags else ""
    tags_rev = _join_with_comma(list(reversed(tags))) if tags else ""
    if auth:
        caption_auth_nl = f"{caption_nl}, by artist {random.choice(auth)}" if caption_nl else f"by artist {random.choice(auth)}"
    else:
        caption_auth_nl = caption_nl
    texts = [tags_fwd, tags_rev, caption_nl, caption_auth_nl]
    mask = [bool(t) for t in texts]
    for j in range(len(texts)):
        for k in range(j):
            if mask[j] and mask[k] and texts[j] == texts[k]:
                mask[j] = False
    mask = np.array(mask, dtype=np.bool_)
    preview_text = random.choice([texts[i] for i in range(len(texts)) if mask[i]]) if any(mask) else ""
    return texts, mask, preview_text

# -------------------------
# 预编码（用 SDXL 的 VAE）
# 输出：index.jsonl + 每张 latent 的 .npz（含 latent, h, w）
# -------------------------
def cmd_encode(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = make_image_transform(args.size)
    os.makedirs(args.out_dir, exist_ok=True)

    base_repo = args.base_repo  # e.g., ./IllustriousEmberveilmix_v10_repo
    print(">> loading SDXL VAE from:", base_repo)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(base_repo, subfolder="vae", torch_dtype=dtype).to(device)
    vae.eval().requires_grad_(False)
    scaling_factor = float(getattr(vae.config, "scaling_factor", 0.13025))  # SDXL 通常 0.13025
    print(f">> VAE scaling_factor={scaling_factor}")

    # --- 新增：加载 BLIP2 组件（只加载一次） ---
    print(">> loading BLIP2 (Salesforce/blip2-opt-2.7b)…")
    blip_dtype = torch.float16 if device.type == "cuda" else torch.float32
    blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=blip_dtype
    ).to(device)
    blip_model.eval()

    # ------- 读取 CSV -------
    if not args.csv:
        raise ValueError("请使用 --csv data.csv，并保证列为 path,caption,author")
    from CSVProcessor import CSVProcessor
    processor = CSVProcessor(args.csv)
    data = processor.get_data()
    print("CSV rows:", len(data))
    if data:
        print("First row:", data[0])

    # 可选过滤
    exclude_word_list = [
        "no humans", "chibi", "character profile", "lineart", "sketch",
        "monochrome", "comic", "text focus", "1990s", "1980s",
        "retro artstyle", "abstract"
    ]

    samples = []
    for row in data:
        try:
            path = row[0]
            caption_tags = row[1] if len(row) > 1 else ""
            author = row[2] if len(row) > 2 else ""
            if any(ex in caption_tags for ex in exclude_word_list):
                continue
            samples.append((path, caption_tags, author))
            if len(samples) >= 16:
                break
        except Exception:
            continue
    random.shuffle(samples)

    index_path = os.path.join(args.out_dir, "index.jsonl")
    idxf = open(index_path, "w", encoding="utf-8")

    # --------- 批处理参数 ----------
    blip_bs = 4
    imgs_batch = []
    metas_batch = []  # (path, caption_tags, author)

    print(f">> encoding {len(samples)} samples… (BLIP batch={blip_bs})")

    def flush_batch():
        """对当前 batch 运行 BLIP2，并写出 index 行。"""
        nonlocal imgs_batch, metas_batch
        if not imgs_batch:
            return
        # 运行 BLIP2 批推理
        try:
            with torch.inference_mode():
                if blip_processor is not None and blip_model is not None:
                    blip_inputs = blip_processor(images=imgs_batch, return_tensors="pt", padding=True).to(device, dtype=blip_dtype)
                    gen_ids = blip_model.generate(
                        **blip_inputs,
                        max_new_tokens=64,
                        # 你可根据需要添加下列解码控制项（会更慢一些）：
                        # no_repeat_ngram_size=3,
                        # repetition_penalty=1.1,
                        # do_sample=True, temperature=0.7, top_p=0.9
                    )
                    captions = blip_processor.batch_decode(gen_ids, skip_special_tokens=True)
                else:
                    captions = [""] * len(imgs_batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("[warn] BLIP2 OOM on batch; fallback to per-image.")
            # OOM 退化到逐张处理
            captions = []
            for im in imgs_batch:
                try:
                    with torch.inference_mode():
                        inp1 = blip_processor(images=im, return_tensors="pt").to(device, dtype=blip_dtype)
                        ids1 = blip_model.generate(**inp1, max_new_tokens=64)
                        cap1 = blip_processor.decode(ids1[0], skip_special_tokens=True).strip()
                except Exception as e:
                    print(f"[warn] BLIP2 single failed: {e}")
                    cap1 = ""
                captions.append(cap1)
        except Exception as e:
            print(f"[warn] BLIP2 batch failed: {e}")
            captions = [""] * len(imgs_batch)

        # 写 index.jsonl
        for (path, caption_tags, author), cap in zip(metas_batch, captions):
            # latent 如需恢复，取消注释以下三行
            pixel = tfm(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=blip_dtype, enabled=(device.type=="cuda")):
                lat = vae.encode(pixel).latent_dist.sample() * scaling_factor
            base = sha1(path) + ".npz"
            np.savez_compressed(os.path.join(args.out_dir, base),
                                latent=lat[0].detach().cpu().to(torch.float16).numpy(),
                                src=np.bytes_(path))

            meta = {
                "npz": base,
                "src": path,
                "caption_tags": caption_tags,
                "caption_nl": cap.strip(),
                "author": author
            }
            idxf.write(json.dumps(meta, ensure_ascii=False) + "\n")

        # 清空 batch
        imgs_batch.clear()
        metas_batch.clear()

    # --------- 主循环：装批 & 刷批 ----------
    for (path, caption_tags, author) in tqdm(samples):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[skip] open {path}: {e}")
            continue

        imgs_batch.append(img)
        metas_batch.append((path, caption_tags, author))

        if len(imgs_batch) >= blip_bs:
            flush_batch()

    # 收尾
    flush_batch()
    idxf.close()
    print(">> done. Saved to", args.out_dir)

    # # 读取 CSV--------------------------------
    # from CSVProcessor import CSVProcessor
    # if not args.csv:
    #     raise ValueError("请使用 --csv data.csv，并保证列为 path,caption,author")
    # data = CSVProcessor(args.csv).get_data()
    # print("CSV rows:", len(data))
    # exclude_word_list = [
    #     "no humans", "chibi", "character profile", "lineart", "sketch",
    #     "monochrome", "comic", "text focus", "1990s", "1980s",
    #     "retro artstyle", "abstract"
    # ]

    # index_path = os.path.join(args.out_dir, "index.jsonl")
    # with open(index_path, "w", encoding="utf-8") as idxf:
    #     for i, row in enumerate(tqdm(data, desc="encode")):
    #         try:
    #             if i > 10:
    #                 break
    #             if any(ex in row[1].lower() for ex in exclude_word_list):
    #                 continue
    #             path = row[0]; caption_tags = row[1] if len(row)>1 else ""; author = row[2] if len(row)>2 else ""
    #             img = Image.open(path).convert("RGB")
    #         except Exception as e:
    #             print("[skip]", path, e); continue

    #         pixel = tfm(img).unsqueeze(0).to(device, dtype=dtype)  # [-1,1]
    #         with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=(device.type=="cuda"), dtype=dtype):
    #             lat = vae.encode(pixel).latent_dist.sample() * scaling_factor  # [1,4,h,w]
    #         lat_np = lat[0].detach().cpu().to(torch.float16).numpy()
    #         base = sha1(path) + ".npz"
    #         np.savez_compressed(os.path.join(args.out_dir, base),
    #                             latent=lat_np, h=img.height, w=img.width, src=np.bytes_(path))
    #         meta = {"npz": base, "src": path, "caption_tags": caption_tags, "caption_nl": "", "author": author}
    #         idxf.write(json.dumps(meta, ensure_ascii=False) + "\n")
    print(">> done. Saved to", args.out_dir)

# -------------------------
# 解码 latent -> 图片（SDXL VAE）
# -------------------------
def cmd_decode(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(args.base_repo, subfolder="vae", torch_dtype=dtype).to(device)
    vae.eval().requires_grad_(False)
    scale = float(getattr(vae.config, "scaling_factor", 0.13025))

    def decode_one(npz_path: str, out_path: str):
        z = np.load(npz_path)
        lat = torch.from_numpy(z["latent"]).to(device=device, dtype=dtype).unsqueeze(0)
        lat = lat / scale
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=(device.type=="cuda"), dtype=dtype):
            img = vae.decode(lat).sample  # [-1,1]
        save_tensor_image(img, out_path)
        print("decoded ->", out_path)

    os.makedirs(args.out_dir, exist_ok=True)
    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, "*.npz")))
        for p in files:
            out = os.path.join(args.out_dir, os.path.splitext(os.path.basename(p))[0] + ".png")
            decode_one(p, out)
    else:
        out = args.out or (os.path.splitext(args.input)[0] + ".png")
        decode_one(args.input, out)

def save_tensor_image(img_t: torch.Tensor, path: str):
    img = (img_t.clamp(-1,1)*0.5+0.5)
    img = (img*255.0).round().byte().cpu().permute(0,2,3,1).numpy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img[0]).save(path)

# -------------------------
# 数据集（读取 latent + 文本字段）
# -------------------------
class LatentCapAuthorDataset(Dataset):
    def __init__(self, index_jsonl: str, root_dir: str):
        self.root = root_dir
        self.items = []
        with open(index_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                fname = j["npz"]; fp = os.path.join(root_dir, fname)
                if os.path.isfile(fp):
                    self.items.append((fname, j.get("caption_tags",""), j.get("caption_nl",""), j.get("author","")))
        print(f"Loaded {len(self.items)} items from {index_jsonl}")
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        fname, cap_tags, cap_nl, auth = self.items[i]
        z = np.load(os.path.join(self.root, fname), allow_pickle=False)
        lat = z["latent"].astype(np.float16)  # (4,H,W) H=W=128 for 1024
        return torch.from_numpy(lat), cap_tags, cap_nl, auth

# -------------------------
# EMA
# -------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.data.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    @torch.no_grad()
    def update(self, model):
        for n,p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0-self.decay)
    def copy_to(self, model):
        for n,p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])

# -------------------------
# Min-SNR
# -------------------------
def compute_snr(alphas_cumprod: torch.FloatTensor, timesteps: torch.LongTensor):
    a = alphas_cumprod.to(timesteps.device)[timesteps]
    return a / (1.0 - a).clamp(min=1e-8)
def min_snr_weights(snr: torch.FloatTensor, gamma: float):
    return torch.minimum(snr, torch.full_like(snr, gamma)) / (snr + 1.0)

# -------------------------
# LR Scheduler
# -------------------------
class CosineLRScheduler:
    def __init__(self, optimizer, max_steps, warmup_steps=1000, min_lr_ratio=0.1):
        self.opt = optimizer
        self.max_steps = max_steps
        self.warm = warmup_steps
        self.min_ratio = min_lr_ratio
        self.step_idx = 0
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
    def step(self):
        self.step_idx += 1
        if self.step_idx < self.warm:
            scale = self.step_idx / max(1, self.warm)
        else:
            progress = (self.step_idx - self.warm) / max(1, self.max_steps - self.warm)
            scale = self.min_ratio + 0.5*(1-self.min_ratio)*(1+math.cos(math.pi*progress))
        for i,g in enumerate(self.opt.param_groups):
            g["lr"] = self.base_lrs[i] * scale

# -------------------------
# 文本编码（SDXL 双编码器）
# -------------------------
class SDXLTextEnc:
    """
    从 base_repo 读取：
      - tokenizer (OpenAI) + text_encoder (CLIP-L/14, hidden 768)
      - tokenizer_2 (OpenCLIP) + text_encoder_2 (bigG, hidden 1280，且带 projection/pooler)
    产出：
      - prompt_embeds: concat([TE1_hidden, TE2_hidden], dim=-1) -> [B,77,2048]
      - pooled_prompt_embeds: TE2 的 pooled_output -> [B,1280]
    """
    def __init__(self, base_repo: str, device, use_amp=True):
        self.device = device
        self.use_amp = use_amp

        self.tokenizer_1 = CLIPTokenizer.from_pretrained(base_repo, subfolder="tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(base_repo, subfolder="tokenizer_2")

        self.text_encoder_1 = CLIPTextModel.from_pretrained(base_repo, subfolder="text_encoder",
                                                            torch_dtype=torch.float16 if device.type=="cuda" else torch.float32).to(device)
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            base_repo, subfolder="text_encoder_2",
            torch_dtype=torch.float16 if device.type=="cuda" else torch.float32
        ).to(device)

        self.text_encoder_1.eval().requires_grad_(False)
        self.text_encoder_2.eval().requires_grad_(False)

        # 预计算无条件
        self.uncond_prompt_embeds, self.uncond_pooled = self.encode_prompts([""])

    @torch.no_grad()
    def encode_prompts(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (prompt_embeds[B,77,2048], pooled_prompt_embeds[B,1280])"""
        ids1 = self.tokenizer_1(texts, padding="max_length", max_length=77,
                                truncation=True, return_tensors="pt").input_ids.to(self.device)
        ids2 = self.tokenizer_2(texts, padding="max_length", max_length=77,
                                truncation=True, return_tensors="pt").input_ids.to(self.device)
        with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type=="cuda" and self.use_amp), dtype=torch.float16):
            z1 = self.text_encoder_1(ids1)[0]                      # [B,77,768]
            z2 = self.text_encoder_2(ids2)                         # BaseModelOutputWithPooling
            z2_hidden = z2.last_hidden_state                        # [B,77,1280]
            z2_pooled = z2.text_embeds if hasattr(z2, "text_embeds") else z2.pooler_output  # [B,1280]
        prompt_embeds = torch.cat([z1, z2_hidden], dim=-1)          # [B,77,2048]
        return prompt_embeds, z2_pooled

# -------------------------
# SDXL 时间/尺寸条件
# -------------------------
def build_time_ids(width: int, height: int, crop_w=0, crop_h=0, target_w=None, target_h=None):
    """
    SDXL 需要 add_time_ids: [6] = (orig_w, orig_h, crop_x, crop_y, target_w, target_h)
    通常设为目标尺寸（宽高相同），无裁剪：crop=(0,0)、orig=target
    """
    if target_w is None: target_w = width
    if target_h is None: target_h = height
    return torch.tensor([width, height, crop_w, crop_h, target_w, target_h], dtype=torch.float32)

# -------------------------
# 训练（SDXL）
# -------------------------
def cmd_train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    # 数据
    index_path = os.path.join(args.data_dir, "index.jsonl")
    dataset = LatentCapAuthorDataset(index_path, args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=(device.type=="cuda"),
                        drop_last=True, persistent_workers=(args.workers>0))

    # SDXL 组件（从 base_repo 读取以对齐 IllustriousEmberveilmix_v10_repo）
    base_repo = args.base_repo
    print(">> loading UNet from:", base_repo)
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(base_repo, subfolder="unet",
                                                                      torch_dtype=torch.float16 if use_amp else torch.float32).to(device)
    unet.enable_gradient_checkpointing()
    print("UNet cross_attention_dim:", unet.config.cross_attention_dim)  # 通常 2048
    print("UNet addition_embed_type:", getattr(unet.config, "addition_embed_type", None))  # 应为 "text_time"

    # 调度器（与 SDXL 兼容）
    prediction_type = "v_prediction" if args.vpred else "epsilon"
    noise_sched = DDPMScheduler(num_train_timesteps=1000,
                                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                prediction_type=prediction_type)

    # 文本编码器（双 encoder）
    txt = SDXLTextEnc(base_repo, device, use_amp=True)

    # 训练超参
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr, betas=(0.9,0.999), weight_decay=1e-2)
    ema = EMA(unet, decay=args.ema)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # LR 调度（按优化器步）
    micro_steps_per_epoch = len(loader)
    opt_steps_per_epoch = max(1, math.ceil(micro_steps_per_epoch / max(1, args.grad_accum)))
    total_opt_steps = max(1, args.epochs * opt_steps_per_epoch)
    warmup_opt_steps = args.warmup if args.warmup>0 else max(100, int(0.03*total_opt_steps))
    lr_sched = CosineLRScheduler(optimizer, max_steps=total_opt_steps, warmup_steps=warmup_opt_steps, min_lr_ratio=0.1)

    # 预览（用 SDXL pipeline 替换 unet 出图）
    pipe = StableDiffusionXLPipeline.from_pretrained(base_repo,
                                                     torch_dtype=torch.float16 if use_amp else torch.float32).to(device)
    pipe.enable_attention_slicing()
    try: pipe.enable_xformers_memory_efficient_attention()
    except: pass
    pipe.scheduler.config.prediction_type = prediction_type

    def save_preview(step: int, prompt: str, use_ema=False, size=1024):
        if not prompt: return
        tmp_unet = UNet2DConditionModel.from_config(unet.config).to(device, dtype=next(unet.parameters()).dtype)
        if use_ema: ema.copy_to(tmp_unet)
        else:        tmp_unet.load_state_dict(unet.state_dict(), strict=True)
        pipe.unet = tmp_unet

        # 编码
        pe, pooled = txt.encode_prompts([prompt])
        neg_pe, neg_pooled = txt.uncond_prompt_embeds, txt.uncond_pooled

        time_ids = build_time_ids(size, size).to(device).unsqueeze(0)  # [1,6]
        add_kw = {"text_embeds": pooled, "time_ids": time_ids}
        add_kw_neg = {"text_embeds": neg_pooled, "time_ids": time_ids}

        g = torch.Generator(device=device.type)
        if args.preview_seed is not None: g = g.manual_seed(args.preview_seed)

        img = pipe(
            prompt_embeds=pe, negative_prompt_embeds=neg_pe,
            pooled_prompt_embeds=pooled, negative_pooled_prompt_embeds=neg_pooled,
            num_inference_steps=args.preview_steps, guidance_scale=args.preview_scale,
            width=size, height=size, generator=g,
            cross_attention_kwargs=None, # 可放入 sdp/noise hints
            **{"added_cond_kwargs": add_kw, "negative_added_cond_kwargs": add_kw_neg}
        ).images[0]

        out_dir = os.path.join(args.out_dir, "preview"); os.makedirs(out_dir, exist_ok=True)
        img.save(os.path.join(out_dir, f"step_{step:08d}_{'ema' if use_ema else 'raw'}.png"))

    # 训练循环
    global_step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)
        for batch in pbar:
            lat, cap_tags, cap_nls, auths = batch
            lat = lat.to(device, dtype=torch.float16 if use_amp else torch.float32)  # [B,4,128,128] for 1024
            B = lat.size(0)

            # 文本变体采样
            chosen = []
            for i in range(B):
                texts_i, mask_i, _ = _build_variants_from_cap_author(cap_tags[i], cap_nls[i], auths[i])
                p = PROBS * mask_i.astype(np.float64); s = p.sum()
                if s<=0: chosen.append("")  # 空提示 -> uncond
                else:
                    p = p/s; idx = int(np.random.choice(len(VARIANT_NAMES), p=p))
                    chosen.append(texts_i[idx] if mask_i[idx] else "")

            # 编码（含无条件 dropout）
            pe_list, pooled_list = [], []
            for s in chosen:
                if (args.cfg_drop>0.0) and (random.random()<args.cfg_drop):
                    pe, pooled = txt.uncond_prompt_embeds, txt.uncond_pooled
                else:
                    pe, pooled = txt.encode_prompts([s]) if s else (txt.uncond_prompt_embeds, txt.uncond_pooled)
                pe_list.append(pe); pooled_list.append(pooled)
            prompt_embeds = torch.cat(pe_list, dim=0)           # [B,77,2048]
            pooled_prompt_embeds = torch.cat(pooled_list, dim=0) # [B,1280]

            # 噪声 & 时间步
            t = torch.randint(0, noise_sched.config.num_train_timesteps, (B,), device=device)
            noise = torch.randn_like(lat)
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.float16):
                noisy = noise_sched.add_noise(lat, noise, t)

                # SDXL 的 added cond
                time_ids = build_time_ids(args.train_size, args.train_size).to(device).unsqueeze(0).expand(B,-1)  # [B,6]
                added_cond = {"text_embeds": pooled_prompt_embeds, "time_ids": time_ids}

                out = unet(noisy, t,
                           encoder_hidden_states=prompt_embeds,
                           added_cond_kwargs=added_cond)
                pred = out.sample

                target = noise if prediction_type=="epsilon" else noise_sched.get_velocity(lat, noise, t)
                loss = F.mse_loss(pred, target, reduction="none").mean(dim=(1,2,3))

                if args.min_snr_gamma>0:
                    snr = compute_snr(noise_sched.alphas_cumprod, t)
                    w = min_snr_weights(snr, gamma=args.min_snr_gamma)
                    loss = (loss * w).mean()
                else:
                    loss = loss.mean()

            scaler.scale(loss/args.grad_accum).backward()

            if ((global_step+1) % args.grad_accum)==0:
                nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr_sched.step()
                ema.update(unet)

            global_step += 1
            pbar.set_postfix({"loss": float(loss.detach().cpu()),
                              "lr": optimizer.param_groups[0]["lr"]})

            if args.save_steps and (global_step % args.save_steps == 0):
                save_ckpt(args, unet, ema, step=global_step, prediction_type=prediction_type)

            if getattr(args, "preview_every_ckpt", False) and (global_step % args.preview_save_steps == 0 or global_step==1):
                # 取 batch[0] 的若干变体作为预览
                texts0, mask0, _ = _build_variants_from_cap_author(cap_tags[0], cap_nls[0], auths[0])
                for i in range(len(texts0)):
                    if mask0[i]:
                        save_preview(global_step+i, texts0[i], use_ema=False, size=args.train_size)
                        save_preview(global_step+i, texts0[i], use_ema=True,  size=args.train_size)

        if args.save_epochs and ((epoch+1) % args.save_epochs == 0):
            save_ckpt(args, unet, ema, step=global_step, prediction_type=prediction_type)

    print(">> training done.")

def save_ckpt(args, unet, ema, step: int, prediction_type: str):
    raw_dir = os.path.join(args.out_dir, f"step_{step}_raw"); os.makedirs(raw_dir, exist_ok=True)
    torch.save(unet.state_dict(), os.path.join(raw_dir, "unet_raw.pt"))

    ema_model = UNet2DConditionModel.from_config(unet.config).to(unet.device, dtype=next(unet.parameters()).dtype)
    ema.copy_to(ema_model)
    ema_dir = os.path.join(args.out_dir, f"step_{step}_ema"); os.makedirs(ema_dir, exist_ok=True)
    safetensors_save({k: v.detach().cpu() for k,v in ema_model.state_dict().items()}, os.path.join(ema_dir, "unet_ema.safetensors"))
    with open(os.path.join(ema_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(unet.config, f, indent=2)
    with open(os.path.join(ema_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"prediction_type": prediction_type, "step": step}, f, indent=2)
    print(f">> saved ckpt @ {ema_dir}")

# -------------------------
# 推理（SDXL：替换 UNet）
# -------------------------
def cmd_infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type=="cuda" else torch.float32
    pipe = StableDiffusionXLPipeline.from_pretrained(args.base_repo, torch_dtype=dtype).to(device)
    pipe.enable_attention_slicing()
    try: pipe.enable_xformers_memory_efficient_attention()
    except: pass

    # 替换 UNet
    print(">> loading trained UNet:", args.unet_path)
    unet = UNNet_from_repo_cfg(args.base_repo, device, dtype)
    sd = safetensors_load(args.unet_path) if args.unet_path.endswith(".safetensors") else torch.load(args.unet_path, map_location="cpu")
    missing, unexpected = unet.load_state_dict(sd, strict=False)
    print("missing:", len(missing), "unexpected:", len(unexpected))
    pipe.unet = unet

    if args.vpred: pipe.scheduler.config.prediction_type = "v_prediction"
    else:          pipe.scheduler.config.prediction_type = "epsilon"

    # SDXL 需要 added_cond_kwargs
    txt = SDXLTextEnc(args.base_repo, device, use_amp=(dtype==torch.float16))
    pe, pooled = txt.encode_prompts([args.prompt])
    neg_pe, neg_pooled = txt.encode_prompts([args.negative_prompt]) if args.negative_prompt else (txt.uncond_prompt_embeds, txt.uncond_pooled)

    time_ids = build_time_ids(args.width, args.height).to(device).unsqueeze(0)
    add_kw     = {"text_embeds": pooled,     "time_ids": time_ids}
    add_kw_neg = {"text_embeds": neg_pooled, "time_ids": time_ids}

    g = torch.Generator(device=device.type)
    if args.seed is not None: g = g.manual_seed(args.seed)

    image = pipe(prompt_embeds=pe, negative_prompt_embeds=neg_pe,
                 pooled_prompt_embeds=pooled, negative_pooled_prompt_embeds=neg_pooled,
                 num_inference_steps=args.steps, guidance_scale=args.scale,
                 width=args.width, height=args.height, generator=g,
                 **{"added_cond_kwargs": add_kw, "negative_added_cond_kwargs": add_kw_neg}).images[0]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    image.save(args.out)
    print(">> saved", args.out)

def UNNet_from_repo_cfg(base_repo: str, device, dtype):
    # 用 repo 的 unet config 构建空模型（确保结构一致）
    cfg_path = Path(base_repo) / "unet" / "config.json"
    if not cfg_path.exists():
        # 兜底：直接 from_pretrained
        return UNet2DConditionModel.from_pretrained(base_repo, subfolder="unet", torch_dtype=dtype).to(device)
    j = json.loads(cfg_path.read_text())
    return UNet2DConditionModel.from_config(j).to(device, dtype=dtype)

# -------------------------
# CLI
# -------------------------
def build_parser():
    p = argparse.ArgumentParser("SDXL UNet Train & Infer (IllustriousEmberveilmix_v10_repo)")
    sub = p.add_subparsers(dest="cmd")

    # encode
    pe = sub.add_parser("encode", help="将图像预编码为 SDXL latent（index.jsonl + .npz）")
    pe.add_argument("--csv", type=str, required=True, help="CSV: path,caption,author")
    pe.add_argument("--out_dir", type=str, required=True)
    pe.add_argument("--base_repo", type=str, required=True, help="SDXL 基模型 REPO 目录（含 unet/vae/text_encoder*/tokenizer*）")
    pe.add_argument("--size", type=int, default=1024)
    pe.set_defaults(func=cmd_encode)

    # train
    pt = sub.add_parser("train", help="微调 SDXL UNet（运行时双编码器现算，CFG dropout）")
    pt.add_argument("--data_dir", type=str, required=True, help="encode 产生的数据目录（含 index.jsonl 和 .npz）")
    pt.add_argument("--base_repo", type=str, required=True, help="用于对齐结构与 tokenizer/encoder 的基模型 REPO 目录")
    pt.add_argument("--out_dir", type=str, required=True)
    pt.add_argument("--train_size", type=int, default=1024, help="训练/预览分辨率（通常 SDXL=1024）")
    pt.add_argument("--batch_size", type=int, default=2)
    pt.add_argument("--workers", type=int, default=4)
    pt.add_argument("--epochs", type=int, default=1)
    pt.add_argument("--lr", type=float, default=1e-4)
    pt.add_argument("--warmup", type=int, default=1000)
    pt.add_argument("--grad_accum", type=int, default=1)
    pt.add_argument("--ema", type=float, default=0.999)
    pt.add_argument("--cfg_drop", type=float, default=0.10)
    pt.add_argument("--min_snr_gamma", type=float, default=0.0)
    pt.add_argument("--vpred", action="store_true")
    pt.add_argument("--seed", type=int, default=1234)
    pt.add_argument("--save_steps", type=int, default=2000)
    pt.add_argument("--save_epochs", type=int, default=0)
    # 预览
    pt.add_argument("--preview_every_ckpt", action="store_true")
    pt.add_argument("--preview_save_steps", type=int, default=2000)
    pt.add_argument("--preview_steps", type=int, default=28)
    pt.add_argument("--preview_scale", type=float, default=5.5)
    pt.add_argument("--preview_seed", type=int, default=None)
    pt.set_defaults(func=cmd_train)

    # infer
    pi = sub.add_parser("infer", help="SDXL 推理（用训练的 UNet 替换）")
    pi.add_argument("--base_repo", type=str, required=True)
    pi.add_argument("--unet_path", type=str, required=True)
    pi.add_argument("--prompt", type=str, required=True)
    pi.add_argument("--negative_prompt", type=str, default="")
    pi.add_argument("--steps", type=int, default=28)
    pi.add_argument("--scale", type=float, default=5.5)
    pi.add_argument("--width", type=int, default=1024)
    pi.add_argument("--height", type=int, default=1024)
    pi.add_argument("--seed", type=int, default=None)
    pi.add_argument("--vpred", action="store_true")
    pi.add_argument("--out", type=str, default="./out.png")
    pi.set_defaults(func=cmd_infer)

    # decode
    pd = sub.add_parser("decode", help="将 latent(.npz) 解码为图片（使用 SDXL VAE）")
    pd.add_argument("--base_repo", type=str, required=True)
    pd.add_argument("--input", type=str, required=True)
    pd.add_argument("--out", type=str, default=None)
    pd.add_argument("--out_dir", type=str, default="./decoded")
    pd.set_defaults(func=cmd_decode)

    return p

def main():
    args = build_parser().parse_args()
    if not hasattr(args, "func"):
        print("Use subcommands: encode | train | infer | decode")
        return
    args.func(args)

if __name__ == "__main__":
    main()
