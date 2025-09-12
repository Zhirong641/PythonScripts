# -*- coding: utf-8 -*-
"""
SDXL UNet Train & Infer (IllustriousEmberveilmix_v10_repo)
- 直接在展开后的 SDXL repo（含 unet/vae/text_encoder*/tokenizer*）上微调 UNet
- 训练：FP32 权重 + autocast(fp16) + GradScaler
- 文本：SDXL 双编码器（OpenAI+OpenCLIP），运行时现算，支持 LRU 缓存
- 条件：added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}
- 预览：StableDiffusionXLPipeline，替换 pipe.unet，dtype 全链路对齐
- 断点恢复：--resume 支持目录/pt/pth/safetensors
- 记录：loss.csv + loss.png（--plot_interval 控制刷新）
"""

import os, io, math, json, glob, argparse, random, hashlib, tempfile
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
from pathlib import Path
from collections import deque, OrderedDict
import contextlib

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from diffusers import (
    UNet2DConditionModel,
    DDPMScheduler,
    AutoencoderKL,
    StableDiffusionXLPipeline,
)
from transformers import (
    CLIPTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from safetensors.torch import save_file as safetensors_save, load_file as safetensors_load

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv as pycsv


# -------------------------
# 基础工具
# -------------------------
def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    try: torch.set_float32_matmul_precision("high")
    except: pass

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def save_tensor_image(img_t: torch.Tensor, path: str):
    img = (img_t.clamp(-1,1)*0.5+0.5)
    img = (img*255.0).round().byte().cpu().permute(0,2,3,1).numpy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img[0]).save(path)

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
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

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
    if random.random() < 0.5 and auth:
        tags_fwd = f"artist:{random.choice(auth)}, {tags_fwd}"
    tags_rev = _join_with_comma(list(reversed(tags))) if tags else ""
    if random.random() < 0.5 and auth:
        tags_rev = f"artist:{random.choice(auth)}, {tags_rev}"
    if auth:
        caption_auth_nl = f"{caption_nl}, by artist {random.choice(auth)}" if caption_nl else f"by artist {random.choice(auth)}"
    else:
        caption_auth_nl = caption_nl
    texts = [tags_fwd, tags_rev, caption_nl, caption_auth_nl]
    mask = [bool(t) for t in texts]
    # 去重屏蔽
    for j in range(len(texts)):
        for k in range(j):
            if mask[j] and mask[k] and texts[j] == texts[k]:
                mask[j] = False
    mask = np.array(mask, dtype=np.bool_)
    preview_text = random.choice([texts[i] for i in range(len(texts)) if mask[i]]) if any(mask) else ""
    return texts, mask, preview_text


# -------------------------
# 数据集（latent + 文本）
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
        lat = z["latent"].astype(np.float16)  # (4,H,W) for SDXL: H=W=128 at 1024
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
# Min-SNR / LR
# -------------------------
def compute_snr(alphas_cumprod: torch.FloatTensor, timesteps: torch.LongTensor):
    a = alphas_cumprod.to(timesteps.device)[timesteps]
    return a / (1.0 - a).clamp(min=1e-8)
def min_snr_weights(snr: torch.FloatTensor, gamma: float):
    return torch.minimum(snr, torch.full_like(snr, gamma)) / (snr + 1.0)

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
# 文本编码（SDXL 双编码器）+ LRU 缓存
# -------------------------
class SDXLTextEnc:
    """
    - tokenizer (OpenAI) + text_encoder (CLIP-L/14, hidden 768)
    - tokenizer_2 (OpenCLIP) + text_encoder_2 (bigG, hidden 1280, 带投影/池化)
    输出：
      - prompt_embeds: concat([TE1_hidden, TE2_hidden], dim=-1) -> [B,77,2048]
      - pooled_prompt_embeds: TE2 的 pooled/text_embeds -> [B,1280]
    支持 LRU 缓存（按文本串）
    """
    def __init__(self, base_repo: str, device, use_amp=True, cache_cap: int = 0, train_te: bool = False):
        self.device = device
        self.use_amp = use_amp
        self.cache_cap = int(cache_cap)
        self.cache: "OrderedDict[str, Tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()
        self.train_te = bool(train_te)

        self.tokenizer_1 = CLIPTokenizer.from_pretrained(base_repo, subfolder="tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(base_repo, subfolder="tokenizer_2")

        dtype = torch.float16 if device.type=="cuda" else torch.float32
        self.text_encoder_1 = CLIPTextModel.from_pretrained(base_repo, subfolder="text_encoder", torch_dtype=dtype).to(device)
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(base_repo, subfolder="text_encoder_2", torch_dtype=dtype).to(device)
        if self.train_te:
            self.text_encoder_1.train().requires_grad_(True)
            self.text_encoder_2.train().requires_grad_(True)
        else:
            self.text_encoder_1.eval().requires_grad_(False)
            self.text_encoder_2.eval().requires_grad_(False)

        # 预计算无条件
        if not self.train_te:
            self.uncond_prompt_embeds, self.uncond_pooled = self.encode_prompts([""])
        else:
            self.uncond_prompt_embeds, self.uncond_pooled = None, None

    def _encode_no_cache(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def encode_prompts(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.train_te:
            return self._encode_no_cache(texts)
        with torch.no_grad():
            return self._encode_no_cache(texts)

    def encode_cached(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.train_te or self.cache_cap <= 0:
            return self._encode_no_cache([text])
        if text in self.cache:
            pe_cpu, pooled_cpu = self.cache.pop(text)
            self.cache[text] = (pe_cpu, pooled_cpu)
        else:
            pe, pooled = self._encode_no_cache([text])     # device
            pe_cpu, pooled_cpu = pe.detach().cpu(), pooled.detach().cpu()
            if len(self.cache) >= self.cache_cap:
                self.cache.popitem(last=False)
            self.cache[text] = (pe_cpu, pooled_cpu)
        pe_cpu, pooled_cpu = self.cache[text]
        return pe_cpu.to(self.device, non_blocking=True), pooled_cpu.to(self.device, non_blocking=True)


# -------------------------
# 时间/尺寸条件（dtype/device 可指定）
# -------------------------
def build_time_ids(width: int, height: int, crop_w=0, crop_h=0, target_w=None, target_h=None,
                   *, dtype=None, device=None):
    if target_w is None: target_w = width
    if target_h is None: target_h = height
    t = torch.tensor([width, height, crop_w, crop_h, target_w, target_h], dtype=torch.float32)
    if dtype is not None:  t = t.to(dtype)
    if device is not None: t = t.to(device)
    return t


# -------------------------
# 断点保存/恢复
# -------------------------
def save_train_state(path, unet, txt, optimizer, lr_sched, ema, global_step, epoch,
                     prediction_type, scaler=None, opt_step=0):
    pkg = {
        "model": {k: v.detach().cpu() for k, v in unet.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "lr_sched_step": getattr(lr_sched, "step_idx", 0),
        "ema": {"decay": getattr(ema, "decay", 0.999),
                "shadow": {k: v.cpu() for k, v in ema.shadow.items()}},
        "global_step": int(global_step),
        "epoch": int(epoch),
        "prediction_type": prediction_type,
        "opt_step": int(opt_step),
        "scaler": (scaler.state_dict() if (scaler is not None and hasattr(scaler, "state_dict")) else None),
        "train_text_encoder": bool(getattr(txt, "train_te", False)),
    }
    if pkg["train_text_encoder"]:
        pkg["text_encoder_1"] = {k: v.detach().cpu() for k, v in txt.text_encoder_1.state_dict().items()}
        pkg["text_encoder_2"] = {k: v.detach().cpu() for k, v in txt.text_encoder_2.state_dict().items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pt", dir=os.path.dirname(path))
    os.close(tmp_fd)
    torch.save(pkg, tmp_path)
    os.replace(tmp_path, path)
    print(f">> saved train state: {path}")

def try_load_train_state(resume_path, unet, txt, optimizer, lr_sched, ema, scaler=None):
    """
    resume_path:
      - 目录：自动选择最新的 .pt/.pth/.safetensors
      - .pt/.pth：完整快照
      - .safetensors：仅模型（可配旁车 <file>.state.json）
    返回: (global_step, epoch, prediction_type, opt_step)
    """
    if resume_path is None:
        return 0, 0, None, 0

    if os.path.isdir(resume_path):
        cands = [os.path.join(resume_path, x) for x in os.listdir(resume_path)
                 if x.endswith((".pth", ".pt", ".safetensors"))]
        if not cands:
            raise FileNotFoundError(f"No *.pth/*.pt/*.safetensors in {resume_path}")
        cands.sort()
        resume_path = cands[-1]

    ext = os.path.splitext(resume_path)[1].lower()

    if ext in (".pt", ".pth"):
        print(f">> resume from packaged: {resume_path}")
        pkg = torch.load(resume_path, map_location="cpu")
        unet.load_state_dict(pkg["model"], strict=True)
        if pkg.get("train_text_encoder"):
            if "text_encoder_1" in pkg:
                txt.text_encoder_1.load_state_dict(pkg["text_encoder_1"], strict=True)
            if "text_encoder_2" in pkg:
                txt.text_encoder_2.load_state_dict(pkg["text_encoder_2"], strict=True)
        if "optimizer" in pkg and pkg["optimizer"]:
            optimizer.load_state_dict(pkg["optimizer"])
        if "lr_sched_step" in pkg:
            lr_sched.step_idx = int(pkg["lr_sched_step"])
        if "ema" in pkg and isinstance(pkg["ema"], dict):
            ema.decay = float(pkg["ema"].get("decay", ema.decay))
            sh = pkg["ema"].get("shadow", {})
            for n, p in ema.shadow.items():
                if n in sh:
                    ema.shadow[n] = sh[n].to(p.device, dtype=p.dtype)
        if scaler is not None and pkg.get("scaler") is not None:
            try: scaler.load_state_dict(pkg["scaler"])
            except Exception as e: print(f"[warn] scaler not restored: {e}")
        return (int(pkg.get("global_step", 0)),
                int(pkg.get("epoch", 0)),
                pkg.get("prediction_type", "epsilon"),
                int(pkg.get("opt_step", 0)))

    if ext == ".safetensors":
        print(f">> resume from safetensors: {resume_path}")
        sd = safetensors_load(resume_path)
        missing, unexpected = unet.load_state_dict(sd, strict=False)
        if missing:    print(f"[info] missing keys: {len(missing)} (show up to 10) {missing[:10]}")
        if unexpected: print(f"[info] unexpected: {len(unexpected)} (show up to 10) {unexpected[:10]}")
        dir_path = os.path.dirname(resume_path)
        te1_path = os.path.join(dir_path, "text_encoder_1.safetensors")
        te2_path = os.path.join(dir_path, "text_encoder_2.safetensors")
        if os.path.isfile(te1_path):
            txt.text_encoder_1.load_state_dict(safetensors_load(te1_path), strict=False)
        if os.path.isfile(te2_path):
            txt.text_encoder_2.load_state_dict(safetensors_load(te2_path), strict=False)
        base_no_ext = os.path.splitext(resume_path)[0]
        meta_path = base_no_ext + ".state.json"
        if os.path.isfile(meta_path):
            try:
                meta = json.load(open(meta_path, "r", encoding="utf-8"))
                if "lr_sched_step" in meta:
                    lr_sched.step_idx = int(meta["lr_sched_step"])
                return (int(meta.get("global_step", 0)),
                        int(meta.get("epoch", 0)),
                        meta.get("prediction_type", "epsilon"),
                        int(meta.get("opt_step", 0)))
            except Exception as e:
                print(f"[warn] read sidecar failed: {e}")
        return 0, 0, None, 0

    raise FileNotFoundError(resume_path)


# -------------------------
# 保存推理用权重（EMA）+ 旁车 json
# -------------------------
def save_ckpt(args, unet, ema, txt, step: int, prediction_type: str, lr_sched_step: int):
    # ⛏️ 自动清理旧的 checkpoint（只保留最近10个）
    ckpt_dirs = sorted(glob.glob(os.path.join(args.out_dir, "step_*_ema")), key=os.path.getmtime)
    max_keep = 1
    if len(ckpt_dirs) > max_keep:
        for old_dir in ckpt_dirs[:-max_keep]:
            try:
                for file in glob.glob(os.path.join(old_dir, "*")):
                    os.remove(file)
                os.rmdir(old_dir)
                print(f"🗑️ deleted old ema ckpt: {old_dir}")
            except Exception as e:
                print(f"❌ failed to delete {old_dir}: {e}")

        raw_dirs = sorted(glob.glob(os.path.join(args.out_dir, "step_*_raw")), key=os.path.getmtime)
        for old_dir in raw_dirs[:-max_keep]:
            try:
                for file in glob.glob(os.path.join(old_dir, "*")):
                    os.remove(file)
                os.rmdir(old_dir)
                print(f"🗑️ deleted old raw ckpt: {old_dir}")
            except Exception as e:
                print(f"❌ failed to delete {old_dir}: {e}")
        
        state_dirs = sorted(glob.glob(os.path.join(args.out_dir, "step_*_state")), key=os.path.getmtime)
        for old_dir in state_dirs[:-max_keep]:
            try:
                for file in glob.glob(os.path.join(old_dir, "*")):
                    os.remove(file)
                os.rmdir(old_dir)
                print(f"🗑️ deleted old state ckpt: {old_dir}")
            except Exception as e:
                print(f"❌ failed to delete {old_dir}: {e}")

    # # RAW
    # raw_dir = os.path.join(args.out_dir, f"step_{step}_raw"); os.makedirs(raw_dir, exist_ok=True)
    # torch.save(unet.state_dict(), os.path.join(raw_dir, "unet_raw.pt"))

    # EMA
    ema_model = UNet2DConditionModel.from_config(unet.config).to(unet.device, dtype=next(unet.parameters()).dtype)
    ema.copy_to(ema_model)
    ema_model = ema_model.to(dtype=torch.float16)
    ema_dir = os.path.join(args.out_dir, f"step_{step}_ema"); os.makedirs(ema_dir, exist_ok=True)
    safetensors_save({k: v.detach().cpu() for k,v in ema_model.state_dict().items()}, os.path.join(ema_dir, "unet_ema.safetensors"))
    with open(os.path.join(ema_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(unet.config, f, indent=2)
    meta = {"prediction_type": prediction_type, "step": step}
    if getattr(args, "train_text_encoder", False):
        safetensors_save({k: v.detach().cpu() for k, v in txt.text_encoder_1.state_dict().items()},
                         os.path.join(ema_dir, "text_encoder_1.safetensors"))
        safetensors_save({k: v.detach().cpu() for k, v in txt.text_encoder_2.state_dict().items()},
                         os.path.join(ema_dir, "text_encoder_2.safetensors"))
        meta["text_encoder_saved"] = True
    with open(os.path.join(ema_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    # 旁车 state.json（便于从 safetensors 恢复）
    with open(os.path.join(ema_dir, "unet_ema.state.json"), "w", encoding="utf-8") as f:
        json.dump({
            "prediction_type": prediction_type,
            "global_step": step,
            "epoch": 0,
            "opt_step": 0,
            "lr_sched_step": lr_sched_step
        }, f, indent=2)
    print(f">> saved ckpt @ {ema_dir}")


# -------------------------
# 预编码：保存 latent，并在 index.jsonl 同时写入
# - caption_tags: 原 CSV 的标签串（保持不变）
# - caption_nl:   用 BLIP2 批量生成的自然语言描述（可选；未开启则写空串）
# - author:       原 CSV 的作者字段
# 仅对 caption_nl 批处理；VAE 单张处理
# -------------------------
def cmd_encode(args):
    import json, os, random
    from PIL import Image
    import numpy as np
    import torch
    from diffusers import AutoencoderKL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    tfm = make_image_transform(args.size)

    base_repo = args.base_repo
    print(">> loading SDXL VAE from:", base_repo)
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        base_repo, subfolder="vae",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device)
    vae.requires_grad_(False).eval()
    scaling_factor = float(getattr(vae.config, "scaling_factor", 0.13025))
    print(f">> VAE scaling_factor={scaling_factor}")

    # ====== BLIP2（可选：仅当 --gen_caption_nl 时加载；批处理仅用于 caption 生成） ======
    blip_enabled = bool(getattr(args, "gen_caption_nl", False))
    blip_model_name = getattr(args, "blip_model", "Salesforce/blip2-opt-2.7b")
    blip_bs = int(getattr(args, "blip_batch", 64))
    blip_processor = None
    blip_model = None

    if blip_enabled:
        try:
            print(f">> loading BLIP2 ({blip_model_name})…")
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            blip_dtype = torch.float16 if device.type == "cuda" else torch.float32
            blip_processor = Blip2Processor.from_pretrained(blip_model_name)
            blip_model = Blip2ForConditionalGeneration.from_pretrained(
                blip_model_name, torch_dtype=blip_dtype
            ).to(device)
            blip_model.eval()
        except Exception as e:
            print(f"[warn] BLIP2 init failed ({e}); fallback to empty caption_nl.")
            blip_enabled = False
            blip_processor = None
            blip_model = None

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
        except Exception:
            continue
    # random.shuffle(samples)

    index_path = os.path.join(args.out_dir, "index.jsonl")
    idxf = open(index_path, "w", encoding="utf-8")

    imgs_batch = []      # 仅用于 BLIP 批处理的 PIL 图像
    metas_batch = []     # (path, caption_tags, author)

    print(f">> encoding {len(samples)} samples… (BLIP enabled={blip_enabled}, blip_batch={blip_bs}; VAE per-image)")

    def _gen_captions_for_batch(pil_list):
        """仅对 caption 做批处理；失败或未启用时返回空串列表。"""
        if not blip_enabled or not pil_list:
            return [""] * len(pil_list)

        try:
            with torch.inference_mode():
                blip_inputs = blip_processor(
                    images=pil_list, return_tensors="pt", padding=True
                ).to(device)
                gen_ids = blip_model.generate(
                    **blip_inputs,
                    max_new_tokens=64,
                    # 需要更强质量可加（更慢）：
                    # no_repeat_ngram_size=3,
                    # repetition_penalty=1.1,
                    # do_sample=True, temperature=0.7, top_p=0.9,
                )
                captions = blip_processor.batch_decode(gen_ids, skip_special_tokens=True)
            # 释放 BLIP 中间张量，避免影响后续 VAE
            del blip_inputs, gen_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return [c.strip() for c in captions]
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[warn] BLIP2 OOM on batch; fallback to per-image.")
        except Exception as e:
            print(f"[warn] BLIP2 batch failed: {e}")

        # 回退到逐张生成 caption（仍然不影响后续 VAE 的逐张处理）
        out = []
        for im in pil_list:
            try:
                with torch.inference_mode():
                    inp1 = blip_processor(images=im, return_tensors="pt").to(device)
                    ids1 = blip_model.generate(**inp1, max_new_tokens=64)
                    cap1 = blip_processor.decode(ids1[0], skip_special_tokens=True).strip()
                del inp1, ids1
            except Exception as e:
                print(f"[warn] BLIP2 single failed: {e}")
                cap1 = ""
            out.append(cap1)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out

    def flush_batch():
        """仅对当前 batch 批量生成 caption；随后逐张做 VAE 编码并写 index。"""
        nonlocal imgs_batch, metas_batch
        if not imgs_batch:
            return

        # 1) 批量生成 caption（只在这里用到 batch）
        captions = _gen_captions_for_batch(imgs_batch)

        # 2) 逐张 VAE 编码 + 写 index.jsonl（严格单张，不做 batch）
        for (im, (path, caption_tags, author), cap) in zip(imgs_batch, metas_batch, captions):
            try:
                pixel = tfm(im).unsqueeze(0).to(
                    device,
                    dtype=(torch.float16 if device.type == "cuda" else torch.float32)
                )
                with torch.no_grad():
                    lat = vae.encode(pixel).latent_dist.sample() * scaling_factor
                base = sha1(path) + ".npz"
                np.savez_compressed(
                    os.path.join(args.out_dir, base),
                    latent=lat[0].detach().cpu().to(torch.float16).numpy(),
                    src=np.bytes_(path)
                )
                meta = {
                    "npz": base,
                    "src": path,
                    "caption_tags": caption_tags,
                    "caption_nl": cap if cap else "",
                    "author": author
                }
                idxf.write(json.dumps(meta, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[skip] encode {path}: {e}")
            finally:
                # 逐张及时释放
                del pixel
                if 'lat' in locals():
                    del lat
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # 3) 清空 batch 容器（保证下一批 caption 生成仍然只占用必要内存）
        imgs_batch.clear()
        metas_batch.clear()

    # --------- 主循环：装批（用于 caption） & 刷批（caption 批 + VAE 单张） ----------
    from tqdm import tqdm
    for (path, caption_tags, author) in tqdm(samples):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[skip] open {path}: {e}")
            continue

        imgs_batch.append(img)                       # 只为 BLIP 批处理收集
        metas_batch.append((path, caption_tags, author))

        if len(imgs_batch) >= blip_bs:
            flush_batch()                            # caption 批量、VAE 单张

    # 收尾
    flush_batch()
    idxf.close()
    print(">> done. Saved to", args.out_dir)



# -------------------------
# 训练（SDXL）
# -------------------------
def cmd_train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.train_text_encoder and not (args.text_encoder_lr < args.lr):
        raise ValueError("--text_encoder_lr must be smaller than --lr")

    # 数据
    index_path = os.path.join(args.data_dir, "index.jsonl")
    dataset = LatentCapAuthorDataset(index_path, args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=(device.type=="cuda"),
                        drop_last=True, persistent_workers=(args.workers>0))

    # UNet（FP32 权重；后续前向用 autocast(fp16)）
    base_repo = args.base_repo
    print(">> loading UNet from:", base_repo)
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(base_repo, subfolder="unet")
    unet = unet.to(device, dtype=torch.float32)
    unet.enable_gradient_checkpointing()
    print("UNet cross_attention_dim:", unet.config.cross_attention_dim)
    print("UNet addition_embed_type:", getattr(unet.config, "addition_embed_type", None))

    # 调度器
    prediction_type = "v_prediction" if args.vpred else "epsilon"
    noise_sched = DDPMScheduler(num_train_timesteps=1000,
                                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                prediction_type=prediction_type)

    # 文本编码器（带 LRU 缓存）
    txt = SDXLTextEnc(
        base_repo,
        device,
        use_amp=True,
        cache_cap=(0 if args.train_text_encoder else args.embed_cache_size),
        train_te=args.train_text_encoder,
    )

    # —— 优化器/EMA/AMP —— #
    unet_params = [p for p in unet.parameters() if p.requires_grad]
    param_groups = [{"params": unet_params, "lr": args.lr}]

    te_params = []
    if args.train_text_encoder:
        te_params = [p for p in itertools.chain(
            txt.text_encoder_1.parameters(),
            txt.text_encoder_2.parameters()
        ) if p.requires_grad]
        param_groups.append({"params": te_params, "lr": args.text_encoder_lr})

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), weight_decay=1e-2)
    ema = EMA(unet, decay=args.ema)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # 合并列表仅用于梯度裁剪（避免重复，用 id 去重）
    train_params_for_clip = list({id(p): p for p in itertools.chain(unet_params, te_params)}.values())

    # LR 调度
    micro_steps_per_epoch = len(loader)
    opt_steps_per_epoch = max(1, math.ceil(micro_steps_per_epoch / max(1, args.grad_accum)))
    total_opt_steps = max(1, args.epochs * opt_steps_per_epoch)
    warmup_opt_steps = args.warmup if args.warmup>0 else max(100, int(0.03*total_opt_steps))
    lr_sched = CosineLRScheduler(optimizer, max_steps=total_opt_steps, warmup_steps=warmup_opt_steps, min_lr_ratio=0.1)

    # 预览（SDXL pipeline；替换 unet；dtype 用新接口）
    pipe = StableDiffusionXLPipeline.from_pretrained(base_repo).to(device)
    pipe.enable_attention_slicing()
    try: pipe.enable_xformers_memory_efficient_attention()
    except: pass
    print(pipe.scheduler.config.prediction_type, "->", prediction_type)
    pipe.scheduler.config.prediction_type = prediction_type

    # 日志/绘图
    plot_interval   = args.plot_interval
    ema_alpha       = args.loss_ema
    csv_path        = os.path.join(args.out_dir, "loss.csv")
    png_path        = os.path.join(args.out_dir, "loss.png")
    MAX_POINTS      = 500000
    loss_steps: deque[int] = deque(maxlen=MAX_POINTS)
    loss_vals:  deque[float] = deque(maxlen=MAX_POINTS)
    ema_vals:   deque[float] = deque(maxlen=MAX_POINTS)
    ema_state = None

    def read_y_range(file_path="range.txt"):
        if not os.path.isfile(file_path):
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                parts = f.read().strip().split()
                if len(parts) != 2:
                    return None
                ymin, ymax = float(parts[0]), float(parts[1])
                if ymin >= ymax:
                    return None
                return (ymin, ymax)
        except Exception:
            return None

    def log_loss(step: int, val: float):
        nonlocal ema_state
        loss_steps.append(step); loss_vals.append(val)
        if ema_alpha >= 1.0:
            ema_vals.append(val)
        else:
            ema_state = val if ema_state is None else (ema_alpha * ema_state + (1 - ema_alpha) * val)
            ema_vals.append(ema_state)

    def flush_csv():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = pycsv.writer(f)
            w.writerow(["step", "loss", f"loss_ema(alpha={ema_alpha})"])
            for s, v, e in zip(loss_steps, loss_vals, ema_vals):
                w.writerow([s, f"{v:.6f}", f"{e:.6f}"])

    def save_plot():
        if not loss_steps: return
        y_range = read_y_range("range.txt")
        plt.figure(figsize=(8,4.5), dpi=150)
        plt.plot(list(loss_steps), list(loss_vals), label="loss", linewidth=1)
        if ema_vals and (ema_alpha < 1.0):
            plt.plot(list(loss_steps), list(ema_vals), label=f"loss EMA (α={ema_alpha})", linewidth=1)
        plt.xlabel("step"); plt.ylabel("loss"); plt.title("Training Loss")
        plt.legend(loc="best"); plt.grid(True, linewidth=0.3)
        if y_range is not None:
            plt.ylim(*y_range)
        plt.tight_layout(); plt.savefig(png_path); plt.close()

    # 预览函数：dtype 对齐
    @torch.no_grad()
    def save_preview(step: int, prompt: str, use_ema: bool = False, size: int = 1024):
        if not prompt:
            return

        # 1) 切 UNet 权重（raw / EMA），并统一 dtype / device
        tmp_dtype = next(pipe.unet.parameters()).dtype
        tmp_unet = UNet2DConditionModel.from_config(unet.config).to(device, dtype=tmp_dtype)
        if use_ema:
            ema.copy_to(tmp_unet)
        else:
            tmp_unet.load_state_dict(unet.state_dict(), strict=True)
        tmp_unet.eval()
        pipe.unet = tmp_unet

        # 2) 正/负向 embeds（不传 dtype；事后 to 到 dtype/device）
        negative = getattr(args, "preview_negative", None)
        do_cfg = float(args.preview_scale) > 1.0

        pe, pooled = txt.encode_prompts([prompt])
        if do_cfg:
            if negative:
                neg_pe, neg_pooled = txt.encode_prompts([negative])
            else:
                neg_pe, neg_pooled = txt.encode_prompts([""])
        else:
            neg_pe = neg_pooled = None

        # 统一 dtype/device
        pe       = pe.to(device=device, dtype=tmp_dtype)
        pooled   = pooled.to(device=device, dtype=tmp_dtype)
        if do_cfg and neg_pe is not None and neg_pooled is not None:
            neg_pe    = neg_pe.to(device=device, dtype=tmp_dtype)
            neg_pooled= neg_pooled.to(device=device, dtype=tmp_dtype)

        # 3) add_time_ids（与尺寸一致；按 batch 扩展）
        try:
            add_time_ids = pipe._get_add_time_ids(
                original_size=(size, size),
                crops_coords_top_left=(0, 0),
                target_size=(size, size),
                dtype=tmp_dtype,
                device=device,
                batch_size=pe.shape[0],
            )
        except TypeError:
            # 老版本兼容：最简 6 维；按 batch repeat
            add_time_ids = torch.tensor(
                [size, size, size, size, 0, 0],
                device=device, dtype=tmp_dtype
            ).unsqueeze(0).repeat(pe.shape[0], 1)

        add_kw = {"text_embeds": pooled, "time_ids": add_time_ids}
        # 只有在 CFG 时才传负向条件/kwargs，避免 None 触发内部断言
        if do_cfg and neg_pooled is not None:
            add_kw_neg = {"text_embeds": neg_pooled, "time_ids": add_time_ids}
        else:
            add_kw_neg = None

        # 4) 随机数发生器
        g = torch.Generator(device=device.type)
        if getattr(args, "preview_seed", None) is not None:
            g = g.manual_seed(args.preview_seed)
        else:
            g = g.manual_seed(torch.seed())

        # 5) 手动传 embeds（不要再传字符串 prompt）
        pipe_kwargs = dict(
            prompt_embeds=pe,
            pooled_prompt_embeds=pooled,
            num_inference_steps=int(args.preview_steps),
            guidance_scale=float(args.preview_scale),
            width=size,
            height=size,
            generator=g,
            added_cond_kwargs=add_kw,
        )
        if do_cfg and neg_pe is not None and add_kw_neg is not None:
            pipe_kwargs.update(
                negative_prompt_embeds=neg_pe,
                negative_pooled_prompt_embeds=neg_pooled,
                negative_added_cond_kwargs=add_kw_neg,
            )

        img = pipe(**pipe_kwargs).images[0]

        # 6) 保存
        out_dir = os.path.join(args.out_dir, "preview")
        os.makedirs(out_dir, exist_ok=True)
        img.save(os.path.join(out_dir, f"step_{step:08d}_{'ema' if use_ema else 'raw'}.png"))
        print(f">> saved preview: step {step} ({'ema' if use_ema else 'raw'})")


    # —— 恢复（可选）
    global_step, start_epoch, pt_pred_type, opt_step = try_load_train_state(
        args.resume, unet, txt, optimizer, lr_sched, ema, scaler
    )
    if pt_pred_type and pt_pred_type != prediction_type:
        print(f"[warn] resume pred_type={pt_pred_type} != current {prediction_type}")

    # 训练
    unet.train()
    optimizer.zero_grad(set_to_none=True)
    VARIANT_NAMES = ["tags_fwd", "tags_rev", "caption_nl", "caption_auth_nl"]
    PROBS = np.array([0.4, 0.2, 0.1, 0.3], dtype=np.float64)

    # —— 调试控制 —— #
    DEBUG = True                    # 全局开关：True 打印；False 安静
    LOG_FIRST_N_STEPS = 3           # 仅前 N 个 step 打印详尽信息
    ABS_MAX_ALERT = 1e6             # 幅值过大阈值（可按需调）

    # —— 辅助函数：统计/打印张量状态 —— #
    def _tensor_stats(t: torch.Tensor):
        with torch.no_grad():
            t2 = torch.nan_to_num(t.detach())
            fin_mask = torch.isfinite(t.detach())
            n = t.numel()
            n_bad = int((~fin_mask).sum().item())
            return {
                "device": str(t.device),
                "dtype": str(t.dtype),
                "shape": tuple(t.shape),
                "min": float(t2.min().item()) if t2.numel() else 0.0,
                "max": float(t2.max().item()) if t2.numel() else 0.0,
                "mean": float(t2.mean().item()) if t2.numel() else 0.0,
                "n_bad": n_bad,
                "n_total": int(n),
            }

    def _log_tensor(name: str, t: torch.Tensor, step: int, force: bool = False):
        need = DEBUG and (force or (step < LOG_FIRST_N_STEPS))
        if not need:
            return
        s = _tensor_stats(t)
        print(f"[step {step}] {name}: device={s['device']} dtype={s['dtype']} shape={s['shape']} "
            f"min={s['min']:.6g} max={s['max']:.6g} mean={s['mean']:.6g} "
            f"nonfinite={s['n_bad']}/{s['n_total']}")
        if abs(s["max"]) > ABS_MAX_ALERT or abs(s["min"]) > ABS_MAX_ALERT:
            print(f"[step {step}] [warn] {name} |min|max 超过阈值（{ABS_MAX_ALERT:g}），请检查幅值是否异常。")

    def _is_finite(name: str, t: torch.Tensor, step: int) -> bool:
        ok = torch.isfinite(t).all().item()
        if not ok:
            _log_tensor(name, t, step, force=True)
            print(f"[step {step}] [NaN] {name} 出现非有限值，跳过该 step。")
        return ok


    # =========================
    # ======= 训练主循环 =======
    # =========================
    for epoch in range(start_epoch, args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            lat, cap_tags, cap_nls, auths = batch
            lat = lat.to(device, dtype=(torch.float16 if use_amp else torch.float32))  # [B,4,128,128]
            B = lat.size(0)

            # 文本变体采样
            chosen = []
            for i in range(B):
                texts_i, mask_i, _ = _build_variants_from_cap_author(cap_tags[i], cap_nls[i], auths[i])
                p = PROBS * mask_i.astype(np.float64)
                s = p.sum()
                if s <= 0:
                    chosen.append("")
                else:
                    p = p / s
                    idx = int(np.random.choice(len(VARIANT_NAMES), p=p))
                    chosen.append(texts_i[idx] if mask_i[idx] else "")

            # --- 编码（含 CFG dropout）---
            pe_list, pooled_list = [], []
            for s in chosen:
                if (args.cfg_drop > 0.0) and (random.random() < args.cfg_drop):
                    pe, pooled = txt.encode_cached("")
                else:
                    pe, pooled = txt.encode_cached(s)
                pe_list.append(pe)
                pooled_list.append(pooled)

            # 与 UNet 保持 dtype/device 一致 + NaN 安全处理
            unet_dtype = next(unet.parameters()).dtype
            prompt_embeds = torch.cat(pe_list, dim=0).to(device=device, dtype=unet_dtype, non_blocking=True).contiguous()  # [B,77,2048]
            pooled_prompt_embeds = torch.cat(pooled_list, dim=0).to(device=device, dtype=unet_dtype, non_blocking=True)   # [B,1280]

            # 将 NaN/Inf 置零，防止传播
            prompt_embeds = torch.nan_to_num(prompt_embeds)
            pooled_prompt_embeds = torch.nan_to_num(pooled_prompt_embeds)

            # SDXL 的 time_ids：尺寸与训练分辨率一致；按 batch 复制 + NaN 安全
            time_ids = build_time_ids(
                args.train_size, args.train_size,
                dtype=unet_dtype, device=device
            ).unsqueeze(0).repeat(prompt_embeds.shape[0], 1)   # [B,6] 或 [B,8]
            time_ids = torch.nan_to_num(time_ids)

            added_cond = {"text_embeds": pooled_prompt_embeds, "time_ids": time_ids}

            # 扩散训练
            t = torch.randint(0, noise_sched.config.num_train_timesteps, (B,), device=device)
            noise = torch.randn_like(lat)

            with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.float16):
                noisy = noise_sched.add_noise(lat, noise, t)
                out = unet(
                    noisy, t,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond
                )
                pred = out.sample

                target = noise if prediction_type == "epsilon" else noise_sched.get_velocity(lat, noise, t)

                # 额外安全：防止 pred/target 中的非有限值影响 loss
                pred = torch.nan_to_num(pred)
                target = torch.nan_to_num(target)

                loss = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))
                if args.min_snr_gamma > 0:
                    snr = compute_snr(noise_sched.alphas_cumprod, t)
                    w = min_snr_weights(snr, gamma=args.min_snr_gamma)
                    loss = (loss * w).mean()
                else:
                    loss = loss.mean()

            # 非有限损失保护：跳过该 step，防止污染优化器
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            loss_scalar = float(loss.detach().cpu())
            log_loss(global_step + 1, loss_scalar)
            if (global_step + 1) % plot_interval == 0:
                flush_csv()
                save_plot()

            # 首次预览
            if getattr(args, "preview_every_ckpt", False) and (global_step == 0):
                texts0, mask0, _ = _build_variants_from_cap_author(cap_tags[0], cap_nls[0], auths[0])
                for i in range(len(texts0)):
                    if mask0[i]:
                        save_preview(global_step + i, texts0[i], use_ema=False, size=args.train_size)
                        save_preview(global_step + i, texts0[i], use_ema=True, size=args.train_size)
                        out_dir = os.path.join(args.out_dir, "preview")
                        os.makedirs(out_dir, exist_ok=True)
                        out_path = os.path.join(out_dir, f"prompt_{global_step + i:08d}.txt")
                        with open(out_path, "w", encoding="utf-8") as f:
                            f.write(texts0[i])

            # 反传 & 优化
            scaler.scale(loss / args.grad_accum).backward()

            if ((global_step + 1) % args.grad_accum) == 0:
                nn.utils.clip_grad_norm_(train_params_for_clip, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr_sched.step()
                ema.update(unet)
                opt_step += 1

            global_step += 1
            pbar.set_postfix({"loss": loss_scalar, "lr": optimizer.param_groups[0]["lr"]})

            # 保存 ckpt + 训练快照 + 预览
            if args.save_steps and (global_step % args.save_steps == 0):
                snap_dir = os.path.join(args.out_dir, f"step_{global_step:08d}_state")
                # 仅保存推理用 ckpt（你之前注释掉了 state 保存）
                save_ckpt(
                    args, unet, ema, txt,
                    step=global_step,
                    prediction_type=prediction_type,
                    lr_sched_step=lr_sched.step_idx
                )
                os.makedirs(snap_dir, exist_ok=True)
                save_train_state(os.path.join(snap_dir, "state.pt"),
                                 unet, txt, optimizer, lr_sched, ema,
                                 global_step=global_step, epoch=epoch,
                                 prediction_type=prediction_type,
                                 scaler=scaler, opt_step=opt_step)

            if getattr(args, "preview_every_ckpt", False) and (global_step % args.preview_save_steps == 0):
                texts0, mask0, _ = _build_variants_from_cap_author(cap_tags[0], cap_nls[0], auths[0])
                for i in range(len(texts0)):
                    if mask0[i]:
                        save_preview(global_step + i, texts0[i], use_ema=False, size=args.train_size)
                        save_preview(global_step + i, texts0[i], use_ema=True, size=args.train_size)
                        out_dir = os.path.join(args.out_dir, "preview")
                        os.makedirs(out_dir, exist_ok=True)
                        out_path = os.path.join(out_dir, f"prompt_{global_step + i:08d}.txt")
                        with open(out_path, "w", encoding="utf-8") as f:
                            f.write(texts0[i])

        if args.save_epochs and ((epoch + 1) % args.save_epochs == 0):
            save_ckpt(
                args, unet, ema, txt,
                step=global_step,
                prediction_type=prediction_type,
                lr_sched_step=lr_sched.step_idx
            )


    print(">> training done.")


# -------------------------
# 推理（替换 UNet）
# -------------------------
def UNNet_from_repo_cfg(base_repo: str, device, dtype):
    cfg_path = Path(base_repo) / "unet" / "config.json"
    if cfg_path.exists():
        j = json.loads(cfg_path.read_text())
        return UNet2DConditionModel.from_config(j).to(device, dtype=dtype)
    # 兜底
    return UNet2DConditionModel.from_pretrained(base_repo, subfolder="unet", torch_dtype=dtype).to(device)

def cmd_infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type=="cuda" else torch.float32
    pipe = StableDiffusionXLPipeline.from_pretrained(args.base_repo, dtype=dtype).to(device)
    pipe.enable_attention_slicing()
    try: pipe.enable_xformers_memory_efficient_attention()
    except: pass

    # 替换 UNet
    print(">> loading trained UNet:", args.unet_path)
    unet = UNNet_from_repo_cfg(args.base_repo, device, next(pipe.unet.parameters()).dtype)
    sd = safetensors_load(args.unet_path) if args.unet_path.endswith(".safetensors") else torch.load(args.unet_path, map_location="cpu")
    missing, unexpected = unet.load_state_dict(sd, strict=False)
    print("missing:", len(missing), "unexpected:", len(unexpected))
    pipe.unet = unet

    pipe.scheduler.config.prediction_type = "v_prediction" if args.vpred else "epsilon"

    # 文本 + 条件
    txt = SDXLTextEnc(args.base_repo, device, use_amp=(dtype==torch.float16), cache_cap=0)
    pe, pooled = txt.encode_prompts([args.prompt])
    if args.negative_prompt:
        neg_pe, neg_pooled = txt.encode_prompts([args.negative_prompt])
    else:
        neg_pe, neg_pooled = txt.uncond_prompt_embeds, txt.uncond_pooled

    tmp_dtype = next(pipe.unet.parameters()).dtype
    pe = pe.to(tmp_dtype); pooled = pooled.to(tmp_dtype)
    neg_pe = neg_pe.to(tmp_dtype); neg_pooled = neg_pooled.to(tmp_dtype)

    time_ids = build_time_ids(args.width, args.height, dtype=tmp_dtype, device=device).unsqueeze(0)
    add_kw     = {"text_embeds": pooled,     "time_ids": time_ids}
    add_kw_neg = {"text_embeds": neg_pooled, "time_ids": time_ids}

    g = torch.Generator(device=device.type)
    if args.seed is not None: g = g.manual_seed(args.seed)

    image = pipe(
        prompt_embeds=pe, negative_prompt_embeds=neg_pe,
        pooled_prompt_embeds=pooled, negative_pooled_prompt_embeds=neg_pooled,
        num_inference_steps=args.steps, guidance_scale=args.scale,
        width=args.width, height=args.height, generator=g,
        **{"added_cond_kwargs": add_kw, "negative_added_cond_kwargs": add_kw_neg}
    ).images[0]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    image.save(args.out)
    print(">> saved", args.out)


# -------------------------
# DECODE：latent -> image（SDXL VAE）
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
     # ✅ 新增：
    pe.add_argument("--gen_caption_nl", action="store_true",
                    help="启用 BLIP2 生成 caption_nl（默认不生成，写空串）")
    pe.add_argument("--blip_batch", type=int, default=4, help="BLIP2 批量大小")
    pe.add_argument("--blip_model", type=str, default="Salesforce/blip2-opt-2.7b",
                    help="BLIP2 模型名")

    # train
    pt = sub.add_parser("train", help="微调 SDXL UNet（双编码器现算，CFG dropout）")
    pt.add_argument("--data_dir", type=str, required=True, help="encode 产生的数据目录（含 index.jsonl 和 .npz）")
    pt.add_argument("--base_repo", type=str, required=True, help="用于对齐结构与 tokenizer/encoder 的基模型 REPO 目录")
    pt.add_argument("--out_dir", type=str, required=True)

    pt.add_argument("--train_size", type=int, default=1024, help="训练/预览分辨率（通常 SDXL=1024）")
    pt.add_argument("--batch_size", type=int, default=2)
    pt.add_argument("--workers", type=int, default=4)
    pt.add_argument("--epochs", type=int, default=1)
    pt.add_argument("--lr", type=float, default=1e-4)
    pt.add_argument("--text_encoder_lr", type=float, default=1e-5,
                    help="text encoder learning rate (must be < --lr)")
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

    # 断点与日志
    pt.add_argument("--resume", type=str, default=None,
                    help="目录或文件：.pt/.pth（完整快照）或 .safetensors（仅UNet；可配旁车 state.json）")
    pt.add_argument("--plot_interval", type=int, default=500, help="每隔多少 step 刷新一次 loss.csv/loss.png")
    pt.add_argument("--loss_ema", type=float, default=0.98, help="loss EMA 平滑系数，1.0 表示关闭平滑")
    pt.add_argument("--embed_cache_size", type=int, default=50000, help="文本嵌入 LRU 缓存上限（按不同文本条目数）")
    pt.add_argument("--train_text_encoder", action="store_true", help="同时训练 Text Encoder")

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
