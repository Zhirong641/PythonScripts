# -*- coding: utf-8 -*-
import os, io, math, json, glob, argparse, random, hashlib
from typing import List
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
from safetensors.torch import save_file as safetensors_save, load_file as safetensors_load

from CSVProcessor import CSVProcessor  # 需提供：返回每行 [path, caption, author]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv as pycsv
from collections import deque, OrderedDict

# -------------------------
# 全局常量（SD1.x惯例）
# -------------------------
VAE_SCALE = 0.18215
MAX_TOKEN = 77
CLIP_MODEL = "openai/clip-vit-large-patch14"
SD15_REPO = "runwayml/stable-diffusion-v1-5"

# 5 桶（去掉空提示）
VARIANT_NAMES = ["tags_fwd", "tags_rev", "author", "both_fwd", "both_rev"]
PROBS_5 = np.array([0.3, 0.15, 0.1, 0.3, 0.15], dtype=np.float64)

# -------------------------
# 图像预处理（Padding到 512x512）
# -------------------------
def make_image_transform(resolution=512):
    def resize_pad(img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = resolution / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BICUBIC)
        from PIL import ImageOps
        pad_w = (resolution - new_w) // 2
        pad_h = (resolution - new_h) // 2
        pad = (pad_w, pad_h, resolution - new_w - pad_w, resolution - new_h - pad_h)
        img = ImageOps.expand(img, border=pad, fill=(0, 0, 0))
        return img

    return transforms.Compose([
        transforms.Lambda(resize_pad),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # [0,1] → [-1,1]
    ])

def save_tensor_image(img_t: torch.Tensor, path: str):
    img = (img_t.clamp(-1, 1) * 0.5 + 0.5)
    img = (img * 255.0).round().byte().cpu().permute(0,2,3,1).numpy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    from PIL import Image
    Image.fromarray(img[0]).save(path)

# -------------------------
# 实用函数
# -------------------------
def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _split_clean_comma_list(s: str) -> List[str]:
    if not s: return []
    items = [x.strip() for x in s.replace("，", ",").split(",")]
    return [x for x in items if x]

def _join_with_comma(items: List[str]) -> str:
    return ", ".join(items)

def _build_variants_from_cap_author(caption: str, author: str):
    """
    基于原始 caption/author 构造 5 类变体文本，并生成可用 mask（去重后）。
    返回 texts(list[str] 长度5), mask(np.bool_ 长度5), preview_text(str)
    """
    tags = _split_clean_comma_list(caption)
    auth = _split_clean_comma_list(author)

    tags_fwd = _join_with_comma(tags) if tags else ""
    tags_rev = _join_with_comma(list(reversed(tags))) if tags else ""
    author_s = _join_with_comma(auth) if auth else ""

    if author_s and tags_fwd:
        both_fwd = f"{author_s}, {tags_fwd}"
    else:
        both_fwd = author_s or tags_fwd

    if author_s and tags_rev:
        both_rev = f"{author_s}, {tags_rev}"
    else:
        both_rev = author_s or tags_rev

    texts = [tags_fwd, tags_rev, author_s, both_fwd, both_rev]
    mask = [bool(t) for t in texts]

    # 去重：后出现者若与前者相同，则屏蔽
    for j in range(len(texts)):
        for k in range(j):
            if mask[j] and mask[k] and texts[j] == texts[k]:
                mask[j] = False

    mask = np.array(mask, dtype=np.bool_)
    preview_text = both_fwd if mask[3] else (tags_fwd if mask[0] else (author_s if mask[2] else ""))

    return texts, mask, preview_text

# -------------------------
# 预编码：只保存 latent，index.jsonl 仅写入 npz/caption/author
# -------------------------
def cmd_encode(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    tfm = make_image_transform(args.size)

    print(">> loading VAE…")
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        SD15_REPO, subfolder="vae", torch_dtype=torch.float16
    ).to(device)
    vae.requires_grad_(False).eval()

    if not args.csv:
        raise ValueError("请使用 --csv data.csv，并保证列为 path,caption,author")
    processor = CSVProcessor(args.csv)
    data = processor.get_data()
    print("CSV rows:", len(data))
    if data:
        print("First row:", data[0])

    # 可选过滤
    exclude_word_list = ["no humans", "chibi", "character profile", "lineart",
                        "sketch", "monochrome", "comic", "text focus", "1990s", "1980s",
                        "retro artstyle", "abstract"]

    # # debug
    # rows = processor.get_rows_by_value("path", "webp/2588678", False)
    # rows = rows[0:1]
    # data = rows

    samples = []
    for row in data:
        try:
            path = row[0]
            caption = row[1] if len(row) > 1 else ""
            author = row[2] if len(row) > 2 else ""
            if any(ex in caption for ex in exclude_word_list):
                continue
            samples.append((path, caption, author))
        except Exception:
            continue
    random.shuffle(samples)

    index_path = os.path.join(args.out_dir, "index.jsonl")
    idxf = open(index_path, "w", encoding="utf-8")

    print(f">> encoding {len(samples)} samples…")
    for (path, caption, author) in tqdm(samples):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[skip] open {path}: {e}")
            continue

        pixel = tfm(img).unsqueeze(0).to(device)  # [1,3,512,512]
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=(device.type=="cuda")):
            lat = vae.encode(pixel).latent_dist.sample() * VAE_SCALE  # [1,4,64,64]

        lat = lat[0].detach().cpu().to(torch.float16).numpy()
        base = sha1(path) + ".npz"
        np.savez_compressed(os.path.join(args.out_dir, base), latent=lat, src=np.bytes_(path))

        # 仅写原始 caption/author（不写任何变体）
        meta = {"npz": base, "src": path, "caption": caption, "author": author}
        idxf.write(json.dumps(meta, ensure_ascii=False) + "\n")

    idxf.close()
    print(">> done. Saved to", args.out_dir)

# -------------------------
# 解码 latent -> 图片（调试用）
# -------------------------
def cmd_decode(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        SD15_REPO, subfolder="vae", torch_dtype=dtype
    ).to(device)
    vae.eval().requires_grad_(False)

    def decode_one(npz_path: str, out_path: str):
        z = np.load(npz_path)
        assert "latent" in z, f"{npz_path} 不含 'latent'"
        lat = torch.from_numpy(z["latent"]).to(device=device, dtype=dtype).unsqueeze(0)  # [1,4,H,W]
        lat = lat / args.scale
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=(dtype==torch.float16), dtype=dtype):
            img = vae.decode(lat).sample  # [-1,1]
        save_tensor_image(img, out_path)
        print(f"decoded -> {out_path}, latent shape: {lat.shape}, image shape: {img.shape}")

    if os.path.isdir(args.input):
        files = sorted([p for p in glob.glob(os.path.join(args.input, "*.npz"))])
        assert files, f"{args.input} 下没有 .npz"
        os.makedirs(args.out_dir, exist_ok=True)
        for p in files:
            base = os.path.splitext(os.path.basename(p))[0]
            out = os.path.join(args.out_dir, base + ".png")
            decode_one(p, out)
    else:
        in_path = args.input
        out_path = args.out or (os.path.splitext(in_path)[0] + ".png")
        decode_one(in_path, out_path)

# -------------------------
# 数据集：读取 latent + 原始 caption/author
# -------------------------
class LatentCapAuthorDataset(Dataset):
    """
    从 index.jsonl 读取 (npz_filename, caption, author)。
    __getitem__ 返回：
      lat: [4,64,64] float16
      caption: str
      author: str
      preview_text: str（基于 caption/author 现算，用于保存预览）
    """
    def __init__(self, index_jsonl: str, root_dir: str):
        self.root = root_dir
        self.items = []
        if not os.path.isfile(index_jsonl):
            raise FileNotFoundError(f"index file not found: {index_jsonl}")
        with open(index_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                    fname = j["npz"]
                    cap = j.get("caption", "")
                    auth = j.get("author", "")
                    full_path = os.path.join(root_dir, fname)
                    if not os.path.isfile(full_path):
                        continue
                    # 为了 DataLoader 预览，预先计算一次 preview_text
                    _, _, preview = _build_variants_from_cap_author(cap, auth)
                    self.items.append((fname, cap, auth, preview))
                except Exception:
                    continue
        print(f"Loaded {len(self.items)} items from {index_jsonl}")

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        fname, cap, auth, preview = self.items[i]
        z = np.load(os.path.join(self.root, fname), allow_pickle=False)
        lat = z["latent"].astype(np.float16)          # (4,64,64)
        return torch.from_numpy(lat), cap, auth, preview

# -------------------------
# EMA
# -------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.detach().clone()
    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
    def copy_to(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])
    def state_dict(self):
        return {"decay": self.decay, "shadow": {k: v.cpu() for k, v in self.shadow.items()}}
    def load_state_dict(self, state):
        self.decay = state["decay"]
        sh = state["shadow"]
        for n, p in self.shadow.items():
            if n in sh:
                self.shadow[n] = sh[n].to(p.device, dtype=p.dtype)

def save_train_state(path, unet, optimizer, lr_sched, ema: EMA, global_step, epoch, prediction_type, scaler=None, opt_step=0):
    pkg = {
        "model": {k: v.detach().cpu() for k, v in unet.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "lr_sched_step": lr_sched.step_idx if hasattr(lr_sched, "step_idx") else 0,
        "ema": ema.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "prediction_type": prediction_type,
        "opt_step": opt_step,
        "scaler": (scaler.state_dict() if (scaler is not None and hasattr(scaler, "state_dict")) else None),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(pkg, path)
    print(f">> saved train state: {path}")

def try_load_train_state(resume_path, unet, optimizer, lr_sched, ema: EMA, scaler=None):
    def _load_pkg(p):
        print(f">> resume from: {p}")
        pkg = torch.load(p, map_location="cpu")
        unet.load_state_dict(pkg["model"], strict=True)
        if "optimizer" in pkg and len(pkg["optimizer"]) > 0:
            optimizer.load_state_dict(pkg["optimizer"])
        if "lr_sched_step" in pkg:
            lr_sched.step_idx = pkg["lr_sched_step"]
        if "ema" in pkg:
            ema.load_state_dict(pkg["ema"])
        if scaler is not None and pkg.get("scaler") is not None:
            try:
                scaler.load_state_dict(pkg["scaler"])
            except Exception as e:
                print(f"[warn] scaler state not restored: {e}")
        return int(pkg.get("global_step", 0)), int(pkg.get("epoch", 0)), pkg.get("prediction_type", "epsilon"), int(pkg.get("opt_step", 0))

    if resume_path is None:
        return 0, 0, None, 0

    if os.path.isdir(resume_path):
        cands = [os.path.join(resume_path, x) for x in os.listdir(resume_path) if x.endswith(".pth") or x.endswith(".pt")]
        if not cands:
            raise FileNotFoundError(f"No *.pth found in {resume_path}")
        cands.sort()
        return _load_pkg(cands[-1])

    if resume_path.endswith(".pth") or resume_path.endswith(".pt"):
        try:
            return _load_pkg(resume_path)
        except Exception:
            print(">> resume file is raw model only; loading UNet weights")
            sd = torch.load(resume_path, map_location="cpu")
            unet.load_state_dict(sd, strict=False)
            return 0, 0, None, 0

    raise FileNotFoundError(resume_path)

# -------------------------
# Min-SNR 权重
# -------------------------
def compute_snr(alphas_cumprod: torch.FloatTensor, timesteps: torch.LongTensor):
    a = alphas_cumprod.to(timesteps.device)[timesteps]  # [B]
    return a / (1.0 - a).clamp(min=1e-8)

def min_snr_weights(snr: torch.FloatTensor, gamma: float):
    return torch.minimum(snr, torch.full_like(snr, gamma)) / (snr + 1.0)

# -------------------------
# 学习率调度（cosine + warmup）
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
            scale = self.min_ratio + 0.5*(1 - self.min_ratio)*(1 + math.cos(math.pi*progress))
        for i, g in enumerate(self.opt.param_groups):
            g["lr"] = self.base_lrs[i] * scale

# -------------------------
# 训练（5 桶 + 训练期现算文本嵌入 + CFG dropout 替代 10% 空提示）
# -------------------------
def cmd_train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")

    os.makedirs(args.out_dir, exist_ok=True)

    # ====== 日志与绘图配置 ======
    plot_interval   = getattr(args, "plot_interval", 500)
    ema_alpha       = getattr(args, "loss_ema", 0.98)
    csv_path        = os.path.join(args.out_dir, "loss.csv")
    png_path        = os.path.join(args.out_dir, "loss.png")

    # ====== 损失历史 ======
    MAX_POINTS = 500000
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
        loss_steps.append(step)
        loss_vals.append(val)
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
        if not loss_steps:
            return
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

    # ====== 数据 ======
    index_path = os.path.join(args.data_dir, "index.jsonl")
    dataset = LatentCapAuthorDataset(index_path, args.data_dir)
    print(">> dataset size:", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=2 if args.workers > 0 else None
    )

    # ====== UNet ======
    print(">> building UNet (SD1.x config)…")
    unet = UNet2DConditionModel.from_config({
        "sample_size": 64, "in_channels": 4, "out_channels": 4,
        "down_block_types": ["CrossAttnDownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D","DownBlock2D"],
        "up_block_types": ["UpBlock2D","CrossAttnUpBlock2D","CrossAttnUpBlock2D","CrossAttnUpBlock2D"],
        "block_out_channels": [320, 640, 1280, 1280],
        "layers_per_block": 2, "cross_attention_dim": 768,
        "mid_block_type": "UNetMidBlock2DCrossAttn"
    }).to(device)
    unet.enable_gradient_checkpointing()

    # ====== 调度器 ======
    prediction_type = "v_prediction" if args.vpred else "epsilon"
    noise_sched = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        prediction_type=prediction_type
    )

    # ====== 优化器 & EMA & AMP ======
    base_lr = args.lr
    optimizer = torch.optim.AdamW(unet.parameters(), lr=base_lr, betas=(0.9, 0.999), weight_decay=1e-2)
    ema = EMA(unet, decay=args.ema)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ====== LR schedule（按优化器步） ======
    micro_steps_per_epoch = len(loader)
    opt_steps_per_epoch   = max(1, math.ceil(micro_steps_per_epoch / max(1, args.grad_accum)))
    total_opt_steps       = max(1, args.epochs * opt_steps_per_epoch)
    warmup_opt_steps = args.warmup if args.warmup > 0 else max(100, int(0.03 * total_opt_steps))
    lr_sched = CosineLRScheduler(optimizer, max_steps=total_opt_steps, warmup_steps=warmup_opt_steps, min_lr_ratio=0.1)

    # ====== 预览图管线 ======
    from diffusers import StableDiffusionPipeline
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(SD15_REPO, torch_dtype=dtype).to(device)
    pipe.safety_checker = None
    pipe.enable_attention_slicing()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    pipe.scheduler.config.prediction_type = prediction_type

    @torch.no_grad()
    def save_preview_image(prompt_text: str, step: int, use_ema: bool = False):
        """
        预览图使用训练期同一套 tokenizer/text_encoder：
        - cond = encode_texts([prompt_text])
        - neg  = encode_texts([args.preview_negative]) 或 uncond_emb（空负面）
        通过 prompt_embeds / negative_prompt_embeds 走管线，避免与 pipe 自带 CLIP 不一致。
        """
        if not prompt_text:
            return
        print(f"DBG: saving preview images for step {step}, prompt: {prompt_text}")

        # 1) 用 RAW / EMA 权重构建临时 UNet 并挂到 pipe 上
        tmp_unet = UNet2DConditionModel.from_config(unet.config).to(device, dtype=dtype)
        if use_ema:
            ema.copy_to(tmp_unet)
        else:
            tmp_unet.load_state_dict(unet.state_dict(), strict=True)
        pipe.unet = tmp_unet

        # 2) 用训练期同一套 CLIP 生成文本嵌入
        cond_emb = encode_texts([prompt_text])                          # [1,77,768]
        neg_text = getattr(args, "preview_negative", "")
        neg_emb  = encode_texts([neg_text]) if neg_text else uncond_emb # [1,77,768]

        # 维度防呆（避免换了 CLIP 维度不一致）
        ca_dim = getattr(unet.config, "cross_attention_dim", None)
        assert ca_dim is None or cond_emb.shape[-1] == ca_dim, \
            f"cond_emb dim {cond_emb.shape[-1]} != UNet cross_attention_dim {ca_dim}"

        # 3) 随机种子
        g = torch.Generator(device=device.type)
        if args.preview_seed is not None:
            g = g.manual_seed(args.preview_seed)

        # 4) 生成并保存
        image = pipe(
            prompt_embeds=cond_emb,
            negative_prompt_embeds=neg_emb,
            num_inference_steps=args.preview_steps,
            guidance_scale=args.preview_scale,
            width=args.preview_size,
            height=args.preview_size,
            generator=g
        ).images[0]

        out_dir = os.path.join(args.out_dir, "preview")
        os.makedirs(out_dir, exist_ok=True)
        prefix = "ema" if use_ema else "raw"
        out_path = os.path.join(out_dir, f"step_{step:08d}_{prefix}.png")
        image.save(out_path)
        print(f">> saved preview: {out_path}")

    # ====== 恢复 ======
    start_step, start_epoch, pt_from_state, start_opt_step = try_load_train_state(args.resume, unet, optimizer, lr_sched, ema, scaler)
    if pt_from_state is not None and pt_from_state != prediction_type:
        print(f"[warn] resume checkpoint pred_type={pt_from_state} != current {prediction_type}")

    # ====== 文本编码器（训练期现算） ======
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL)
    text_encoder = CLIPTextModel.from_pretrained(
        CLIP_MODEL, torch_dtype=torch.float16 if device.type=="cuda" else torch.float32
    ).to(device)
    text_encoder.eval().requires_grad_(False)

    @torch.no_grad()
    def encode_texts(text_list: List[str]) -> torch.Tensor:
        ids = tokenizer(text_list, padding="max_length", max_length=MAX_TOKEN,
                        truncation=True, return_tensors="pt").input_ids.to(device)
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.float16):
            return text_encoder(ids)[0]  # [B,77,768]

    # 预计算 uncond
    uncond_emb = encode_texts([""])  # [1,77,768]

    # 文本嵌入 LRU 缓存
    CACHE_CAP = getattr(args, "embed_cache_size", 50000)
    cache = OrderedDict()
    def get_emb_cached(s: str) -> torch.Tensor:
        if CACHE_CAP <= 0:
            return encode_texts([s])  # 禁用缓存

        if s in cache:
            e_cpu = cache.pop(s); cache[s] = e_cpu
        else:
            e = encode_texts([s])         # on device
            e_cpu = e.detach().cpu()      # 移到 CPU
            if len(cache) >= CACHE_CAP:
                cache.popitem(last=False)
            cache[s] = e_cpu
        return cache[s].to(device, non_blocking=True)  # 用时再搬回 GPU

    def sample_variant_index5(mask_i: np.ndarray) -> int:
        p = PROBS_5 * mask_i.astype(np.float64)
        s = p.sum()
        if s <= 0:
            # 该样本完全无文本 -> 返回任意索引，但稍后会走 uncond
            return int(np.random.choice(5))
        p = p / s
        return int(np.random.choice(5, p=p))

    # ====== 训练循环 ======
    global_step = start_step
    opt_step = start_opt_step
    unet.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            # batch: lat [B,4,64,64], cap(list[str])，auth(list[str])，preview_text(list[str])
            lat, caps, auths, preview_texts = batch
            lat = lat.to(device, dtype=torch.float16)
            B = lat.size(0)

            # —— 为每个样本按 5 桶概率挑一条文本
            chosen_texts = []
            for i in range(B):
                texts_i, mask_i, _ = _build_variants_from_cap_author(caps[i], auths[i])
                idx = sample_variant_index5(mask_i)
                s = texts_i[idx] if (idx < len(texts_i) and mask_i[idx]) else ""
                # print(f"DBG {epoch} chosen idx {idx}: {s}")
                chosen_texts.append(s)

            # —— 批量/缓存编码
            embs_list = []
            for s in chosen_texts:
                if not s:
                    embs_list.append(uncond_emb.expand(1, -1, -1))
                else:
                    embs_list.append(get_emb_cached(s))
            emb = torch.cat(embs_list, dim=0)  # [B,77,768]

            # —— CFG dropout：等价 10% 空提示（可通过 --cfg_drop 调整）
            if args.cfg_drop > 0.0:
                m = (torch.rand(B, device=device) < args.cfg_drop).view(B, 1, 1)
                emb = torch.where(m, uncond_emb.expand(B, -1, -1), emb)

            # —— 扩散训练
            t = torch.randint(0, noise_sched.config.num_train_timesteps, (B,), device=device)
            noise = torch.randn_like(lat)

            with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.float16):
                noisy = noise_sched.add_noise(lat, noise, t)
                pred = unet(noisy, t, encoder_hidden_states=emb).sample

                target = noise if prediction_type == "epsilon" else noise_sched.get_velocity(lat, noise, t)

                loss = F.mse_loss(pred, target, reduction="none").mean(dim=(1,2,3))
                if args.min_snr_gamma > 0:
                    snr = compute_snr(noise_sched.alphas_cumprod, t)
                    w = min_snr_weights(snr, gamma=args.min_snr_gamma)
                    loss = (loss * w).mean()
                else:
                    loss = loss.mean()

            # —— 记录/反传/优化
            loss_scalar = float(loss.detach().cpu())
            log_loss(global_step + 1, loss_scalar)
            if (global_step + 1) % plot_interval == 0:
                flush_csv()
                save_plot()

            scaler.scale(loss / args.grad_accum).backward()

            if (global_step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr_sched.step()
                ema.update(unet)
                opt_step += 1

            global_step += 1
            pbar.set_postfix({"loss": loss_scalar, "lr": optimizer.param_groups[0]["lr"]})

            # 保存（按微步 step 命名）
            if args.save_steps and (global_step % args.save_steps == 0):
                save_ckpt(args, unet, ema, step=global_step, prediction_type=prediction_type)

            # 预览图（用 batch 内第一条的 preview_text）
            if getattr(args, "preview_every_ckpt", False) and global_step % args.preview_save_steps == 0:
                ptxt = preview_texts[0] if isinstance(preview_texts, list) and len(preview_texts) > 0 else ""
                save_preview_image(ptxt, global_step, use_ema=False)
                save_preview_image(ptxt, global_step, use_ema=True)

        # 每个 epoch 结束也保存一次
        if args.save_epochs and ((epoch + 1) % args.save_epochs == 0):
            save_ckpt(args, unet, ema, step=global_step, prediction_type=prediction_type)

    print(">> training done.")

def save_ckpt(args, unet, ema: EMA, step: int, prediction_type: str):
    # 保存当前 raw 权重
    raw_dir = os.path.join(args.out_dir, f"step_{step}_raw")
    os.makedirs(raw_dir, exist_ok=True)
    torch.save(unet.state_dict(), os.path.join(raw_dir, "unet_raw.pt"))

    # 保存 EMA 权重（推荐推理用）
    ema_model = UNet2DConditionModel.from_config(unet.config).to(unet.device)
    ema.copy_to(ema_model)
    ema_dir = os.path.join(args.out_dir, f"step_{step}_ema")
    os.makedirs(ema_dir, exist_ok=True)

    # 用 safetensors 存一份
    state = {k: v.detach().cpu() for k, v in ema_model.state_dict().items()}
    safetensors_save(state, os.path.join(ema_dir, "unet_ema.safetensors"))

    # 保存 config 和 meta.json
    with open(os.path.join(ema_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(unet.config, f, indent=2)
    with open(os.path.join(ema_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"prediction_type": prediction_type, "step": step}, f, indent=2)

    print(f">> saved ckpt @ {ema_dir}")

    # ⛏️ 自动清理旧的 checkpoint（只保留最近10个）
    ckpt_dirs = sorted(glob.glob(os.path.join(args.out_dir, "step_*_ema")), key=os.path.getmtime)
    max_keep = 10
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

# -------------------------
# 推理（替换 UNet）
# -------------------------
def cmd_infer(args):
    from diffusers import StableDiffusionPipeline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(SD15_REPO, torch_dtype=dtype).to(device)
    pipe.safety_checker = None
    pipe.enable_attention_slicing()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    print(">> loading trained UNet…")
    unet = UNet2DConditionModel.from_config(pipe.unet.config).to(device, dtype=dtype)

    if args.unet_path.endswith(".safetensors"):
        sd = safetensors_load(args.unet_path)
    else:
        sd = torch.load(args.unet_path, map_location="cpu")
    missing, unexpected = unet.load_state_dict(sd, strict=False)
    print("missing:", len(missing), "unexpected:", len(unexpected))
    pipe.unet = unet

    if args.vpred:
        pipe.scheduler.config.prediction_type = "v_prediction"
    else:
        pipe.scheduler.config.prediction_type = "epsilon"

    g = torch.Generator(device=device.type)
    if args.seed is not None:
        g = g.manual_seed(args.seed)

    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.scale,
        width=args.width, height=args.height,
        generator=g
    ).images[0]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    image.save(args.out)
    print(">> saved", args.out)

# -------------------------
# CLI
# -------------------------
def build_parser():
    p = argparse.ArgumentParser("SD1.5 UNet Train & Infer (5-way conditioning, runtime text encode)")
    sub = p.add_subparsers(dest="cmd")

    # encode
    pe = sub.add_parser("encode", help="预编码图像为 latent（CSV: path,caption,author），文本仅写入 index.jsonl")
    pe.add_argument("--csv", type=str, required=True, help="CSV with columns: path,caption,author")
    pe.add_argument("--out_dir", type=str, required=True)
    pe.add_argument("--size", type=int, default=512)
    pe.set_defaults(func=cmd_encode)

    # train
    pt = sub.add_parser("train", help="训练 UNet（5 桶采样 + 训练期现算文本嵌入 + CFG dropout）")
    pt.add_argument("--data_dir", type=str, required=True, help="encode 产生的数据目录（含 index.jsonl 和 .npz）")
    pt.add_argument("--out_dir", type=str, required=True)
    pt.add_argument("--batch_size", type=int, default=8)
    pt.add_argument("--workers", type=int, default=4)
    pt.add_argument("--epochs", type=int, default=5)
    pt.add_argument("--lr", type=float, default=2e-4)
    pt.add_argument("--warmup", type=int, default=1000)
    pt.add_argument("--grad_accum", type=int, default=1)
    pt.add_argument("--ema", type=float, default=0.999)
    pt.add_argument("--cfg_drop", type=float, default=0.10, help="无条件 dropout 比例（等价原 10% 空提示）")
    pt.add_argument("--min_snr_gamma", type=float, default=0.0, help=">0 启用 Min-SNR 权重（如 5.0）")
    pt.add_argument("--vpred", action="store_true", help="使用 v-prediction 训练（默认 epsilon）")
    pt.add_argument("--seed", type=int, default=1234)
    pt.add_argument("--save_steps", type=int, default=2000)
    pt.add_argument("--save_epochs", type=int, default=0, help=">0 则每 N 个 epoch 保存一次")
    pt.add_argument("--embed_cache_size", type=int, default=50000, help="训练期文本嵌入 LRU 缓存上限条目数")

    pt.add_argument("--plot_interval", type=int, default=500, help="每隔多少 step 保存一次损失曲线")
    pt.add_argument("--loss_ema", type=float, default=0.98, help="loss EMA 平滑系数，1.0 表示关闭平滑")

    pt.add_argument("--resume", type=str, default=None, help="从保存的训练状态恢复（.pth/.pt 或目录）")

    # 预览图相关参数
    pt.add_argument("--preview_every_ckpt", action="store_true", help="保存权重时同时保存预览图")
    pt.add_argument("--preview_save_steps", type=int, default=2000, help="每 N 个 step 保存一次预览图")
    pt.add_argument("--preview_steps", type=int, default=30, help="预览图推理步数")
    pt.add_argument("--preview_scale", type=float, default=7.5, help="CFG scale")
    pt.add_argument("--preview_seed", type=int, default=12345, help="预览图随机种子")
    pt.add_argument("--preview_size", type=int, default=512, help="预览图边长(正方形)")
    pt.add_argument("--preview_negative", type=str, default="", help="预览图的负面提示词")

    pt.set_defaults(func=cmd_train)

    # infer
    pi = sub.add_parser("infer", help="替换管线 UNet 推理出图")
    pi.add_argument("--unet_path", type=str, required=True, help="训练产出的 EMA/RAW 权重（.safetensors 或 .pt）")
    pi.add_argument("--prompt", type=str, required=True)
    pi.add_argument("--negative_prompt", type=str, default="")
    pi.add_argument("--steps", type=int, default=30)
    pi.add_argument("--scale", type=float, default=7.5)
    pi.add_argument("--width", type=int, default=512)
    pi.add_argument("--height", type=int, default=512)
    pi.add_argument("--seed", type=int, default=None)
    pi.add_argument("--vpred", action="store_true", help="推理用 v-prediction（需与训练对齐）")
    pi.add_argument("--out", type=str, default="./out.png")
    pi.set_defaults(func=cmd_infer)

    # decode
    pd = sub.add_parser("decode", help="将 latent(.npz) 解码为图片（使用 SD1.5 的 VAE）")
    pd.add_argument("--input", type=str, required=True, help="单个 .npz 文件或目录")
    pd.add_argument("--out", type=str, default=None, help="单文件模式的输出路径（.png）")
    pd.add_argument("--out_dir", type=str, default="./decoded", help="目录模式下的输出目录")
    pd.add_argument("--scale", type=float, default=VAE_SCALE, help="latent 缩放（SD1.x=0.18215）")
    pd.set_defaults(func=cmd_decode)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        print("Use one of subcommands: encode | train | infer | decode. Example:\n"
              "  python sd15_unet_runtime_text.py encode --csv data.csv --out_dir ./latent_db\n"
              "  python sd15_unet_runtime_text.py train  --data_dir ./latent_db --out_dir ./ckpts --vpred --min_snr_gamma 5 --preview_every_ckpt\n"
              "  python sd15_unet_runtime_text.py infer  --unet_path ./ckpts/step_20000_ema/unet_ema.safetensors "
              "--prompt 'best quality, 1girl, miko' --vpred --out sample.png\n"
              "  python sd15_unet_runtime_text.py decode --input ./latent_db --out_dir ./decoded")
        return
    args.func(args)

if __name__ == "__main__":
    main()
