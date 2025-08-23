# -*- coding: utf-8 -*-
import os, io, csv, math, json, glob, argparse, random, hashlib
from dataclasses import dataclass
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
from transformers import CLIPTokenizer, CLIPTextModel
from safetensors.torch import save_file as safetensors_save, load_file as safetensors_load

from CSVProcessor import CSVProcessor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
from collections import deque

# -------------------------
# 全局常量（SD1.x惯例）
# -------------------------
VAE_SCALE = 0.18215
MAX_TOKEN = 77
CLIP_MODEL = "openai/clip-vit-large-patch14"
SD15_REPO = "runwayml/stable-diffusion-v1-5"

# # -------------------------
# # 图像预处理（到 512x512）
# # -------------------------
# def make_image_transform(resolution=512):
#     return transforms.Compose([
#         transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
#         transforms.CenterCrop(resolution),
#         transforms.ToTensor(),  # [0,1]
#         transforms.Normalize([0.5], [0.5])  # -> [-1,1]
#     ])

# 图像预处理（Padding到 512x512）
def make_image_transform(resolution=512):
    def resize_pad(img: Image.Image) -> Image.Image:
        # 等比缩放
        w, h = img.size
        scale = resolution / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BICUBIC)

        # Padding 到 resolution×resolution
        pad_w = (resolution - new_w) // 2
        pad_h = (resolution - new_h) // 2
        pad = (pad_w, pad_h, resolution - new_w - pad_w, resolution - new_h - pad_h)
        img = ImageOps.expand(img, border=pad, fill=(0, 0, 0))
        return img

    from PIL import ImageOps
    return transforms.Compose([
        transforms.Lambda(resize_pad),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # [0,1] → [-1,1]
    ])

def save_tensor_image(img_t: torch.Tensor, path: str):
    """
    img_t: [B,3,H,W] in [-1, 1]
    """
    img = (img_t.clamp(-1, 1) * 0.5 + 0.5)  # [-1,1] -> [0,1]
    img = (img * 255.0).round().byte().cpu().permute(0,2,3,1).numpy()  # [B,H,W,3]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    from PIL import Image
    Image.fromarray(img[0]).save(path)
# -------------------------
# 实用函数
# -------------------------
def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# -------------------------
# 预编码子命令
# 输入：CSV（两列：path,caption），或给出目录（caption=空）
# 输出：每张图一个 .npz（latent(4,64,64,fp16) & text_emb(77,768,fp16)），以及 uncond_emb.npy
# -------------------------
def cmd_encode(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    tfm = make_image_transform(args.size)

    print(">> loading VAE & CLIP…")
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(SD15_REPO, subfolder="vae", torch_dtype=torch.float16).to(device)
    vae.requires_grad_(False).eval()
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL)
    text_encoder = CLIPTextModel.from_pretrained(CLIP_MODEL, torch_dtype=torch.float16).to(device)
    text_encoder.requires_grad_(False).eval()

    # 解析数据源
    samples = []
    if args.csv:
        # CSV 文件；必须包含 path, caption 列
        processor = CSVProcessor(args.csv)
        rows = processor.get_rows_by_value("path", "webp/2588678", False)
        rows = rows[0:1]
        print("Matching rows len:", len(rows))
        for row in rows:
            samples.append((row[0], row[1]))
    else:
        # 图像目录；caption 置空
        img_ext = {".jpg",".jpeg",".png",".webp",".bmp"}
        for p in glob.glob(os.path.join(args.data_dir, "**", "*"), recursive=True):
            if os.path.splitext(p)[1].lower() in img_ext:
                samples.append((p, ""))

    # 预先计算 uncond 嵌入
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        ids_uncond = tokenizer([""], padding="max_length", max_length=MAX_TOKEN, truncation=True, return_tensors="pt").input_ids.to(device)
        uncond_emb = text_encoder(ids_uncond)[0].detach().cpu().to(torch.float16).numpy()
        np.save(os.path.join(args.out_dir, "uncond_emb.npy"), uncond_emb)

    index_path = os.path.join(args.out_dir, "index.jsonl")
    idxf = open(index_path, "w", encoding="utf-8")

    print(f">> encoding {len(samples)} samples…")
    for (path, caption) in tqdm(samples):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[skip] open {path}: {e}")
            continue

        pixel = tfm(img).unsqueeze(0).to(device)  # [1,3,512,512]
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            # VAE -> latent
            lat = vae.encode(pixel).latent_dist.sample() * VAE_SCALE  # [1,4,64,64]
            # 文本嵌入
            ids = tokenizer(caption, padding="max_length", max_length=MAX_TOKEN, truncation=True, return_tensors="pt").input_ids.to(device)
            emb = text_encoder(ids)[0]  # [1,77,768]

        lat = lat[0].detach().cpu().to(torch.float16).numpy()
        emb = emb[0].detach().cpu().to(torch.float16).numpy()

        base = sha1(path) + ".npz"
        npz_path = os.path.join(args.out_dir, base)
        np.savez_compressed(npz_path, latent=lat, text_emb=emb, src=path)
        idxf.write(json.dumps({"npz": base, "src": path, "caption": caption}, ensure_ascii=False) + "\n")
    idxf.close()
    print(">> done. Saved to", args.out_dir)


def cmd_decode(args):
    """
    将 latent 解码回图片。
    支持输入：
      - 单个 .npz（包含 'latent' 键）
      - 目录：批量对其中所有 .npz 进行解码
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # 载入 VAE（SD1.5）
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        SD15_REPO, subfolder="vae", torch_dtype=dtype
    ).to(device)
    vae.eval().requires_grad_(False)

    def decode_one(npz_path: str, out_path: str):
        z = np.load(npz_path)
        assert "latent" in z, f"{npz_path} 不含 'latent'"
        lat = torch.from_numpy(z["latent"]).to(device=device, dtype=dtype)  # [4,h,w]
        lat = lat.unsqueeze(0)  # [1,4,H,W]
        # 训练/采样时 latent 乘了 0.18215，这里要除回去
        lat = lat / args.scale

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=(dtype==torch.float16), dtype=dtype):
            img = vae.decode(lat).sample  # [-1,1], [1,3,H*8,W*8]（若H=64则输出512）
        save_tensor_image(img, out_path)
        print(f"decoded -> {out_path}, latent shape: {lat.shape}, image shape: {img.shape}")

    # 输入可以是文件或目录
    if os.path.isdir(args.input):
        files = sorted([p for p in glob.glob(os.path.join(args.input, "*.npz"))])
        assert files, f"{args.input} 下没有 .npz"
        os.makedirs(args.out_dir, exist_ok=True)
        for p in files:
            base = os.path.splitext(os.path.basename(p))[0]
            out = os.path.join(args.out_dir, base + ".png")
            decode_one(p, out)
    else:
        # 单文件
        in_path = args.input
        out_path = args.out
        if out_path is None:
            # 默认输出到同目录同名 .png
            out_path = os.path.splitext(in_path)[0] + ".png"
        decode_one(in_path, out_path)

# -------------------------
# 训练数据集（读 .npz）
# -------------------------
class LatentCLIPDataset(Dataset):
    def __init__(self, index_jsonl: str, root_dir: str):
        self.root = root_dir
        with open(index_jsonl, "r", encoding="utf-8") as f:
            self.items = [json.loads(x) for x in f]
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p = os.path.join(self.root, self.items[i]["npz"])
        z = np.load(p)
        lat = z["latent"].astype(np.float16)          # (4,64,64)
        emb = z["text_emb"].astype(np.float16)        # (77,768)
        return torch.from_numpy(lat), torch.from_numpy(emb)

# -------------------------
# EMA
# -------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.collected_params = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.detach().clone()
    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if not p.requires_grad: 
                continue
            assert n in self.shadow
            self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
    def copy_to(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])
    def state_dict(self):
        # 只存 shadow 权重，体积小、恢复简单
        return {"decay": self.decay, "shadow": {k: v.cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, state):
        self.decay = state["decay"]
        sh = state["shadow"]
        # 保证 key 完整
        for n, p in self.shadow.items():
            if n in sh:
                self.shadow[n] = sh[n].to(p.device, dtype=p.dtype)

def save_train_state(path, unet, optimizer, lr_sched, ema: EMA, global_step, epoch, prediction_type, scaler=None):
    pkg = {
        "model": {k: v.detach().cpu() for k, v in unet.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "lr_sched_step": lr_sched.step_idx if hasattr(lr_sched, "step_idx") else 0,
        "ema": ema.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "prediction_type": prediction_type,
        "scaler": (scaler.state_dict() if (scaler is not None and hasattr(scaler, "state_dict")) else None),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(pkg, path)
    print(f">> saved train state: {path}")

def try_load_train_state(resume_path, unet, optimizer, lr_sched, ema: EMA, scaler=None):
    """
    支持三种输入：
      1) 训练状态包 .pth/.pt（推荐）
      2) 目录：自动找其中的 *.pth（按文件名排序取最后一个）
      3) 仅 raw 模型权重 .pt（只能恢复模型）
    """
    def _load_pkg(p):
        print(f">> resume from: {p}")
        pkg = torch.load(p, map_location="cpu")
        unet.load_state_dict(pkg["model"], strict=True)
        # optimizer / lr / ema / scaler 如果存在就恢复
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
        return int(pkg.get("global_step", 0)), int(pkg.get("epoch", 0)), pkg.get("prediction_type", "epsilon")

    if resume_path is None:
        return 0, 0, None

    if os.path.isdir(resume_path):
        cands = [os.path.join(resume_path, x) for x in os.listdir(resume_path) if x.endswith(".pth") or x.endswith(".pt")]
        if not cands:
            raise FileNotFoundError(f"No *.pth found in {resume_path}")
        cands.sort()
        return _load_pkg(cands[-1])

    # 文件
    if resume_path.endswith(".pth") or resume_path.endswith(".pt"):
        # 尝试按“完整训练包”加载
        try:
            return _load_pkg(resume_path)
        except Exception:
            # 退化为“仅模型权重”
            print(">> resume file is raw model only; loading UNet weights")
            sd = torch.load(resume_path, map_location="cpu")
            unet.load_state_dict(sd, strict=False)
            return 0, 0, None

    raise FileNotFoundError(resume_path)

# -------------------------
# Min-SNR 权重
# weight = min(snr, gamma) / (snr + 1)
# -------------------------
def compute_snr(alphas_cumprod: torch.FloatTensor, timesteps: torch.LongTensor):
    # snr_t = alpha_bar_t / (1 - alpha_bar_t)
    a = alphas_cumprod.to(timesteps.device)[timesteps]  # [B]
    return a / (1.0 - a).clamp(min=1e-8)

def min_snr_weights(snr: torch.FloatTensor, gamma: float):
    # 论文中多种写法；这里选一个常见实现
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
# 训练子命令
# -------------------------
def cmd_train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)

    # ====== 新增：日志与绘图配置（可在 argparse 里配，也可用 getattr 提供默认）======
    plot_interval   = getattr(args, "plot_interval", 500)     # 每隔多少 step 存一次图
    ema_alpha       = getattr(args, "loss_ema", 0.98)          # EMA 平滑系数（0~1，越大越平滑；=1 关闭）
    csv_path        = os.path.join(args.out_dir, "loss.csv")
    png_path        = os.path.join(args.out_dir, "loss.png")

    # ====== 新增：损失历史 ======
    loss_steps: list[int] = []
    loss_vals:  list[float] = []
    ema_vals:   list[float] = []
    ema_state = None

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
        # 追加写入或覆盖保存都可以；这里用覆盖，保证是“最新全集”
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step", "loss", f"loss_ema(alpha={ema_alpha})"])
            for s, v, e in zip(loss_steps, loss_vals, ema_vals):
                w.writerow([s, f"{v:.6f}", f"{e:.6f}"])

    def save_plot():
        if not loss_steps:
            return
        plt.figure(figsize=(8,4.5), dpi=150)
        plt.plot(loss_steps, loss_vals, label="loss", linewidth=1)
        if ema_vals and (ema_alpha < 1.0):
            plt.plot(loss_steps, ema_vals, label=f"loss EMA (α={ema_alpha})", linewidth=1)
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("Training Loss")
        plt.legend(loc="best")
        plt.grid(True, linewidth=0.3)
        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()

    # 数据
    dataset = LatentCLIPDataset(os.path.join(args.data_dir, "index.jsonl"), args.data_dir)
    uncond_emb = torch.from_numpy(np.load(os.path.join(args.data_dir, "uncond_emb.npy"))).to(torch.float16)  # (1,77,768)
    print(">> dataset size:", len(dataset))

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

    # UNet
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

    # scheduler（训练）
    prediction_type = "v_prediction" if args.vpred else "epsilon"
    noise_sched = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        prediction_type=prediction_type
    )

    # 优化器 & EMA & AMP
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)
    ema = EMA(unet, decay=args.ema)
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    # LR schedule
    total_steps = args.epochs * (len(loader))
    lr_sched = CosineLRScheduler(optimizer, max_steps=total_steps, warmup_steps=args.warmup)

    # ===== 恢复训练（如果传了 --resume）=====
    start_step, start_epoch, pt_from_state = try_load_train_state(args.resume, unet, optimizer, lr_sched, ema, scaler)
    if pt_from_state is not None and pt_from_state != prediction_type:
        print(f"[warn] resume checkpoint pred_type={pt_from_state} != current {prediction_type}")

    # 训练循环
    global_step = 0
    unet.train()

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for lat, emb in pbar:
            lat = lat.to(device, dtype=torch.float16)                 # [B,4,64,64], 已含 0.18215 缩放
            emb = emb.to(device, dtype=torch.float16)                 # [B,77,768]

            B = lat.size(0)

            # CFG dropout：按 p_drop 替换为 uncond
            if args.cfg_drop > 0.0:
                mask = (torch.rand(B, device=device) < args.cfg_drop).view(B, 1, 1).to(torch.bool)
                un = uncond_emb.to(device).repeat(B, 1, 1)            # [B,77,768]
                emb = torch.where(mask, un, emb)

            # 采样 t 与噪声
            t = torch.randint(0, noise_sched.config.num_train_timesteps, (B,), device=device)
            noise = torch.randn_like(lat)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                noisy = noise_sched.add_noise(lat, noise, t)

                # 前向
                pred = unet(noisy, t, encoder_hidden_states=emb).sample

                # 目标
                if prediction_type == "epsilon":
                    target = noise
                else:
                    # v = alpha^0.5 * noise - (1-alpha)^0.5 * x0  => diffusers提供转换API
                    target = noise_sched.get_velocity(lat, noise, t)

                loss = F.mse_loss(pred, target, reduction="none")
                loss = loss.mean(dim=(1,2,3))  # [B]

                # Min-SNR reweighting（可选）
                if args.min_snr_gamma > 0:
                    snr = compute_snr(noise_sched.alphas_cumprod, t)  # [B]
                    w = min_snr_weights(snr, gamma=args.min_snr_gamma)
                    loss = (loss * w).mean()
                else:
                    loss = loss.mean()
            
            # ====== 新增：记录 loss & 画图 ======
            loss_scalar = float(loss.detach().cpu())
            log_loss(global_step + 1, loss_scalar)
            if (global_step + 1) % plot_interval == 0:
                flush_csv()
                save_plot()

            scaler.scale(loss / args.grad_accum).backward()
            # (loss / args.grad_accum).backward()

            # 梯度累积
            if (global_step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                # optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_sched.step()
                ema.update(unet)

            global_step += 1
            pbar.set_postfix({"loss": float(loss.detach().cpu()), "lr": optimizer.param_groups[0]["lr"]})

            # 保存
            if args.save_steps and (global_step % args.save_steps == 0):
                save_ckpt(args, unet, ema, step=global_step, prediction_type=prediction_type)

        # 每个 epoch 结束也保存一次
        if args.save_epochs and (epoch + 1) % args.save_epochs == 0:
            save_ckpt(args, unet, ema, step=global_step, prediction_type=prediction_type)

    print(">> training done.")

def save_ckpt(args, unet, ema: EMA, step: int, prediction_type: str):
    # 保存当前权重
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

    # 也存一下 config 和一个简单的元数据（包含prediction_type）
    with open(os.path.join(ema_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(unet.config, f, indent=2)
    with open(os.path.join(ema_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"prediction_type": prediction_type, "step": step}, f, indent=2)
    print(f">> saved ckpt @ {ema_dir}")

# -------------------------
# 推理子命令（diffusers 管线，替换 UNet）
# -------------------------
def cmd_infer(args):
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 载入管线
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(SD15_REPO, torch_dtype=dtype).to(device)
    pipe.safety_checker = None  # 可选
    pipe.enable_attention_slicing()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    # 加载自训 UNet
    print(">> loading trained UNet…")
    unet = UNet2DConditionModel.from_config(pipe.unet.config).to(device, dtype=dtype)

    # 既支持 .safetensors 也支持 .pt
    if args.unet_path.endswith(".safetensors"):
        sd = safetensors_load(args.unet_path)
    else:
        sd = torch.load(args.unet_path, map_location="cpu")
    missing, unexpected = unet.load_state_dict(sd, strict=False)
    print("missing:", len(missing), "unexpected:", len(unexpected))
    pipe.unet = unet

    # scheduler 配置要和训练对齐
    if args.vpred:
        pipe.scheduler.config.prediction_type = "v_prediction"
    else:
        pipe.scheduler.config.prediction_type = "epsilon"

    # 出图
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
    p = argparse.ArgumentParser("SD1.5 UNet Train & Infer (single file)")
    sub = p.add_subparsers(dest="cmd")

    # encode
    pe = sub.add_parser("encode", help="预编码图像为 latent & 文本嵌入")
    pe.add_argument("--data_dir", type=str, default=None, help="图像目录（若不提供 --csv）")
    pe.add_argument("--csv", type=str, default=None, help="CSV with columns: path,caption")
    pe.add_argument("--out_dir", type=str, required=True)
    pe.add_argument("--size", type=int, default=512)
    pe.set_defaults(func=cmd_encode)

    # train
    pt = sub.add_parser("train", help="训练 UNet（仅 latent 空间）")
    pt.add_argument("--data_dir", type=str, required=True, help="encode 产生的数据目录（含 index.jsonl/uncond_emb.npy）")
    pt.add_argument("--out_dir", type=str, required=True)
    pt.add_argument("--batch_size", type=int, default=8)
    pt.add_argument("--workers", type=int, default=4)
    pt.add_argument("--epochs", type=int, default=5)
    pt.add_argument("--lr", type=float, default=2e-4)
    pt.add_argument("--warmup", type=int, default=1000)
    pt.add_argument("--grad_accum", type=int, default=1)
    pt.add_argument("--ema", type=float, default=0.999)
    pt.add_argument("--cfg_drop", type=float, default=0.1, help="Classifier-Free Guidance 训练时的 unconditional dropout 概率")
    pt.add_argument("--min_snr_gamma", type=float, default=0.0, help=">0 启用 Min-SNR 权重（如 5.0）")
    pt.add_argument("--vpred", action="store_true", help="使用 v-prediction 训练（默认 epsilon）")
    pt.add_argument("--seed", type=int, default=1234)
    pt.add_argument("--save_steps", type=int, default=2000)
    pt.add_argument("--save_epochs", type=int, default=0, help=">0 则每 N 个 epoch 保存一次")

    pt.add_argument("--plot_interval", type=int, default=500, help="每隔多少 step 保存一次损失曲线")
    pt.add_argument("--loss_ema", type=float, default=0.98, help="loss EMA 平滑系数，1.0 表示关闭平滑")

    pt.add_argument("--resume", type=str, default=None, help="从保存的训练状态恢复（.pth/.pt 或目录）")

    pt.set_defaults(func=cmd_train)

    # infer
    pi = sub.add_parser("infer", help="替换管线 UNet 推理出图")
    pi.add_argument("--unet_path", type=str, required=True, help="训练产出的 EMA 权重（.safetensors 或 .pt）")
    pi.add_argument("--prompt", type=str, required=True)
    pi.add_argument("--negative_prompt", type=str, default="")
    pi.add_argument("--steps", type=int, default=30)
    pi.add_argument("--scale", type=float, default=7.5)
    pi.add_argument("--width", type=int, default=512)
    pi.add_argument("--height", type=int, default=512)
    pi.add_argument("--seed", type=int, default=None)
    pi.add_argument("--vpred", action="store_true", help="推理用 v-prediction（需与训练对齐）")
    pi.add_argument("--out", type=str, default="out.png")
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
        print("Use one of subcommands: encode | train | infer. Example:\n"
              "  python sd15_unet_train_infer.py encode --csv data.csv --out_dir ./latent_db\n"
              "  python sd15_unet_train_infer.py train --data_dir ./latent_db --out_dir ./ckpts --vpred --min_snr_gamma 5\n"
              "  python sd15_unet_train_infer.py infer --unet_path ./ckpts/step_20000_ema/unet_ema.safetensors "
              "--prompt 'best quality, 1girl, miko' --vpred --out sample.png")
        return
    args.func(args)

if __name__ == "__main__":
    main()
