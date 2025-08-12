import sys, os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL

device = "cuda" if torch.cuda.is_available() else "cpu"
supported_ext = [".jpg", ".jpeg", ".png", ".webp"]
MODEL_ID = "stabilityai/sd-vae-ft-mse"  # 更高保真；SDXL 可改为 "stabilityai/sdxl-vae"

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def to_multiple_of_8_wh(img: Image.Image):
    # VAE 下采样因子为 8；不是 8 的倍数也能跑，但建议对齐以得出“整齐”的 latent 尺寸
    w, h = img.size
    nw, nh = (w // 8) * 8, (h // 8) * 8
    if nw == 0 or nh == 0:
        nw = max(8, nw); nh = max(8, nh)
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), Image.BICUBIC)
    return img

def encode_image(vae: AutoencoderKL, input_path, output_path, use_align=True):
    img = Image.open(input_path).convert("RGB")
    if use_align:
        img = to_multiple_of_8_wh(img)
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)  # [B,3,H,W], 0..1
    x = tensor * 2.0 - 1.0  # [-1,1], KL-VAE 期望输入

    sf = float(getattr(vae.config, "scaling_factor", 0.18215))  # SD1.x=0.18215, SDXL=0.13025

    with torch.no_grad():
        posterior = vae.encode(x).latent_dist
        z = posterior.sample() * sf  # 按 scaling_factor 缩放后的 latent
        latent = z.squeeze(0).cpu().numpy()  # [4, H/8, W/8]

    ensure_dir(output_path)
    np.save(output_path, latent)
    print(f"✅ Saved latent: {output_path}.npy | shape={latent.shape}, sf={sf}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python encode.py <input_image_or_dir> <output_latent_or_dir>")
        return
    print("Using device:", device)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    vae = AutoencoderKL.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)
    vae.eval()
    print(f"Loaded VAE: {MODEL_ID} | scaling_factor={getattr(vae.config, 'scaling_factor', 'N/A')}")

    if os.path.isfile(input_path):
        # 单文件：output_path 可是文件名（不带 .npy 也行，np.save 会补）
        encode_image(vae, input_path, output_path)
    else:
        # 目录：保持相对目录结构，前缀 latent_
        for root, _, files in os.walk(input_path):
            for file in files:
                if not any(file.lower().endswith(ext) for ext in supported_ext):
                    continue
                in_file = os.path.join(root, file)
                rel_path = os.path.relpath(in_file, input_path)
                base_name = os.path.splitext(rel_path)[0]
                out_path = os.path.join(output_path, f"latent_{base_name}")
                encode_image(vae, in_file, out_path)

if __name__ == "__main__":
    main()
