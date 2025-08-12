import sys, os
import torch
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "stabilityai/sd-vae-ft-mse"  # 与 encode 使用同一类 VAE

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def decode_latent(vae: AutoencoderKL, input_path, output_path):
    latent_np = np.load(input_path)  # [4, H/8, W/8]
    z = torch.tensor(latent_np).unsqueeze(0).to(device)  # [1,4,h,w]

    sf = float(getattr(vae.config, "scaling_factor", 0.18215))

    with torch.no_grad():
        recon = vae.decode(z / sf).sample  # 先去缩放
        recon = (recon / 2.0 + 0.5).clamp(0, 1)  # 回到 [0,1]
    img_np = (recon.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    ensure_dir(output_path)
    Image.fromarray(img_np).save(output_path)
    print(f"🖼️ Saved image: {output_path} | from {os.path.basename(input_path)} | sf={sf}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python decode.py <input_npy_file_or_dir> <output_image_file_or_dir>")
        return
    print("Using device:", device)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    vae = AutoencoderKL.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)
    vae.eval()
    print(f"Loaded VAE: {MODEL_ID} | scaling_factor={getattr(vae.config, 'scaling_factor', 'N/A')}")

    if os.path.isfile(input_path):
        decode_latent(vae, input_path, output_path)
    else:
        for root, _, files in os.walk(input_path):
            for file in files:
                if not file.endswith(".npy"):
                    continue
                in_file = os.path.join(root, file)
                rel_path = os.path.relpath(in_file, input_path)
                base_name = os.path.splitext(rel_path)[0]
                out_path = os.path.join(output_path, f"{base_name}.png")
                decode_latent(vae, in_file, out_path)

if __name__ == "__main__":
    main()
