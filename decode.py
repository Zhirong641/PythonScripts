# decode.py
import sys, os
import torch
import numpy as np
from PIL import Image
from diffusers import AutoencoderTiny

device = "cuda" if torch.cuda.is_available() else "cpu"

def decode_latent(model, input_path, output_path):
    latent_np = np.load(input_path)  # [4, 64, 64]
    latent = torch.tensor(latent_np).unsqueeze(0).to(device)

    with torch.no_grad():
        recon = model.decode(latent).sample.clamp(0, 1)
    img_np = (recon.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(img_np).save(output_path)
    print(f"🖼️ Saved image: {output_path}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python decoder.py <input_npy_file_or_dir> <output_image_file_or_dir>")
        return
    print("Using device:", device)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", cache_dir="./taesd_model", torch_dtype=torch.float32).to(device)
    vae.eval()

    if os.path.isfile(input_path):
        fname = os.path.splitext(os.path.basename(input_path))[0]
        out_img = output_path
        decode_latent(vae, input_path, out_img)
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
