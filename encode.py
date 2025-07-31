# encode.py
import sys, os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderTiny

device = "cuda" if torch.cuda.is_available() else "cpu"
supported_ext = [".jpg", ".jpeg", ".png", ".webp"]

def encode_image(model, input_path, output_path):
    img = Image.open(input_path).convert("RGB") #.resize((512, 512))
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        latent = model.encode(tensor).latents.squeeze(0).cpu().numpy()  # [4, 64, 64]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, latent)
    print(f"✅ Saved latent: {output_path}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python encode.py <input_image_or_dir> <output_latent_or_dir>")
        return
    print("Using device:", device)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", cache_dir = "./taesd_model", torch_dtype=torch.float32).to(device)
    vae.eval()

    if os.path.isfile(input_path):
        fname = os.path.splitext(os.path.basename(input_path))[0]
        encode_image(vae, input_path, output_path)
    else:
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
