# 一次性导出
from diffusers import UNet2DConditionModel
unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet")
import torch, os
os.makedirs("./sdxl_unet_init", exist_ok=True)
torch.save(unet.state_dict(), "./sdxl_unet_init/unet_raw.pt")
