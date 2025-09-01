# 一次性导出
from diffusers import UNet2DConditionModel
unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
import torch, os
os.makedirs("./sd15_unet_init", exist_ok=True)
torch.save(unet.state_dict(), "./sd15_unet_init/unet_raw.pt")
