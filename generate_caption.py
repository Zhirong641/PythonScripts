from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).cuda()

img = Image.open("/mnt/shared/data/webp/3422304/image_96.webp").convert("RGB")
inputs = processor(images=img, return_tensors="pt").to("cuda", torch.float16)
out = model.generate(**inputs, max_new_tokens=77)
caption = processor.decode(out[0], skip_special_tokens=True)
print(caption)
