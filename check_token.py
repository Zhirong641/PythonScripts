from transformers import CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

caption = "single hair intake, school uniform, snowflakes, 1girl, red hair, long hair, open mouth, smile, braid, skirt, ponytail, hair bow, copyright name, ribbon, bow, juliet sleeves, long sleeves, night, puffy sleeves, outdoors, standing, :d, snow, white skirt, company name, blue ribbon, blush, frills, solo, pleated skirt, tree, dutch angle, breasts, very long hair, black bow, looking at viewer, winter, fang, black ribbon, standing on one leg, hair intakes, snowing, sky, neck ribbon, waving, leg up, starry sky, frilled skirt, miniskirt, hair ribbon, bare tree, blue eyes, star (sky), single braid, blue bow, hand up, floating hair, shirt"
tokens = tokenizer(caption, truncation=False, return_tensors="pt")
print("token 数:", tokens.input_ids.shape[1])

# from diffusers import StableDiffusionPipeline
# StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", cache_dir="./sd15_base")