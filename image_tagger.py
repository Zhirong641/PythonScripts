# Image Tagger using DeepDanbooru
import os
import glob
import deepdanbooru as dd
import tensorflow as tf
from PIL import Image

# Configuration
image_dir = '/mnt/shared/webp_250722'               # Image directory
output_dir = '/mnt/shared/tag_250722'               # Tag output directory
model_dir = '/mnt/shared/deepdanbooru-v3-20211112-sgd-e28'  # DeepDanbooru model directory
threshold = 0.5                                     # Threshold

os.makedirs(output_dir, exist_ok=True)

# Load model
print('Loading model...')
model = dd.project.load_model_from_project(model_dir, compile_model=False)
tags = dd.project.load_tags_from_project(model_dir)
width = model.input_shape[2]
height = model.input_shape[1]

# Get image list
count = 0
valid_exts = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']

# Traverse all image files in the specified directory
image_files = [
    f for f in glob.glob(os.path.join(image_dir, '**', '*'), recursive=True)
    if os.path.isfile(f) and os.path.splitext(f)[1].lower() in valid_exts
]

# Read image_files list from file
# file_list_path = '/mnt/shared/list_no_numbers.txt'
# if os.path.exists(file_list_path):
#     with open(file_list_path, 'r', encoding='utf-8') as f:
#         image_files = [line.strip() for line in f if line.strip() and os.path.isfile(line.strip())]
# else:
#     print(f'File list {file_list_path} does not exist.')
#     image_files = []

print(f'Found {len(image_files)} images.')
input("Press any key to start processing...")

for image_path in image_files:
    try:
        relative_path = os.path.relpath(image_path, image_dir)
        output_path = os.path.join(output_dir, os.path.splitext(relative_path)[0] + '.txt')
        output_folder = os.path.dirname(output_path)
        os.makedirs(output_folder, exist_ok=True)
        
        count += 1
        print(f'{count}. Processing {relative_path}...')
        ext = os.path.splitext(image_path)[1].lower()
        if ext == '.webp':
            # Use PIL to process webp animation, only take the first frame
            with Image.open(image_path) as im:
                im.seek(0)  # Only process the first frame
                im = im.convert('RGB')
                im = im.resize((width, height), Image.LANCZOS)
                image = tf.keras.preprocessing.image.img_to_array(im) / 255.0
        else:
            image = dd.data.load_image_for_evaluate(image_path, width=width, height=height)
        image = image.reshape((1, height, width, 3))
        # # Save the processed image as PNG
        # save_png_path = 'test.png'
        # im_to_save = Image.fromarray((image[0] * 255).astype('uint8'))
        # im_to_save.save(save_png_path)
        # input("Press Enter to continue...")
        # continue
        
        y = model.predict(image)[0]
        
        result_lines = []
        for i, tag in enumerate(tags):
            if y[i] >= threshold:
                result_lines.append(f'{tag} {y[i]:.3f}')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(result_lines))
        
        # print(f'Saved tags to {output_path}')
    except Exception as e:
        print(f'Error processing {image_path}: {e}')
        with open('not_processed_list.txt', 'a', encoding='utf-8') as error_log:
            error_log.write(f'{image_path}\n')

print('All done.')
