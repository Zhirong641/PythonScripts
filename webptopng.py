from PIL import Image, ImageOps
import os

# 定义源目录和目标目录
source_dir = "/mnt/shared/data/webp"  # 替换为包含 .webp 图片的根目录
target_dir = "/mnt/shared/data/128x128_png"  # 替换为保存 .png 图片的根目录
list_file = "/mnt/shared/source/WebpToPng/build/failed_files.txt.old"
target_size = (128, 128)  # 目标输出尺寸，宽和高相等

def convert_images_to_square(source_path, target_path, size):
    # 确保目标目录存在，如果不存在则创建
    os.makedirs(target_path, exist_ok=True)
    count = 0  # 用于计数转换的图片数量
    
    # 遍历源目录中的所有文件和子目录
    for root, dirs, files in os.walk(source_path):
        for file in files:
            if file.endswith(".webp"):
                count += 1
                # 构建完整的源文件路径
                webp_path = os.path.join(root, file)
                
                # 构建目标文件路径，保持与源目录结构一致
                relative_path = os.path.relpath(root, source_path)
                target_subdir = os.path.join(target_path, relative_path)
                os.makedirs(target_subdir, exist_ok=True)
                png_filename = os.path.splitext(file)[0] + ".png"
                png_path = os.path.join(target_subdir, png_filename)
                if os.path.exists(png_path):
                    print(f"Skipping {png_path}, already exists.")
                    continue
                
                # 打开 .webp 图片
                img = Image.open(webp_path)
                
                # 保持宽高比并调整为正方形，使用黑色填充
                img_square = ImageOps.pad(img, size, color=(0, 0, 0))

                # 保存为 .png
                img_square.save(png_path, "PNG")
                print(f"{count}. Converted {webp_path} to {png_path} with square size {size}")

def convert_images_from_list(list_file, source_path, target_path, size):
    # 确保目标目录存在
    os.makedirs(target_path, exist_ok=True)

    # 读取 txt 文件中所有路径
    with open(list_file, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]

    count = 0  # 计数

    for webp_path in image_paths:
        if not webp_path.endswith(".webp"):
            continue
        if not os.path.exists(webp_path):
            print(f"❌ 跳过 {webp_path}，文件不存在。")
            continue

        count += 1

        # 计算相对路径
        relative_path = os.path.relpath(os.path.dirname(webp_path), source_path)
        target_subdir = os.path.join(target_path, relative_path)
        os.makedirs(target_subdir, exist_ok=True)

        png_filename = os.path.splitext(os.path.basename(webp_path))[0] + ".png"
        png_path = os.path.join(target_subdir, png_filename)

        if os.path.exists(png_path):
            print(f"Skipping {png_path}, already exists.")
            continue

        try:
            img = Image.open(webp_path)
            img_square = ImageOps.pad(img, size, color=(0, 0, 0))
            img_square.save(png_path, "PNG")
            print(f"{count}. Converted {webp_path} -> {png_path}")
        except Exception as e:
            print(f"❌ 处理 {webp_path} 时出错: {e}")

# 调用函数进行转换
# convert_images_to_square(source_dir, target_dir, target_size)
convert_images_from_list(list_file, source_dir, target_dir, target_size)

print("All images have been converted to square size.")
