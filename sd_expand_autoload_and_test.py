# -*- coding: utf-8 -*-
import argparse, os, json, time, sys
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    AutoencoderKL,
)
from pathlib import Path

def expand_single_file_to_repo(model_file: str, repo_out: str, fp16: bool):
    """
    用 from_single_file 加载单文件并保存为 REPO 结构（幂等：会覆盖同名目录）
    返回保存后的 repo 路径
    """
    from diffusers import StableDiffusionXLPipeline  # 仅为加载 from_single_file
    dtype = torch.float16 if fp16 else torch.float32
    print(f"[1/3] 从单文件加载: {model_file}")
    pipe = StableDiffusionXLPipeline.from_single_file(
        model_file,
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=False,
        # 一些社区混合模型可能需要放开尺寸校验（必要时取消注释）：
        # ignore_mismatched_sizes=True,
    )
    os.makedirs(repo_out, exist_ok=True)
    print(f"[2/3] 保存为 REPO 结构: {repo_out}")
    pipe.save_pretrained(repo_out, safe_serialization=True)
    del pipe
    torch.cuda.empty_cache()
    print("[3/3] 展开完成")
    return repo_out

def detect_model_type(repo_path: str) -> str:
    """
    通过 model_index.json / 目录结构检测模型类型
    返回值：'sdxl' 或 'sd15'
    """
    mi = Path(repo_path) / "model_index.json"
    if not mi.exists():
        # 兜底：看是否存在 text_encoder_2 目录
        if (Path(repo_path) / "text_encoder_2").exists():
            return "sdxl"
        return "sd15"

    with open(mi, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 典型 SDXL: 有 text_encoder_2 / tokenizer_2 等；或者 _class_name 包含 XL
    # 尽量稳妥地多重判断
    submodels = data.get("_expected_submodels", []) or list(data.keys())
    class_name = data.get("_class_name", "") or data.get("_class_name", "")
    has_te2 = any("text_encoder_2" in str(x) for x in submodels) or (Path(repo_path) / "text_encoder_2").exists()
    if "XL" in class_name or has_te2:
        return "sdxl"
    return "sd15"

def maybe_enable_speedups(pipe):
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print(">> xFormers 已启用")
    except Exception:
        print(">> xFormers 未启用（可选）")
    pipe.enable_attention_slicing()
    # 显存特别吃紧时再启用（会更慢）：
    # pipe.enable_model_cpu_offload()

def auto_default_hw(model_type: str, width: int, height: int):
    """
    若未显式指定分辨率（传入 <=0），按模型类型给默认分辨率
    """
    if width > 0 and height > 0:
        return width, height
    if model_type == "sdxl":
        return 1024, 1024
    else:
        return 512, 512

def load_pipeline(repo_out: str, model_type: str, fp16: bool):
    dtype = torch.float16 if fp16 else torch.float32
    if model_type == "sdxl":
        print(">> 识别为 SDXL，使用 StableDiffusionXLPipeline 加载")
        pipe = StableDiffusionXLPipeline.from_pretrained(repo_out, torch_dtype=dtype)
    else:
        print(">> 识别为 SD1.5，使用 StableDiffusionPipeline 加载")
        pipe = StableDiffusionPipeline.from_pretrained(repo_out, torch_dtype=dtype)
    return pipe

def run_generate(repo_out: str, args):
    model_type = detect_model_type(repo_out)
    pipe = load_pipeline(repo_out, model_type, args.fp16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    maybe_enable_speedups(pipe)

    w, h = auto_default_hw(model_type, args.width, args.height)
    if args.seed >= 0:
        g = torch.Generator(device=device).manual_seed(args.seed)
    else:
        g = None

    os.makedirs(args.out_dir, exist_ok=True)

    # 统一的推理参数；SDXL 与 SD1.5 都支持这些字段
    print(f"[gen] type={model_type} | size={w}x{h} | steps={args.steps} | cfg={args.cfg}")
    print(f"[gen] prompt: {args.prompt}")
    if args.negative_prompt:
        print(f"[gen] neg   : {args.negative_prompt}")

    out = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        width=w,
        height=h,
        num_images_per_prompt=args.num_images,
        generator=g,
    )
    images = out.images

    ts = time.strftime("%Y%m%d_%H%M%S")
    paths = []
    for i, im in enumerate(images):
        fn = os.path.join(args.out_dir, f"{model_type}_{ts}_{i+1:02d}.png")
        im.save(fn)
        paths.append(fn)

    print("✅ 生成完成：")
    for p in paths:
        print(" -", p)

def main():
    ap = argparse.ArgumentParser("Expand .safetensors/.ckpt to REPO, auto-detect SDXL/SD1.5, and test-generate")
    ap.add_argument("--model_file", type=str, help="单文件权重路径（.safetensors/.ckpt）")
    ap.add_argument("--repo_out", type=str, default="./expanded_repo", help="展开/已有的 REPO 目录")
    ap.add_argument("--expand_only", action="store_true", help="只展开不出图（需要提供 --model_file）")
    ap.add_argument("--skip_expand", action="store_true", help="跳过展开，直接从 --repo_out 读取（当已是 REPO）")
    ap.add_argument("--fp16", action="store_true", help="使用 FP16（GPU 推荐）")

    # 推理参数
    ap.add_argument("--prompt", type=str, default="a cinematic anime-style girl, soft lighting, highly detailed, masterpiece")
    ap.add_argument("--negative_prompt", type=str, default="low quality, worst quality, blurry, extra fingers")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=5.5)
    ap.add_argument("--width", type=int, default=-1, help="未指定或<=0时按模型类型给默认: SD1.5→512, SDXL→1024")
    ap.add_argument("--height", type=int, default=-1)
    ap.add_argument("--num_images", type=int, default=1)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out_dir", type=str, default="./samples")

    args = ap.parse_args()

    if not args.skip_expand:
        if not args.model_file:
            print("ERROR: 未提供 --model_file（当不使用 --skip_expand 时必须提供）", file=sys.stderr)
            sys.exit(1)
        expand_single_file_to_repo(args.model_file, args.repo_out, args.fp16)
        if args.expand_only:
            print("完成：仅展开未出图（--expand_only）")
            return

    # 直接从 REPO 出图
    run_generate(args.repo_out, args)

if __name__ == "__main__":
    main()
