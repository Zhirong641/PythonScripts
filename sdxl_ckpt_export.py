#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sdxl_ckpt_export.py
- 组件合并为单文件 .safetensors（fp16/bf16/fp32）
- 可把微调组件注入 SDXL 基座并导出完整 pipeline
- 默认不使用 variant 后缀，直接写默认文件名，避免加载时还要传 variant
"""
import argparse, json, os, sys, shutil, glob
from pathlib import Path
from typing import Optional

import torch
from diffusers import UNet2DConditionModel, AutoencoderKL, StableDiffusionXLPipeline

# 文本编码器来自 transformers
try:
    from transformers import CLIPTextModel, CLIPTextModelWithProjection
except Exception:
    CLIPTextModel = None
    CLIPTextModelWithProjection = None


def parse_dtype(s: str) -> torch.dtype:
    s = s.lower()
    if s == "fp16":
        return torch.float16
    if s == "bf16":
        return torch.bfloat16
    if s == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s} (choose fp16/bf16/fp32)")


def infer_component_type_from_config(cfg_path: Path) -> Optional[str]:
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        klass = cfg.get("_class_name", "")
        if "UNet2DConditionModel" in klass:
            return "unet"
        if "AutoencoderKL" in klass:
            return "vae"
        if "CLIPTextModelWithProjection" in klass:
            return "text_encoder_2"
        if "CLIPTextModel" in klass:
            return "text_encoder"
    except Exception:
        pass
    return None


def load_component(src: str, comp_type: str, dtype: torch.dtype):
    ct = comp_type.lower()
    if ct == "unet":
        return UNet2DConditionModel.from_pretrained(src, torch_dtype=dtype)
    if ct == "vae":
        return AutoencoderKL.from_pretrained(src, torch_dtype=dtype)
    if ct == "text_encoder":
        if CLIPTextModel is None:
            raise RuntimeError("transformers 未安装或版本不支持 CLIPTextModel。")
        return CLIPTextModel.from_pretrained(src, torch_dtype=dtype)
    if ct == "text_encoder_2":
        if CLIPTextModelWithProjection is None:
            raise RuntimeError("transformers 未安装或版本不支持 CLIPTextModelWithProjection。")
        return CLIPTextModelWithProjection.from_pretrained(src, torch_dtype=dtype)
    raise ValueError(f"Unknown component type: {comp_type}")


def _default_filename_for(comp_type: str) -> str:
    # 与 diffusers 目录命名对齐
    if comp_type in ("unet", "vae"):
        return "diffusion_pytorch_model.safetensors"
    if comp_type in ("text_encoder", "text_encoder_2"):
        return "model.safetensors"
    return "pytorch_model.safetensors"


def _cleanup_old_shards(dst_dir: str, comp_type: str):
    base = _default_filename_for(comp_type)
    idx = base + ".index.json" if base.endswith(".safetensors") else None

    patterns = []
    # 常见分片与索引名
    patterns += ["diffusion_pytorch_model-*.safetensors",
                 "diffusion_pytorch_model.safetensors.index.json",
                 "pytorch_model-*.bin",
                 "model-*.safetensors",
                 "model.safetensors.index.json"]
    if idx:
        patterns.append(idx)

    for pat in patterns:
        for f in glob.glob(os.path.join(dst_dir, pat)):
            try:
                os.remove(f)
            except Exception:
                pass


def save_component_single(model, dst: str, comp_type: str, dtype: torch.dtype,
                          variant: Optional[str], max_shard_size: str,
                          make_default_copy: bool, clean_old_shards: bool):
    os.makedirs(dst, exist_ok=True)
    model = model.to(dtype=dtype).to("cpu")

    # 先保存（若给了 variant，会写成 *.{variant}.safetensors）
    kw = dict(safe_serialization=True, max_shard_size=max_shard_size)
    if variant:  # 仅当用户显式要求时才写 variant 后缀
        kw["variant"] = variant
    model.save_pretrained(dst, **kw)

    # 目标文件名（无后缀的“默认名”）
    default_name = _default_filename_for(comp_type)
    default_path = os.path.join(dst, default_name)

    if variant:
        # 计算带后缀输出名并做一份默认名拷贝（兼容不传 variant 的加载器）
        stem, ext = os.path.splitext(default_name)
        with_suffix = os.path.join(dst, f"{stem}.{variant}{ext}")
        if os.path.exists(with_suffix) and make_default_copy:
            shutil.copy2(with_suffix, default_path)
    else:
        # 未使用 variant：确保存在默认名；并清理旧分片索引避免干扰
        if clean_old_shards:
            _cleanup_old_shards(dst, comp_type)


def cmd_component(args):
    src = args.src
    dst = args.dst
    comp_type = args.type
    dtype = parse_dtype(args.dtype)
    variant = args.variant  # 现在默认 None：不再自动等于 dtype
    max_shard_size = args.max_shard_size

    if not comp_type or comp_type == "auto":
        cfg = Path(src) / "config.json"
        if not cfg.exists():
            raise SystemExit("无法自动推断组件类型：缺少 config.json。请显式 --type 指定。")
        comp_type = infer_component_type_from_config(cfg)
        if not comp_type:
            raise SystemExit("无法从 config.json 推断组件类型，请使用 --type 指定（unet/vae/text_encoder/text_encoder_2）。")

    print(f"[INFO] Loading component: type={comp_type}, src={src}, dtype={args.dtype}")
    model = load_component(src, comp_type, dtype)

    print(f"[INFO] Saving single-file to: {dst} (dtype={args.dtype}, "
          f"variant={'<none>' if not variant else variant}, max_shard_size={max_shard_size})")

    save_component_single(
        model=model, dst=dst, comp_type=comp_type, dtype=dtype,
        variant=variant, max_shard_size=max_shard_size,
        make_default_copy=args.make_default_copy,
        clean_old_shards=args.clean_old_shards,
    )
    print("[OK] Done.")


def _fix_pipeline_variant_files(root: str, variant: str, make_default_copy: bool, clean_old_shards: bool):
    if not variant:
        if clean_old_shards:
            for comp in ("unet", "vae", "text_encoder", "text_encoder_2"):
                d = os.path.join(root, comp)
                if os.path.isdir(d):
                    _cleanup_old_shards(d, comp)
        return

    if not make_default_copy:
        return

    for comp in ("unet", "vae", "text_encoder", "text_encoder_2"):
        d = os.path.join(root, comp)
        if not os.path.isdir(d):
            continue
        default_name = _default_filename_for(comp)
        stem, ext = os.path.splitext(default_name)
        with_suffix = os.path.join(d, f"{stem}.{variant}{ext}")
        default_path = os.path.join(d, default_name)
        if os.path.exists(with_suffix) and not os.path.exists(default_path):
            shutil.copy2(with_suffix, default_path)


def cmd_pipeline(args):
    base = args.base
    dst = args.dst
    dtype = parse_dtype(args.dtype)
    variant = args.variant  # 默认 None，不写后缀
    max_shard_size = args.max_shard_size

    print(f"[INFO] Loading base SDXL pipeline: {base} (dtype={args.dtype})")
    pipe = StableDiffusionXLPipeline.from_pretrained(base, torch_dtype=dtype)

    if args.unet:
        print(f"[INFO] Replacing UNet from: {args.unet}")
        pipe.unet = UNet2DConditionModel.from_pretrained(args.unet, torch_dtype=dtype)
    if args.vae:
        print(f"[INFO] Replacing VAE from: {args.vae}")
        pipe.vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=dtype)
    if args.text_encoder:
        if CLIPTextModel is None:
            raise RuntimeError("transformers 缺少 CLIPTextModel，无法替换 text_encoder。")
        print(f"[INFO] Replacing text_encoder from: {args.text_encoder}")
        pipe.text_encoder = CLIPTextModel.from_pretrained(args.text_encoder, torch_dtype=dtype)
    if args.text_encoder2:
        if CLIPTextModelWithProjection is None:
            raise RuntimeError("transformers 缺少 CLIPTextModelWithProjection，无法替换 text_encoder_2。")
        print(f"[INFO] Replacing text_encoder_2 from: {args.text_encoder2}")
        pipe.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.text_encoder2, torch_dtype=dtype)

    pipe = pipe.to(dtype).to("cpu")

    print(f"[INFO] Saving full pipeline to: {dst} (variant={'<none>' if not variant else variant}; max_shard_size={max_shard_size})")
    kw = dict(safe_serialization=True, max_shard_size=max_shard_size)
    if variant:
        kw["variant"] = variant
    pipe.save_pretrained(dst, **kw)

    _fix_pipeline_variant_files(dst, variant, args.make_default_copy, args.clean_old_shards)
    print("[OK] Pipeline saved.")


def main():
    parser = argparse.ArgumentParser(description="SDXL checkpoint exporter (merge shards, change dtype, save single-file)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("component", help="合并并导出组件（unet/vae/te1/te2）为单文件 .safetensors")
    p1.add_argument("--src", required=True, help="组件目录（包含 config.json 与 shards/index.json）")
    p1.add_argument("--dst", required=True, help="输出目录（会写入单个 safetensors + config.json）")
    p1.add_argument("--type", default="auto", choices=["auto", "unet", "vae", "text_encoder", "text_encoder_2"], help="组件类型（默认 auto 从 config.json 推断）")
    p1.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"], help="导出数据类型")
    p1.add_argument("--variant", default=None, help="保存时的 variant 标记；默认不用 variant，以生成默认文件名")
    p1.add_argument("--max-shard-size", default="100GB", help="避免再次分片（设大一些）")
    p1.add_argument("--make-default-copy", action="store_true",
                    help="当使用 --variant 时，同时写一份默认文件名的拷贝，便于不传 variant 的加载器使用")
    p1.add_argument("--clean-old-shards", action="store_true",
                    help="保存后清理目标目录里的旧分片与 *.index.json（只清理目标目录）")
    p1.set_defaults(func=cmd_component)

    p2 = sub.add_parser("pipeline", help="把组件注入到 SDXL 基础模型并一次性导出完整 pipeline")
    p2.add_argument("--base", required=True, help="基础 SDXL pipeline（repoId 或本地路径）")
    p2.add_argument("--dst", required=True, help="输出目录")
    p2.add_argument("--unet", default=None, help="替换用的 UNet 目录")
    p2.add_argument("--vae", default=None, help="替换用的 VAE 目录")
    p2.add_argument("--text-encoder", default=None, help="替换用的 text_encoder 目录（CLIPTextModel）")
    p2.add_argument("--text-encoder2", default=None, help="替换用的 text_encoder_2 目录（CLIPTextModelWithProjection）")
    p2.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"], help="导出数据类型")
    p2.add_argument("--variant", default=None, help="保存时的 variant 标记；默认不用 variant，以生成默认文件名")
    p2.add_argument("--max-shard-size", default="100GB", help="避免再次分片（设大一些）")
    p2.add_argument("--make-default-copy", action="store_true",
                    help="当使用 --variant 时，同时为各组件写默认文件名的拷贝")
    p2.add_argument("--clean-old-shards", action="store_true",
                    help="保存后清理各组件目录的旧分片与 *.index.json")
    p2.set_defaults(func=cmd_pipeline)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
