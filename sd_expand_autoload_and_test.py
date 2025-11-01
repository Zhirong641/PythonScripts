# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from compel_sdxl_utils import get_compel_for_sdxl

try:
    from safetensors import safe_open  # type: ignore
except Exception:
    safe_open = None

_VALID_PREDICTION_TYPES = {"epsilon", "v_prediction"}
_BOOL_TRUE = {"1", "true", "yes", "y", "on"}
_BOOL_FALSE = {"0", "false", "no", "n", "off"}


def normalize_prediction_type_value(value):
    if value is None:
        return None
    v = str(value).strip().lower()
    mapping = {
        "epsilon": "epsilon",
        "eps": "epsilon",
        "e": "epsilon",
        "noise": "epsilon",
        "epsilon_prediction": "epsilon",
        "v_prediction": "v_prediction",
        "vpred": "v_prediction",
        "v_pred": "v_prediction",
        "v": "v_prediction",
        "v-pred": "v_prediction",
        "velocity": "v_prediction",
    }
    return mapping.get(v)


def prediction_type_arg(value: str) -> str:
    if value is None:
        return "auto"
    v = str(value).strip()
    if not v:
        return "auto"
    v_lower = v.lower()
    if v_lower in {"auto", "default", "none"}:
        return "auto"
    normalized = normalize_prediction_type_value(v_lower)
    if not normalized:
        raise argparse.ArgumentTypeError(
            f"Unsupported prediction_type '{value}'. Use auto/epsilon/v_prediction."
        )
    return normalized


def detect_prediction_type_from_metadata(weight_path: str | None) -> str | None:
    if not weight_path or not os.path.exists(weight_path):
        return None
    if not weight_path.lower().endswith(".safetensors"):
        return None
    if safe_open is None:
        return None
    try:
        with safe_open(weight_path, framework="pt", device="cpu") as handler:
            metadata = handler.metadata() or {}
    except Exception:
        return None
    keys = ("prediction_type", "parameterization", "ss_v_pred", "v_pred", "sd_prediction_type")
    for key in keys:
        if key not in metadata:
            continue
        raw = metadata[key]
        normalized = normalize_prediction_type_value(raw)
        if normalized:
            return normalized
        raw_str = str(raw).strip().lower()
        if key in ("ss_v_pred", "v_pred"):
            if raw_str in _BOOL_TRUE:
                return "v_prediction"
            if raw_str in _BOOL_FALSE:
                return "epsilon"
    return None


def detect_prediction_type_from_repo(repo_path: str) -> str | None:
    cfg_path = Path(repo_path) / "scheduler" / "scheduler_config.json"
    if not cfg_path.exists():
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return normalize_prediction_type_value(data.get("prediction_type"))


def maybe_apply_prediction_type(pipe, prediction_type: str | None) -> bool:
    if prediction_type not in _VALID_PREDICTION_TYPES:
        return False
    current = getattr(pipe.scheduler.config, "prediction_type", None)
    if current == prediction_type:
        print(f">> scheduler prediction_type 已为 {prediction_type}")
        return True
    print(f">> 设置 scheduler prediction_type = {prediction_type}")
    applied = False
    try:
        pipe.scheduler.register_to_config(prediction_type=prediction_type)
        applied = True
    except Exception as e:
        print(f">> 警告: register_to_config 失败: {e}")
    try:
        pipe.scheduler.config.prediction_type = prediction_type
        pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
        applied = True
    except Exception as e:
        print(f">> 警告: 重新初始化 scheduler 失败: {e}")
    return applied


def expand_single_file_to_repo(
    model_file: str, repo_out: str, fp16: bool, prediction_type_hint: str | None = None
):
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
    )

    detected_pred = detect_prediction_type_from_metadata(model_file)
    pred_to_apply = prediction_type_hint or detected_pred
    if prediction_type_hint:
        print(f">> 使用命令行指定的 prediction_type={prediction_type_hint}")
    elif detected_pred:
        print(f">> safetensors metadata 指示 prediction_type={detected_pred}")
    if pred_to_apply:
        maybe_apply_prediction_type(pipe, pred_to_apply)

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

    submodels = data.get("_expected_submodels", []) or list(data.keys())
    class_name = data.get("_class_name", "") or data.get("_class_name", "")
    has_te2 = any("text_encoder_2" in str(x) for x in submodels) or (
        Path(repo_path) / "text_encoder_2"
    ).exists()
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


def auto_default_hw(model_type: str, width: int, height: int):
    """
    若未显式指定分辨率（传入 <=0），按模型类型给默认分辨率
    """
    if width > 0 and height > 0:
        return width, height
    if model_type == "sdxl":
        return 1024, 1024
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

    manual_pred = args.prediction_type if args.prediction_type != "auto" else None
    repo_pred = detect_prediction_type_from_repo(repo_out)
    metadata_pred = None
    if not manual_pred and not repo_pred:
        metadata_pred = detect_prediction_type_from_metadata(
            getattr(args, "model_file", None)
        )

    final_pred = manual_pred or repo_pred or metadata_pred
    if manual_pred:
        print(f">> 命令行指定 prediction_type={manual_pred}")
    elif repo_pred:
        print(f">> repo scheduler 配置指示 prediction_type={repo_pred}")
    elif metadata_pred:
        print(f">> safetensors metadata 指示 prediction_type={metadata_pred}")
    else:
        print(">> 未检测到 prediction_type，沿用 scheduler 默认设置")
    if final_pred:
        maybe_apply_prediction_type(pipe, final_pred)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    maybe_enable_speedups(pipe)

    w, h = auto_default_hw(model_type, args.width, args.height)
    if args.seed >= 0:
        g = torch.Generator(device=device).manual_seed(args.seed)
    else:
        g = None

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[gen] type={model_type} | size={w}x{h} | steps={args.steps} | cfg={args.cfg}")
    print(f"[gen] prompt: {args.prompt}")
    if args.negative_prompt:
        print(f"[gen] neg   : {args.negative_prompt}")

    # =========================
    # 【新增】长提示分块+拼接（SDXL 双编码器）
    # =========================
    compel, empty_conditioning = get_compel_for_sdxl(
        [pipe.tokenizer, pipe.tokenizer_2],
        [pipe.text_encoder, pipe.text_encoder_2],
        device=device,
    )

    # 正向
    prompt_embeds, pooled_prompt_embeds = compel(args.prompt)
    # 负向（若无，则用空串得到对齐形状）
    neg_text = args.negative_prompt if args.negative_prompt is not None else ""
    negative_prompt_embeds, negative_pooled_prompt_embeds = compel(neg_text)
    (
        prompt_embeds,
        negative_prompt_embeds,
    ) = compel.pad_conditioning_tensors_to_same_length(
        [prompt_embeds, negative_prompt_embeds], precomputed_padding=empty_conditioning
    )

    time_ids = torch.tensor(
        [
            [
                h,
                w,
                0,
                0,
                h,
                w,
            ]
        ],
        device=device,
        dtype=prompt_embeds.dtype,
    )

    # ========== 用嵌入调用 ==========
    out = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        width=w,
        height=h,
        num_images_per_prompt=args.num_images,
        generator=g,
        added_cond_kwargs={"time_ids": time_ids},
    )
    # =========================
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
    ap = argparse.ArgumentParser(
        "Expand .safetensors/.ckpt to REPO, auto-detect SDXL/SD1.5, and test-generate"
    )
    ap.add_argument("--model_file", type=str, help="单文件权重路径（.safetensors/.ckpt）")
    ap.add_argument("--repo_out", type=str, default="./expanded_repo", help="展开/已有的 REPO 目录")
    ap.add_argument("--expand_only", action="store_true", help="只展开不出图（需要提供 --model_file）")
    ap.add_argument("--skip_expand", action="store_true", help="跳过展开，直接从 --repo_out 读取（当已是 REPO）")
    ap.add_argument("--fp16", action="store_true", help="使用 FP16（GPU 推荐）")
    ap.add_argument(
        "--prediction_type",
        type=prediction_type_arg,
        default="auto",
        help="scheduler prediction_type: auto/epsilon/v_prediction (vpred 模型请选择 v_prediction)",
    )

    ap.add_argument("--prompt", type=str, default="karyl (princess connect!), by miwa futaba, rating:questionable, 1girl, animal ear fluff, animal ears, bar censor, black hair, blush, bottomless, breasts, cat ears, cat girl, cat tail, cowboy shot, cropped shirt, day, twintails, small breasts, page number, wet hair, tail, outdoors, ripples, water, nipples, skin fang, partially submerged, sleeveless, shirt, tail raised, open mouth, looking at viewer, navel, sidelocks, streaked hair, long hair, stomach, wet, fang, era:newest")
    ap.add_argument("--negative_prompt", type=str, default="")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=5.5)
    ap.add_argument("--width", type=int, default=-1, help="未指定或<=0时按模型类型给默认: SD1.5→512, SDXL→1024")
    ap.add_argument("--height", type=int, default=-1)
    ap.add_argument("--num_images", type=int, default=1)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out_dir", type=str, default="./samples")

    args = ap.parse_args()

    manual_pred = args.prediction_type if args.prediction_type != "auto" else None
    metadata_pred = (
        detect_prediction_type_from_metadata(args.model_file)
        if args.model_file and not manual_pred
        else None
    )

    if not args.skip_expand:
        if not args.model_file:
            print("ERROR: 未提供 --model_file（当不使用 --skip_expand 时必须提供）", file=sys.stderr)
            sys.exit(1)
        expand_pred_hint = manual_pred or metadata_pred
        expand_single_file_to_repo(
            args.model_file,
            args.repo_out,
            args.fp16,
            prediction_type_hint=expand_pred_hint,
        )
        if args.expand_only:
            print("完成：仅展开未出图（--expand_only）")
            return

    run_generate(args.repo_out, args)


if __name__ == "__main__":
    main()
