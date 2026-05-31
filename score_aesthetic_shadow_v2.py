# -*- coding: utf-8 -*-
"""
使用 aesthetic-shadow-v2 为图片打分，并可将分数离散成 aesthetic tag，
用于 SDXL 训练数据预处理。

特性：
- 边处理边写 CSV / JSONL
- RGBA / LA / 带 transparency 的 P 模式图片会先合成到底色
- 保持原始长宽比：等比缩放 + padding 到模型输入尺寸
- 不会把长图/宽图直接拉伸变形
- aesthetic 标签使用下划线形式
- 支持额外废弃阈值：hq_score > discard_threshold 时返回 discard_tag
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from PIL import Image, ImageFile, ImageOps
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

VALID_EXTS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"
}


@dataclass
class ScoreResult:
    path: str
    hq_score: float
    lq_score: float
    aesthetic_tag: str
    error: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch score images with aesthetic-shadow-v2.")
    p.add_argument("--input", required=True, help="图片目录、单张图片或 JSONL 文件")
    p.add_argument("--output-csv", required=True, help="输出 CSV 路径")
    p.add_argument("--output-jsonl", default="", help="可选：输出带分数的新 JSONL")
    p.add_argument(
        "--model",
        default="NeoChen1024/aesthetic-shadow-v2-backup",
        help="Hugging Face 模型名",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="推理设备",
    )
    p.add_argument("--batch-size", type=int, default=4, help="批大小")
    p.add_argument("--recursive", action="store_true", help="输入为目录时递归扫描")
    p.add_argument(
        "--prepend-target",
        default="none",
        choices=["none", "general", "caption"],
        help="将 aesthetic tag 前置到哪个字段，供训练使用",
    )
    p.add_argument(
        "--separator",
        default=", ",
        help="前置 aesthetic tag 时使用的分隔符",
    )
    p.add_argument(
        "--flush-every",
        type=int,
        default=1,
        help="每处理多少个 batch 后 flush 一次输出文件，默认 1",
    )
    p.add_argument(
        "--bg-color",
        default="255,255,255",
        help="alpha 合成与 padding 使用的背景色，格式如 255,255,255",
    )
    p.add_argument(
        "--force-size",
        default="",
        help="可选，手动指定输入尺寸，格式如 1024 或 1024,1024；默认读取模型预处理配置",
    )
    p.add_argument(
        "--discard-threshold",
        type=float,
        default=None,
        help="可选：当 hq_score 大于该阈值时，返回废弃标签",
    )
    p.add_argument(
        "--discard-tag",
        default="discard_image",
        help="超过 discard-threshold 时使用的标签名",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="跳过已在 --output-jsonl 中存在且含 aesthetic_tag 的记录",
    )
    return p.parse_args()


def parse_rgb_color(s: str) -> Tuple[int, int, int]:
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"无效颜色格式: {s}，应为 R,G,B")
    vals = tuple(int(x) for x in parts)
    if any(v < 0 or v > 255 for v in vals):
        raise ValueError(f"颜色分量必须在 0~255: {s}")
    return vals  # type: ignore


def iter_images_from_dir(root: Path, recursive: bool) -> Iterator[Dict]:
    pattern = "**/*" if recursive else "*"
    for p in sorted(root.glob(pattern)):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            yield {"path": str(p)}


def iter_records_from_jsonl(jsonl_path: Path) -> Iterator[Dict]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[WARN] JSONL 第 {line_no} 行解析失败: {e}")
                continue
            if not isinstance(obj, dict):
                print(f"[WARN] JSONL 第 {line_no} 行不是对象，已跳过")
                continue
            if "path" not in obj:
                print(f"[WARN] JSONL 第 {line_no} 行缺少 path 字段，已跳过")
                continue
            yield obj


def iter_records(input_path: Path, recursive: bool) -> Iterator[Dict]:
    if input_path.is_dir():
        yield from iter_images_from_dir(input_path, recursive)
        return
    if input_path.is_file() and input_path.suffix.lower() == ".jsonl":
        yield from iter_records_from_jsonl(input_path)
        return
    if input_path.is_file() and input_path.suffix.lower() in VALID_EXTS:
        yield {"path": str(input_path)}
        return
    raise ValueError(f"不支持的输入: {input_path}")


def batched_iter(it: Iterator[Dict], batch_size: int) -> Iterator[List[Dict]]:
    batch: List[Dict] = []
    for item in it:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def score_to_tag(
    hq: float,
    discard_threshold: Optional[float] = None,
    discard_tag: str = "discard_image",
) -> str:
    # 按你的要求：大于该阈值时返回废弃标签
    if discard_threshold is not None and hq > discard_threshold:
        return discard_tag

    if hq > 0.71:
        return "very_aesthetic"
    if hq > 0.45:
        return "aesthetic"
    if hq > 0.27:
        return "displeasing"
    return "very_displeasing"


def prepend_tag(old_value: str, tag: str, sep: str) -> str:
    old_value = (old_value or "").strip()
    if not old_value:
        return tag

    if sep == ", ":
        parts = [x.strip() for x in old_value.split(",")]
        if tag in parts:
            return old_value
    else:
        if old_value == tag or old_value.startswith(tag + sep):
            return old_value

    return f"{tag}{sep}{old_value}"


def resolve_input_size(processor, force_size: str) -> Tuple[int, int]:
    if force_size:
        parts = [x.strip() for x in force_size.split(",")]
        if len(parts) == 1:
            s = int(parts[0])
            return s, s
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
        raise ValueError(f"无效 --force-size: {force_size}")

    size = getattr(processor, "size", None)

    if isinstance(size, dict):
        if "height" in size and "width" in size:
            return int(size["width"]), int(size["height"])
        if "shortest_edge" in size:
            s = int(size["shortest_edge"])
            return s, s

    if isinstance(size, int):
        return size, size

    return 1024, 1024


def make_model(model_name: str, device_mode: str, force_size: str):
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    if device_mode == "auto":
        use_cuda = torch.cuda.is_available()
    else:
        use_cuda = (device_mode == "cuda")

    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("指定了 --device cuda，但当前环境没有可用 CUDA")

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
    )

    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    model.eval()

    input_w, input_h = resolve_input_size(processor, force_size)

    id2label = getattr(model.config, "id2label", {})
    label_to_idx = {}
    for k, v in id2label.items():
        label_to_idx[str(v).strip().lower()] = int(k)

    if "hq" not in label_to_idx or "lq" not in label_to_idx:
        raise RuntimeError(f"模型标签中未找到 hq/lq，当前标签: {id2label}")

    return processor, model, device, use_cuda, input_w, input_h, label_to_idx


def has_alpha(im: Image.Image) -> bool:
    if im.mode in ("RGBA", "LA"):
        return True
    if im.mode == "P" and "transparency" in im.info:
        return True
    return False


def open_rgb_image_keep_ratio(
    path: str,
    target_w: int,
    target_h: int,
    bg_color: Tuple[int, int, int],
) -> Image.Image:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)

        if has_alpha(im):
            rgba = im.convert("RGBA")
            canvas_rgba = Image.new("RGBA", rgba.size, bg_color + (255,))
            rgb = Image.alpha_composite(canvas_rgba, rgba).convert("RGB")
        else:
            rgb = im.convert("RGB")

    src_w, src_h = rgb.size
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"非法图像尺寸: {src_w}x{src_h}")

    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    resized = rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), bg_color)

    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    return canvas


def score_batch(
    processor,
    model,
    device,
    label_to_idx: Dict[str, int],
    batch_records: Sequence[Dict],
    target_w: int,
    target_h: int,
    bg_color: Tuple[int, int, int],
    discard_threshold: Optional[float],
    discard_tag: str,
) -> List[Tuple[Dict, ScoreResult]]:
    import torch

    paths = [str(x["path"]) for x in batch_records]
    pil_images: List[Image.Image] = []
    valid_records: List[Tuple[int, Dict]] = []
    results: List[Optional[ScoreResult]] = [None] * len(batch_records)

    for i, rec in enumerate(batch_records):
        path = paths[i]
        try:
            img = open_rgb_image_keep_ratio(
                path=path,
                target_w=target_w,
                target_h=target_h,
                bg_color=bg_color,
            )
            pil_images.append(img)
            valid_records.append((i, rec))
        except Exception as e:
            results[i] = ScoreResult(
                path=path,
                hq_score=0.0,
                lq_score=0.0,
                aesthetic_tag="",
                error=str(e),
            )

    if pil_images:
        try:
            inputs = processor(images=pil_images, return_tensors="pt", do_resize=False)
        except TypeError:
            inputs = processor(images=pil_images, return_tensors="pt")

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu()

        hq_idx = label_to_idx["hq"]
        lq_idx = label_to_idx["lq"]

        for out_idx, (rec_idx, rec) in enumerate(valid_records):
            hq = float(probs[out_idx, hq_idx].item())
            lq = float(probs[out_idx, lq_idx].item())
            tag = score_to_tag(
                hq,
                discard_threshold=discard_threshold,
                discard_tag=discard_tag,
            )

            results[rec_idx] = ScoreResult(
                path=str(rec["path"]),
                hq_score=hq,
                lq_score=lq,
                aesthetic_tag=tag,
                error="",
            )

    merged: List[Tuple[Dict, ScoreResult]] = []
    for rec, result in zip(batch_records, results):
        if result is None:
            result = ScoreResult(
                path=str(rec["path"]),
                hq_score=0.0,
                lq_score=0.0,
                aesthetic_tag="",
                error="unknown error",
            )
        merged.append((rec, result))
    return merged


def write_csv_header(csv_writer) -> None:
    csv_writer.writerow(["path", "hq_score", "lq_score", "aesthetic_tag", "error"])


def write_csv_row(csv_writer, score: ScoreResult) -> None:
    csv_writer.writerow([
        score.path,
        f"{score.hq_score:.6f}",
        f"{score.lq_score:.6f}",
        score.aesthetic_tag,
        score.error,
    ])


def build_output_record(
    rec: Dict,
    score: ScoreResult,
    prepend_target: str,
    sep: str,
) -> Dict:
    new_rec = dict(rec)
    new_rec["aesthetic_score"] = round(score.hq_score, 6)
    new_rec["aesthetic_tag"] = score.aesthetic_tag

    if prepend_target != "none" and score.aesthetic_tag and not score.error:
        old_value = str(new_rec.get(prepend_target, ""))
        new_rec[prepend_target] = prepend_tag(old_value, score.aesthetic_tag, sep)

    return new_rec


def load_done_paths(jsonl_path: Path) -> set:
    done: set = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "path" in obj and "aesthetic_tag" in obj:
                    done.add(str(obj["path"]))
            except Exception:
                pass
    return done


def count_records_if_possible(input_path: Path, recursive: bool) -> Optional[int]:
    try:
        if input_path.is_dir():
            pattern = "**/*" if recursive else "*"
            return sum(
                1 for p in input_path.glob(pattern)
                if p.is_file() and p.suffix.lower() in VALID_EXTS
            )
        if input_path.is_file() and input_path.suffix.lower() == ".jsonl":
            count = 0
            with input_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
            return count
        if input_path.is_file():
            return 1
    except Exception:
        return None
    return None


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    bg_color = parse_rgb_color(args.bg_color)

    processor, model, device, use_cuda, input_w, input_h, label_to_idx = make_model(
        model_name=args.model,
        device_mode=args.device,
        force_size=args.force_size,
    )

    print(
        f"[INFO] device={'cuda' if use_cuda else 'cpu'}, "
        f"model={args.model}, input_size={input_w}x{input_h}, bg_color={bg_color}, "
        f"discard_threshold={args.discard_threshold}, discard_tag={args.discard_tag}"
    )

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    if args.output_jsonl:
        Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    total = count_records_if_possible(input_path, args.recursive)

    processed = 0
    ok_count = 0
    err_count = 0
    batch_idx = 0

    done_paths: set = set()
    if args.resume and args.output_jsonl:
        done_paths = load_done_paths(Path(args.output_jsonl))
        if done_paths:
            print(f"[INFO] resume 模式：已跳过 {len(done_paths)} 条已处理记录")
    elif args.resume and not args.output_jsonl:
        print("[WARN] --resume 需要配合 --output-jsonl 使用，已忽略")

    record_iter = iter_records(input_path, args.recursive)
    if done_paths:
        record_iter = (rec for rec in record_iter if str(rec["path"]) not in done_paths)
    batch_iter = batched_iter(record_iter, args.batch_size)

    csv_mode = "a" if (args.resume and Path(args.output_csv).exists()) else "w"
    with open(args.output_csv, csv_mode, encoding="utf-8", newline="") as csv_fp:
        csv_writer = csv.writer(csv_fp)
        if csv_mode == "w":
            write_csv_header(csv_writer)

        jsonl_fp = None
        try:
            if args.output_jsonl:
                jsonl_mode = "a" if (args.resume and Path(args.output_jsonl).exists()) else "w"
                jsonl_fp = open(args.output_jsonl, jsonl_mode, encoding="utf-8")

            total_batches = None if total is None else (total + args.batch_size - 1) // args.batch_size

            for batch in tqdm(batch_iter, total=total_batches, desc="scoring"):
                merged = score_batch(
                    processor=processor,
                    model=model,
                    device=device,
                    label_to_idx=label_to_idx,
                    batch_records=batch,
                    target_w=input_w,
                    target_h=input_h,
                    bg_color=bg_color,
                    discard_threshold=args.discard_threshold,
                    discard_tag=args.discard_tag,
                )

                for rec, score in merged:
                    write_csv_row(csv_writer, score)

                    if jsonl_fp is not None:
                        out_rec = build_output_record(
                            rec=rec,
                            score=score,
                            prepend_target=args.prepend_target,
                            sep=args.separator,
                        )
                        jsonl_fp.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

                    processed += 1
                    if score.error:
                        err_count += 1
                    else:
                        ok_count += 1

                batch_idx += 1
                if args.flush_every > 0 and (batch_idx % args.flush_every == 0):
                    csv_fp.flush()
                    if jsonl_fp is not None:
                        jsonl_fp.flush()

            csv_fp.flush()
            if jsonl_fp is not None:
                jsonl_fp.flush()

        finally:
            if jsonl_fp is not None:
                jsonl_fp.close()

    print(f"[INFO] CSV 已写出: {args.output_csv}")
    if args.output_jsonl:
        print(f"[INFO] JSONL 已写出: {args.output_jsonl}")
    print(f"[INFO] processed={processed}, ok={ok_count}, error={err_count}")


if __name__ == "__main__":
    main()