#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pip3 install -U onnxruntime-gpu huggingface_hub pillow tqdm numpy

"""
Tag images with Camais03/camie-tagger-v2 (ONNX).
- Auto-downloads model + metadata from Hugging Face
- ImageNet normalization, keep aspect ratio with padding
- Supports single image or directory (recursive), batching, CUDA/CPU, CSV/JSONL

Author: you :)
"""

import argparse
import csv
import json
import os
import sys
import ctypes
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
from PIL import Image
from tqdm import tqdm
import onnxruntime as ort
from huggingface_hub import hf_hub_download

HF_REPO = "Camais03/camie-tagger-v2"
MODEL_FILE = "camie-tagger-v2.onnx"
META_FILE = "camie-tagger-v2-metadata.json"

# ImageNet mean/std
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

CATEGORY_ORDER = ["general", "character", "copyright", "artist", "meta", "year", "rating"]

# -------------------------
# IO helpers
# -------------------------

def list_images(root: Path, recursive: bool, exts: Tuple[str, ...]) -> List[Path]:
    if root.is_file():
        return [root]
    if recursive:
        return [p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file()]
    else:
        return [p for p in root.glob("*") if p.suffix.lower() in exts and p.is_file()]

def load_metadata(meta_path: str) -> Tuple[List[str], Dict[str, str], int, int]:
    """
    Returns:
      idx_to_tag: index -> tag string (list length = total_tags)
      tag_to_category: tag string -> category name
      total_tags
      img_size (recommended)
    """
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    ds = meta["dataset_info"]
    tag_map = ds["tag_mapping"]
    idx_to_tag_map: Dict[str, str] = tag_map["idx_to_tag"]      # keys are str indices
    tag_to_category: Dict[str, str] = tag_map["tag_to_category"]  # tag -> category name
    total_tags = int(ds["total_tags"])

    # ensure a dense list aligned by index
    idx_to_tag: List[str] = [""] * total_tags
    for k, v in idx_to_tag_map.items():
        i = int(k)
        if 0 <= i < total_tags:
            idx_to_tag[i] = v

    img_size = int(meta.get("model_info", {}).get("img_size", 512))
    return idx_to_tag, tag_to_category, total_tags, img_size

# -------------------------
# Preprocess
# -------------------------

def preprocess_imagenet_nchw(img: Image.Image, img_size: int) -> np.ndarray:
    """
    - Convert to RGB, handle alpha
    - Keep aspect ratio, center pad to square using ImageNet mean color (~[124,116,104])
    - Resize to img_size with LANCZOS
    - Normalize to ImageNet mean/std
    - Return CHW float32
    """
    # handle alpha/palette
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    else:
        img = img.copy()

    w, h = img.size
    ar = w / h if h > 0 else 1.0

    if ar >= 1.0:
        new_w = img_size
        new_h = max(1, int(round(new_w / ar)))
    else:
        new_h = img_size
        new_w = max(1, int(round(new_h * ar)))

    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # pad color close to ImageNet mean (in 0..255)
    pad_color = (124, 116, 104)
    canvas = Image.new("RGB", (img_size, img_size), pad_color)
    paste_x = (img_size - new_w) // 2
    paste_y = (img_size - new_h) // 2
    canvas.paste(img, (paste_x, paste_y))

    arr = np.asarray(canvas).astype(np.float32) / 255.0  # HWC in [0,1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD          # normalize
    arr = np.transpose(arr, (2, 0, 1))                  # CHW
    return arr

def collate_nchw(batch: List[np.ndarray]) -> np.ndarray:
    return np.stack(batch, axis=0).astype(np.float32)  # N,C,H,W

# -------------------------
# Inference & Postprocess
# -------------------------

def run_onnx(session: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    """
    Camie v2 ONNX may output:
      - [logits] OR
      - [initial_logits, refined_logits, selected_candidates]
    We use refined if available, otherwise the single output.
    """
    inputs = {session.get_inputs()[0].name: x}
    outs = session.run(None, inputs)
    if len(outs) >= 2:
        logits = outs[1]
    else:
        logits = outs[0]
    # sigmoid -> probabilities
    probs = 1.0 / (1.0 + np.exp(-logits))
    return probs  # [N, num_tags]

def format_tags_per_category(
    probs_1d: np.ndarray,
    idx_to_tag: List[str],
    tag_to_category: Dict[str, str],
    thr_map: Dict[str, float],
    topk_map: Dict[str, int],
    replace_underscore: bool
) -> Dict[str, List[Tuple[str, float]]]:
    buckets: Dict[str, List[Tuple[str, float]]] = {}
    for i, p in enumerate(probs_1d.tolist()):
        tag = idx_to_tag[i]
        if not tag:
            continue
        cat = tag_to_category.get(tag, "general")
        thr = thr_map.get(cat, thr_map.get("default", 0.5))
        if p >= thr:
            if replace_underscore:
                tag_out = tag.replace("_", " ")
            else:
                tag_out = tag
            buckets.setdefault(cat, []).append((tag_out, float(p)))

    # sort & clip per category
    for cat, items in buckets.items():
        items.sort(key=lambda x: x[1], reverse=True)
        k = topk_map.get(cat, topk_map.get("default", 64))
        buckets[cat] = items[:k]
    return buckets

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Image tagger using Camais03/camie-tagger-v2 (ONNX)")
    ap.add_argument("-i", "--input", required=True, help="Image file or directory")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories when input is a directory")
    ap.add_argument("--batch-size", type=int, default=4, help="Batch size")
    ap.add_argument("--use-gpu", action="store_true", help="Use CUDAExecutionProvider if available")

    # thresholds
    ap.add_argument("--threshold", type=float, default=0.492, help="Global threshold (default: macro-optimized 0.492)")
    ap.add_argument("--thr-general", type=float, default=None)
    ap.add_argument("--thr-character", type=float, default=None)
    ap.add_argument("--thr-copyright", type=float, default=None)
    ap.add_argument("--thr-artist", type=float, default=None)
    ap.add_argument("--thr-meta", type=float, default=None)
    ap.add_argument("--thr-rating", type=float, default=None)
    ap.add_argument("--thr-year", type=float, default=None)

    # top-k per category for printing / exporting
    ap.add_argument("--topk-default", type=int, default=64)
    ap.add_argument("--topk-rating", type=int, default=4)
    ap.add_argument("--replace-underscore", action="store_true", help="Replace '_' with space in tag strings")

    # outputs
    ap.add_argument("--out-csv", type=str, default=None, help="Write results to CSV")
    ap.add_argument("--out-jsonl", type=str, default=None, help="Write results to JSONL")
    ap.add_argument("--repo", type=str, default=HF_REPO, help="Hugging Face repo id")
    ap.add_argument("--exts", type=str, default=",".join(DEFAULT_EXTS), help="Comma-separated file extensions")
    ap.add_argument("--resume", action="store_true", help="Skip already-processed images if output file exists")
    args = ap.parse_args()

    # download artifacts
    model_path = hf_hub_download(repo_id=args.repo, filename=MODEL_FILE)
    meta_path  = hf_hub_download(repo_id=args.repo, filename=META_FILE)

    # load meta
    idx_to_tag, tag_to_category, num_tags, img_size = load_metadata(meta_path)
    print(f"> Loaded metadata: total_tags={num_tags}, img_size={img_size}")

    # providers
    def tensorrt_lib_available() -> bool:
        for name in ("libnvinfer.so.10", "libnvinfer.so"):
            try:
                ctypes.CDLL(name)
                return True
            except OSError:
                continue
        return False

    def cuda_deps_available() -> Tuple[bool, List[str]]:
        missing: List[str] = []

        def loadable(candidates: Tuple[str, ...]) -> bool:
            for name in candidates:
                try:
                    ctypes.CDLL(name)
                    return True
                except OSError:
                    continue
            return False

        if not loadable(("libcudart.so.12", "libcudart.so")):
            missing.append("CUDA runtime (libcudart.so)")
        if not loadable(("libcudnn.so.9", "libcudnn.so")):
            missing.append("cuDNN 9 (libcudnn.so)")
        return (len(missing) == 0, missing)

    providers = ["CPUExecutionProvider"]
    if args.use_gpu:
        available = set(ort.get_available_providers())
        providers = []
        if "TensorrtExecutionProvider" in available and tensorrt_lib_available():
            providers.append("TensorrtExecutionProvider")
        elif "TensorrtExecutionProvider" in available:
            print("[WARN] libnvinfer not found; skipping TensorRT provider.", file=sys.stderr)
        if "CUDAExecutionProvider" in available:
            cuda_ok, cuda_missing = cuda_deps_available()
            if cuda_ok:
                providers.append("CUDAExecutionProvider")
            else:
                print(
                    f"[WARN] Missing CUDA dependencies ({'; '.join(cuda_missing)}); skipping CUDA provider.",
                    file=sys.stderr,
                )
        providers.append("CPUExecutionProvider")

    sess_opt = ort.SessionOptions()
    try:
        session = ort.InferenceSession(model_path, sess_options=sess_opt, providers=providers)
    except Exception as e:
        if (
            args.use_gpu
            and "TensorrtExecutionProvider" in providers
            and any(substr in str(e) for substr in ("libnvinfer", "TensorRT", "tensorrt"))
        ):
            print("[WARN] TensorRT provider unavailable, retrying without it.", file=sys.stderr)
            fallback_providers = [p for p in providers if p != "TensorrtExecutionProvider"]
            session = ort.InferenceSession(model_path, sess_options=sess_opt, providers=fallback_providers)
            providers = fallback_providers
        else:
            raise

    print(f"> Using providers: {session.get_providers()}")

    # collect images
    exts = tuple([e.strip().lower() for e in args.exts.split(",") if e.strip()])
    inputs = list_images(Path(args.input), args.recursive, exts)
    if not inputs:
        print("No images found.", file=sys.stderr)
        sys.exit(1)

    # thresholds map
    thr_map = {"default": args.threshold}
    if args.thr_general is not None:    thr_map["general"] = args.thr_general
    if args.thr_character is not None:  thr_map["character"] = args.thr_character
    if args.thr_copyright is not None:  thr_map["copyright"] = args.thr_copyright
    if args.thr_artist is not None:     thr_map["artist"] = args.thr_artist
    if args.thr_meta is not None:       thr_map["meta"] = args.thr_meta
    if args.thr_rating is not None:     thr_map["rating"] = args.thr_rating
    if args.thr_year is not None:       thr_map["year"] = args.thr_year

    # topk map
    topk_map = {"default": args.topk_default, "rating": args.topk_rating}

    # collect already-processed paths for resume
    done_paths: set = set()
    if args.resume:
        if args.out_jsonl and Path(args.out_jsonl).exists():
            with open(args.out_jsonl, "r", encoding="utf-8") as _f:
                for _line in _f:
                    _line = _line.strip()
                    if _line:
                        try:
                            done_paths.add(json.loads(_line)["path"])
                        except Exception:
                            pass
        elif args.out_csv and Path(args.out_csv).exists():
            with open(args.out_csv, "r", encoding="utf-8") as _f:
                for _row in csv.DictReader(_f):
                    done_paths.add(_row.get("path", ""))
    if done_paths:
        _total = len(inputs)
        inputs = [p for p in inputs if str(p) not in done_paths]
        print(f"> Resuming: {_total - len(inputs)} already done, {len(inputs)} remaining.")

    # outputs
    csv_writer = None
    csv_fp = None
    if args.out_csv:
        _csv_exists = args.resume and Path(args.out_csv).exists()
        csv_fp = open(args.out_csv, "a" if _csv_exists else "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_fp)
        if not _csv_exists:
            csv_writer.writerow(["path", "rating", "general", "character", "copyright", "artist", "meta", "year"])

    if args.out_jsonl:
        _jsonl_exists = args.resume and Path(args.out_jsonl).exists()
        jsonl_fp = open(args.out_jsonl, "a" if _jsonl_exists else "w", encoding="utf-8")
    else:
        jsonl_fp = None

    # buffers
    batch_imgs: List[np.ndarray] = []
    batch_paths: List[Path] = []

    def flush():
        if not batch_imgs:
            return
        x = collate_nchw(batch_imgs)  # N,C,H,W
        probs = run_onnx(session, x)  # [N, num_tags]
        if probs.shape[1] != num_tags:
            raise RuntimeError(f"Probability shape mismatch: got {probs.shape}, expected num_tags={num_tags}")

        for i in range(probs.shape[0]):
            per = format_tags_per_category(
                probs[i],
                idx_to_tag,
                tag_to_category,
                thr_map=thr_map,
                topk_map=topk_map,
                replace_underscore=args.replace_underscore,
            )

            # build pretty strings in fixed order
            texts: Dict[str, str] = {}
            for cat in CATEGORY_ORDER:
                if cat in per:
                    texts[cat] = ", ".join([f"{t}:{p:.3f}" for t, p in per[cat]])
                else:
                    texts[cat] = ""
            if len(inputs) <= 10:
                print(f"\n[{batch_paths[i]}]")
                for cat in ["rating", "general", "character", "copyright", "artist", "meta", "year"]:
                    label = f"{cat:<10}"
                    print(f"  {label}: {texts[cat]}")

            if csv_writer:
                csv_writer.writerow([
                    str(batch_paths[i]),
                    texts["rating"], texts["general"], texts["character"],
                    texts["copyright"], texts["artist"], texts["meta"], texts["year"],
                ])
            if jsonl_fp:
                obj = {"path": str(batch_paths[i])}
                for cat in CATEGORY_ORDER:
                    obj[cat] = per.get(cat, [])
                jsonl_fp.write(json.dumps(obj, ensure_ascii=False) + "\n")

        batch_imgs.clear()
        batch_paths.clear()

    # iterate
    for p in tqdm(inputs, desc="Tagging"):
        try:
            with Image.open(p) as img:
                arr = preprocess_imagenet_nchw(img, img_size)
            batch_imgs.append(arr)
            batch_paths.append(p)
            if len(batch_imgs) >= args.batch_size:
                flush()
        except Exception as e:
            print(f"[WARN] Failed to process {p}: {e}", file=sys.stderr)

    flush()

    if csv_fp:
        csv_fp.close()
    if jsonl_fp:
        jsonl_fp.close()

if __name__ == "__main__":
    main()
