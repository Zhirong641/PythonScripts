#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tag images with SmilingWolf/wd-eva02-large-tagger-v3 (ONNX).
- Auto-downloads model and tag list from Hugging Face.
- Auto-detects input layout (NHWC vs NCHW) and applies the right preprocessing.
- Supports single image or directory (recursive), batching, CUDA/CPU, CSV/JSONL outputs.

Author: you :)
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
import onnxruntime as ort
from huggingface_hub import hf_hub_download

HF_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"
MODEL_FILE = "model.onnx"
TAGS_FILE = "selected_tags.csv"

IMG_SIZE = 448
DEFAULT_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# -------------------------
# Utility
# -------------------------

def is_nhwc_input(session: ort.InferenceSession) -> bool:
    """
    Inspect ONNX input shape to detect whether the model expects NHWC or NCHW.
    NHWC example: [None, 448, 448, 3]
    NCHW example: [None, 3, 448, 448]
    """
    ishape = session.get_inputs()[0].shape
    # Some models may have dynamic dims as 'None' or 'None' strings.
    if len(ishape) != 4:
        # Fallback: most wd-eva02* ONNX on HF use NHWC
        return True
    try:
        # Treat 'None' / None as sentinel values; we only care relative positions
        c1 = ishape[1]
        c3 = ishape[3]
        # If channel last: last dim should be 3, middle dim should be 448 (or not 3)
        if c3 == 3:
            return True
        # If channel first: second dim is 3
        if c1 == 3:
            return False
    except Exception:
        pass
    # Conservative default: NHWC for this repo
    return True

def list_images(root: Path, recursive: bool, exts: Tuple[str, ...]) -> List[Path]:
    if root.is_file():
        return [root]
    if recursive:
        return [p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file()]
    else:
        return [p for p in root.glob("*") if p.suffix.lower() in exts and p.is_file()]

def load_tags(tags_csv_path: str) -> Tuple[List[str], List[int]]:
    """
    Read selected_tags.csv -> names + categories(int).
    Category encoding in this model:
      general = 0, character = 4, rating = 9
    """
    names: List[str] = []
    cats: List[int] = []
    with open(tags_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # expected columns: name, category, ...
        for row in reader:
            names.append(row["name"])
            # some CSVs may have Category / category as string
            cat_str = row.get("category") or row.get("Category") or "0"
            try:
                cats.append(int(cat_str))
            except Exception:
                # fallback if category is text like 'general'
                low = str(cat_str).strip().lower()
                if low.startswith("char"):
                    cats.append(4)
                elif low.startswith("rating"):
                    cats.append(9)
                else:
                    cats.append(0)
    return names, cats

# -------------------------
# Preprocess
# -------------------------

def pad_to_square_white(img_rgb: Image.Image) -> Image.Image:
    """
    Pad to square with white background.
    """
    w, h = img_rgb.size
    if w == h:
        return img_rgb
    m = max(w, h)
    canvas = Image.new("RGB", (m, m), (255, 255, 255))
    canvas.paste(img_rgb, ((m - w) // 2, (m - h) // 2))
    return canvas

def preprocess_nhwc_bgr(img: Image.Image, img_size: int = IMG_SIZE) -> np.ndarray:
    """
    NHWC path for wd-eva02-large-tagger-v3 (per official Space):
      - Remove transparency on white background
      - Pad to square on white
      - Resize (bicubic) to 448x448
      - Convert to numpy HWC, channels **BGR**, float32 in [0..255]
    """
    # Handle transparency by compositing over white
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    canvas = Image.new("RGBA", img.size, (255, 255, 255, 255))
    canvas.alpha_composite(img)
    img = canvas.convert("RGB")

    img = pad_to_square_white(img)
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size), Image.BICUBIC)

    arr = np.asarray(img, dtype=np.float32)  # HWC, RGB, 0..255
    arr = arr[:, :, ::-1]                    # HWC, BGR
    return arr  # HWC

def preprocess_nchw_rgb_norm(img: Image.Image, img_size: int = IMG_SIZE) -> np.ndarray:
    """
    Fallback NCHW path (if model declares NCHW):
      - Convert to RGB, white background for alpha
      - Pad square white, resize to 448
      - Convert to numpy CHW, float32 normalized to [-1, 1]
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    canvas = Image.new("RGBA", img.size, (255, 255, 255, 255))
    canvas.alpha_composite(img)
    img = canvas.convert("RGB")

    img = pad_to_square_white(img)
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size), Image.BICUBIC)

    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC RGB in [0,1]
    arr = arr * 2.0 - 1.0                            # [-1,1]
    arr = np.transpose(arr, (2, 0, 1))               # CHW
    return arr  # CHW

def collate_nhwc(batch: List[np.ndarray]) -> np.ndarray:
    return np.stack(batch, axis=0).astype(np.float32)  # N,H,W,C (BGR, 0..255)

def collate_nchw(batch: List[np.ndarray]) -> np.ndarray:
    return np.stack(batch, axis=0).astype(np.float32)  # N,C,H,W (RGB, [-1,1])

# -------------------------
# Inference & Postprocess
# -------------------------

def run_onnx(session: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    inputs = {session.get_inputs()[0].name: x}
    out = session.run(None, inputs)
    return out[0]  # [N, num_tags]

def split_and_filter(
    probs_1d: np.ndarray,
    tag_names: List[str],
    tag_cats: List[int],
    general_th: float,
    character_th: float
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Split by category and apply thresholds.
      rating: all (sorted)
      general: keep >= general_th
      character: keep >= character_th
    """
    rating, general, character = [], [], []
    for idx, (name, cat) in enumerate(zip(tag_names, tag_cats)):
        p = float(probs_1d[idx])
        if cat == 9:          # rating
            rating.append((name, p))
        elif cat == 4:        # character
            if p >= character_th:
                character.append((name, p))
        else:                 # general (0)
            if p >= general_th:
                general.append((name, p))

    rating.sort(key=lambda x: x[1], reverse=True)
    general.sort(key=lambda x: x[1], reverse=True)
    character.sort(key=lambda x: x[1], reverse=True)
    return {"rating": rating, "general": general, "character": character}

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Image tagger using SmilingWolf/wd-eva02-large-tagger-v3 (ONNX)")
    ap.add_argument("-i", "--input", required=True, help="Image file or directory")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories when input is a directory")
    ap.add_argument("--batch-size", type=int, default=4, help="Batch size")
    ap.add_argument("--use-gpu", action="store_true", help="Use CUDAExecutionProvider if available")
    ap.add_argument("--general-threshold", type=float, default=0.35, help="Threshold for general tags")
    ap.add_argument("--character-threshold", type=float, default=0.85, help="Threshold for character tags")
    ap.add_argument("--out-csv", type=str, default=None, help="Write results to CSV file")
    ap.add_argument("--out-jsonl", type=str, default=None, help="Write results to JSONL file")
    ap.add_argument("--repo", type=str, default=HF_REPO, help="Hugging Face repo id")
    ap.add_argument("--exts", type=str, default=",".join(DEFAULT_EXTS), help="Comma-separated file extensions")
    args = ap.parse_args()

    # download artifacts
    model_path = hf_hub_download(repo_id=args.repo, filename=MODEL_FILE)
    tags_path = hf_hub_download(repo_id=args.repo, filename=TAGS_FILE)

    # load tags
    tag_names, tag_cats = load_tags(tags_path)
    num_tags = len(tag_names)

    # providers
    if args.use_gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    sess_opt = ort.SessionOptions()
    session = ort.InferenceSession(model_path, sess_options=sess_opt, providers=providers)

    # detect layout
    nhwc = is_nhwc_input(session)
    layout_str = "NHWC (BGR, 0..255)" if nhwc else "NCHW (RGB, [-1,1])"
    print(f"> Detected model input layout: {layout_str}")

    # collect images
    exts = tuple([e.strip().lower() for e in args.exts.split(",") if e.strip()])
    inputs = list_images(Path(args.input), args.recursive, exts)
    if not inputs:
        print("No images found.", file=sys.stderr)
        sys.exit(1)

    # outputs
    csv_writer = None
    csv_fp = None
    if args.out_csv:
        csv_fp = open(args.out_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_fp)
        csv_writer.writerow(["path", "rating", "general", "character"])

    jsonl_fp = open(args.out_jsonl, "w", encoding="utf-8") if args.out_jsonl else None

    # buffers
    batch_imgs: List[np.ndarray] = []
    batch_paths: List[Path] = []

    def flush():
        if not batch_imgs:
            return
        x = collate_nhwc(batch_imgs) if nhwc else collate_nchw(batch_imgs)
        probs = run_onnx(session, x)  # [N, num_tags]
        if probs.shape[1] != num_tags:
            raise RuntimeError(f"Probability shape mismatch: got {probs.shape}, expected num_tags={num_tags}")

        for i in range(probs.shape[0]):
            per = split_and_filter(
                probs[i], tag_names, tag_cats,
                general_th=args.general_threshold,
                character_th=args.character_threshold
            )
            rating_text = ", ".join([f"{t}:{p:.3f}" for t, p in per["rating"][:4]])
            general_text = ", ".join([f"{t}:{p:.3f}" for t, p in per["general"][:64]])
            character_text = ", ".join([f"{t}:{p:.3f}" for t, p in per["character"][:64]])

            if len(inputs) <= 10:
                print(f"\n[{batch_paths[i]}]")
                print("  rating   :", rating_text)
                print("  general  :", general_text)
                print("  character:", character_text)

            if csv_writer:
                csv_writer.writerow([str(batch_paths[i]), rating_text, general_text, character_text])
            if jsonl_fp:
                obj = {
                    "path": str(batch_paths[i]),
                    "rating": per["rating"],
                    "general": per["general"],
                    "character": per["character"],
                }
                jsonl_fp.write(json.dumps(obj, ensure_ascii=False) + "\n")

        batch_imgs.clear()
        batch_paths.clear()

    # iterate
    for p in tqdm(inputs, desc="Tagging"):
        try:
            with Image.open(p) as img:
                if nhwc:
                    arr = preprocess_nhwc_bgr(img, IMG_SIZE)
                else:
                    arr = preprocess_nchw_rgb_norm(img, IMG_SIZE)
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
