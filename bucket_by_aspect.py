#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
按图片宽高比分桶（JSONL 输入）。
- 读取每行 JSON，打开 "src" 对应图片，仅读取尺寸（不会完整解码）
- 计算宽高比，按照模式（gcd / nearest / bin）分桶
- 统计每桶数量，并把原始 JSON 行写入各桶对应的 JSONL 文件
- 支持 LRU 文件句柄缓存（避免同时打开过多文件）
- 对无法打开的图片会写入 error.jsonl，方便后续排查

依赖：
  - Pillow
  - tqdm

安装：
  pip install pillow tqdm
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter, OrderedDict
from math import gcd
from typing import Dict, Tuple, Optional

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


COMMON_RATIOS = {
    "1:1": 1.0,
    "3:4": 3/4,
    "4:3": 4/3,
    "2:3": 2/3,
    "3:2": 3/2,
    "9:16": 9/16,
    "16:9": 16/9,
    "5:7": 5/7,
    "7:5": 7/5,
    "1:2": 1/2,
    "2:1": 2/1,
    "21:9": 21/9,
    "9:21": 9/21,
    "5:4": 5/4,
    "4:5": 4/5,
}

# BIN_RANGES = [
#     # (name, min_ratio_inclusive, max_ratio_exclusive)
#     ("ultra_tall",    0.0,    0.50),  # 非常狭长的竖图（w/h < 0.5）
#     ("tall",          0.50,   0.75),
#     ("portrait",      0.75,   0.95),
#     ("square",        0.95,   1.05),
#     ("landscape",     1.05,   1.33),
#     ("wide",          1.33,   1.78),
#     ("ultra_wide",    1.78,   float("inf")),
# ]

BIN_RANGES = [
    ("tachi-e", 0.2, 0.95),          # 竖向（w/h < 0.95）→ 立绘
    ("cg",      0.95, 3), # 方图与横向（w/h >= 0.95）→ 一般CG
]



def parse_args():
    ap = argparse.ArgumentParser(description="按图片宽高比分桶并输出每桶 JSONL")
    ap.add_argument("jsonl", help="输入 JSONL 文件路径（每行一个 JSON 对象，包含 'src' 键）")
    ap.add_argument("--outdir", default="buckets", help="桶文件输出目录（默认：buckets）")
    ap.add_argument("--mode", choices=["gcd", "nearest", "bin"], default="gcd",
                    help="分桶模式：gcd=精确比例；nearest=就近常见比例；bin=范围分箱")
    ap.add_argument("--tolerance", type=float, default=0.015,
                    help="nearest 模式下的相对误差阈值（默认 0.015 = 1.5%）")
    ap.add_argument("--max-open", type=int, default=64,
                    help="同时打开的桶文件最大数量（默认 64）")
    ap.add_argument("--save-counts", default=None,
                    help="把桶计数另存为 CSV（可选），例如 counts.csv")
    ap.add_argument("--errors-file", default="errors.jsonl",
                    help="无法读取图片的行写到该文件（默认 errors.jsonl）")
    return ap.parse_args()


def ratio_label_gcd(w: int, h: int) -> str:
    g = gcd(w, h) or 1
    return f"{w // g}:{h // g}"


def ratio_label_nearest(w: int, h: int, tolerance: float) -> str:
    r = w / h
    # 选相对误差最小的常见比例
    best_name: Optional[str] = None
    best_rel_err = float("inf")
    for name, val in COMMON_RATIOS.items():
        rel_err = abs(r - val) / val
        if rel_err < best_rel_err:
            best_rel_err = rel_err
            best_name = name
    # 若相对误差超出阈值，回落到精确 gcd 比例
    if best_name is None or best_rel_err > tolerance:
        return  "unknown"  # ratio_label_gcd(w, h)
    return best_name


def ratio_label_bin(w: int, h: int) -> str:
    r = w / h
    for name, lo, hi in BIN_RANGES:
        if lo <= r < hi:
            return name
    return "unknown"


def sanitize_bucket_name(label: str) -> str:
    # 文件名友好：把 "3:4" → "3x4"，其余保留
    return label.replace(":", "x").replace("/", "_")


class LRUFileCache:
    """限制同时打开的文件数量，超过则最久未使用的句柄会被关闭。"""
    def __init__(self, capacity: int):
        self.capacity = max(1, capacity)
        self._cache: "OrderedDict[str, any]" = OrderedDict()

    def get(self, path: Path):
        key = str(path)
        if key in self._cache:
            f = self._cache.pop(key)
            self._cache[key] = f  # move to end (recently used)
            return f
        # open new
        if len(self._cache) >= self.capacity:
            old_key, old_f = self._cache.popitem(last=False)  # LRU
            try:
                old_f.close()
            except Exception:
                pass
        f = open(path, "a", encoding="utf-8")
        self._cache[key] = f
        return f

    def close_all(self):
        for _, f in self._cache.items():
            try:
                f.close()
            except Exception:
                pass
        self._cache.clear()


def main():
    args = parse_args()
    in_path = Path(args.jsonl)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    errors_fp = open(args.errors_file, "w", encoding="utf-8")

    # 统计计数
    counts: Counter = Counter()

    # LRU 文件缓存
    file_cache = LRUFileCache(capacity=args.max_open)

    # 逐行流式处理
    total = sum(1 for _ in open(in_path, "r", encoding="utf-8", errors="ignore"))
    with open(in_path, "r", encoding="utf-8", errors="ignore") as fin, tqdm(total=total, desc="Bucketing") as pbar:
        for line in fin:
            line = line.strip()
            pbar.update(1)
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # 非法 JSON 行
                errors_fp.write(line + "\n")
                continue

            src = obj.get("src") or obj.get("path") or ""
            if not src:
                errors_fp.write(json.dumps({"error": "missing src", "line": obj}, ensure_ascii=False) + "\n")
                continue

            # 读取图片尺寸（不会完整解码）
            try:
                with Image.open(src) as im:
                    w, h = im.size
            except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
                errors_fp.write(json.dumps({"error": f"{type(e).__name__}: {str(e)}", "src": src, "line": obj}, ensure_ascii=False) + "\n")
                continue

            if w <= 0 or h <= 0:
                errors_fp.write(json.dumps({"error": "invalid size", "size": [w, h], "src": src, "line": obj}, ensure_ascii=False) + "\n")
                continue

            # 生成桶标签
            if args.mode == "gcd":
                label = ratio_label_gcd(w, h)
            elif args.mode == "nearest":
                label = ratio_label_nearest(w, h, args.tolerance)
            else:
                label = ratio_label_bin(w, h)

            counts[label] += 1

            # 写入对应桶的 JSONL
            bucket_name = f"ratio_{sanitize_bucket_name(label)}.jsonl"
            bucket_path = outdir / bucket_name
            fp = file_cache.get(bucket_path)
            # 原样写入输入行（不改变字段内容）
            fp.write(line + "\n")

    # 关闭所有文件
    file_cache.close_all()
    errors_fp.close()

    # 打印统计结果（按数量降序）
    print("\n=== Bucket counts ===")
    for label, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{label:>10s} : {c}")

    # 可选导出到 CSV
    if args.save_counts:
        csv_path = Path(args.save_counts)
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("bucket,count\n")
            for label, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
                f.write(f"{label},{c}\n")
        print(f"\n已保存计数到: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
