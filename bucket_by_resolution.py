#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
按图片“分辨率(宽x高)”分桶（JSONL 输入）。

特性：
- 读取每行 JSON，优先从 JSON 字段读取 width/height（可选），否则按需打开图片仅读尺寸
- 多种分桶模式：exact / snap / short / long / mp / grid
- 统计每桶数量，并把原始 JSON 行写入各桶对应的 JSONL 文件
- 支持 LRU 文件句柄缓存（避免同时打开过多文件）
- 对无法打开或尺寸异常的图片写入 errors.jsonl，便于排查

依赖：
  - Pillow
  - tqdm

安装：
  pip install pillow tqdm
"""

import argparse
import json
from pathlib import Path
from collections import Counter, OrderedDict
from typing import Dict, Tuple, Optional, List

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


# 常见“目标分辨率”（可按需增删），用于 --mode snap
COMMON_RESOLUTIONS = [
    (512, 512),
    (640, 360), (640, 480),
    (720, 480),
    (800, 600),
    (896, 512),
    (960, 540),
    (1024, 576), (1024, 640), (1024, 1024),
    (1152, 648), (1152, 768),
    (1280, 720), (1280, 800), (1280, 960),
    (1344, 768),
    (1536, 864), (1536, 960),
    (1600, 900), (1600, 1200),
    (1792, 1024),
    (1920, 1080), (1920, 1200),
    (2048, 1152), (2048, 1536),
    (2560, 1440), (2560, 1600),
    (2816, 1536),
    (3200, 1800),
    (3840, 2160),
]

# 默认短边/长边分箱边界（单位：像素）
DEFAULT_SHORT_BINS = [0, 256, 512, 768, 1024, 1536, 2048, 10_000_000]
DEFAULT_LONG_BINS  = [0, 512, 768, 1024, 1536, 2048, 3072, 4096, 10_000_000]

# 默认 MP（百万像素）分箱边界，单位：百万像素
DEFAULT_MP_BINS = [0.0, 0.5, 1.0, 2.0, 4.0, 9999.0]


def parse_bins(arg: Optional[str], default_bins: List[float]) -> List[float]:
    """
    解析逗号分隔的边界列表，返回递增边界（最后自动补大值）。
    示例：--short-bins 0,256,512,768,1024,1536,2048,10000000
    """
    if not arg:
        return default_bins
    vals = []
    for part in arg.split(","):
        v = float(part.strip())
        vals.append(v)
    if sorted(vals) != vals:
        raise ValueError("bin 边界必须递增")
    return vals


def label_from_bin(value: float, bins: List[float], unit: str = "") -> str:
    """
    返回形如 "[a,b)" / ">=a" 的区间标签。
    """
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if lo <= value < hi:
            a = int(lo) if unit != "MP" else lo
            b = int(hi) if unit != "MP" else hi
            return f"[{a}{unit},{b}{unit})"
    return f">={int(bins[-1])}{unit}" if unit != "MP" else f">={bins[-1]}{unit}"


def sanitize_bucket_name(label: str) -> str:
    # 更友好的文件名
    return (
        label.replace(" ", "")
             .replace(":", "x")
             .replace("/", "_")
             .replace("[", "")
             .replace(")", "")
             .replace(",", "_")
             .replace("]", "")
             .replace(">=", "ge")
    )


class LRUFileCache:
    """限制同时打开的文件数量，超过则最久未使用的句柄会被关闭。"""
    def __init__(self, capacity: int):
        self.capacity = max(1, capacity)
        self._cache: "OrderedDict[str, any]" = OrderedDict()

    def get(self, path: Path):
        key = str(path)
        if key in self._cache:
            f = self._cache.pop(key)
            self._cache[key] = f  # move to end
            return f
        if len(self._cache) >= self.capacity:
            _, old_f = self._cache.popitem(last=False)
            try:
                old_f.close()
            except Exception:
                pass
        path.parent.mkdir(parents=True, exist_ok=True)
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


def parse_res_list(s: Optional[str]) -> List[Tuple[int, int]]:
    """
    解析分辨率列表字符串，例如：
      "1024x1024,1152x648,1280x720"
    """
    if not s:
        return COMMON_RESOLUTIONS
    out = []
    for part in s.split(","):
        part = part.strip().lower()
        if "x" not in part:
            raise ValueError(f"非法分辨率：{part}")
        w, h = part.split("x", 1)
        out.append((int(w), int(h)))
    return out


def nearest_resolution_label(w: int, h: int, cand: List[Tuple[int, int]], rel_tol: float) -> str:
    """
    找到与 (w,h) 在相对误差意义下最近的候选分辨率。
    误差度量：max(|w-w0|/w0, |h-h0|/h0)
    """
    best = None
    best_err = float("inf")
    for w0, h0 in cand:
        if w0 == 0 or h0 == 0:
            continue
        ew = abs(w - w0) / w0
        eh = abs(h - h0) / h0
        err = max(ew, eh)
        if err < best_err:
            best_err = err
            best = (w0, h0)
    if best is None or best_err > rel_tol:
        return "unknown"
    return f"{best[0]}x{best[1]}"


def grid_round(v: int, base: int) -> int:
    """
    四舍五入到最接近的 base 倍数。
    """
    if base <= 1:
        return v
    q, r = divmod(v, base)
    if r >= base / 2:
        return (q + 1) * base
    return q * base


def try_get_wh_from_json(obj: dict, keys_w=("width", "w"), keys_h=("height", "h")) -> Optional[Tuple[int, int]]:
    """
    优先从 JSON 对象中直接读取宽高（减少磁盘 IO）。
    """
    w = None
    h = None
    for k in keys_w:
        if k in obj:
            try:
                w = int(obj[k])
                break
            except Exception:
                pass
    for k in keys_h:
        if k in obj:
            try:
                h = int(obj[k])
                break
            except Exception:
                pass
    if w and h and w > 0 and h > 0:
        return w, h
    return None


def parse_args():
    ap = argparse.ArgumentParser(description="按图片分辨率分桶并输出每桶 JSONL")
    ap.add_argument("jsonl", help="输入 JSONL 文件路径（每行一个 JSON 对象，包含 'src' 或 'path' 键）")
    ap.add_argument("--outdir", default="buckets_res", help="桶文件输出目录（默认：buckets_res）")
    ap.add_argument("--mode", choices=["exact", "snap", "short", "long", "mp", "grid"], default="exact",
                    help="分桶模式："
                         "exact=精确 WxH；"
                         "snap=就近常见分辨率；"
                         "short=按短边分箱；"
                         "long=按长边分箱；"
                         "mp=按百万像素分箱；"
                         "grid=四舍五入到指定步长（如 64）再按 WxH 分桶")
    ap.add_argument("--prefer-json-size", action="store_true",
                    help="若 JSON 行包含 width/height，则直接使用，避免打开图片")
    ap.add_argument("--snap-candidates",
                    help="逗号分隔的分辨率列表用于 snap 模式，例如：1024x1024,1152x648,1280x720")
    ap.add_argument("--snap-tolerance", type=float, default=0.02,
                    help="snap 模式相对误差阈值（默认 0.02 = 2%）")
    ap.add_argument("--short-bins", help="短边分箱边界，逗号分隔（像素）。不传则用默认边界")
    ap.add_argument("--long-bins", help="长边分箱边界，逗号分隔（像素）。不传则用默认边界")
    ap.add_argument("--mp-bins", help="MP 分箱边界，逗号分隔（单位：百万像素），如 0,0.5,1,2,4,9999")
    ap.add_argument("--grid-base", type=int, default=64, help="grid 模式的步长（默认 64）")
    ap.add_argument("--max-open", type=int, default=64, help="同时打开的桶文件最大数量（默认 64）")
    ap.add_argument("--save-counts", default=None, help="把桶计数另存为 CSV（例如 counts_res.csv）")
    ap.add_argument("--errors-file", default="errors.jsonl", help="无法读取图片的行写到该文件（默认 errors.jsonl）")
    return ap.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.jsonl)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    errors_fp = open(args.errors_file, "w", encoding="utf-8")
    counts: Counter = Counter()
    file_cache = LRUFileCache(capacity=args.max_open)

    # 预处理参数
    snap_list = parse_res_list(args.snap_candidates) if args.mode == "snap" else []
    short_bins = parse_bins(args.short_bins, DEFAULT_SHORT_BINS)
    long_bins = parse_bins(args.long_bins, DEFAULT_LONG_BINS)
    mp_bins = parse_bins(args.mp_bins, DEFAULT_MP_BINS)

    total = sum(1 for _ in open(in_path, "r", encoding="utf-8", errors="ignore"))
    with open(in_path, "r", encoding="utf-8", errors="ignore") as fin, tqdm(total=total, desc="Bucketing (resolution)") as pbar:
        for line in fin:
            pbar.update(1)
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                errors_fp.write(s + "\n")
                continue

            src = obj.get("src") or obj.get("path") or ""
            if not src and not args.prefer-json-size:  # 若没有路径，且不能从 JSON 取尺寸，也无法处理
                errors_fp.write(json.dumps({"error": "missing src", "line": obj}, ensure_ascii=False) + "\n")
                continue

            wh = try_get_wh_from_json(obj) if args.prefer_json_size else None
            if wh is None:
                if not src:
                    errors_fp.write(json.dumps({"error": "missing src and no width/height", "line": obj}, ensure_ascii=False) + "\n")
                    continue
                # 打开图片仅读尺寸
                try:
                    with Image.open(src) as im:
                        w, h = im.size
                except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
                    errors_fp.write(json.dumps({"error": f"{type(e).__name__}: {str(e)}", "src": src, "line": obj}, ensure_ascii=False) + "\n")
                    continue
            else:
                w, h = wh

            if w <= 0 or h <= 0:
                errors_fp.write(json.dumps({"error": "invalid size", "size": [w, h], "src": src, "line": obj}, ensure_ascii=False) + "\n")
                continue

            # 生成桶标签
            if args.mode == "exact":
                label = f"{w}x{h}"
            elif args.mode == "snap":
                label = nearest_resolution_label(w, h, snap_list, args.snap_tolerance)
            elif args.mode == "short":
                short_side = min(w, h)
                label = f"short_{label_from_bin(short_side, short_bins)}"
            elif args.mode == "long":
                long_side = max(w, h)
                label = f"long_{label_from_bin(long_side, long_bins)}"
            elif args.mode == "mp":
                mp = (w * h) / 1_000_000.0
                label = f"mp_{label_from_bin(mp, mp_bins, unit='MP')}"
            else:  # grid
                gw = grid_round(w, args.grid_base)
                gh = grid_round(h, args.grid_base)
                label = f"{gw}x{gh}"

            counts[label] += 1

            # 写入桶
            bucket_name = f"res_{sanitize_bucket_name(label)}.jsonl"
            bucket_path = outdir / bucket_name
            fp = file_cache.get(bucket_path)
            fp.write(s + "\n")

    file_cache.close_all()
    errors_fp.close()

    # 打印统计
    print("\n=== Bucket counts (resolution) ===")
    for label, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{label:>18s} : {c}")

    # 导出 CSV
    if args.save_counts:
        csv_path = Path(args.save_counts)
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("bucket,count\n")
            for label, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
                f.write(f"{label},{c}\n")
        print(f"\n已保存计数到: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
