#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
csv2jsonl_tags.py

把含 Danbooru 字段的 CSV 转为训练用 JSONL，字段映射如下：
- path        <- saved_path
- general     <- tags_general（以空格分隔）→ 输出用“, ”分隔
- rating      <- rating（s/q/e/g 映射为 sensitive/questionable/explicit/general；其他或空→""）
- meta        <- tags_meta（空格→“, ”）
- year        <- created_at（ISO8601，取年份，输出为 "year_YYYY"）
- character   <- tags_character（空格→“, ”）
- artist      <- artist_input（按原样输出）
- copyright   <- tags_copyright（空格→“, ”）

其他说明：
- 输入 CSV 列必须包含：saved_path, rating, created_at, tags_general, tags_meta, tags_character,
  tags_copyright, artist_input（若缺失会自动当作空）
- 所有“tags_*”源列中，源是**空格分隔**；输出统一用“, ”分隔。
- 空值统一输出为 ""（空字符串）。
"""

import argparse, csv, json, sys
from datetime import datetime

def parse_args():
    ap = argparse.ArgumentParser(description="Convert CSV to JSONL for tag training.")
    ap.add_argument("--csv", required=True, help="输入 CSV 路径")
    ap.add_argument("--out", required=True, help="输出 JSONL 路径")
    ap.add_argument("--encoding", default="utf-8", help="CSV 编码（默认 utf-8）")
    ap.add_argument("--dialect", default="excel", help="csv 方言（默认 excel）")
    return ap.parse_args()

def norm_tags_space_to_commas(s: str) -> str:
    """
    源为以空格分隔的 tags 字符串，转换为 ', ' 分隔。
    - 会去除多余空白；若为空或全空白 → 返回 ""。
    - 不去重、不排序（保持原始顺序）。
    """
    if not s:
        return ""
    # 拆分时将连续空白当作分隔符
    toks = [t for t in s.strip().split() if t]
    return ", ".join(toks) if toks else ""

def map_rating(r: str) -> str:
    """
    Danbooru rating -> 目标字符串：
      s -> sensitive
      q -> questionable
      e -> explicit
      g -> general
    其他/空 -> ""
    """
    if not r:
        return ""
    r = r.strip().lower()
    table = {
        "s": "sensitive",
        "q": "questionable",
        "e": "explicit",
        "g": "general",
    }
    return table.get(r, "")

def year_from_created_at(s: str) -> str:
    """
    从 ISO8601 时间取年份，返回 'year_YYYY'。
    失败或空 → ""。
    例：'2025-03-18T18:49:40.883-04:00' -> 'year_2025'
    """
    if not s:
        return ""
    s = s.strip()
    # Python 3.8+ 支持 datetime.fromisoformat 含偏移
    try:
        y = datetime.fromisoformat(s).year
        return f"year_{y}"
    except Exception:
        # 有些场景可能带 Z 或毫秒格式不同；可以做兜底简单截取
        try:
            y = int(s[:4])
            return f"year_{y}"
        except Exception:
            return ""

def get(field: str, row: dict) -> str:
    """安全取列值，None → ""，去掉首尾空白。"""
    v = row.get(field, "")
    if v is None:
        return ""
    return str(v).strip()

def row_to_example(row: dict) -> dict:
    saved_path   = get("saved_path", row)  # 作为输出 path
    rating_raw   = get("rating", row)
    created_at   = get("created_at", row)
    tags_general = get("tags_general", row)
    tags_meta    = get("tags_meta", row)
    tags_char    = get("tags_character", row)
    tags_copy    = get("tags_copyright", row)
    artist_input = get("artist_input", row)  # 指定使用 artist_input

    example = {
        "path":      saved_path,
        "general":   norm_tags_space_to_commas(tags_general),
        "rating":    map_rating(rating_raw),
        "meta":      norm_tags_space_to_commas(tags_meta),
        "year":      year_from_created_at(created_at),
        "character": norm_tags_space_to_commas(tags_char),
        "artist":    artist_input,  # 原样输出
        "copyright": norm_tags_space_to_commas(tags_copy),
    }
    return example

def main():
    args = parse_args()
    # 逐行读取、逐行写出，低内存
    with open(args.csv, "r", encoding=args.encoding, newline="") as f_in, \
         open(args.out, "w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in, dialect=args.dialect)
        # 容错：列名大小写/意外空白
        reader.fieldnames = [fn.strip() for fn in reader.fieldnames] if reader.fieldnames else None

        # 必要列缺失时给出提示但不强制退出（缺失列会被当作空）
        required = ["saved_path", "rating", "created_at", "tags_general", "tags_meta",
                    "tags_character", "tags_copyright", "artist_input"]
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            print(f"[WARN] CSV 缺少列：{missing}（将按空处理）", file=sys.stderr)

        count = 0
        for row in reader:
            ex = row_to_example(row)
            f_out.write(json.dumps(ex, ensure_ascii=False) + "\n")
            count += 1

    print(f"[OK] 已写出 {count} 行到 {args.out}")

if __name__ == "__main__":
    main()
