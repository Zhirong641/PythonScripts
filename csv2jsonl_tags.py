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
- artist      <- tags_artist（空格→“, ”；同时用 alias.csv 转为 hitomi 格式）
- copyright   <- tags_copyright（空格→“, ”）

其他说明：
- 输入 CSV 列必须包含：saved_path, rating, created_at, tags_general, tags_meta, tags_character,
  tags_copyright, tags_artist（若缺失会自动当作空）
- 所有“tags_*”源列中，源是**空格分隔**；输出统一用“, ”分隔。
- 空值统一输出为 ""（空字符串）。
"""

import argparse, csv, json, sys
from datetime import datetime
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser(description="Convert CSV to JSONL for tag training.")
    ap.add_argument("--csv", required=True, help="输入 CSV 路径")
    ap.add_argument("--out", required=True, help="输出 JSONL 路径")
    ap.add_argument("--encoding", default="utf-8", help="CSV 编码（默认 utf-8）")
    ap.add_argument("--dialect", default="excel", help="csv 方言（默认 excel）")
    ap.add_argument("--alias", default="csv/alias.csv", help="artist 别名表，danbooru -> hitomi（默认 csv/alias.csv）")
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

def load_alias_map(path: str) -> dict:
    """读取 artist 别名，返回 {danbooru: hitomi} 映射。缺失或错误时返回空映射。"""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[WARN] alias 文件不存在：{path}（将直接使用原 tags_artist）", file=sys.stderr)
        return {}
    alias = {}
    try:
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dan = (row.get("artist_danbooru") or "").strip()
                hitomi = (row.get("artist_hitomi") or "").strip()
                if dan and hitomi:
                    alias[dan] = hitomi
    except Exception as e:
        print(f"[WARN] 读取 alias 文件失败：{e}（将直接使用原 tags_artist）", file=sys.stderr)
    return alias

def map_artists(tags_artist: str, alias_map: dict) -> str:
    """
    将 tags_artist（空格分隔）映射为 hitomi 格式并用 ', ' 连接。
    - 空或失败返回 ""。
    """
    if not tags_artist:
        return ""
    toks = [t for t in tags_artist.strip().split() if t]
    if not toks:
        return ""
    mapped = [alias_map.get(t, t) for t in toks]
    return ", ".join(mapped)

def row_to_example(row: dict, alias_map: dict) -> dict:
    saved_path   = get("saved_path", row)  # 作为输出 path
    rating_raw   = get("rating", row)
    created_at   = get("created_at", row)
    tags_general = get("tags_general", row)
    tags_meta    = get("tags_meta", row)
    tags_char    = get("tags_character", row)
    tags_copy    = get("tags_copyright", row)
    tags_artist  = get("tags_artist", row)

    example = {
        "path":      saved_path,
        "general":   norm_tags_space_to_commas(tags_general),
        "rating":    map_rating(rating_raw),
        "meta":      norm_tags_space_to_commas(tags_meta),
        "year":      year_from_created_at(created_at),
        "character": norm_tags_space_to_commas(tags_char),
        "artist":    map_artists(tags_artist, alias_map),
        "copyright": norm_tags_space_to_commas(tags_copy),
        "type":      "danbooru",
    }
    return example

def main():
    args = parse_args()
    alias_map = load_alias_map(args.alias)
    # 逐行读取、逐行写出，低内存
    with open(args.csv, "r", encoding=args.encoding, newline="") as f_in, \
         open(args.out, "w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in, dialect=args.dialect)
        # 容错：列名大小写/意外空白
        reader.fieldnames = [fn.strip() for fn in reader.fieldnames] if reader.fieldnames else None

        # 必要列缺失时给出提示但不强制退出（缺失列会被当作空）
        required = ["saved_path", "rating", "created_at", "tags_general", "tags_meta",
                    "tags_character", "tags_copyright", "tags_artist"]
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            print(f"[WARN] CSV 缺少列：{missing}（将按空处理）", file=sys.stderr)

        count = 0
        for row in reader:
            ex = row_to_example(row, alias_map)
            f_out.write(json.dumps(ex, ensure_ascii=False) + "\n")
            count += 1

    print(f"[OK] 已写出 {count} 行到 {args.out}")

if __name__ == "__main__":
    main()
