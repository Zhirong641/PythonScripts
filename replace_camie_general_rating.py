#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
replace_camie_general_rating_preserve_naked_skirt.py

功能
- 用 tags_wd.jsonl 中同一 path 的 general 与 rating 完整替换 tags_camie.jsonl 中对应条目的 general 与 rating。
- 特例：保留 camie 的 general 里名为 "naked_skirt" 的标签及其 score（wd 中通常没有该标签），
  若 wd.general 里没有该标签，则把它补回去（按分数排序）。
- 其它字段（character/copyright/artist/meta/year 等）保持不变。
- 若某个 path 在 wd 中不存在，则保留 camie 原值并计数提示。
- 支持 .jsonl 与 .jsonl.gz。
"""

import argparse, gzip, json, os, sys
from typing import Dict, Any, List, Tuple, Optional

# 只在 general 组里保留这些标签（可扩展）
PRESERVE_GENERAL_TAGS = {"naked_skirt"}

def open_auto(path: str, mode: str = "rt", encoding: str = "utf-8"):
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding=encoding)
    return open(path, mode, encoding=encoding)

def load_jsonl_map(wd_path: str) -> Dict[str, Dict[str, Any]]:
    """
    读取 wd jsonl，构建 path -> {"general": [...], "rating": [...]} 的映射
    """
    m: Dict[str, Dict[str, Any]] = {}
    total = 0
    with open_auto(wd_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            p = obj.get("path")
            if not p:
                continue
            m[p] = {
                "general": obj.get("general"),
                "rating": obj.get("rating"),
            }
            total += 1
            if total % 500000 == 0:
                print(f"[wd] loaded {total} rows...", file=sys.stderr)
    print(f"[wd] total loaded: {total}", file=sys.stderr)
    return m

def normalize_pairs(x: Optional[List]) -> List[Tuple[str, float]]:
    """
    归一化一个形如 [["tag", score], ...] 的列表，过滤非法项，转成 [(tag, score), ...]
    """
    out: List[Tuple[str, float]] = []
    if not x:
        return out
    for it in x:
        if isinstance(it, (list, tuple)) and len(it) >= 2:
            tag = str(it[0])
            try:
                score = float(it[1])
            except Exception:
                continue
            out.append((tag, score))
    return out

def dump_pairs(pairs: List[Tuple[str, float]]) -> List[List[Any]]:
    """
    把 [(tag, score), ...] 转回 [["tag", score], ...]
    """
    return [[t, s] for t, s in pairs]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camie", required=True, help="tags_camie.jsonl(.gz)")
    ap.add_argument("--wd", required=True, help="tags_wd.jsonl(.gz)")
    ap.add_argument("--out", required=True, help="output jsonl(.gz allowed)")
    ap.add_argument("--indent", type=int, default=None, help="调试用：缩进输出（默认 None 更省空间）")
    args = ap.parse_args()

    wd_map = load_jsonl_map(args.wd)

    tmp_out = args.out + ".tmp"
    replaced, missing, total = 0, 0, 0

    out_f = gzip.open(tmp_out, "wt", encoding="utf-8") if args.out.endswith(".gz") else open(tmp_out, "wt", encoding="utf-8")

    with open_auto(args.camie, "rt", encoding="utf-8") as f_in, out_f:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # 非法 JSON 行跳过
                continue

            p = obj.get("path")
            total += 1

            if p and p in wd_map:
                wd_entry = wd_map[p]

                # 1) 从 camie.general 里抓取需要保留的标签及 score
                camie_general_pairs = normalize_pairs(obj.get("general"))
                preserve_from_camie: Dict[str, float] = {}
                for tag, score in camie_general_pairs:
                    if tag in PRESERVE_GENERAL_TAGS:
                        preserve_from_camie[tag] = score

                # 2) 用 wd 的 general/rating 覆盖
                wd_general_pairs = normalize_pairs(wd_entry.get("general"))
                wd_rating_pairs = normalize_pairs(wd_entry.get("rating"))

                # 3) 把保留标签补回 general（若 wd.general 里没有）
                wd_general_dict = {t: s for t, s in wd_general_pairs}
                changed = False
                for tag, score in preserve_from_camie.items():
                    if tag not in wd_general_dict:
                        wd_general_dict[tag] = score
                        changed = True

                # 如有补回/变动，重建并按 score 降序排序，保持稳定
                if changed:
                    wd_general_pairs = sorted(wd_general_dict.items(), key=lambda x: x[1], reverse=True)

                # 4) 写回对象
                obj["general"] = dump_pairs(wd_general_pairs)
                obj["rating"]  = dump_pairs(wd_rating_pairs)
                replaced += 1
            else:
                # 没有 wd 记录：保留 camie 原值
                missing += 1

            out_f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), indent=args.indent) + "\n")

            if total % 500000 == 0:
                print(f"[camie] processed {total} rows... (replaced={replaced}, missing={missing})", file=sys.stderr)

    # 原子替换
    if os.path.exists(args.out):
        os.remove(args.out)
    os.rename(tmp_out, args.out)

    print(f"Done. total={total}, replaced={replaced}, missing_in_wd={missing}", file=sys.stderr)
    print(f"Output: {args.out}")

if __name__ == "__main__":
    main()
