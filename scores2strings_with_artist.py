#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scores2strings_with_artist.py

功能
- 读取带分数标签的 JSON/JSONL（每条记录含 path、general、rating、meta、year、artist 等）。
- 各类别（除 artist 外）按分数降序，低于阈值过滤，拼成逗号分隔字符串。
- rating：仅保留分最高的一个，并去掉 'rating_' 等前缀。
- artist：按给定 artists.csv（id, "artist1, artist2"）
  1) 若 JSON 的 artist 与该 id 对应 CSV 中任意一项“同名”（见下方同名规则）→ 输出该 JSON 里的匹配 artist（原始字符串）；
  2) 否则若 JSON 的最高分 artist 分数 > 0.8 → 输出该 artist；
  3) 否则 → 输出 CSV 中该 id 的全部 artist（逗号分隔）。
- 若该 id 在 CSV 中不存在：
  - 仅执行规则 (2)；不满足则输出空字符串。

输入格式
- JSONL（每行一条 JSON），或一个 JSON 数组文件。

artists.csv 格式
810328,"marui, mikoto akemi",GroupA
810327,"kobuichi",GroupB
（第三列可选，表示 group；若文件含有该列，则输出 JSON 会额外包含 group 字段。）

注意：id 为 path 中 /webp/<id>/... 的这个目录名。

用法示例
python scores2strings_with_artist.py \
  --input tags_scored.jsonl \
  --output tags_flat.jsonl \
  --artists-csv artists.csv \
  --default-thresh 0.2 \
  --thresh general=0.3 meta=0.4 year=0.0 rating=0.6 \
  --topk 64
"""

import argparse, csv, io, json, os, re, sys
from typing import Any, Dict, Iterable, List, Optional, Tuple
# ====== 每类默认阈值（可被 --thresh 覆盖）======
CATEGORY_DEFAULT_THRESH = {
    "rating": 0.65,
    "character": 0.83,
    "meta": 0.6,
    "year": 0.6,
}

# ========== 通用解析 ==========

def parse_thresholds(th_args: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for s in th_args:
        if '=' not in s:
            raise ValueError(f"Invalid --thresh item '{s}', expected key=value")
        k, v = s.split('=', 1)
        out[k.strip()] = float(v)
    return out

def iter_records(path: str) -> Iterable[Dict[str, Any]]:
    with io.open(path, 'r', encoding='utf-8') as f:
        head = f.read(1)
        if not head:
            return
        f.seek(0)
        if head == '[':
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON starts with '[' but is not a list")
            for obj in data:
                if isinstance(obj, dict):
                    yield obj
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

def is_pair_list(val: Any) -> bool:
    if not isinstance(val, list):
        return False
    if len(val) == 0:
        return True
    first = val[0]
    return isinstance(first, (list, tuple)) and len(first) == 2 and isinstance(first[0], str)

# ========== rating 清洗 ==========

def normalize_rating(tag: str) -> str:
    t = tag.strip().lower()
    for pref in ("rating_", "rating-", "rate_", "rate-"):
        if t.startswith(pref):
            return t[len(pref):]
    return t

# ========== 从 path 抽取 id ==========

_WEBP_ID_RE = re.compile(r"/webp/([^/]+)/")

def extract_webp_id(path: str) -> Optional[str]:
    m = _WEBP_ID_RE.search(path)
    return m.group(1) if m else None

# ========== 读 artists.csv ==========

def load_artists_csv(csv_path: str) -> Tuple[Dict[str, Dict[str, Any]], bool]:
    """
    返回 ({id: {"artists": [...], "group": "xxx"}}, 是否存在第三列 group)
    """
    mapping: Dict[str, Dict[str, Any]] = {}
    has_group_col = False
    if not csv_path:
        return mapping, has_group_col
    with io.open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            id_ = row[0].strip()
            if not id_:
                continue
            artists_field = ""
            group_field: Optional[str] = None
            if len(row) >= 2:
                artists_field = row[1]
            if len(row) >= 3:
                group_field = row[2]
                has_group_col = True
            # 若 artist 字段在 CSV 中未被引号包裹导致被拆成多列，则把多余部分重新拼回去
            if len(row) > 3 and not group_field:
                artists_extra = row[2:]
                artists_field = ",".join([artists_field] + artists_extra)
            # 去掉外层可能的引号
            artists_field = artists_field.strip().strip('"').strip("'")
            group_field = group_field.strip().strip('"').strip("'") if group_field else None
            # 以逗号切分，并 strip
            artists = [a.strip() for a in artists_field.split(",") if a.strip()]
            mapping[id_] = {"artists": artists, "group": group_field or ""}
    return mapping, has_group_col

# ========== 同名判断（忠实复刻你的 C++ 逻辑） ==========

def to_lower_str(s: str) -> str:
    return s.lower()

def keep_alnum_only(s: str) -> str:
    return "".join(ch for ch in s if ("a" <= ch <= "z") or ("0" <= ch <= "9"))

def extract_base_norm(s: str) -> str:
    low = to_lower_str(s)
    p = low.find('(')
    base = low if p == -1 else low[:p]
    return keep_alnum_only(base)

def extract_aliases_norm(s: str) -> List[str]:
    low = to_lower_str(s)
    res: List[str] = []
    i = 0
    while True:
        l = low.find('(', i)
        if l == -1:
            break
        r = low.find(')', l + 1)
        if r == -1:
            break
        inside = low[l + 1 : r]
        token = []
        def flush():
            if token:
                k = keep_alnum_only("".join(token))
                if k:
                    res.append(k)
                token.clear()
        for c in inside:
            if c.isalnum():
                token.append(c)
            else:
                flush()
        flush()
        i = r + 1
    # 去重
    res = sorted(set(res))
    return res

def normalize_combined(s: str) -> str:
    return keep_alnum_only(to_lower_str(s))

def same_artist(a: str, b: str) -> bool:
    if normalize_combined(a) == normalize_combined(b):
        return True
    abase = extract_base_norm(a)
    bbase = extract_base_norm(b)
    if abase and abase == bbase:
        return True
    aals = extract_aliases_norm(a)
    bals = extract_aliases_norm(b)
    # 任意 a.alias == b.base
    if bbase and any(x == bbase for x in aals):
        return True
    # 任意 b.alias == a.base
    if abase and any(y == abase for y in bals):
        return True
    # 任意 alias 对 alias
    if any(x == y for x in aals for y in bals):
        return True
    return False

# ========== artist 专用决策 ==========

def parse_artist_pairs(value: Any) -> List[Tuple[str, float]]:
    """
    把 JSON 中的 artist 字段解析成 [(name, score), ...]，并按分数降序。
    允许兼容 ["a","b"] 或单纯字符串/空。
    """
    pairs: List[Tuple[str, float]] = []
    if is_pair_list(value):
        for x in value:
            try:
                pairs.append((str(x[0]), float(x[1])))
            except Exception:
                pass
    elif isinstance(value, list) and all(isinstance(t, str) for t in value):
        pairs = [(t, 1.0) for t in value]
    elif isinstance(value, str) and value.strip():
        pairs = [(value.strip(), 1.0)]
    pairs.sort(key=lambda p: (-p[1], p[0]))
    return pairs

def find_artist_for_record(artist_pairs: List[Tuple[str, float]],
                           artists_csv_list: List[str]) -> List[str]:
    """
    按三条规则决定输出的 artist 列表：
      1) 若任一 JSON 中的 artist 与 CSV 列表同名 → 返回该 JSON 里的那个 artist（立即返回）
      2) 否则若 JSON 最高分 > 0.8 → 返回该最高分的 artist
      3) 否则 → 返回 CSV 列表（可多项）
    若 CSV 列表为空：仅走 (2)，否则返回空。
    """
    # 1) 同名
    if artists_csv_list:
        for danbooru_name, _score in artist_pairs:
            for hitomi_name in artists_csv_list:
                if same_artist(hitomi_name, danbooru_name):
                    return [hitomi_name]
        if len(artists_csv_list) == 1:
            # 仅有一项时，若同名失败，则不再尝试第二条规则，直接回退到 CSV
            return list(artists_csv_list)

    # 2) 顶项 > 0.90
    if artist_pairs and artist_pairs[0][1] > 0.90:
        return [artist_pairs[0][0]]

    # 3) 回退到 CSV
    if artists_csv_list:
        return list(artists_csv_list)

    return []

# ========== 其他类别处理（延续原逻辑） ==========

def format_category(name: str, value: Any, thr: float, topk: Optional[int]) -> str:
    if is_pair_list(value):
        pairs: List[Tuple[str, float]] = []
        for it in value:
            try:
                tag, score = str(it[0]), float(it[1])
            except Exception:
                continue
            if score >= thr:
                pairs.append((tag, score))
        pairs.sort(key=lambda x: (-x[1], x[0]))
        if name.lower() == "rating":
            return normalize_rating(pairs[0][0]) if pairs else ""
        if topk is not None and topk > 0:
            pairs = pairs[:topk]
        return ", ".join(t for t, _ in pairs)

    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        if name.lower() == "rating":
            return normalize_rating(value[0]) if value else ""
        return ", ".join(value)

    if isinstance(value, str):
        return value

    return ""

def process_record(rec: Dict[str, Any],
                   thresholds: Dict[str, float],
                   default_thr: float,
                   topk: Optional[int],
                   artists_map: Dict[str, Dict[str, Any]],
                   artists_has_group: bool,
                   type_map: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": rec.get("path", "")}

    # 先处理非 artist
    for k, v in rec.items():
        if k in ("path", "artist"):
            continue
        thr = thresholds.get(k, CATEGORY_DEFAULT_THRESH.get(k, default_thr))
        out[k] = format_category(k, v, thr, topk)

    # 处理 artist（特殊逻辑）
    path = out["path"]
    rec_id = extract_webp_id(path) or ""
    # 判断artists_map是否包含该 id
    if rec_id not in artists_map:
        print(f"!! Warning: id '{rec_id}' from path '{path}' not found in artists CSV.")
    entry = artists_map.get(rec_id)
    artists_csv_list = entry["artists"] if entry else []

    artist_pairs = parse_artist_pairs(rec.get("artist", []))
    chosen = find_artist_for_record(artist_pairs, artists_csv_list)
    out["artist"] = ", ".join(chosen) if chosen else ""
    if artists_has_group:
        out["group"] = entry.get("group", "") if entry else ""

    # 附加 type（基于额外 CSV，id 为 path 中的 webp 目录名）
    out["type"] = type_map.get(rec_id, "")

    return out

# ========== 主程序 ==========

def main():
    ap = argparse.ArgumentParser(description="Convert scored tag JSON/JSONL to string-tag JSONL (with special artist rules).")
    ap.add_argument("--input", required=True, help="输入 JSON/JSONL 文件")
    ap.add_argument("--output", required=True, help="输出 JSONL 文件")
    ap.add_argument("--artists-csv", default="", help="artists.csv 路径（id, \"artist1, artist2\"）")
    ap.add_argument("--type-csv", default="", help="包含 id 与类型映射的 CSV，倒数第二列视为 id，倒数第三列视为 type（例：... ,Game CG,2871902,1）")
    ap.add_argument("--default-thresh", type=float, default=0.0, help="除已单独指定外的默认阈值")
    ap.add_argument("--thresh", nargs="*", default=[], help="逐类别阈值，例如：general=0.3 meta=0.4 year=0 rating=0.6")
    ap.add_argument("--topk", type=int, default=0, help="非 rating 类别的 Top-K，0 表示不限")
    args = ap.parse_args()

    thresholds = parse_thresholds(args.thresh)
    topk = args.topk if args.topk > 0 else None

    if args.artists_csv:
        artists_map, has_group_column = load_artists_csv(args.artists_csv)
    else:
        artists_map, has_group_column = {}, False

    type_map: Dict[str, str] = {}
    if args.type_csv:
        try:
            with io.open(args.type_csv, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 3:
                        continue
                    id_col = row[-2].strip()
                    type_col = row[-3].strip()
                    if id_col and type_col:
                        type_map[id_col] = type_col
        except FileNotFoundError:
            print(f"!! Warning: type CSV '{args.type_csv}' not found, type 字段将为空。")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    cnt_in = 0
    cnt_out = 0
    with io.open(args.output, 'w', encoding='utf-8') as fw:
        for rec in iter_records(args.input):
            cnt_in += 1
            out = process_record(
                rec,
                thresholds,
                args.default_thresh,
                topk,
                artists_map,
                has_group_column,
                type_map,
            )
            fw.write(json.dumps(out, ensure_ascii=False))
            fw.write("\n")
            cnt_out += 1

    print(f"Done. Read {cnt_in}, wrote {cnt_out} → {args.output}")

if __name__ == "__main__":
    main()
