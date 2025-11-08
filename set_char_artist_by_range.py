#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re, sys, tempfile, os
from typing import List, Tuple, Optional

# 需要修改的区间（目录ID, 起始序号, 结束序号，角色，画师，均闭区间）
RANGES: List[Tuple[int, int, int, str, str]] = [
    (727768, 345, 461, "rindou tsubame", "chikotam"),
    (727768, 630, 647, "rindou tsubame", "chikotam"),
    (727768, 3, 25, "takakura anzu", "primil"),
    (727768, 31, 106, "takakura anzu", "primil"),
    (727768, 107, 220, "takakura anri", "primil"),
    (688336, 625, 741, "rindou tsubame", "chikotam"),
    (688336, 34, 147, "takakura anri", "primil"),
    (688336, 173, 195, "takakura anzu", "primil"),
    (688336, 173, 195, "takakura anzu", "primil"),
    (688336, 201, 276, "takakura anzu", "primil"),
    (1093883, 1130, 1829, "misuzu sasa", "chikotam"),
    (1146404, 12, 13, None, "shiramori yuse"),
    (1146404, 407, 582, None, "shiramori yuse"),
    (1146404, 891, 1017, None, "shiramori yuse"),
    (3000014, 35, 137, None, "kimishima ao"),
    (3000014, 220, 324, None, "shiratama"),
    (2746430, 13, 74, None, "shiratama"),
    (2746430, 219, 259, None, "shiratama"),
    (2746430, 278, 340, None, "shiratama"),
    (1920176, 2, 66, None, "shiratama"),
    (1920176, 116, 212, None, "shiratama"),
    (1920176, 259, 380, None, "shiratama"),
    (1920176, 429, 474, None, "shiratama"),
    (1562101, 50, 89, None, "shiratama"),
    (1562101, 155, 196, None, "shiratama"),
    (868607, 39, 141, None, "kimishima ao"),
    (868607, 224, 328, None, "shiratama"),
    (1245707, 13, 23, "amanogawa saya", "yashima takahiro"),
    (1245707, 48, 72, "amanogawa saya", "yashima takahiro"),
    (1245707, 107, 130, "amanogawa saya", "yashima takahiro"),
    (1245707, 305, 318, "amanogawa saya", "yashima takahiro"),
    (943537, 618, 837, "amanogawa saya", "yashima takahiro"),
    (900491, 1283, 1540, "amanogawa saya", "yashima takahiro"),
    (634594, 11, 13, "futaba hisui", "nanase meruchi"),
    (634594, 17, 19, "futaba hisui", "nanase meruchi"),
    (634594, 221, 292, "futaba hisui", "nanase meruchi"),
    (899895, 3, 26, "yanase hitomi", "primil"),
    (522375, 46, 191, "hondou ayano", "primil"),
    (522375, 298, 440, "amamoto louis", "primil"),
    (847794, 1, 418, "natsuki rino", None),
    (979189, 1, 53, "mito kohaku", None),
    (979189, 54, 111, "saijou hifumi", None),
    (979189, 228, 250, "mito kohaku", None),
    (979189, 251, 271, "saijou hifumi", None),

    (1056434, 22, 32, "mito kohaku", None),
    (1056434, 33, 43, "saijou hifumi", None),
    (1056434, 54, 65, "yunohana nano", None),
    (1056434, 91, 132, "mito kohaku", None),
    (1056434, 133, 172, "saijou hifumi", None),
    (1056434, 211, 247, "yunohana nano", None),

    (1993857, 49, 71, "mito kohaku", None),
    (1993857, 72, 92, "saijou hifumi", None),
    (1993857, 119, 137, "yunohana nano", None),
    (1993857, 138, 141, "yunohana nano, mito kohaku", None),
    (1993857, 143, 193, "mito kohaku", None),
    (1993857, 194, 251, "saijou hifumi", None),
    (1993857, 312, 367, "yunohana nano", None),

    (633524, 2, 141, "luce yami asutarite", "yamakaze ran"),
    (633524, 142, 277, "julia lin road", "sakurazaka tsuchiyu"),
    (633524, 278, 405, "mitsu no tama yori hime", "yamakaze ran"),
    (633524, 406, 553, "amagi karin", "yamakaze ran"),
    (633524, 672, 815, "shirahase yuuna", "yamakaze ran"),

    (1217027, 3, 142, "luce yami asutarite", "yamakaze ran"),
    (1217027, 143, 278, "julia lin road", "sakurazaka tsuchiyu"),
    (1217027, 279, 406, "mitsu no tama yori hime", "yamakaze ran"),
    (1217027, 407, 554, "amagi karin", "yamakaze ran"),
    (1217027, 673, 816, "shirahase yuuna", "yamakaze ran"),

    (634833, 3, 142, "luce yami asutarite", "yamakaze ran"),
    (634833, 143, 278, "julia lin road", "sakurazaka tsuchiyu"),
    (634833, 279, 406, "mitsu no tama yori hime", "yamakaze ran"),
    (634833, 407, 554, "amagi karin", "yamakaze ran"),
    (634833, 673, 816, "shirahase yuuna", "yamakaze ran"),

    (1805418, 1, 2000, None, "kaniya shiku, konomi, yuzuna hiyo"),
    (1805420, 1, 2000, None, "kaniya shiku, konomi, yuzuna hiyo"),
    (2313627, 1, 2000, None, "mizuno sao, satasama, yuzuna hiyo"),
    (1537715, 1, 2000, None, "yuzuna hiyo"),
    (1537491, 1, 2000, None, "yuzuna hiyo"),
    (1537567, 1, 2000, None, "yuzuna hiyo"),
]

# 提取目录与图片序号：.../webp/<dir>/image_<num>.webp
PATH_RE = re.compile(r"/webp/(\d+)/image_(\d+)\.webp$")

def lookup_targets(dir_id: int, num: int) -> Optional[Tuple[str, str]]:
    for d, start, end, character, artist in RANGES:
        if dir_id == d and start <= num <= end:
            return character, artist
    return None

def process_file(in_path: str, out_path: str) -> int:
    modified = 0
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for lineno, line in enumerate(fin, 1):
            s = line.strip()
            if not s:
                fout.write(line)
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                # 非法 JSON，原样写回并提示
                sys.stderr.write(f"[WARN] Line {lineno}: JSON decode error: {e}\n")
                fout.write(line)
                continue

            path = obj.get("path", "")
            m = PATH_RE.search(path)
            if m:
                dir_id = int(m.group(1))
                num = int(m.group(2))
                targets = lookup_targets(dir_id, num)
                if targets:
                    target_character, target_artist = targets
                    # 命中范围：覆盖 character 与 artist
                    if (target_character and obj.get("character") != target_character) or (target_artist and obj.get("artist") != target_artist):
                        obj["character"] = target_character if target_character else obj.get("character", "")
                        obj["artist"] = target_artist if target_artist else obj.get("artist", "")
                        modified += 1

            # 紧凑写回，保持一行一个 JSON
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return modified

def main():
    ap = argparse.ArgumentParser(description="Set character/artist for specific image ranges in JSONL.")
    ap.add_argument("input", help="input JSONL path")
    ap.add_argument("output", nargs="?", help="output JSONL path (omit when using --inplace)")
    ap.add_argument("--inplace", action="store_true", help="overwrite the input file in place")
    args = ap.parse_args()

    if args.inplace:
        if args.output:
            ap.error("Do not provide OUTPUT when using --inplace.")
        # 写到临时文件再替换，避免半写入损坏
        dir_ = os.path.dirname(os.path.abspath(args.input)) or "."
        fd, tmp_path = tempfile.mkstemp(prefix=".jsonl_tmp_", dir=dir_, text=True)
        os.close(fd)
        try:
            changed = process_file(args.input, tmp_path)
            os.replace(tmp_path, args.input)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        print(f"Done. Modified lines: {changed}")
    else:
        if not args.output:
            ap.error("OUTPUT is required unless using --inplace.")
        changed = process_file(args.input, args.output)
        print(f"Done. Modified lines: {changed}")

if __name__ == "__main__":
    main()
