#!/usr/bin/env python3
# csv_to_jsonl_exact.py
import csv, json, sys, io

def open_text(path, mode, encoding):
    if path == "-":
        buf = sys.stdin.buffer if "r" in mode else sys.stdout.buffer
        return io.TextIOWrapper(buf, encoding=encoding, newline="")
    return open(path, mode, encoding=encoding, newline="")

def to_jsonl(input_csv: str, output_jsonl: str = "-", encoding: str = "utf-8-sig"):
    # 读取时不去空格，保持原样；让 csv 处理引号与逗号
    with open_text(input_csv, "r", encoding) as f, open_text(output_jsonl, "w", "utf-8") as out:
        # 自动识别分隔符（逗号/制表符/分号/竖线）；失败则退回逗号
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
        except csv.Error:
            dialect = csv.excel
        dialect.skipinitialspace = False  # 关键：不去掉分隔符后的空格

        reader = csv.DictReader(f, dialect=dialect)
        fieldnames = reader.fieldnames or []

        for row in reader:
            # 将 None 转为 ""；其余全部转为 str，保持原样
            obj = {k: ("" if row.get(k) is None else str(row.get(k))) for k in fieldnames}
            out.write(json.dumps(obj, ensure_ascii=False))
            out.write("\n")

if __name__ == "__main__":
    # 用法：
    #   python csv_to_jsonl_exact.py input.csv > output.jsonl
    #   python csv_to_jsonl_exact.py - -  # 从 stdin 到 stdout
    in_path = sys.argv[1] if len(sys.argv) > 1 else "-"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "-"
    to_jsonl(in_path, out_path)
