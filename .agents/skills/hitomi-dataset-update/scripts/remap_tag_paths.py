#!/usr/bin/env python3
"""Atomically remap scored JSONL image paths from staging to the dataset root."""

import argparse
import json
import os
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--from-root", type=Path, required=True)
    parser.add_argument("--to-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()

    source_root = args.from_root.resolve()
    target_root = args.to_root.resolve()
    fd, name = tempfile.mkstemp(prefix=args.jsonl.name + ".", suffix=".tmp", dir=args.jsonl.parent)
    os.close(fd)
    temporary = Path(name)
    count = 0
    paths = set()
    try:
        with args.jsonl.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as output:
            for line_number, line in enumerate(source, start=1):
                obj = json.loads(line)
                relative = Path(obj["path"]).relative_to(source_root)
                target = target_root / relative
                if not target.is_file():
                    raise ValueError(f"Missing target image at line {line_number}: {target}")
                path = str(target)
                if path in paths:
                    raise ValueError(f"Duplicate target image path: {path}")
                paths.add(path)
                obj["path"] = path
                output.write(json.dumps(obj, ensure_ascii=False) + "\n")
                count += 1
            output.flush()
            os.fsync(output.fileno())
        if count != args.expected_count:
            raise ValueError(f"Expected {args.expected_count} rows, got {count}")
        os.replace(temporary, args.jsonl)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Remapped {count} rows in {args.jsonl}")


if __name__ == "__main__":
    main()
