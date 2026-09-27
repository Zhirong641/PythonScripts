#!/usr/bin/env python3
"""Require WD and Camie JSONL outputs to cover the image manifest exactly."""

import argparse
import json
from pathlib import Path


def check_output(path, expected, fields):
    seen = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            obj = json.loads(line)
            image_path = obj.get("path")
            if image_path not in expected or image_path in seen:
                raise ValueError(f"{path}:{line_number}: unexpected or duplicate {image_path}")
            if not all(isinstance(obj.get(field), list) for field in fields):
                raise ValueError(f"{path}:{line_number}: missing tag category")
            seen.add(image_path)
    missing = expected - seen
    if missing:
        raise ValueError(f"{path}: missing {len(missing)} paths, e.g. {sorted(missing)[:3]}")
    print(f"{path}: {len(seen)} unique paths, complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wd", type=Path, required=True)
    parser.add_argument("--camie", type=Path, required=True)
    args = parser.parse_args()

    lines = args.manifest.read_text(encoding="utf-8").splitlines()
    expected = set(lines)
    if not expected or len(lines) != len(expected):
        raise ValueError("Empty or duplicate image manifest")
    check_output(args.wd, expected, ("general", "rating", "character"))
    check_output(
        args.camie, expected,
        ("general", "rating", "character", "copyright", "artist", "meta", "year"),
    )


if __name__ == "__main__":
    main()
