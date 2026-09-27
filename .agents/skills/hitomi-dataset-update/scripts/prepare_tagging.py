#!/usr/bin/env python3
"""Build a stable image manifest for an incremental Hitomi tagging run."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_path(root, game_id, index):
    directory = root / game_id
    matches = [path for path in directory.glob(f"image_{index}.*") if path.is_file()]
    if len(matches) != 1 or matches[0].stat().st_size == 0:
        raise ValueError(f"Expected one nonempty image for {game_id}/{index}, got {matches}")
    return matches[0].resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-csv", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--existing-tags", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    batch_copy = run_dir / "batch.csv"
    source_hash = file_hash(args.batch_csv)
    if batch_copy.exists():
        if file_hash(batch_copy) != source_hash:
            raise ValueError(f"Existing {batch_copy} differs from --batch-csv")
    else:
        shutil.copy2(args.batch_csv, batch_copy)

    paths = []
    seen = set()
    with batch_copy.open(newline="", encoding="utf-8") as source:
        for line_number, row in enumerate(csv.reader(source), start=1):
            if len(row) != 6 or not row[4].isdigit() or not row[5].isdigit():
                raise ValueError(f"Invalid batch CSV row {line_number}: {row}")
            key = (row[4], row[5])
            if key in seen:
                raise ValueError(f"Duplicate gallery/image index in batch CSV: {key}")
            seen.add(key)
            paths.append(str(image_path(args.images_root, *key)))

    wanted = set(paths)
    already_tagged = set()
    with args.existing_tags.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            obj = json.loads(line)
            path = obj.get("path")
            if path in wanted:
                if path in already_tagged:
                    raise ValueError(f"Duplicate existing tag path at line {line_number}: {path}")
                already_tagged.add(path)
    missing = [path for path in paths if path not in already_tagged]
    manifest = run_dir / "images.txt"
    content = "".join(path + "\n" for path in missing)
    if manifest.exists() and manifest.read_text(encoding="utf-8") != content:
        raise ValueError(f"Existing {manifest} has a different image set")
    manifest.write_text(content, encoding="utf-8")

    summary = {
        "batch_csv_sha256": source_hash,
        "batch_rows": len(paths),
        "already_tagged": len(already_tagged),
        "images_to_tag": len(missing),
        "manifest": str(manifest),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
