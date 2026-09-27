#!/usr/bin/env python3
"""Atomically add a checked batch of tags and remove aesthetic_tag everywhere."""

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path


def count_lines(path):
    with path.open("rb") as source:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: source.read(1024 * 1024), b""))


def count_images(root):
    count = 0
    for _, _, names in os.walk(root):
        count += len(names)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-tags", type=Path, required=True)
    parser.add_argument("--new-tags", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cumulative-csv", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()

    expected = set(args.manifest.read_text(encoding="utf-8").splitlines())
    if not expected or len(expected) != count_lines(args.manifest):
        raise ValueError("Empty or duplicate image manifest")
    canonical_root = args.images_root.resolve(strict=True)
    for image_path in expected:
        path = Path(image_path)
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(canonical_root)
        except ValueError as exc:
            raise ValueError(f"Image path is outside the cumulative webp root: {image_path}") from exc
        if (not path.is_absolute() or str(resolved) != image_path or
                len(relative.parts) != 2 or not relative.parts[0].isdigit() or
                not re.fullmatch(r"image_\d+\.(?:webp|jpg|jpeg|png|gif|avif)", relative.name)):
            raise ValueError(f"Noncanonical image path in manifest: {image_path}")

    new_paths = set()
    with args.new_tags.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            obj = json.loads(line)
            path = obj.get("path")
            if path not in expected or path in new_paths:
                raise ValueError(f"Unexpected or duplicate new tag path at line {line_number}: {path}")
            if "aesthetic_tag" in obj:
                raise ValueError(f"aesthetic_tag in new tag at line {line_number}")
            if not all(isinstance(obj.get(key), str) for key in (
                "general", "character", "copyright", "meta", "year",
                "rating", "artist", "group", "type",
            )):
                raise ValueError(f"Invalid tag fields at line {line_number}")
            if not Path(path).is_file():
                raise ValueError(f"Missing image at line {line_number}: {path}")
            new_paths.add(path)
    if new_paths != expected:
        raise ValueError(f"New tags missing {len(expected - new_paths)} images")

    old_count = count_lines(args.existing_tags)
    expected_total = old_count + len(new_paths)
    csv_count = count_lines(args.cumulative_csv)
    image_count = count_images(args.images_root)
    if expected_total != csv_count or expected_total != image_count:
        raise ValueError(
            f"Count mismatch: old tags {old_count} + new {len(new_paths)} = {expected_total}, "
            f"CSV {csv_count}, images {image_count}"
        )
    if args.backup.exists():
        raise FileExistsError(f"Backup already exists: {args.backup}")

    original_stat = args.existing_tags.stat()
    shutil.copy2(args.existing_tags, args.backup)
    descriptor, name = tempfile.mkstemp(
        prefix=args.existing_tags.name + ".", suffix=".tmp",
        dir=args.existing_tags.parent,
    )
    os.close(descriptor)
    temporary = Path(name)
    seen = set()
    try:
        with args.backup.open(encoding="utf-8") as old, \
                args.new_tags.open(encoding="utf-8") as new, \
                temporary.open("w", encoding="utf-8") as output:
            for line_number, line in enumerate(old, start=1):
                obj = json.loads(line)
                path = obj.get("path")
                if not isinstance(path, str) or path in seen:
                    raise ValueError(f"Invalid or duplicate existing path at line {line_number}: {path}")
                seen.add(path)
                obj.pop("aesthetic_tag", None)
                output.write(json.dumps(obj, ensure_ascii=False) + "\n")
            if len(seen) != old_count or seen & new_paths:
                raise ValueError("Existing tags contain duplicate or overlapping paths")
            for line in new:
                output.write(line)
            output.flush()
            os.fsync(output.fileno())

        current_stat = args.existing_tags.stat()
        if (current_stat.st_size, current_stat.st_mtime_ns) != (
            original_stat.st_size, original_stat.st_mtime_ns
        ):
            raise RuntimeError("Existing tags changed during finalization")
        if count_lines(temporary) != expected_total:
            raise RuntimeError("Candidate tag file has the wrong line count")
        os.chmod(temporary, original_stat.st_mode)
        os.replace(temporary, args.existing_tags)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Updated {args.existing_tags}: {expected_total} tags, no aesthetic_tag")
    print(f"Backup: {args.backup}")


if __name__ == "__main__":
    main()
