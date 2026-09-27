"""Identify completed Hitomi galleries without requiring a legacy ids.csv file."""

import csv
import os
import re
from collections import Counter
from pathlib import Path


IMAGE_NAME = re.compile(r"image_(\d+)\.(?:webp|jpg|jpeg|png|gif|avif)", re.IGNORECASE)


def read_duplicate_ids(path: Path) -> dict[str, str]:
    """Map excluded duplicate gallery IDs to the already-kept gallery ID."""
    if not path.is_file():
        return {}
    result = {}
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not {"duplicate_id", "canonical_id"} <= set(reader.fieldnames or []):
            raise ValueError(f"Invalid duplicate gallery CSV: {path}")
        for row in reader:
            duplicate = (row.get("duplicate_id") or "").strip()
            canonical = (row.get("canonical_id") or "").strip()
            if not duplicate.isdigit() or not canonical.isdigit() or duplicate == canonical:
                raise ValueError(f"Invalid duplicate gallery mapping: {row}")
            if duplicate in result and result[duplicate] != canonical:
                raise ValueError(f"Conflicting duplicate gallery mapping: {duplicate}")
            result[duplicate] = canonical
    return result


def read_legacy_ids(path: Path) -> set[str]:
    """Read the gallery ID in column five when the optional legacy file exists."""
    if not path.is_file():
        return set()
    ids = set()
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.reader(source):
            if len(row) > 4 and row[4].strip().isdigit():
                ids.add(row[4].strip())
    return ids


def read_recorded_image_keys(path: Path) -> set[tuple[str, int]]:
    """Read the current batch CSV so resumed downloads can repair missing rows."""
    if not path.is_file():
        return set()
    keys = set()
    with path.open(newline="", encoding="utf-8") as source:
        for line_number, row in enumerate(csv.reader(source), start=1):
            if len(row) != 6 or not row[4].isdigit() or not row[5].isdigit():
                raise ValueError(f"Invalid image CSV row {line_number} in {path}: {row}")
            key = (row[4], int(row[5]))
            if key in keys:
                raise ValueError(f"Duplicate image CSV row for {key} in {path}")
            keys.add(key)
    return keys


def read_verified_cumulative_ids(csv_path: Path, images_root: Path) -> set[str]:
    """Trust an ID only when its CSV row count matches contiguous, nonempty images."""
    if not csv_path.is_file() or not images_root.is_dir():
        return set()

    expected = Counter()
    with csv_path.open(newline="", encoding="utf-8") as source:
        for row in csv.reader(source):
            if len(row) > 5 and row[4].strip().isdigit() and row[5].strip().isdigit():
                expected[row[4].strip()] += 1

    complete = set()
    for game_id, count in expected.items():
        directory = images_root / game_id
        if not directory.is_dir():
            continue
        indices = set()
        files = 0
        valid = True
        for entry in os.scandir(directory):
            match = IMAGE_NAME.fullmatch(entry.name)
            if not match or not entry.is_file(follow_symlinks=False):
                continue
            if entry.stat().st_size == 0:
                valid = False
                break
            files += 1
            indices.add(int(match.group(1)))
        if valid and files == count and indices == set(range(1, count + 1)):
            complete.add(game_id)
    return complete
