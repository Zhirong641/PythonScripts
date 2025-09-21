#!/usr/bin/env python3

"""Extract artist tags from Camie tagger metadata into a CSV file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List


def load_artist_names(metadata_path: Path) -> List[str]:
    """Return sorted artist tag names that belong to the `artist` category."""

    with metadata_path.open(encoding="utf-8") as fp:
        metadata = json.load(fp)

    tag_to_category = (
        metadata
        .get("dataset_info", {})
        .get("tag_mapping", {})
        .get("tag_to_category", {})
    )

    if not tag_to_category:
        raise ValueError(
            "`tag_to_category` mapping not found in metadata file: "
            f"{metadata_path}"
        )

    artists = {
        tag for tag, category in tag_to_category.items() if category == "artist"
    }

    return sorted(artists)

def load_tags(metadata_path: Path, categories: List[str]) -> List[str]:
    """Return sorted tag names that belong to the specified categories."""

    with metadata_path.open(encoding="utf-8") as fp:
        metadata = json.load(fp)

    tag_to_category = (
        metadata
        .get("dataset_info", {})
        .get("tag_mapping", {})
        .get("tag_to_category", {})
    )

    if not tag_to_category:
        raise ValueError(
            "`tag_to_category` mapping not found in metadata file: "
            f"{metadata_path}"
        )

    tags = {
        tag for tag, category in tag_to_category.items() if category in categories
    }

    return sorted(tags)

def write_csv(artists: Iterable[str], output_path: Path) -> None:
    """Write artist names to `output_path` with a single `artist` column."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["artist"])
        for name in artists:
            writer.writerow([name])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract all artist tags from camie-tagger metadata JSON and save as CSV"
        )
    )
    parser.add_argument(
        "metadata_json",
        type=Path,
        help="Path to camie-tagger-v2-metadata.json",
    )
    parser.add_argument(
        "output_csv",
        type=Path,
        help="Destination CSV file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # artists = load_artist_names(args.metadata_json)
    all_tags = load_tags(args.metadata_json, ["general", "character"])
    write_csv(all_tags, args.output_csv)
    print(f"Saved {len(all_tags)} tags to {args.output_csv}")


if __name__ == "__main__":
    main()

# import csv, sys
# seen = set()
# out = []
# with open('../data/artists.csv', newline='', encoding='utf-8') as f:
#     for row in csv.reader(f):
#         if len(row) < 2: 
#             continue
#         for name in row[1].split(','):
#             n = name.strip()
#             k = n.casefold()
#             if n and k not in seen:
#                 seen.add(k)
#                 out.append(n)

# with open('artist_hitomi.csv', 'w', newline='', encoding='utf-8') as g:
#     w = csv.writer(g)
#     w.writerow(['name'])
#     for n in out:
#         w.writerow([n])




