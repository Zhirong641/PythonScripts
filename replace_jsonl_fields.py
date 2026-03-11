#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
replace_jsonl_fields.py

Replace selected fields in base JSONL using another JSONL as an update source,
matched by a key (default: "path").

Added:
- Preserve specific tags for a tag-string field (e.g. general) when replacing:
    --preserve-field general --preserve-tags no_bra,no_pantsu

Example:
  python replace_jsonl_fields.py \
    --base file1.jsonl --update file2.jsonl -o out.jsonl \
    --key path --fields general \
    --preserve-field general --preserve-tags no_bra
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Any


@dataclass
class Stats:
    base_lines: int = 0
    update_lines: int = 0
    update_indexed: int = 0
    update_bad_lines: int = 0
    base_bad_lines: int = 0
    matched: int = 0
    changed_records: int = 0
    field_replacements: int = 0
    preserved_tags_appended: int = 0
    skipped_empty_updates: int = 0
    skipped_missing_fields: int = 0


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def parse_fields(fields_str: str) -> List[str]:
    fields = [f.strip() for f in fields_str.split(",") if f.strip()]
    if not fields:
        raise ValueError("No fields specified.")
    return fields


def iter_jsonl(path: str, encoding: str = "utf-8") -> Iterable[Tuple[int, str]]:
    with open(path, "r", encoding=encoding) as f:
        for lineno, line in enumerate(f, start=1):
            yield lineno, line.rstrip("\n")


def json_load_line(line: str) -> Optional[dict]:
    line = line.strip()
    if not line:
        return None
    return json.loads(line)


def is_empty_value(v) -> bool:
    # treat "", None, [] and {} as empty; keep 0/False as non-empty
    if v is None:
        return True
    if isinstance(v, str) and v == "":
        return True
    if isinstance(v, (list, dict)) and len(v) == 0:
        return True
    return False


def _dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def parse_tags(value: Any) -> List[str]:
    """
    Parse tags from a field value.
    Supports:
      - "a, b, c"
      - "a b c" (fallback if no commas)
      - ["a","b"]
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if "," in s:
            parts = [p.strip() for p in s.split(",")]
            return [p for p in parts if p]
        # fallback: whitespace split
        parts = [p.strip() for p in s.split()]
        return [p for p in parts if p]
    # other types: treat as single token
    t = str(value).strip()
    return [t] if t else []


def format_tags(tags: List[str], original_value: Any) -> Any:
    """
    Format tags back to the same style as original_value.
    If original was a list, return list.
    Otherwise return comma+space string.
    """
    tags = _dedup_keep_order(tags)
    if isinstance(original_value, list):
        return tags
    return ", ".join(tags)


def merge_preserved_tags(
    old_value: Any,
    new_value: Any,
    preserve_set: set,
) -> Tuple[Any, int]:
    """
    Keep tags that exist in old_value AND in preserve_set, even after replacement.
    Returns (merged_value, appended_count).
    """
    old_tags = parse_tags(old_value)
    new_tags = parse_tags(new_value)

    keep = [t for t in old_tags if t in preserve_set]
    # append those not already in new
    appended = [t for t in keep if t not in set(new_tags)]
    if appended:
        merged_tags = new_tags + appended
        # Keep output style based on new_value (so replace result remains consistent)
        return format_tags(merged_tags, new_value), len(appended)
    return new_value, 0


class UpdateIndexMemory:
    def __init__(self):
        self._map: Dict[str, dict] = {}

    def put(self, key: str, obj: dict, on_duplicate: str):
        if key in self._map:
            if on_duplicate == "first":
                return
            if on_duplicate == "error":
                raise ValueError(f"Duplicate key in update: {key}")
        self._map[key] = obj  # last

    def get(self, key: str) -> Optional[dict]:
        return self._map.get(key)

    def size(self) -> int:
        return len(self._map)


class UpdateIndexSqlite:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS updates (
                k TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            );
        """)
        self.conn.commit()

    def put(self, key: str, obj: dict, on_duplicate: str):
        payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        if on_duplicate == "first":
            self.conn.execute(
                "INSERT OR IGNORE INTO updates(k, payload) VALUES(?, ?);",
                (key, payload),
            )
        elif on_duplicate == "error":
            cur = self.conn.execute("SELECT 1 FROM updates WHERE k=?;", (key,))
            if cur.fetchone() is not None:
                raise ValueError(f"Duplicate key in update: {key}")
            self.conn.execute(
                "INSERT INTO updates(k, payload) VALUES(?, ?);",
                (key, payload),
            )
        else:
            self.conn.execute(
                "INSERT INTO updates(k, payload) VALUES(?, ?) "
                "ON CONFLICT(k) DO UPDATE SET payload=excluded.payload;",
                (key, payload),
            )

    def get(self, key: str) -> Optional[dict]:
        cur = self.conn.execute("SELECT payload FROM updates WHERE k=?;", (key,))
        row = cur.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def size(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM updates;")
        return int(cur.fetchone()[0])

    def commit(self):
        self.conn.commit()

    def close(self):
        try:
            self.conn.commit()
        finally:
            self.conn.close()


def build_update_index(
    update_path: str,
    key_field: str,
    mode: str,
    sqlite_path: Optional[str],
    on_duplicate: str,
    ignore_errors: bool,
    stats: Stats,
) -> object:
    if mode == "sqlite":
        if not sqlite_path:
            sqlite_path = "update_index.sqlite"
        idx = UpdateIndexSqlite(sqlite_path)
    else:
        idx = UpdateIndexMemory()

    for lineno, line in iter_jsonl(update_path):
        stats.update_lines += 1
        if not line.strip():
            continue
        try:
            obj = json_load_line(line)
            if obj is None:
                continue
            if not isinstance(obj, dict):
                raise ValueError("JSON line is not an object")
            k = obj.get(key_field)
            if not isinstance(k, str) or not k:
                raise ValueError(f"Missing/invalid key field '{key_field}'")
            idx.put(k, obj, on_duplicate=on_duplicate)
        except Exception as ex:
            stats.update_bad_lines += 1
            if not ignore_errors:
                raise RuntimeError(
                    f"[update] parse error at {update_path}:{lineno}: {ex}\nLINE: {line}"
                ) from ex

    if mode == "sqlite":
        idx.commit()
    stats.update_indexed = idx.size()
    return idx


def replace_fields(
    base_path: str,
    out_path: str,
    update_index: object,
    key_field: str,
    fields: List[str],
    skip_empty_update: bool,
    ignore_errors: bool,
    encoding: str,
    preserve_field: Optional[str],
    preserve_tags: List[str],
    stats: Stats,
):
    preserve_set = set([t.strip() for t in preserve_tags if t.strip()])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", encoding=encoding) as out:
        for lineno, line in iter_jsonl(base_path, encoding=encoding):
            stats.base_lines += 1
            if not line.strip():
                out.write("\n")
                continue

            try:
                obj = json_load_line(line)
                if obj is None:
                    out.write("\n")
                    continue
                if not isinstance(obj, dict):
                    raise ValueError("JSON line is not an object")

                k = obj.get(key_field)
                if not isinstance(k, str) or not k:
                    out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    continue

                upd = update_index.get(k)
                if upd is None:
                    out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    continue

                stats.matched += 1
                changed = False

                for f in fields:
                    if f not in upd:
                        stats.skipped_missing_fields += 1
                        continue

                    new_v = upd.get(f)
                    if skip_empty_update and is_empty_value(new_v):
                        stats.skipped_empty_updates += 1
                        continue

                    old_v = obj.get(f, None)

                    # Preserve tags only when:
                    # - preserve_field matches current field
                    # - preserve_tags provided
                    if preserve_set and preserve_field and f == preserve_field:
                        merged_v, appended_cnt = merge_preserved_tags(
                            old_value=old_v,
                            new_value=new_v,
                            preserve_set=preserve_set,
                        )
                        if appended_cnt:
                            stats.preserved_tags_appended += appended_cnt
                        new_v = merged_v

                    if old_v != new_v:
                        obj[f] = new_v
                        stats.field_replacements += 1
                        changed = True

                if changed:
                    stats.changed_records += 1

                out.write(json.dumps(obj, ensure_ascii=False) + "\n")

            except Exception as ex:
                stats.base_bad_lines += 1
                if not ignore_errors:
                    raise RuntimeError(
                        f"[base] parse/replace error at {base_path}:{lineno}: {ex}\nLINE: {line}"
                    ) from ex
                out.write(line + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Replace JSONL fields in base file using update file matched by a key."
    )
    ap.add_argument("--base", required=True, help="Base JSONL (to be modified).")
    ap.add_argument("--update", required=True, help="Update JSONL (source of new values).")
    ap.add_argument("-o", "--out", required=True, help="Output JSONL.")
    ap.add_argument("--key", default="path", help="Key field name used for matching. Default: path")
    ap.add_argument(
        "--fields",
        default="general",
        help='Comma-separated field names to replace. Default: "general"',
    )
    ap.add_argument(
        "--skip-empty-update",
        action="store_true",
        help="If set, do not overwrite with empty values (''/null/[]/{}).",
    )
    ap.add_argument(
        "--mode",
        choices=["memory", "sqlite"],
        default="memory",
        help="Index mode for update file. Default: memory. Use sqlite for very large update files.",
    )
    ap.add_argument(
        "--sqlite-path",
        default=None,
        help="SQLite DB path when --mode sqlite (default: update_index.sqlite).",
    )
    ap.add_argument(
        "--on-duplicate",
        choices=["last", "first", "error"],
        default="last",
        help="How to handle duplicate keys in update file. Default: last",
    )
    ap.add_argument(
        "--ignore-errors",
        action="store_true",
        help="Ignore bad JSON lines and keep going (bad base lines are copied as-is).",
    )
    ap.add_argument("--encoding", default="utf-8", help="File encoding. Default: utf-8")

    # NEW: preserve tags on a specific field
    ap.add_argument(
        "--preserve-field",
        default=None,
        help="Field name to apply tag-preserve behavior (e.g. general). Default: disabled",
    )
    ap.add_argument(
        "--preserve-tags",
        default="",
        help='Comma-separated tags to preserve from base when replacing preserve-field (e.g. "no_bra,no_pantsu").',
    )

    args = ap.parse_args()
    fields = parse_fields(args.fields)
    preserve_tags = [t.strip() for t in args.preserve_tags.split(",") if t.strip()]

    stats = Stats()

    eprint(f"[1/2] Building update index ({args.mode}) from: {args.update}")
    idx = build_update_index(
        update_path=args.update,
        key_field=args.key,
        mode=args.mode,
        sqlite_path=args.sqlite_path,
        on_duplicate=args.on_duplicate,
        ignore_errors=args.ignore_errors,
        stats=stats,
    )
    eprint(f"  indexed keys: {stats.update_indexed:,} (update lines: {stats.update_lines:,}, bad: {stats.update_bad_lines:,})")

    eprint(f"[2/2] Replacing fields {fields} in: {args.base} -> {args.out}")
    if args.preserve_field and preserve_tags:
        eprint(f"  preserve tags on field '{args.preserve_field}': {preserve_tags}")

    replace_fields(
        base_path=args.base,
        out_path=args.out,
        update_index=idx,
        key_field=args.key,
        fields=fields,
        skip_empty_update=args.skip_empty_update,
        ignore_errors=args.ignore_errors,
        encoding=args.encoding,
        preserve_field=args.preserve_field,
        preserve_tags=preserve_tags,
        stats=stats,
    )

    if args.mode == "sqlite":
        idx.close()

    eprint("\nDone.")
    eprint(f"base lines:        {stats.base_lines:,} (bad: {stats.base_bad_lines:,})")
    eprint(f"matched records:   {stats.matched:,}")
    eprint(f"changed records:   {stats.changed_records:,}")
    eprint(f"field replacements:{stats.field_replacements:,}")
    if stats.preserved_tags_appended:
        eprint(f"preserved tags appended: {stats.preserved_tags_appended:,}")
    if stats.skipped_empty_updates:
        eprint(f"skipped empty upd: {stats.skipped_empty_updates:,}")
    if stats.skipped_missing_fields:
        eprint(f"missing fields in upd: {stats.skipped_missing_fields:,}")


if __name__ == "__main__":
    main()