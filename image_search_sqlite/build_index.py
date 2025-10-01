#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a SQLite index for 1M+ images with low RAM usage.
Usage:
  python build_index.py --json /path/data.jsonl --tags-csv /path/tags_ja.csv --db /path/tags.db \
    --min-score 0.3 --top-per-image 128

Notes:
- Streams JSONL/JSON; commits in batches.
- Keeps at most top-N tags per image by score (configurable) to limit DB size.
- Records doc frequency (how many images contain each tag) for suggestion sorting.
"""

import argparse, csv, json, os, sqlite3, sys
from collections import defaultdict

def ensure_schema(cur):
    cur.executescript("""
    PRAGMA journal_mode=WAL;
    PRAGMA synchronous=OFF;
    PRAGMA temp_store=MEMORY;
    PRAGMA mmap_size=30000000000;

    CREATE TABLE IF NOT EXISTS meta (
      key TEXT PRIMARY KEY,
      value TEXT
    );

    CREATE TABLE IF NOT EXISTS tags (
      id INTEGER PRIMARY KEY,
      name TEXT UNIQUE NOT NULL,
      ja   TEXT,
      freq INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS images (
      id INTEGER PRIMARY KEY,
      path TEXT UNIQUE NOT NULL
    );

    CREATE TABLE IF NOT EXISTS image_tags (
      image_id INTEGER NOT NULL,
      tag_id   INTEGER NOT NULL,
      score    REAL NOT NULL,
      grp      TEXT NOT NULL,
      PRIMARY KEY (image_id, tag_id),
      FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE,
      FOREIGN KEY(tag_id) REFERENCES tags(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);
    CREATE INDEX IF NOT EXISTS idx_image_tags_tag  ON image_tags(tag_id);
    CREATE INDEX IF NOT EXISTS idx_image_tags_img  ON image_tags(image_id);
    CREATE INDEX IF NOT EXISTS idx_image_tags_score ON image_tags(score);
    """)

def upsert_tag(cur, cache, name, ja=None):
    tid = cache.get(name)
    if tid is not None:
        return tid
    cur.execute("INSERT OR IGNORE INTO tags(name, ja) VALUES(?, COALESCE((SELECT ja FROM tags WHERE name=?), ?))",
                (name, name, ja))
    cur.execute("SELECT id FROM tags WHERE name=?", (name,))
    tid = cur.fetchone()[0]
    cache[name] = tid
    return tid

def stream_records(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        head = f.read(1024)
        f.seek(0)
        if head.strip().startswith("["):
            data = json.load(f)
            for obj in data:
                yield obj
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    # tolerate minor formatting
                    if not line.startswith("{"):
                        line = "{" + line
                    if not line.endswith("}"):
                        line = line + "}"
                    obj = json.loads(line)
                yield obj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--tags-csv", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--min-score", type=float, default=0.3, help="drop tags with score < this to shrink DB")
    ap.add_argument("--top-per-image", type=int, default=128, help="keep only top-N tags per image")
    ap.add_argument("--commit-every", type=int, default=5000)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.db) or ".", exist_ok=True)
    con = sqlite3.connect(args.db)
    cur = con.cursor()
    ensure_schema(cur)

    # load JA map
    ja = {}
    with open(args.tags_csv, "r", encoding="utf-8") as f:
        rd = csv.reader(f)
        for row in rd:
            if len(row) >= 2:
                ja[row[0].strip()] = row[1].strip()

    # seed JA
    # (Optional) We don't bulk insert to avoid huge memory; insert on demand.
    tag_cache = {}

    dfreq = defaultdict(int)  # doc freq
    total = 0
    batch = 0

    for obj in stream_records(args.json):
        path = obj.get("path") or obj.get("image_path") or obj.get("file")
        if not path:
            continue

        # collect all tags across groups
        all_pairs = []
        for grp in ("rating", "general", "character", "meta" , "year", "artist"):
            for it in obj.get(grp, []) or []:
                if isinstance(it, (list, tuple)) and len(it) >= 2:
                    tag = str(it[0])
                    try:
                        score = float(it[1])
                    except Exception:
                        continue
                    if score >= args.min_score:
                        all_pairs.append((grp, tag, score))

        if not all_pairs:
            continue

        # keep top-N by score
        all_pairs.sort(key=lambda x: x[2], reverse=True)
        all_pairs = all_pairs[:args.top_per_image]

        # insert image
        cur.execute("INSERT OR IGNORE INTO images(path) VALUES(?)", (path,))
        cur.execute("SELECT id FROM images WHERE path=?", (path,))
        img_id = cur.fetchone()[0]

        seen_tags = set()
        for grp, tag, score in all_pairs:
            tid = upsert_tag(cur, tag_cache, tag, ja.get(tag))
            cur.execute("INSERT OR REPLACE INTO image_tags(image_id, tag_id, score, grp) VALUES(?,?,?,?)",
                        (img_id, tid, score, grp))
            if tid not in seen_tags:
                dfreq[tid] += 1
                seen_tags.add(tid)

        total += 1
        batch += 1
        if batch >= args.commit_every:
            # update freq in bulk
            for tid, cnt in dfreq.items():
                cur.execute("UPDATE tags SET freq = COALESCE(freq,0) + ? WHERE id=?", (cnt, tid))
            dfreq.clear()
            con.commit()
            batch = 0
            print(f"Committed {total} images...", flush=True)

    # final commit + freq
    for tid, cnt in dfreq.items():
        cur.execute("UPDATE tags SET freq = COALESCE(freq,0) + ? WHERE id=?", (cnt, tid))
    con.commit()

    print(f"Done. Indexed {total} images.", flush=True)

if __name__ == "__main__":
    main()
