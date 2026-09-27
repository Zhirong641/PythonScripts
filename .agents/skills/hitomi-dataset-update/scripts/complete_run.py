#!/usr/bin/env python3
"""Validate model outputs, merge tags, and publish one complete Hitomi batch."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def still_running(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def call(*args):
    print("Running:", " ".join(map(str, args)), flush=True)
    subprocess.run([str(arg) for arg in args], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/shared/data"))
    parser.add_argument("--wait-pid", action="append", type=int, default=[])
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--backup", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    data_root = args.data_root.resolve()
    project_root = Path(__file__).resolve().parents[4]
    backup = args.backup or data_root / (
        "hitomi_tags.jsonl.bak_" + datetime.now(timezone.utc).strftime("%Y%m%d")
    )
    with (run_dir / "complete.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while any(still_running(pid) for pid in args.wait_pid):
            print("Waiting for taggers:", args.wait_pid, flush=True)
            time.sleep(max(args.poll_seconds, 1))

        scripts = Path(__file__).parent
        call(
            sys.executable, scripts / "check_model_outputs.py",
            "--manifest", run_dir / "images.txt",
            "--wd", run_dir / "wd.jsonl",
            "--camie", run_dir / "camie.jsonl",
        )
        call(
            sys.executable, project_root / "replace_camie_general_rating.py",
            "--camie", run_dir / "camie.jsonl",
            "--wd", run_dir / "wd.jsonl",
            "--out", run_dir / "merged_scored.jsonl",
        )
        call(
            sys.executable, project_root / "scores2strings_with_artist.py",
            "--input", run_dir / "merged_scored.jsonl",
            "--output", run_dir / "new_tags.jsonl",
            "--artists-csv", data_root / "artists_with_group.csv",
            "--type-csv", run_dir / "batch.csv",
        )
        call(
            sys.executable, scripts / "finalize_tags.py",
            "--existing-tags", data_root / "hitomi_tags.jsonl",
            "--new-tags", run_dir / "new_tags.jsonl",
            "--manifest", run_dir / "images.txt",
            "--cumulative-csv", data_root / "hitomi_260801.csv",
            "--images-root", data_root / "webp",
            "--backup", backup,
        )
        (run_dir / "complete.json").write_text(
            json.dumps({
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "backup": str(backup),
            }, indent=2) + "\n", encoding="utf-8"
        )
        print("Hitomi tagging update complete.", flush=True)


if __name__ == "__main__":
    main()
