"""Regression checks for failures found during Hitomi dataset updates."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from hitomi_download_state import read_duplicate_ids, read_recorded_image_keys
from hitomi_targets import canonical_gallery_type, parse_gg_routing


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINALIZER = PROJECT_ROOT / ".agents/skills/hitomi-dataset-update/scripts/finalize_tags.py"


class HitomiWorkflowGuards(unittest.TestCase):
    def test_duplicate_gallery_mapping_is_persistent_data(self):
        mapping = read_duplicate_ids(PROJECT_ROOT / "config/hitomi_duplicate_gallery_ids.csv")
        self.assertEqual(mapping["4210997"], "4211174")

    def test_cdn_router_handles_both_default_branches(self):
        for default, override in ((0, 1), (1, 0)):
            script = (
                "gg = { m: function(g) { "
                f"var o = {default}; switch (g) {{ case 123: case 456: "
                f"o = {override}; break; }} return o; }}, b: '123456/' }};"
            )
            prefix, fallback, cases = parse_gg_routing(script)
            self.assertEqual((prefix, fallback, cases), (
                "123456/", default, {123: override, 456: override}
            ))

    def test_type_casing_is_canonical(self):
        allowed = ["Game CG", "Image Set", "Artist CG"]
        self.assertEqual(canonical_gallery_type("game CG", allowed), "Game CG")
        self.assertEqual(canonical_gallery_type(" GAME CG ", allowed), "Game CG")
        self.assertIsNone(canonical_gallery_type("Manga", allowed))

    def test_existing_batch_rows_are_loaded_and_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batch.csv"
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output)
                writer.writerow(["source", "title", "url", "Game CG", "123", "1"])
                writer.writerow(["source", "title", "url", "Game CG", "123", "2"])
            self.assertEqual(read_recorded_image_keys(path), {("123", 1), ("123", 2)})
            with path.open("a", newline="", encoding="utf-8") as output:
                csv.writer(output).writerow(["source", "title", "url", "Game CG", "123", "2"])
            with self.assertRaisesRegex(ValueError, "Duplicate image CSV row"):
                read_recorded_image_keys(path)

    def test_staging_path_cannot_be_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "webp" / "123"
            staging = root / "staging" / "123"
            images.mkdir(parents=True)
            staging.mkdir(parents=True)
            for image in (images / "image_1.webp", images / "image_2.webp", staging / "image_2.webp"):
                image.write_bytes(b"webp")
            existing = root / "tags.jsonl"
            incoming = root / "new.jsonl"
            manifest = root / "manifest.txt"
            cumulative = root / "images.csv"
            backup = root / "tags.bak"
            existing.write_text(json.dumps({"path": str(images / "image_1.webp")}) + "\n")
            incoming.write_text(json.dumps({
                "path": str(staging / "image_2.webp"),
                **{key: "" for key in (
                    "general", "character", "copyright", "meta", "year",
                    "rating", "artist", "group", "type",
                )},
            }) + "\n")
            manifest.write_text(str(staging / "image_2.webp") + "\n")
            cumulative.write_text("row1\nrow2\n")
            result = subprocess.run([
                sys.executable, str(FINALIZER),
                "--existing-tags", str(existing),
                "--new-tags", str(incoming),
                "--manifest", str(manifest),
                "--cumulative-csv", str(cumulative),
                "--images-root", str(root / "webp"),
                "--backup", str(backup),
            ], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("outside the cumulative webp root", result.stderr)
            self.assertFalse(backup.exists())
            self.assertEqual(len(existing.read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
