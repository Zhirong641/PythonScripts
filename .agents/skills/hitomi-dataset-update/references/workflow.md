# Hitomi source image dataset update

## Full update contract

For a full update request, carry out sections 1–4 in one run without asking the
user to prompt the next stage. First record the current cumulative counts and
check for active crawlers, `stop`, and failed-gallery files. Use the source and
download location the user specified; otherwise choose based on current access.
Preview the source and identify new IDs before downloading. If none are new,
report a verified no-op instead of rewriting cumulative files. Keep a dated run
directory with the batch, logs, model revisions, validation results, and backup
paths. Continue until the completion checks below pass, or report the concrete
stage and condition that prevents progress.

When a stage exposes a repeatable failure, repair its code and the relevant
workflow rule in this same run. Add or adjust a focused regression check where
it guards the failure, run validation, then continue from the preserved
checkpoint. A one-off manual workaround is not the final workflow fix.

## Data layout

- Working scripts: `/mnt/shared/PythonScripts`.
- Cumulative images: `/mnt/shared/data/webp/<gallery-id>/image_<index>.webp`.
- Cumulative image CSV: `/mnt/shared/data/hitomi_260801.csv`, six columns:
  source URL, title, gallery URL, type, gallery ID, image index.
- Metadata: `/mnt/shared/data/artists.csv` (`id,artists`) and
  `/mnt/shared/data/artists_with_group.csv` (`id,artists,group`). No header.
- Final tags: `/mnt/shared/data/hitomi_tags.jsonl`, one JSON object per image
  path. Keep `general`, `character`, `copyright`, `meta`, `year`, `rating`,
  `artist`, `group`, and `type`; do not include `aesthetic_tag`.
- Use a dated run directory such as `/mnt/shared/data/hitomi_update_runs/YYYY-MM-DD`
  for a batch CSV, image manifest, intermediate tags, and logs.

## 1. Acquire a batch

Determine which galleries are new. `get_images_from_hitomi.py` supports three
targeting modes:

- No target flag: scan the configured `base_urls` listing pages.
- `--url URL`: scan a Hitomi listing URL, or download one gallery when the URL
  is a gallery/reader page. The URL must be under `https://hitomi.la`.
- `--title "Exact Work Title"`: search the site and select only exact title
  matches; `--title-match contains` explicitly broadens matching to variants.
  A title with no matches exits nonzero. `--list-only` previews results without
  downloading. Title/listing modes need Chrome, Selenium, and
  `webdriver-manager`; ID mode only needs HTTP requests.
- `--id 1234567`: fetch one gallery's metadata directly by ID and download it
  without scanning search results. Repeat `--id` or `--title` for several works.
  An explicitly selected ID bypasses the listing type filter; title/listing
  results still follow it. Use `--type "Game CG"` to restrict a run explicitly.

For example:

```bash
python get_images_from_hitomi.py --id 4211174 --list-only
python get_images_from_hitomi.py --title "Ousama Ren'Ai" --list-only
python get_images_from_hitomi.py --title "Ousama Ren" --title-match contains \
  --output-csv "$RUN/batch.csv"
```

The ID route reads `galleries/<id>.js` metadata directly. The site search page
uses `search.html?<encoded terms>` and renders 25 results per page; the title
route searches all result pages before applying the requested title match.
The search parser treats `:` as a namespace operator and leading `-` as an
exclusion, so title searches neutralize those operators in the query but still
compare the returned work title against the original requested title.
For a new batch, set `--output-csv` to the run's six-column CSV instead of
appending to an older dated output file.

For local title/listing runs, use the project crawler environment (Chrome is
also required):

```bash
python3 -m venv .venv-hitomi
.venv-hitomi/bin/python -m pip install -r requirements-hitomi.txt
```

Then replace `python` in the examples above with `.venv-hitomi/bin/python`.
Direct `--id` requests do not load Selenium and can also run in a Python
environment with just `requests` installed.

`--retry-failed` handles the failure list separately. `ids.csv` is optional: when
present, its fifth column supplies legacy completed IDs. The local default also
checks `/mnt/shared/data/hitomi_260801.csv` against the actual `webp` folders
and skips IDs only when the recorded image count and contiguous nonempty files
agree. For a different dataset location, pass `--existing-csv` together with
`--existing-images-root`. Known duplicate galleries are mapped to their retained
ID in `config/hitomi_duplicate_gallery_ids.csv` and skipped on every run; keep
the canonical ID in the dataset instead of writing a false completion marker for
the duplicate. Completion markers in `webp_complete/` also cause a
skip. A bare directory without a marker or verified CSV is partial and must
resume, not be skipped. IDs completed earlier in the same run are skipped on
subsequent URL results. Check the selected URL, type filter, output CSV name,
`stop` file, failure list, and active process before starting.
Listing type text is normalized to the configured spelling before writing the
batch CSV (for example, `game CG` becomes `Game CG`). On a resumed gallery,
existing nonempty images are checked against the batch CSV and any missing rows
are repaired; duplicate `(gallery ID, image index)` rows are rejected. A gallery
is marked complete only after both its images and batch CSV rows are complete.

Download locally when appropriate. If network conditions favor a server, keep a
per-run six-column CSV there, sync only the completed gallery directories into
`/mnt/shared/data/webp`, and copy that CSV locally. Do not assume one fixed host.
For large transfers, use resumable `rsync`, then compare every file's relative
name and size against a source manifest. Only after the images are verified,
deduplicate `(gallery ID, image index)` against the cumulative CSV and append
the missing rows through a staged atomic replacement. The image count and CSV
row count must match. Keep completion markers outside `webp` so they do not
inflate image counts.

## 2. Artist and group metadata

For every new gallery ID missing from the two metadata CSVs, fetch its
`https://ltn.gold-usergeneratedcontent.net/galleries/<id>.js` metadata or use
`get_hitomi_artists.py` and `get_hitomi_groups.py`. The gallery metadata exposes
`artists` and `groups`; join artist names with `, ` and group names with `,`,
lowercase as in the existing CSVs. Preserve existing IDs and keep the two CSVs
aligned. An absent artist field becomes an empty string; do not invent a name.
Back up the originals and check all downloaded IDs appear exactly once.

## 3. Tag only new images

Create a run manifest from the batch CSV. The helper checks image existence and
excludes paths already in the cumulative tag file:

```bash
python .agents/skills/hitomi-dataset-update/scripts/prepare_tagging.py \
  --batch-csv /path/to/new_batch.csv \
  --images-root /mnt/shared/data/webp \
  --existing-tags /mnt/shared/data/hitomi_tags.jsonl \
  --run-dir /mnt/shared/data/hitomi_update_runs/YYYY-MM-DD
```

Run `image_tagger.py` (WD) and `image_camie_tagger.py` (Camie) on that
`images.txt` with `--input-list`, `--out-jsonl`, and `--resume`. GPU runs on this
machine have worked with `--batch-size 32 --workers 8 --use-gpu` and
`HF_HUB_DISABLE_XET=1`. Check actual providers and benchmark the current host;
CPU-only or a different GPU may need other settings. Keep distinct WD and Camie
output files. Each must contain exactly one path from the manifest.
Record resolved Hugging Face model revisions and actual thresholds in the run
directory so a later rerun can explain any tag differences.

For example, from `/mnt/shared/PythonScripts` with `RUN` set to the dated run
directory (run the two model commands in separate persistent terminals):

```bash
HF_HUB_DISABLE_XET=1 python image_tagger.py --input-list "$RUN/images.txt" \
  --batch-size 32 --workers 8 --use-gpu --resume --out-jsonl "$RUN/wd.jsonl"
HF_HUB_DISABLE_XET=1 python image_camie_tagger.py --input-list "$RUN/images.txt" \
  --batch-size 32 --workers 8 --use-gpu --resume --out-jsonl "$RUN/camie.jsonl"
```

Do not merge until both files cover the exact manifest path set with no
duplicates or malformed lines. A skipped/unreadable image needs repair and a
resume run; accepting a shorter output would silently lose its tags.
When tagging staged images before they are copied into the cumulative `webp`
root, remap the scored JSONL `path` fields with `scripts/remap_tag_paths.py`
after copying the images and before resuming against the canonical manifest.
Never publish a staging path into `hitomi_tags.jsonl`.
The finalizer enforces that every manifest path is canonical and directly under
the cumulative `webp/<id>/` root, so an omitted remap stops before creating a
backup or replacing the cumulative tags.

```bash
python .agents/skills/hitomi-dataset-update/scripts/check_model_outputs.py \
  --manifest "$RUN/images.txt" --wd "$RUN/wd.jsonl" --camie "$RUN/camie.jsonl"
```

Merge the scored outputs with `replace_camie_general_rating.py`: use Camie for
character, copyright, artist, meta, and year; use WD for general and rating;
retain Camie's `naked_skirt` general tag if WD lacks it. Then run
`scores2strings_with_artist.py` with `--artists-csv` pointing at
`artists_with_group.csv` and `--type-csv` pointing at the batch CSV. Its default
thresholds and artist matching reproduce the established flat-tag format.

Example continuation after both model outputs finish:

```bash
python replace_camie_general_rating.py --camie RUN/camie.jsonl \
  --wd RUN/wd.jsonl --out RUN/merged_scored.jsonl
python scores2strings_with_artist.py --input RUN/merged_scored.jsonl \
  --output RUN/new_tags.jsonl \
  --artists-csv /mnt/shared/data/artists_with_group.csv \
  --type-csv RUN/batch.csv
```

Replace `RUN` with the actual run directory. Validate exact path coverage before
publishing. The finalizer backs up and atomically replaces the cumulative tags,
removes `aesthetic_tag` from historical records, and checks image/CSV/tag counts:

For a full run, `scripts/complete_run.py --run-dir "$RUN"` performs the model
output check, scored merge, string conversion, and finalization in that order.
It can wait for active model processes with repeated `--wait-pid PID`; on a model
error or incomplete output it stops before changing the cumulative tag file.
Its success completes raw tagging only: continue through section 4 and the
completion checks. The individual commands remain useful for inspecting or
repairing a stage.

```bash
python .agents/skills/hitomi-dataset-update/scripts/finalize_tags.py \
  --existing-tags /mnt/shared/data/hitomi_tags.jsonl \
  --new-tags RUN/new_tags.jsonl --manifest RUN/images.txt \
  --cumulative-csv /mnt/shared/data/hitomi_260801.csv \
  --images-root /mnt/shared/data/webp \
  --backup /mnt/shared/data/hitomi_tags.jsonl.bak_YYYY-MM-DD
```

## 4. Character/artist ranges and deterministic filtering

In a full update, always run the current
`set_char_artist_by_range.py` over the full cumulative `hitomi_tags.jsonl`.
Also run it when the user specifically requests only the derived filtered tags.
The script's `RANGES` and files under `filter_lists/` are user-maintained rules;
preserve them. Compile the script and verify range tuple structure first, since
one malformed range can prevent the whole run from starting.

Write to a candidate file in the run directory, not directly over the existing
filtered file:

```bash
python set_char_artist_by_range.py /mnt/shared/data/hitomi_tags.jsonl \
  "$RUN/hitomi_tags_filtered.candidate.jsonl" > "$RUN/filter_full.log" 2>&1
```

Validate every candidate JSON object, unique `path` values, absence of
`aesthetic_tag`, and `input rows - reported filtered rows = output rows`.
Check representative new range assignments and reconcile old-path differences
with changed filter lists. Back up the previous filtered file, then atomically
replace it with the validated candidate. Record the script and filter-list
versions or hashes with the run so the filtered result is reproducible. The
filtered count need not match the raw image/CSV/tag counts.

## Completion checks

Before reporting a full update complete, confirm all new gallery IDs have the
expected unique, nonempty image files and batch CSV rows, with known duplicate
IDs absent. The cumulative `webp` file count, `hitomi_260801.csv` row count, and
raw tag row count must agree. Every new raw tag path must be canonical and unique
under `/mnt/shared/data/webp`; `aesthetic_tag` must be absent. Both metadata
CSVs must contain each new ID once, with source-provided group and artist data
(an absent source artist stays empty). The filtered row count must equal raw
rows minus the script's reported filter count. Confirm no required process is
still running, record backups and results in the run directory, and report any
remaining failures rather than claiming success.

## Failure lessons

- The source reader can hang or crash Chrome on a small-memory server. Fetching
  gallery metadata directly avoids loading thousands of images in a browser.
- A reader dropdown can be empty before the page finishes loading. Do not mark
  an empty gallery directory as completed.
- Browser page transitions can return the previous image URL. Generate image
  URLs from gallery metadata rather than trusting the visible DOM after a hash
  change.
- The site's `gg.js` CDN routing script can flip its default branch between
  `w1` and `w2`. Parse the current switch values; do not hardcode which listed
  cases map to either host. Verify a small direct-ID download after changes.
- A CDN 503 may be transient; use bounded retries and preserve partial progress.
- Keep large cumulative outputs untouched until the new batch is fully checked.
- Run `python -m unittest discover -s tests -p test_hitomi_workflow_guards.py`
  after changing type normalization, resumed CSV writing, or tag publication.
