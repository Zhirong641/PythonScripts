---
name: hitomi-dataset-update
description: Update the Hitomi image dataset in this workspace, including downloads, artist and group metadata, WebP synchronization, and incremental image tagging. Use for Hitomi dataset maintenance requests.
---

# Hitomi dataset update

Read [the workflow reference](references/workflow.md) before changing dataset files.
Identify the actual source of the new images for the current run: downloads may
be local or on a remote host.

When the user asks for a full update or points to new galleries without narrowing
the request, complete every stage in the reference through the filtered JSONL
and final checks. Routine transitions between stages are already authorized;
do not wait for the user to prompt each one. For a request explicitly limited to
one stage, perform that stage and its necessary checks.

This skill and its supporting workflow are maintainable project files. If a run
reveals a concrete flaw, changed site behavior, or a manual repair that should
be repeatable, update the relevant skill/reference, implementation, and focused
regression check as part of the same task. Verify the change before continuing;
the user has already authorized these project-local workflow edits. Preserve the
requested task scope and unrelated user rules.

Key invariants:

- Use `(gallery ID, image index)` and absolute image paths to prevent duplicate
  records. Apply `config/hitomi_duplicate_gallery_ids.csv` before downloading;
  a directory's existence alone does not prove a gallery is complete.
- Normalize listing types before writing the batch CSV. On resume, repair CSV
  rows missing for existing images, reject duplicate rows, and require all image
  files and rows before marking a gallery complete. Parse the current `gg.js`
  routing instead of assuming a fixed CDN branch.
- Preserve existing records. Stage, validate, and back up cumulative files before
  replacement. A failed partial run must be safe to resume.
- Check the source batch against its images, then check the cumulative image,
  CSV, and raw tag counts after a full update. The derived filtered tag file is
  intentionally smaller; reconcile its size with the reported filter count.
- Preserve the WD/Camie tag decisions described in the reference. The final
  `/mnt/shared/data/hitomi_tags.jsonl` must have no `aesthetic_tag` key or paths
  outside the canonical cumulative `webp` root.

The deterministic scripts in [scripts](scripts) prepare the new image manifest
and finalize the tag file. Follow the commands and decision points in the
reference for the other stages.
