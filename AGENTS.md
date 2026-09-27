# Hitomi dataset workflow

For requests about updating the Hitomi source image dataset in this workspace, read
[the project skill](.agents/skills/hitomi-dataset-update/SKILL.md) and its linked
workflow reference. This includes gallery downloads, artist/group metadata, WebP
sync, and image tagging. The image download may run locally or on a remote server;
check the current source before acting.

When the user requests a **complete Hitomi dataset update** or asks to update
new source galleries without limiting the scope, carry out the entire workflow
autonomously: discover new IDs, download and verify images, update the cumulative
image CSV and artist/group CSVs, tag new images, merge tags, rebuild the filtered
JSONL, and verify the published files. Do not stop after one stage to ask for the
next instruction. If the user explicitly requests only one stage, honor that
scope. A missing file, failed validation, or unavailable dependency is a reason
to repair or report a concrete blocker, not to declare the update complete.

The user authorizes maintenance of this project's Hitomi workflow instructions,
skill, references, helper scripts, and regression checks. When a real defect or
site change makes the workflow incomplete or unsafe, fix the relevant files and
validate the fix during the task without waiting for a separate request. Keep
the change limited to the observed issue and report what was updated.

Keep the cumulative data under `/mnt/shared/data` consistent: each image should
have one row in `hitomi_260801.csv` and one row in `hitomi_tags.jsonl` after a
complete update. Preserve existing rows, use a backup and an atomic replacement
for large output files, and verify the batch before publishing it. The final
`hitomi_tags.jsonl` must not contain an `aesthetic_tag` field. The derived
`hitomi_tags_filtered.jsonl` is intentionally smaller because filtering removes
images; validate its output count against the script's filtered count.
