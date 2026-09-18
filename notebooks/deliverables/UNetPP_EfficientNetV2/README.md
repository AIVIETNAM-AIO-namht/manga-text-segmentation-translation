# UNet++ + EfficientNetV2 Benchmark Deliverable

Status: **EXPLORATORY / NOT ADMISSIBLE**

Method: UNet++ + `tu-efficientnetv2_rw_m`
Repository: https://github.com/ContemporaryCat/Manga-Text-Segmentation.git
Commit: `6cfa907b0721faa398054be527de98171746ef1b`
Checkpoint: `model.pth`
Checkpoint SHA256: `a0a895dc385608554a81ca765b0c62d654462c2bac661abdff63634985ab37fc`

## Input
Only raw images are extracted/read from `/content/drive/MyDrive/DIP/data/raw/Manga109s_released_2026_05_21-001.zip`. Ground-truth is not extracted or read by the runner.

## Alignment and preprocessing
Top-left crop to **1654 x 1170**, no resize; ImageNet normalization; bottom/right padding to multiple of 32; sigmoid; threshold 0.5.

## Output masks
PNG, single-channel, uint8, binary `(0, 255)`, size 1654 x 1170. Mapping: `masks/<manga>/<page_id>.png`.

## Metadata
One JSON sidecar per page under `metadata/<manga>/<page_id>.json`, including identity fields, input checksum, checkpoint provenance, environment, GPU, seed and per-page timings.

## Evaluation
This runner does **not** calculate metrics and does not create `metrics_per_page.csv` or `metrics_summary.json`.

## Visualization
Original + prediction mask + prediction overlay only. Ground-truth is not included.

## Run
Page-list source: `ground-truth name-matched discovery`
Pages attempted: 390
Successful: 0
Failed: 390
Total runner time: 150.08 seconds
GPU: Tesla T4

Weight license is intentionally not guessed; verify it from the Hugging Face model card before admissible submission.
