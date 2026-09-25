# unetpp-efficientnetv2 Benchmark Deliverable

Status: **ADMISSIBLE**

Method: `unetpp-efficientnetv2`
Repository: `https://github.com/ContemporaryCat/Manga-Text-Segmentation`
Commit: `6cfa907b0721faa398054be527de98171746ef1b`
Checkpoint: `model.pth`
Checkpoint SHA256: `a0a895dc385608554a81ca765b0c62d654462c2bac661abdff63634985ab37fc`
Code License: `none`
Weight License: `none`

## Input
Consumes official 390 pages from `benchmark/page-list.json`. Raw images only. Ground-truth is not read.

## Alignment and Preprocessing
Top-left crop to **1654 x 1170**, no resize; ImageNet norm; pad multiple of 32; sigmoid threshold 0.5.

## Deliverables
- `masks/<manga>/<stem>.png`: Single-channel uint8, binary {0, 255}, shape (1170, 1654).
- `metadata/<manga>/<stem>.json`: Provenance sidecar for all 390 pages.
- `provenance.json`: Root hand-off provenance record.
- `errors.json`: Execution log.
- `visualizations/`: Original + prediction + overlay (No ground truth).
