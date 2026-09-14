# Contract: Output Layout + Mask Sidecar

**Normative for**: FR-015–FR-019, FR-024, FR-032–FR-033 and the ONBOARDING output contract.

## Layout

```text
outputs/segmentation/<run_id>/
├── manifest.json                 # 390 pairs, (manga, stem) ordered
├── validation-report.json        # 60 orphans categorized, 0 corrupt expected
├── metrics.csv                   # per-image rows (see metrics.schema.json)
├── summaries.json                # per-method summaries
├── failures.json                 # per-method failure lists (FR-031)
├── <method>/<manga>/<stem>.png   # prediction mask: single-channel uint8 {0,255},
│                                 #   size == raw native 1170w x 1654h
├── <method>/<manga>/<stem>.json  # sidecar (below), same basename
└── viz/<method>/<mode>-<n>/<manga>_<stem>.png   # 4-panel composites (FR-032)
```

## Sidecar (`<stem>.json`, beside each prediction mask)

```jsonc
{
  "image_id": "<manga>/<stem>",
  "method": "otsu",
  "preprocessing": { "grayscale": true, "denoise": "none" },
  "postprocessing": { "fill_holes": false },
  "threshold": "otsu-auto",
  "alignment": { "strategy": "crop-topleft", "pre_size": [1176, 1656], "post_size": [1170, 1654] },
  "inference_time_seconds": 0.42,
  "status": "ok"
  // on failure: "status": "failed", "error": "<message>"
}
```

## Rules
1. Prediction masks are PNG, single-channel, uint8, values ∈ {0, 255} only.
2. Mask size MUST equal the aligned post size (default 1170×1654); pre/post sizes recorded.
3. Nothing is written under `data/`; `outputs/` is gitignored.
4. `report` and `visualize` run from persisted `metrics.csv` without re-running inference.
