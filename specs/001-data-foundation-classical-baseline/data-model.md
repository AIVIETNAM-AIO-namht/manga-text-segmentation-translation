# Data Model: Data Foundation & Classical Baseline

**Date**: 2026-09-13 | **Spec**: [spec.md](spec.md) | **Research**: [research.md](research.md)

## Entities

### GroundTruthMask (source, read-only)
- `manga: str` — folder name under `data/groundtruth/post-processed/`
- `stem: str` — zero-padded page number (`"003"`)
- `path: str` — absolute source path
- `width, height: int` — native pixels (observed 1176×1656)
- `encoding: enum(magenta-black | magenta-only | near-black-only | all-white-empty)` — detected text encoding
- Source of truth; never modified (FR-009).

### RawPage (source, read-only)
- `manga: str`, `stem: str`, `path: str`
- `width, height: int` — native pixels (observed 1170×1654)
- Loaded BGR via OpenCV (`imaging.py`).

### ImageMaskPair (manifest row)
- `manga: str`, `stem: str`
- `raw_path: str`, `mask_path: str`
- `mask_encoding: GroundTruthMask.encoding`
- Ordering key: `(manga, stem)` — deterministic manifest order (FR-004).
- 390 rows expected; orphans excluded but reported.

### ValidationReport
- `total_masks: int` (= 450), `paired: int` (= 390), `orphan: int` (= 60)
- `orphans: list[{manga, stem, path}]` — per-file, categorized (FR-006)
- `corrupt: list[{path, error}]` — expected empty (FR-007: 0 unreadable)
- `generated_at_run: str` (run id, not wall-clock — determinism)

### NormalizedMask (derived, in-memory + cached optionally)
- `pair: ImageMaskPair`
- `array: uint8[H, W]` values ∈ {0, 255}; background = 0, text = 255 (FR-012/FR-014)
- `normalizer_version: str` — recorded for reproducibility

### AlignedPair (derived)
- `pair: ImageMaskPair`
- `pre_size: (W, H)` mask native, `post_size: (W, H)` = raw native 1170×1654
- `strategy: enum(crop-topleft [default] | resize)` (FR-017)
- `array: uint8[1654, 1170]` values ∈ {0, 255}

### SegmentationMethod (registry entry)
- `name: str` — one of `otsu | adaptive | mser | edges | components | morphology | pipeline:*`
- `params: dict` — from run config; all params logged to sidecar
- Interface: `segment(grayscale: uint8[H, W]) -> uint8[H, W]` ∈ {0, 255} (FR-035)

### PredictionMask (output artifact)
- `pair: ImageMaskPair`, `method: str`, `run_id: str`
- `path: outputs/segmentation/<run_id>/<method>/<manga>/<stem>.png`
- `sidecar: <stem>.json` — `{image_id, method, preprocessing, postprocessing, threshold, alignment, pre_size, post_size, inference_time_seconds, status, error?}` (ONBOARDING contract)
- `status: enum(ok | failed)`; failures recorded, never abort the sweep (FR-031)

### PerImageMetrics (CSV row)
- `manga, stem, method, run_id: str`
- `iou, precision, recall, f1: float` ∈ [0, 1]
- `tp, fp, fn: int` (tn omitted — background-dominated, unused)
- `inference_time_seconds: float`
- Empty-mask conventions per research R3.

### MethodSummary
- `method, run_id: str`, `n_images: int`
- `mean_iou/std_iou, mean_precision, mean_recall, mean_f1, mean_time_seconds: float`
- Persisted JSON + CSV (FR-030).

### FailureReport
- `method, run_id: str`
- `failures: list[{manga, stem, error, stage}]`
- Rerunnable: `report`/`visualize` reuse persisted metrics (FR-031).

### VisualizationComposite
- `pair: ImageMaskPair`, `method: str`
- 4 panels: raw | GT | prediction | overlay (FR-032)
- Selection: `{mode: first-N [default] | best-N | worst-N, N: int}` ranked by F1 (FR-033)

### RunConfig (single configuration file, FR-036)
- `dataset: {raw_root, gt_root}`, `output_root: str`, `run_id: str`
- `alignment: {strategy, ...}`, `preprocessing: {denoise, ...}`
- `methods: list[{name, params}]`, `metrics: [iou, precision, recall, f1]`
- `visualization: {mode, n}`

## Relationships

```text
GroundTruthMask (450) ─┐
                       ├─► ImageMaskPair (390) ─► NormalizedMask ─► AlignedPair ─┬─► PredictionMask (× methods)
RawPage (8519) ────────┘         │                                               │         │
                                 │                                               │         ▼
                    ValidationReport (60 orphans)                        PerImageMetrics ─► MethodSummary
                                                                                    └─► VisualizationComposite (subset)
RunConfig drives every stage; Manifest persists ImageMaskPair list.
```

## Validation rules (entity-level)
- Manifest rows unique on `(manga, stem)`; sorted ascending — byte-identical across runs.
- Every PredictionMask pixel ∈ {0, 255}; single channel; size == AlignedPair.post_size.
- PerImageMetrics floats ∈ [0, 1]; no NaN (R3 conventions applied at write time).
- No artifact path under `data/`; `data/no-need-to-read/` never opened.
