# Data Model: Text Removal & Image Inpainting

**Date**: 2026-09-24 | **Spec**: [spec.md](spec.md) | **Research**: [research.md](research.md)

Entities reused from Spec 1 — `RawPage`, `GroundTruthMask`, `NormalizedMask`, `AlignedPair`,
`PerImageMetrics`, `FailureReport` — are defined in
[Spec 1's data model](../001-data-foundation-classical-baseline/data-model.md) and are **not restated
here**. Spec 2's `DistributablePageList`, `ImageIdentityRecord` and `Provenance` are defined in
[Spec 2's data model](../002-deep-learning-segmentation-benchmark/data-model.md) and are referenced by
name for the same reason. They are read by this feature, never written by it. Ground truth is read only
by this project's own metric code, never exported and never drawn into an output (FR-033).

## Entities

### InpaintRun (FR-027)

One identified execution. Persisted as `outputs/inpainting/<run_id>/run.json`.

- `run_id` — the directory name; the only time-identifying field in the run (research R3).
- `methods` — the four method identities processed, in the fixed FR-026 order.
- `method_sources` — per identity, where its masks were read from and the identity it resolved to. For
  `classical_baseline` this records the Spec 1 method chosen and why (research R1); for the three DL
  identities it records the admitted `run_id` under `deliverables/`.
- `mask_processing_config` — the `MaskProcessingConfig` used, recorded once and referenced by every
  sample (SC-003).
- `inpainting_config` — the `InpaintingConfig` used.
- `page_list_identity` — quoted **verbatim** from `benchmark/page-list.json`. Never recomputed.
- `input_image_identity` — the `ImageIdentityRecord` for every page consumed, quoted verbatim from
  `benchmark/image-identity.json`. Never recomputed.
- `counts` — samples attempted, succeeded, failed, by method × algorithm.
- `ablation` — `false` for a main-benchmark run, `true` for an FR-024 run (which lives under
  `outputs/inpainting/ablation/<run_id>/` instead).

*Identity is the run ID, not a timestamp, so two runs of the same inputs are comparable by name and
the tree stays reproducible (research R3, SC-012).*

### SegmentationMethodInput (FR-026)

One method identity's mask collection as this feature receives it. Not a new file — an in-memory
resolution over `InpaintRun.method_sources`.

- `method` — one of `classical_baseline`, `manga_text_segmentation`, `comic_text_detector`,
  `unetpp_efficientnetv2`. Fixed names, matching `configs/dl.json`'s adapter keys for the three DL
  identities.
- `source_root` — `deliverables/<run_id>/<method>/` for the DL identities;
  `outputs/segmentation/default/<spec1_method>/` for `classical_baseline`.
- `masks_dir` / `metadata_dir` — `<source_root>/masks/` and `<source_root>/metadata/` in both cases,
  so one intake code path serves all four (research R1, R5).
- `masks` — the `MaskRecord` list, in page-list order.

*The four identities differ only in name and source root, which is why they are configuration rather
than four code paths (plan.md Structure Decision).*

### MaskRecord (FR-008–FR-015, FR-018)

One prediction mask for one page, and the outcome of validating it.

- `image_id` — `<manga>/<NNN>`, resolved by manga folder plus zero-padded stem. Never inferred by
  similarity or guesswork (FR-008).
- `manga`, `stem` — the two components the mapping was made from, kept so an unmappable mask can be
  reported with the names it actually had.
- `mask_path` — where the mask was read from.
- `raw_size` — the shape as read, before alignment (FR-018).
- `aligned_size` — the shape after alignment, `(1170, 1654)`.
- `alignment_rule` — `pass-through` when the raw size already equals the aligned size, or
  `crop-topleft` when it is the known multiple-of-8 GT canvas (FR-015).
- `value_set_valid` — whether the normalized values are within `{0, 255}` (FR-014).
- `empty` — whether the mask contains no text pixel (FR-010's emptiness check).
- `dilation` — the applied `DilationConfig`, or `null` when dilation was disabled (FR-016).
- `status` — `accepted`, or the FR-012 category that rejected it.
- `reason` — the specific reason, populated on rejection.

*A rejected record still exists. FR-013 forbids silently resizing, re-thresholding, re-cropping,
renaming or re-mapping a rejected input, so the record carries the raw facts and the rejection rather
than a corrected version of them.*

### MaskProcessingConfig (FR-014–FR-018)

The shared preprocessing configuration. Held in `configs/inpainting.json`, recorded in `run.json`,
and identical for every page and every method in the main benchmark (FR-017).

- `normalize` — background `0`, text `255`; out-of-set values are a validation error, never coerced
  (FR-014).
- `alignment` — `crop-topleft` for the GT canvas size, `pass-through` at the aligned size, error
  otherwise (FR-015).
- `dilation.enabled`, `dilation.kernel_shape`, `dilation.kernel_size`, `dilation.iterations`
  (FR-016).

*Anything that differs from this configuration is an ablation (FR-024) and runs in the separated
namespace — never in the main benchmark (FR-017).*

### InpaintingConfig (FR-019–FR-021)

- `algorithms` — `["INPAINT_TELEA", "INPAINT_NS"]`, both always run over the same input set in the
  main benchmark (FR-019).
- `radius` — the inpaint radius, identical across all four methods (FR-021).
- `output_format` — default `png`, lossless (FR-028, research R4).

*Per-method tuning is prohibited (FR-021), so this configuration has no per-method override and the
record has no place to put one.*

### ProcessedSample (FR-025, FR-026)

One page through the pipeline, for one method, for one algorithm. The unit that produces artifacts.

- `image_id`, `method`, `algorithm`
- `page_path` — the original page image, copied into the run (FR-025).
- `raw_mask_path` — the **raw prediction mask copy**, byte-identical to the admitted mask it came from
  (FR-025, contract C1).
- `processed_mask_path` — the mask after normalization, alignment and dilation; the mask actually
  handed to `cv2.inpaint`. This is Spec 5's "processed mask" (contract C1).
- `inpainted_path` — the `cv2.inpaint` result.
- `metadata_path` — its own `SampleMetadata` sidecar.
- `status`, `error_message` — `error_message` is populated only on failure (FR-012).

### SampleMetadata (FR-025)

The per-sample JSON sidecar written beside the artifacts. Its field list is normative — contract C1
fixes it — and is the union of FR-025's list and FR-018's mask-processing record.

- Identity: `image_id`, `method`, `algorithm`
- Provenance: `model_repository`, `experiment_id`
- Paths: `input_image_path`, `prediction_mask_path`
- Mask: `mask_size` (raw), `alignment.pre_size`, `alignment.post_size`, `alignment.rule`
- Dilation: `dilation.enabled`, `dilation.kernel_shape`, `dilation.kernel_size`,
  `dilation.iterations`
- Inpainting: `inpaint_algorithm`, `inpaint_radius`, `output_format`
- Outcome: `processing_time_seconds`, `status`, `error_message`

*`processing_time_seconds` is recorded because FR-022 and FR-025 require it, and is excluded from the
byte-wise determinism comparison because it measures the machine, not the input (research R3).*

### PerformanceSummary (FR-022)

The aggregate over a run, written once per run.

- Grouped by `method` × `algorithm`.
- `mean_processing_time_seconds`, `sample_count`, `failure_count`, `runtime_device`.

### ErrorReport (FR-011, FR-012)

All failures of one run, written once as `outputs/inpainting/<run_id>/errors.json`.

- Grouped by the **nine** FR-012 categories: missing source image; missing prediction mask; wrong
  image ID / unmappable or ambiguous mapping; wrong size; empty mask; non-binary mask; corrupt or
  undecodable image; inpainting failure; output write failure.
- Each entry: `image_id`, `category`, `reason`.
- A failing sample never stops the batch (FR-011); the report is the record of every one.

### SampleSelectionRule (FR-034)

One of five strategies. Resolved per method, N = 5 (research R2).

- `highest_iou_f1`, `lowest_iou_f1`, `most_false_positives`, `most_false_negatives` — computed from
  Spec 2's per-page metrics for that method.
- `manual_artifact_flags` — read from a human-supplied list in the selection configuration. It has no
  computed candidate set, so it selects **at most** N and the selection report records the shortfall
  rather than inventing flags.
- Each rule's output records `method`, `rule`, the selected `image_id`s and, when fewer than N were
  available, the shortfall.

*`classical_baseline`'s per-page metrics come from Spec 1's `outputs/segmentation/default/metrics.csv`
for the method it resolved to, so the selection is computed against the same masks that were inpainted
(research R1).*

### ComparisonBoard (FR-033)

One selected sample's five-panel image, written under the run's `boards/` directory.

- `image_id`, `method`, `rule`
- Five labelled panels: original page; raw prediction mask; mask after dilation; TELEA result; NS
  result.
- Labels carry method, algorithm, mask-processing configuration and `image_id`.

*No panel is a ground-truth mask and no panel is a prediction-vs-GT overlay — the board shows inputs
and outputs, never correctness (FR-033, REISSUE §6).*

### QualitativeReport (FR-032, FR-035)

The pre-labelled scaffold, written once per run as a human-editable document plus its data file.

- One entry per assessed sample, labelled with method, algorithm, mask-processing configuration and
  `image_id`.
- Seven criteria, each on one documented ordinal scale, with the rating field **empty** and a written
  observation field **empty**.
- No system-generated rating, no PSNR/SSIM figure, no synthetic ground truth, no single-score method
  ranking (FR-031, FR-032, SC-008).

*The scaffold is generated once and then belongs to the reviewer; regenerating a run must not
overwrite a filled-in one (FR-027).*

### AblationRun (FR-024)

One supplementary sweep. Same entities as a main run, under `outputs/inpainting/ablation/<run_id>/`.

- `varied` — which configuration key was varied (dilation parameters or radius) and its values.
- `baseline_config` — the main-benchmark configuration it is compared against.
- Excluded from the main comparison by namespace, and marked `ablation: true` in its `run.json`.

## Relationships

```text
InpaintRun 1 ── 4 SegmentationMethodInput ── * MaskRecord ── 1 ProcessedSample (×2 algorithms)
ProcessedSample 1 ── 1 SampleMetadata
InpaintRun 1 ── 1 PerformanceSummary, 1 ErrorReport, 1 QualitativeReport, * ComparisonBoard
SampleSelectionRule * ── * ComparisonBoard
AblationRun 1 ── its own InpaintRun-shaped tree, namespaced apart
```

## Validation Rules

Enforced on receipt, per FR-010 and FR-012, and never corrected silently (FR-013):

| Check | Rejection category |
|---|---|
| Source image exists and decodes | missing source image / corrupt or undecodable image |
| Prediction mask exists and decodes | missing prediction mask / corrupt or undecodable image |
| Mask maps to exactly one manifest page | wrong image ID / unmappable or ambiguous mapping |
| Raw size is the aligned size or the GT canvas size | wrong size |
| Normalized values ⊆ `{0, 255}` | non-binary mask |
| Mask contains at least one text pixel | empty mask |
| `cv2.inpaint` returns a result | inpainting failure |
| Artifacts and metadata are written | output write failure |

A run whose output directory already exists refuses to proceed unless overwrite was explicitly
requested (FR-027, SC-006).
