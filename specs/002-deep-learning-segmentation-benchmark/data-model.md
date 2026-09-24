# Data Model: Deep Learning Segmentation Benchmark

**Date**: 2026-09-19 | **Spec**: [spec.md](spec.md) | **Research**: [research.md](research.md)

Entities reused from Spec 1 — `GroundTruthMask`, `RawPage`, `ImageMaskPair`, `NormalizedMask`,
`AlignedPair`, `PerImageMetrics`, `MethodSummary`, `FailureReport`, `VisualizationComposite` — are
defined in [Spec 1's data model](../001-data-foundation-classical-baseline/data-model.md) and are
**not restated here**. They are referenced by name. Ground truth is read only by the project, never
exported to a runner (FR-054a).

## Entities

### DistributablePageList (exported artifact, FR-054)

- `page_list_identity: str` — sha256 over the list's own canonical serialisation; computed by the
  project, quoted verbatim by runners, never recomputed by them (ONBOARDING §6).
- `generated_at_run: str` — run id that produced the export, not wall-clock (determinism).
- `aligned_size: [int, int]` — `[1654, 1170]` (W×H); the size every prediction mask must be returned at.
- `alignment: enum(crop-topleft)` — the convention a runner applies to reach `aligned_size`.
- `mask_convention: {dtype: uint8, channels: 1, values: [0, 255], text: 255, background: 0}` — FR-004,
  restated because the runner has no access to Spec 1's contract files.
- `pages: list[{manga, stem, image_id, image_ref, input_image_identity}]` — evaluation order, sorted
  `(manga, stem)`. `image_id` = `"<manga>/<stem>"`. `image_ref` is relative to the image bundle root.
- *"Carries no `gt_root`, no `mask_path`, no ground-truth reference and no ground-truth pixel data.
  A runner needs to know which pages to process and what size to return, not what the answers are."*

### ImageBundle (distribution, FR-054a)

- `root: path` — `benchmark/dist/images/` (gitignored; matching the existing `*.zip` rule).
- `identity_record: path` — `benchmark/image-identity.json` (**committed**), one
  `{image_id, sha256, size_bytes}` per page.
- `total_bytes: int` — measured 144,364,052 (137.7 MiB) for the 390 paired raws; largest single file
  627,003 bytes.
- *"Page list and image bundle are complementary: neither alone is sufficient to run a method. The
  committed identity record makes any copy of the bundle verifiable on receipt."*

### MethodAdapter (code, registry entry)

- `name: str` — `manga-text-segmentation | comic-text-detector | unetpp-efficientnetv2`.
- `interface: segment(image: uint8[H, W, 3]) -> uint8[1654, 1170] ∈ {0, 255}` — the DL counterpart of
  Spec 1's frozen `segment(image) -> uint8[H, W]`; it returns a mask **at aligned size**, having
  inverted its own padding internally.
- `config: MethodConfiguration` — injected, not read from module scope.
- `availability: AvailabilityVerdict` — set by the availability check, not by inference success.
- `error_status: {status: enum(ok | failed), error: str | null}`.
- *"The only place method-specific behaviour lives (FR-051). Metric, reporting and visualisation
  modules MUST NOT branch on method identity; US1 scenario 6 is the test that proves it."*

### MethodConfiguration (per-method, outside source)

- `checkpoint: {path, sha256, size_bytes}` — recorded separately from the availability check's copy.
- `repository: {url, revision: str[40 hex], clone_path}` — clone lives outside the repository;
  nothing vendored, copied or committed (FR-055, SC-013).
- `device: {type: enum(cpu | cuda), name: str}` — `name` is required, `"cuda"` alone is not (FR-060).
- `input_size: [int, int]` — network input canvas.
- `padding_multiple: int` — A: 8, B: letterbox to 1024×1024 stride-64, C: 32. Recorded and inverted
  **per method separately** (FR-024a; aligned size is a multiple of 8 but not of 32).
- `threshold: float` — the configured value.
- `effective_threshold: float` — the value **verified against the produced mask**, which may differ
  from `threshold` where upstream ignores the parameter (Method B's hardcoded ≈0.235 cutoff).
- `channel_order: enum(rgb | bgr)` — the order **actually used**; a suspected reversal degrades
  accuracy without raising an error, so it is recorded rather than silently corrected.
- `preprocessing: dict` — fixed for the whole run, never tuned per page (FR-026/FR-029/FR-030).
- `postprocessing: dict` — same constraint.
- `consumed_output: str` — which pipeline output is consumed; Method B consumes only the refined
  segmentation mask, its text-line polygons are discarded (FR-013/FR-014).
- `label_collapse: {rule: str, verified: bool}` — required where the method's training label space
  distinguishes sub-classes the binary GT has collapsed; verification, not assumption (FR-024b).
- `code_license: str`, `weight_license: str` — two separate fields (FR-056). A: MIT / MIT;
  B: GPL-3.0 / GPL-3.0; C: `none` / `none`.
- `cache_location: path` — outside the repository.
- *"Under FR-052 the configuration travels with the method: the copy a runner uses is authoritative
  for that method's result; the copy retained here records what was configured at admission."*

### AvailabilityVerdict (pre-flight, per method, per environment)

- `method: str`.
- `verdict: enum(available | unavailable)`.
- `environment: {device: {type, name}, interpreter: str, packages: dict}` — the identity of the
  environment the check ran in (FR-017). A verdict is true of *that* environment only.
- `checkpoint_status: {present: bool, missing_artefact: str | null, obtain_from: str | null,
  size_bytes: int | null}`.
- `repository_status: {present: bool, revision_matches: bool, detail: str | null}`.
- `dependency_status: {satisfied: bool, conflict: str | null}` — names the version conflict rather
  than deferring it to inference time.
- `license_status: {code: str, weights: str, restriction: str | null}`.
- `remediation: list[str]` — what to install or obtain to flip the verdict.
- `checked_at: str` — timestamp.
- *"Persisted as JSON so verdicts are retrievable later without re-running (FR-022). A method needing
  an accelerator or legacy interpreter this machine lacks MUST be re-checked in a suitable
  environment before being declared unavailable (FR-021)."*

### ReturnedResult (one method's hand-off, FR-053/FR-058)

- `method: str`.
- `page_list_identity: str` — quoted verbatim from the page list.
- `input_image_identity: dict` — per page, the identity record's value for that page (FR-054a).
- `masks: list[{image_id, path}]` — one binary mask per page, aligned size, FR-004 convention.
- `metadata: list[PageMetadata]` — one sidecar per page (ONBOARDING §6).
- `provenance: ProvenanceRecord` — one per hand-off.
- `validation_verdict: {admitted: bool, refused_reason: str | null, missing_field: str | null}`.
- *"A runner returns prediction masks and provenance, not scores (FR-057)."*

### PageMetadata (per-page sidecar, refusal-grade — ONBOARDING §6)

- `image_id: str`.
- `page_list_identity: str` — **điều kiện sống còn**; wrong or missing ⇒ refused even if the mask is
  correct.
- `input_image_identity: str` — same.
- `method: str`.
- `repository: str`, `commit: str[40 hex]`.
- `code_license: str`.
- `checkpoint: {name, source, size_bytes, sha256, weight_license, loaded_evidence}` — `sha256`, not
  just the filename (FR-044); `loaded_evidence` is **positive** proof by identity, not an inference
  from "the code ran without error" (FR-018b).
- `fold_attribution: str | null` — Method A only; FR-058 refuses a Method A result missing or
  inconsistent in it.
- `input_size: [int, int]`, `output_size: [int, int]`.
- `preprocessing: dict`, `postprocessing: dict`, `threshold: float`, `alignment: dict`.
- `device: {type, name}`, `interpreter: str`, `packages: dict` — the **real** versions of the
  environment that ran, not those in `requirements.txt` (FR-044).
- `seed: int`.
- `run_timestamp: str`.
- `inference_time_seconds: float`, `preprocessing_time_seconds: float`,
  `postprocessing_time_seconds: float` — per page; the project aggregates, the runner does not.
- `status: enum(success | failed)`, `error: str | null`.
- *"This is far richer than Spec 1's `PredictionMask.sidecar`. The DL path carries its own sidecar
  rather than widening the frozen Spec 1 shape."*

### ProvenanceRecord (environment evidence, FR-056/FR-018b)

- `repository: {url, revision: str[40 hex]}` — external repo and exact revision.
- `checkpoint: {identity, size_bytes, sha256}`.
- `code_license: str`, `weight_license: str`.
- `checkpoint_load_evidence: {evidenced: bool, method: str, detail: str}` — distinguishes "evidenced"
  from "not evidenced"; a plausible mask is not evidence.
- `device: {type, name}`.
- `interpreter: str`, `packages: dict`.
- `seed: int`.
- `run_timestamp: str`.
- `deviation: {applied: bool, what: str | null, why: str | null, changes_output: bool}` — a workaround
  applied in the runner's own environment is recorded here as a deviation from the pinned revision
  (FR-055); one that changes the model's output makes the result a **different method**.
- *"A result whose provenance record is missing or incomplete MUST be refused rather than accepted
  with gaps. Completeness is machine-checked; the refusal names the missing field (SC-014)."*

### RunRecord (experiment, FR-044/FR-059)

- `run_id: str` — names the run; no earlier run is ever modified.
- `page_list_identity: str`.
- `methods_admitted: list[str]`, `methods_awaited: list[str]`.
- `method_configuration: dict[str, MethodConfiguration]` — snapshot at admission.
- `method_provenance: dict[str, ProvenanceRecord]`.
- `method_device: dict[str, {type, name}]`.
- `admission_timestamps: dict[str, str]`.
- `started_at: str`, `ended_at: str | null`.
- `nondeterminism_sources: list[str]`.
- *"Holds no single environment or package-version field of its own, because under FR-052 there is no
  single one. Remains reportable while methods are outstanding."*

### AdmittedPrediction (per page per method, after receipt validation)

- `image_id: str`, `method: str`, `run_id: str`.
- `mask_path: str` — `deliverables/<run_id>/<method>/masks/<manga>/<stem>.png`; full aligned
  resolution, FR-004 convention, deterministic path, no run-specific metadata, so Spec 003 can consume
  it from page id and method name alone (FR-043).
- `page_metadata: PageMetadata`.
- `admitted_at: str`.

### DLPerPageMetrics (extends Spec 1's `PerImageMetrics`)

- Reuses `manga, stem, method, run_id, iou, precision, recall, f1, tp, fp, fn,
  inference_time_seconds` from Spec 1's `PerImageMetrics`.
- Adds `degenerate_convention: str | null` — the convention flag applied to that page.
- Adds `device: {type, name}` — the producing device, carried through from the sidecar so timing is
  never reported without it (FR-060).
- *"Computed only by `metrics.compute_metrics`, after admission, never transcribed from a runner."*

### DLMethodSummary (extends Spec 1's `MethodSummary`)

- Reuses `method, run_id, n_images, mean_iou/std_iou, mean_precision, mean_recall, mean_f1,
  mean_time_seconds`.
- Adds `std_precision, std_recall, std_f1` — standard deviation for each of the four metrics.
- Adds `mean_preprocessing_time_seconds`, `mean_postprocessing_time_seconds` where measurable.
- Adds `total_processing_time_seconds`, `successful_pages: int`, `failed_pages: int`.
- Adds `device: {type, name}`, `training_data_overlap: enum(full | partial | unknown | none)` — the
  disclosure flag required next to the scores (SC-011).
- *"A method that did not run produces no summary row with numeric values; it appears as "not run"
  with its reason. No placeholder or imputed value anywhere (FR-036)."*

### DLFailureRecord (extends Spec 1's `FailureReport`)

- `method: str`, `run_id: str`.
- `scope: enum(method-level | page-level)` — the distinction US4 scenario 5 requires: "pages that
  failed inside a method that ran" versus "method did not run".
- `failures: list[{image_id, error, stage, reason}]`.
- `remediation: str | null` — where applicable.
- *"An absent method is never presented as having zero failed pages."*

### ComparisonSummary (cross-method view)

- `run_id: str`, `page_list_identity: str`.
- `rows: list[{method, metrics: {iou, precision, recall, f1} with mean and std, timing, device,
  status: enum(scored | not_run), disclosure: str | null}]` — one row per method **including the
  classical baseline**.
- *"Accuracy is comparable — one evaluation procedure, same pages. Timing is not comparable across
  three machines, so each timing carries its producing device and no cross-device ranking is
  presented. Pixel Accuracy is absent as a primary ranking metric and no OCR/translation/inpainting
  measure ranks any method."*

### VisualisationCase (one selected qualitative example)

- `image_id: str`, `method: str`, `run_id: str`.
- `panels: [raw, ground_truth, prediction, prediction_vs_ground_truth_overlay]` — four panels.
- `selection: {mode: enum(first-N | best-N | worst-N), n: int}` — configurable case count, ranked by
  F1, reusing Spec 1's `visualization` config block.
- `case_kind: enum(good | average | failure)` — the written report separates good from failure cases.

## Relationships

```
DistributablePageList ──┬── ImageBundle ──► runner environment (×3, separately hosted)
                        │
                        └─ page_list_identity ─────────────┐
                                                          │
ReturnedResult ──┬── ProvenanceRecord                     │
                 ├── PageMetadata (×390)                  │
                 └── masks (×390)                         │
                        │                                 │
                        ▼                                 │
              receipt validation (FR-058) ◄───────────────┘
                        │  refusals → DLFailureRecord, nothing written
                        ▼  admissions
              AdmittedPrediction ──► metrics.compute_metrics ──► DLPerPageMetrics
                        │                                              │
                        │                                              ▼
                        │                                      DLMethodSummary
                        ▼
              RunRecord (named run; admitted incrementally, FR-059)
                        │
                        ├──► ComparisonSummary (+ Spec 1 classical baseline)
                        └──► VisualisationCase

MethodConfiguration + AvailabilityVerdict ──► MethodAdapter ──► runner environment
   (Spec 1: ImageMaskPair ─► NormalizedMask ─► AlignedPair ─► GroundTruthMask, project-side only)
```

## Validation rules (entity-level)

- `page_list_identity` is computed once, by the project, and quoted verbatim. A runner that recomputes
  it differently produces a different string and its result is refused.
- The page list and the image bundle are both required to run a method; neither alone is sufficient.
- No artifact under `data/` is ever referenced by an exported or returned artifact; `data/groundtruth/`
  and `data/no-need-to-read/` are never opened on the runner path.
- `effective_threshold` is verified against the produced mask, not copied from configuration.
- `padding_multiple` and its inverse are recorded and applied per method separately.
- Every admitted mask is single-channel uint8, values ⊆ {0, 255}, size == `aligned_size`. Receipt
  validation enforces this **before** `compute_metrics` is called; a mismatch is refused, never
  silently resized, corrected or re-thresholded.
- `code_license` and `weight_license` are distinct and both non-empty; Method C's value is `none`,
  which is a recorded position, not a missing field.
- No `DLMethodSummary` row exists with numeric metrics for a method that did not run.
- No timing figure exists without a producing `device.name`.
- The stored tree contains no third-party model source file and no weight file of any size (SC-013);
  per method it holds only configuration, returned standardised results and documentation.
