# Feature Specification: Data Foundation & Classical Baseline

**Feature Branch**: `001-data-foundation-classical-baseline`

**Created**: 2026-09-09

**Status**: Draft — clarifications resolved, ready for `/speckit-plan`

**Input**: User description: "Xây dựng feature 'Data Foundation và Classical Baseline' cho hệ thống phân đoạn văn bản trong ảnh manga. Xây dựng data loader dùng chung, ánh xạ ảnh gốc với ground-truth mask, kiểm tra tính hợp lệ, chuẩn hóa mask nhị phân, xây dựng các phương pháp phân đoạn cổ điển, và tạo output/interface chung để Spec 2 tích hợp model deep learning. Sử dụng toàn bộ ground-truth hiện có làm benchmark; không tạo benchmark mới; không thay đổi dữ liệu gốc."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Reproducible Data Manifest & Validation (Priority: P1)

As a researcher, I want the system to discover every ground-truth mask in the existing
dataset, pair each one with its source manga page, validate every pair, and emit a
deterministic, reusable manifest together with a validation report — so that I have a
trustworthy index of the benchmark that downstream specs can rely on.

**Why this priority**: Every other capability (normalization, segmentation, metrics,
visualization, Spec-2 integration) consumes the manifest. Without a correct, stable,
validated data foundation there is nothing to evaluate against. This is the MVP slice.

**Independent Test**: Run discovery against the dataset; confirm the manifest contains
exactly the valid image–mask pairs, in a stable order, identical across two consecutive
runs; confirm the validation report enumerates every anomaly (orphan masks, missing or
corrupt files, name mismatches) with no silent drops; confirm no source file changed.

**Acceptance Scenarios**:

1. **Given** the ground-truth root with 45 manga folders and 450 masks, **When** discovery
   runs, **Then** all 450 masks are found and each is classified as either paired (source
   image located) or orphan (no source image), with counts reported.
2. **Given** a mask whose source image exists, **When** pairing runs, **Then** the pair is
   recorded with manga name, page stem, raw-image path, mask path, and both file sizes.
3. **Given** discovery has completed, **When** it is run a second time with no data change,
   **Then** the manifest content and ordering are byte-for-byte identical (deterministic).
4. **Given** the dataset, **When** validation runs, **Then** the report lists, by category:
   masks missing a source image; source images without a mask; corrupt/undecodable files;
   filename mismatches; manga-folder mismatches — and no entry is silently removed.

---

### User Story 2 - Correct Mask Normalization & Alignment (Priority: P2)

As a researcher, I want each ground-truth mask loaded into a single consistent binary
convention — regardless of which of the dataset's text encodings it uses — and spatially
aligned to its source image, so that any prediction can be compared fairly against it.

**Why this priority**: Metrics are meaningless if text pixels are lost during normalization
or if prediction and ground truth are not the same size and registration. This unblocks
correct evaluation for all methods.

**Independent Test**: Feed reference masks covering every observed encoding (white
background; magenta `(255,1,255)`; near-black `(1,1,1)`; an all-white empty mask) and assert
the normalized text-pixel set is exactly the non-white pixels (no loss). Feed an image/mask
pair with differing dimensions and assert the aligned mask matches the target size, with
pre- and post-alignment sizes recorded.

**Acceptance Scenarios**:

1. **Given** a mask using magenta `(255,1,255)` text on white, **When** normalized, **Then**
   every magenta pixel becomes text and no text is lost by an equality-only test.
2. **Given** a mask using near-black `(1,1,1)` text on white, **When** normalized, **Then**
   every near-black pixel becomes text.
3. **Given** a mask containing both magenta and near-black text, **When** normalized,
   **Then** both are captured under the same single convention.
4. **Given** an all-white mask with no text, **When** normalized, **Then** it yields an
   all-background mask and is handled without error in downstream metrics.
5. **Given** a raw image (1170×1654 H×W) and its mask (1176×1656 H×W), **When** aligned with the
   default crop strategy, **Then** the mask is cropped top-left to 1170×1654 so prediction and
   ground truth share one identical size before any metric, zero text pixels are lost (the
   cropped padding is verified empty), and pre/post-alignment sizes are recorded.

---

### User Story 3 - Configurable Classical Baseline Segmentation (Priority: P3)

As a researcher, I want a configurable set of classical image-processing methods that each
take a raw manga page and return a binary text mask, runnable individually or as a full
baseline sweep over the manifest — so that I have reproducible classical results.

**Why this priority**: Provides the baseline against which Spec 2's deep-learning models will
be compared. Depends on the manifest (P1) and normalization/alignment (P2).

**Independent Test**: Run one method on the manifest and assert it produces, per image, a
binary mask at the aligned size plus a recorded processing time; then run all methods and
assert each is independently selectable and all share the same manifest and output contract.

**Acceptance Scenarios**:

1. **Given** a configured method, **When** it processes a raw page, **Then** it returns a
   binary text mask at the aligned size and records the per-image processing time.
2. **Given** any method, **When** it writes output, **Then** masks go to a separate output
   directory and no source data is overwritten.
3. **Given** the method registry, **When** invoked, **Then** a single named method or the
   entire baseline set can be run over the same manifest.
4. **Given** a method's parameters, **When** configured, **Then** thresholds, kernel sizes,
   iteration counts, and method-specific parameters are settable via configuration.

---

### User Story 4 - Per-Image Metrics & Per-Method Summary (Priority: P4)

As a researcher, I want IoU, Precision, Recall, and F1 computed per image between each
prediction and the normalized/aligned ground truth, persisted per image, and aggregated per
method with mean, standard deviation, and average processing time — so that I can compare
baselines objectively.

**Why this priority**: Turns raw masks into decision-grade results. Depends on P2 (alignment)
and P3 (predictions).

**Independent Test**: Given predictions and ground truth for a small manifest, assert per-image
metrics are persisted, the per-method summary reports mean/std and average time, Pixel
Accuracy is absent from primary metrics, and that one failing image is logged while the rest
still complete.

**Acceptance Scenarios**:

1. **Given** a prediction and its aligned ground truth, **When** evaluated, **Then** IoU,
   Precision, Recall, and F1 are computed and persisted for that image.
2. **Given** all per-image results for a method, **When** summarized, **Then** the summary
   reports mean and standard deviation of each metric plus average per-image processing time.
3. **Given** an image that fails during processing, **When** evaluation runs, **Then** the
   error is recorded in the report and the remaining images are still processed.
4. **Given** the metric set, **When** reported, **Then** Pixel Accuracy is not used as a
   primary metric.

---

### User Story 5 - Visualization & Stable Spec-2 Interface (Priority: P5)

As a researcher, I want overlay visualizations (raw, ground-truth mask, prediction mask, and
prediction-vs-ground-truth overlay) and a stable output/interface contract, so that I can
inspect results and Spec 2 can add deep-learning models without changing metrics or output
format.

**Why this priority**: Enables qualitative checking and guarantees forward compatibility for
Spec 2. Depends on the full pipeline.

**Independent Test**: Generate visualizations for the configured number of images and assert
all four panels are present; register a stub "method" implementing the common interface and
assert it produces metrics through the unchanged pipeline and output format.

**Acceptance Scenarios**:

1. **Given** a processed image, **When** visualization runs, **Then** it produces the raw
   image, the ground-truth mask, the prediction mask, and a prediction-vs-ground-truth overlay.
2. **Given** a configured visualization limit, **When** visualization runs, **Then** at most
   that many images are rendered, while metrics still run on the entire valid manifest.
3. **Given** a new method implementing the common interface (e.g., a Spec-2 deep-learning
   stub), **When** integrated, **Then** it produces predictions consumed by the same metric
   and output pipeline with no change to metric definitions or output format.

---

### Edge Cases

- A ground-truth mask exists but its source image is absent from the entire dataset (60 such
  masks across 6 manga were found): recorded in the validation report; excluded from the
  benchmark (no source image means nothing to evaluate) but never dropped silently.
- A source image exists but has no corresponding ground-truth mask: recorded as
  "image without mask" (these are not part of the GT-driven benchmark).
- A mask uses only one text encoding on one page and a different encoding on another page of
  the same manga (observed: encoding varies per mask): normalization must be per-mask, not
  per-manga.
- An all-white mask with zero text pixels: metrics must not divide by zero; the case is
  reported and handled deterministically.
- A corrupt or truncated image/mask file that cannot be decoded: logged, skipped, run continues.
- Raw image and mask sizes differ (the normal case here: 1170×1654 vs 1176×1656): alignment
  required before metrics.
- Page stems with different zero-padding or non-numeric names between the two trees: detected
  as a filename mismatch rather than silently paired.
- Manga folder name present in one tree but not the other: detected as a folder mismatch.
- Two consecutive discovery runs must never differ (no timestamps, locale ordering, or
  filesystem-walk order leaking into the manifest order).

## Clarifications

### Session 2026-09-10

- Q: Benchmark scope for Spec 001 evaluation — all 390 valid pairs or a ~100–200 subset per
  the course outline (đề cương §5.1)? → A: All 390 valid pairs are evaluated in Spec 001; any
  subsampling for deep-learning evaluation is Spec 2's decision, made via the same manifest
  without changing Spec 001.
- Q: Should the classical baseline include an explicit preprocessing stage before
  segmentation? → A: Yes — a separate, configurable preprocessing module (grayscale
  conversion, optional denoising) runs before all classical methods (per đề cương §5.2
  "tiền xử lý").
- Q: Which images should be visualized when the visualization count is limited? → A: A
  configurable selection mode — `first-N` in manifest order (default) or `best-N`/`worst-N`
  ranked by F1 — to support the typical success/failure case analysis required by the course
  outline (đề cương §6).
- Q: Does Spec 001 need a cross-method comparison artifact (table/chart) beyond the per-method
  summary? → A: No — FR-030's per-method summary suffices for the classical-only scope; the
  course outline's (đề cương §6) cross-method comparison tables/charts are produced later from
  the persisted per-image outputs once Spec 2's deep-learning results exist.

## Requirements *(mandatory)*

### Functional Requirements

**Data discovery, mapping & manifest**

- **FR-001**: System MUST discover all ground-truth masks under the configured ground-truth
  root, recursively, identifying each by manga folder name and page stem.
- **FR-002**: System MUST map each mask to its source image by manga folder name plus page
  stem (page `NNN.png` ↔ page `NNN.jpg` within the same manga folder), without assuming a
  fixed extension and without renaming source data.
- **FR-003**: System MUST scan the entire existing ground-truth set (all 450 masks); it MUST
  NOT create a new benchmark or select a subset. Spec 001's evaluation scope is all valid
  pairs found (390); a narrower evaluation subset for a downstream spec is that spec's
  decision, made via the manifest, without re-running discovery or changing Spec 001.
- **FR-004**: System MUST generate the manifest deterministically from the data pairs actually
  found, ordered stably (by manga name, then page stem) so repeated runs produce identical output.
- **FR-005**: System MUST persist the manifest as a reusable artifact with a stable schema
  consumable by downstream specs.

**Validation & data safety**

- **FR-006**: System MUST validate each candidate pair: the source image exists and decodes;
  the mask exists and decodes.
- **FR-007**: System MUST produce a validation report enumerating, by category: masks missing a
  source image; source images without a mask; corrupt/undecodable files; filename mismatches;
  manga-folder mismatches.
- **FR-008**: System MUST NOT automatically delete or silently drop erroneous data; every issue
  MUST be recorded in the validation report.
- **FR-009**: System MUST treat all source data (`data/raw/...`, `data/groundtruth/...`) as
  read-only and MUST NOT modify, move, or delete any original file.
- **FR-010**: System MUST NOT read or use data under `data/no-need-to-read/` unless explicitly
  requested.

**Mask normalization**

- **FR-011**: System MUST read each ground-truth PNG in a mode that preserves all channel
  information needed to distinguish every observed text encoding (the masks are RGB).
- **FR-012**: System MUST normalize each mask to a single binary convention — **background = 0,
  text = 255** (single-channel `uint8`) — such that ALL observed encodings are captured:
  white background `(255,255,255)`, magenta text `(255,1,255)`, and near-black text `(1,1,1)`.
  Text MUST be defined as any pixel that is not pure white, so normalization MUST NOT lose text
  by testing pixel equality against only `255` or only `1`, nor by a grayscale conversion that
  collapses magenta to a mid value. This single convention (text = 255) MUST be used consistently
  across the data loader, classical methods, metric calculation, visualization, and the Spec-2
  interface.
- **FR-013**: System MUST handle an all-background (empty) mask without error and represent it
  as a mask with zero text pixels.
- **FR-014**: System MUST apply the binary convention consistently across all consumers (loader,
  classical methods, metrics, visualization, Spec-2 interface).

**Alignment**

- **FR-015**: System MUST compare the source-image size and the mask size for each pair and MUST
  NOT assume they are equal.
- **FR-016**: System MUST guarantee that a prediction mask and its ground-truth mask have
  identical dimensions before any metric is computed, using a configured alignment strategy.
  The measured cause is a uniform canvas-size difference — the GT mask is +2px wider and +6px
  taller than the raw page (1656×1176 vs 1654×1170 W×H, constant across all 390 pairs). Data
  inspection confirmed the mask is top-left aligned with the raw page and that the extra padding
  region (right 2 columns, bottom 6 rows) contains zero text pixels on every mask (max text
  y = 1169 < 1170, max text x = 1653 < 1654). The chosen strategy is therefore to crop the GT
  mask to the raw native size (top-left 1654×1170), which is verified lossless: no text pixel is
  discarded and the binary mask undergoes no resampling. All classical methods run on the raw
  page at native size, and all metrics are computed in that same raw native space (1654×1170).
  Resize MUST NOT be the default strategy, because it would introduce resampling artifacts into
  a binary mask where a lossless crop suffices. The alignment strategy MUST remain configurable
  so a future dataset with different registration can select another strategy (resize, pad)
  without changing metric definitions.
- **FR-017**: System MUST apply a distortion-free alignment for the default strategy — cropping
  the binary mask with no resampling — and, for any alternative configured strategy, MUST use a
  binary-preserving resampling (nearest-neighbor) so mask values stay in {0, 255}. The chosen
  method MUST be documented in the configuration and the validation/alignment report.
- **FR-018**: System MUST record, per pair, the size before and after alignment.
- **FR-019**: System MUST include tests for alignment across differing-size cases.

**Classical baseline segmentation**

- **FR-020**: System MUST provide a dedicated, configurable preprocessing stage that runs
  before all classical methods — grayscale conversion plus optional denoising (e.g., blur),
  with parameters settable via configuration — and System MUST provide these classical
  text-segmentation capabilities: Otsu thresholding; adaptive thresholding; MSER; edge
  detection (where appropriate); connected components; morphological operations (dilation,
  erosion, opening, closing); and configurable multi-step pipelines that compose these stages.
  The preprocessing output is the input page delivered to each segmentation method.
- **FR-021**: Each method MUST accept a raw manga page image and return a binary text mask.
- **FR-022**: Each method's output mask MUST match the aligned size used for evaluation.
- **FR-023**: Each method MUST expose its own configuration (thresholds, kernel size, iteration
  count, and method-specific parameters).
- **FR-024**: Methods MUST write masks to a separate output directory and MUST NOT overwrite
  source data.
- **FR-025**: System MUST record the processing time for each image, per method.
- **FR-026**: System MUST allow running a single named method independently, or running the
  entire baseline set over the same manifest.

**Metrics & evaluation**

- **FR-027**: System MUST compute IoU, Precision, Recall, and F1-score between each prediction
  mask and the normalized, aligned ground-truth mask.
- **FR-028**: System MUST NOT use Pixel Accuracy as a primary metric.
- **FR-029**: System MUST persist metrics per image.
- **FR-030**: System MUST produce a per-method summary table including mean and standard
  deviation of each metric and the average per-image processing time.
- **FR-031**: When an individual image fails, System MUST log the error to the report and
  continue processing the remaining images.

**Visualization**

- **FR-032**: System MUST produce, for visual inspection, an image containing the raw page, the
  ground-truth mask, the prediction mask, and an overlay of prediction vs ground truth.
- **FR-033**: System MUST allow the number of visualized images to be limited by configuration,
  while metrics still run on the entire valid manifest. The visualization set MUST be chosen by
  a configurable selection mode: `first-N` in manifest order (the deterministic default), or
  `best-N`/`worst-N` ranked by F1 (for typical success/failure case analysis).

**Architecture & operability**

- **FR-034**: System MUST separate concerns into distinct modules: dataset discovery; dataset
  validation; manifest; image loading; mask loading; mask normalization; alignment;
  preprocessing; classical segmentation; metrics; visualization; configuration; CLI.
- **FR-035**: System MUST provide a common method interface so that Spec 2 can add deep-learning
  models without modifying metric definitions or the output format.
- **FR-036**: All data and output paths MUST live in a configuration file and MUST NOT be
  hard-coded in logic.
- **FR-037**: System MUST run on CPU (no GPU required).

### Out of Scope

- Deep-learning model integration (deferred to Spec 2 via the common interface).
- Inpainting, OCR, machine translation, and text rendering.
- Model training of any kind.
- Creating or modifying ground-truth data.
- Creating a new benchmark or a new benchmark subset.
- Cross-method comparison tables/charts spanning classical and deep-learning methods (deferred
  until Spec 2's results exist; they will be derived from Spec 001's persisted per-image
  outputs without changing Spec 001's metric definitions or output format).
- Reading or using `data/no-need-to-read/`.

### Key Entities

- **Manga**: A title identified by its folder name; groups pages.
- **Page**: A single page identified by its zero-padded stem (`NNN`) within a manga.
- **ImageMaskPair**: A raw image path, a ground-truth mask path, manga name, page stem, raw
  size, mask size, and a validity status.
- **Manifest**: The deterministic, ordered collection of valid pairs plus generation metadata;
  reusable by downstream specs.
- **ValidationReport**: All detected issues grouped by category, with counts and file references.
- **BinaryMask**: A normalized mask with its convention (background/text value) and dimensions.
- **SegmentationMethod**: An identifiable method with configuration and the contract
  `segment(image) -> binary mask`.
- **Prediction**: A method's output for a pair — predicted mask reference and processing time.
- **ImageMetrics**: Per pair, per method — IoU, Precision, Recall, F1, aligned size, status.
- **MethodSummary**: Per method — mean and standard deviation of each metric, average time.
- **Visualization**: The rendered raw / GT / prediction / overlay composite for a pair.
- **Configuration**: Paths, alignment strategy, binary convention, per-method parameters,
  visualization limit, and visualization selection mode (`first-N` / `best-N` / `worst-N` by F1).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Discovery finds 100% of the ground-truth masks present (450) and classifies each as
  paired or orphan; the manifest reproducibly contains exactly the valid pairs (expected 390),
  byte-for-byte identical across repeated runs.
- **SC-002**: The validation report enumerates every data anomaly (orphan masks, missing/corrupt
  files, name mismatches) with zero silent drops, and no source file is modified (verified by
  checksum or modification-time invariance over a full run).
- **SC-003**: Normalization classifies 100% of non-white pixels as text for every observed
  encoding, and a reference mask's normalized text count equals its manually verified text count
  (no text lost); the empty mask normalizes to zero text pixels without error.
- **SC-004**: For 100% of evaluated pairs, prediction and ground-truth masks have identical
  dimensions at metric time, with pre- and post-alignment sizes recorded.
- **SC-005**: Each classical method runs both independently and over the full manifest, producing
  a binary mask and a timing for every valid pair it processes; a single failing image does not
  abort the run (the run completes and the error is logged).
- **SC-006**: Per-image IoU, Precision, Recall, and F1 are persisted for every evaluated pair;
  each per-method summary reports mean, standard deviation, and average per-image processing
  time; Pixel Accuracy is absent from the primary metric set.
- **SC-007**: Visualizations are produced for the configured number of images, each showing all
  four panels (raw, ground truth, prediction, overlay).
- **SC-008**: A new method implementing the common interface (a Spec-2 stub) produces metrics
  through the unchanged pipeline, with no modification to metric definitions or output format
  (verified by integrating the stub).

## Assumptions

- **Technology preference (not mandate)**: Python with OpenCV, NumPy, pandas, and matplotlib is
  preferred, per the user's "Ưu tiên" (prefer) wording; the environment also has Pillow and SciPy
  available. Final stack is fixed at `/speckit-plan`.
- **Data layout is fixed** as provided: ground-truth under `data/groundtruth/post-processed/<manga>/<NNN>.png`;
  source images under `data/raw/Manga109s_released_2026_05_21/images/<manga>/<NNN>.jpg`.
- **Mapping rule (verified)**: a mask pairs with the source image sharing the same manga folder
  name and the same zero-padded page stem; `.png` ↔ `.jpg`.
- **Uniform sizes (verified)**: all 450 ground-truth masks are 1176×1656 (H×W); all 390 pairable
  source images are 1170×1654. The mask is larger by +6px (H) and +2px (W) — constant across
  every pair, never varying per manga or page.
- **Alignment geometry (verified)**: the mask is top-left aligned with the raw page; its extra
  padding region (right 2 columns, bottom 6 rows) contains zero text pixels on all 390 pairs
  (max text y = 1169, max text x = 1653). Cropping the mask top-left to 1170×1654 is therefore
  lossless, and no pair needs translation, resize, or padding. The alignment strategy still
  remains configurable for future datasets.
- **Binary convention (decided)**: normalized masks are single-channel with background = 0 and
  text = 255 (`uint8`), used identically by loader, methods, metrics, visualization, and Spec 2.
- **GT text encodings (verified, exhaustive)**: across all 450 masks only three unique pixel
  colors exist — background pure white `(255,255,255)`, text magenta `(255,1,255)` (BGR), and
  text near-black `(1,1,1)` — with no anti-aliased intermediates. Encoding varies per mask:
  357 masks use both text colors, 89 magenta-only, 3 near-black-only
  (`BurariTessenTorimonocho/005`, `EvaLady/007`, `YoumaKourin/002`), and 1 all-white empty mask
  (`EvaLady/006`), which is the reference case for the empty-mask edge handling.
- **Six manga lack source images**: `Belmondo`, `BokuHaSitatakaKun`, `ByebyeC-BOY`,
  `GOOD_KISS_Ver2`, `TotteokiNoABC`, `YouchienBoueigumi` have ground-truth masks but no images
  anywhere in the release (absent from `books.txt`, annotations, and the images tree). Their 60
  masks are excluded from the benchmark (no source image to evaluate against) and appear only
  in the validation report.
- **Determinism** means ordered by (manga name, page stem) with no timestamps, locale-dependent
  ordering, or filesystem-walk order affecting the manifest.
- **CPU-only execution**; no GPU is available or required.
- **Spec language**: this specification is written in English for tooling consistency; the user
  communicates in Vietnamese.
