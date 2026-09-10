# Feature Specification: Text Removal & Image Inpainting

**Feature Branch**: `003-text-removal-inpainting`

**Created**: 2026-09-10

**Status**: Draft — ready for `/speckit-plan`. Two clarification rounds completed 2026-09-10:
round 1 (hand-off form: per-page metadata sidecar; default config: dilation 3×3 ellipse ×1,
radius 3), round 2 (qualitative assessment by human reviewer on a system-generated
pre-labelled scaffold, no auto-scoring; representative-sample selection: fixed N per rule ×
segmentation method, N=5 initial default finalized at planning; performance summary grouped
by method × algorithm with mean time per image). 39 FRs, 12 SCs, 6 user stories, 13 edge
cases, 0 `[NEEDS CLARIFICATION]` markers.

**Input**: User description: "Xây dựng feature 'Text Removal và Image Inpainting' cho pipeline xử lý ảnh Manga. Spec 1 đã xác định dataset, manifest, image-mask mapping, mask convention và alignment convention. Spec 2 cung cấp prediction mask từ: Classical baseline, Manga-Text-Segmentation, comic-text-detector, UNet++ EfficientNetV2 (các model chạy độc lập trên Google Colab). Feature này không chạy lại model segmentation; chỉ nhận prediction mask và xử lý tiếp ảnh manga: xóa vùng văn bản, phục hồi nền bằng OpenCV inpainting (TELEA, NS), so sánh kết quả giữa các phương pháp, tạo output chuẩn cho Spec 4, tạo visualization và báo cáo qualitative. Không tạo benchmark mới, không thay đổi dữ liệu gốc, không dùng data/no-need-to-read/."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Receive and Validate Spec 2 Prediction Masks (Priority: P1)

As a researcher, I want the system to take each segmentation method's stored prediction
masks, map every mask to exactly one manifest page by the agreed identity rule, and
validate each input (image present, mask present, size and binary convention correct) —
so that inpainting never runs on a mis-mapped, mis-sized or corrupt input, and every
rejection is recorded with a named reason instead of silently corrected.

**Why this priority**: Nothing downstream is trustworthy if the mapping or the mask
validity is wrong. This is the intake gate for the whole feature and the MVP slice.

**Independent Test**: Feed a small set of prediction masks — including one correctly
mapped, one whose page identity mismatches the manifest, one of the wrong size, one
non-binary, and one with a missing source image — and assert exactly the intended ones
pass, every failure is recorded under its named error category, no input is silently
resized, re-thresholded or renamed, and valid processing continues.

**Acceptance Scenarios**:

1. **Given** a prediction mask stored per the Spec 2 output contract, **When** intake
   runs, **Then** it is mapped to exactly one manifest page via manga folder name plus
   page stem (`image_id` = `<manga>/<NNN>`) and paired with the original page image.
2. **Given** a prediction mask whose page identity does not match the manifest mapping,
   **When** intake runs, **Then** the sample is recorded as failed (image-ID mismatch)
   and is not processed.
3. **Given** a prediction mask at a size other than the aligned page size or the known
   ground-truth padded size, **When** intake runs, **Then** it is recorded as failed
   (size mismatch) and is not silently resized.
4. **Given** a mask containing pixel values outside the binary set, **When** intake
   runs, **Then** it is recorded as failed (non-binary mask) and is not silently
   thresholded.
5. **Given** a valid sample whose source image file is missing or undecodable, **When**
   intake runs, **Then** the failure is recorded under its category and the remaining
   samples are still processed.

---

### User Story 2 - Inpaint Text Regions with TELEA and NS (Priority: P1)

As a researcher, I want each validated prediction mask normalized to the single binary
convention, aligned into the aligned page space, optionally dilated by one shared
configuration, and used to remove the text and restore the background with both
OpenCV `INPAINT_TELEA` and `INPAINT_NS` — so that every segmentation method's masks go
through the identical text-removal procedure and produce comparable restored pages.

**Why this priority**: This is the core capability the feature exists for. It depends
only on US 1 and delivers the first usable inpainted pages.

**Independent Test**: Run the core processing over a small fixture (a few pages, a
synthetic or stored prediction mask each) and assert that, per sample, both algorithm
outputs are produced at exactly the aligned page size, the processed (dilated) mask
differs from the raw mask only by the configured dilation, all processing parameters
appear in the per-sample metadata, and identical inputs with identical configuration
produce byte-identical outputs.

**Acceptance Scenarios**:

1. **Given** a validated sample, **When** processing runs, **Then** the mask is
   normalized to the binary convention, aligned into the aligned page space, and both
   the raw prediction mask and the processed (post-dilation) mask are recorded.
2. **Given** a validated sample, **When** inpainting runs, **Then** both a TELEA result
   and an NS result are produced, each exactly the aligned page size, in the configured
   output format.
3. **Given** dilation is configured (kernel shape, size, iterations), **When** mask
   processing runs, **Then** the same configuration is applied to every page of every
   segmentation method in the main benchmark and is recorded in the metadata.
4. **Given** a fully background (empty) mask after normalization, **When** processing
   runs, **Then** the case is recorded under its error/edge category and handled
   consistently across methods without aborting the batch.
5. **Given** the same input and configuration, **When** processing runs twice, **Then**
   the outputs are identical (deterministic, reproducible).

---

### User Story 3 - Batch Runs, Error Isolation and Run Separation (Priority: P2)

As a researcher, I want to process a whole segmentation method or all prediction masks
in batch, with every failure logged to an error report without stopping the run, and
with each run identified by a run ID that never overwrites a previous run — so that
large-scale processing is robust and every experiment remains on disk.

**Why this priority**: Turns the per-sample capability into the actual benchmark-scale
operation; depends on US 1–2.

**Independent Test**: Run a batch containing a deliberate failure of each error
category; assert the run completes over all remaining samples, the error report
enumerates every induced failure by category with the sample identifier and reason, a
second run with a different run ID coexists with the first, and reusing an existing run
ID is refused unless overwrite was explicitly requested.

**Acceptance Scenarios**:

1. **Given** a batch where one sample fails, **When** the batch runs, **Then** all other
   samples complete and the failure is recorded in the error report.
2. **Given** the nine defined error categories, **When** each is induced, **Then** the
   error report records each occurrence with category, sample identifier and reason.
3. **Given** a run ID that already exists on disk, **When** a new run is started, **Then**
   the system refuses to overwrite the prior results unless explicitly requested.
4. **Given** two completed runs, **When** both are inspected, **Then** their output
   trees and metadata remain intact and independent.

---

### User Story 4 - Comparative Visualization Boards (Priority: P3)

As a researcher, I want a comparison board for each selected representative sample
showing the original page, the raw prediction mask, the dilated mask, the TELEA result
and the NS result, with samples selectable by Spec 2 metric profiles — so that the
differences between segmentation methods and inpainting algorithms can be inspected
side by side.

**Why this priority**: Makes the comparison human-reviewable; depends on US 2–3.

**Independent Test**: Generate boards for samples selected by each selection rule and
assert each board contains all five labeled panels with the correct image ID, method
and configuration labels.

**Acceptance Scenarios**:

1. **Given** a selected sample, **When** a board is generated, **Then** it shows all
   five panels — original page, raw prediction mask, mask after dilation, TELEA result,
   NS result — each labeled.
2. **Given** the selection rules, **When** sample selection runs, **Then** samples can
   be chosen by: highest IoU/F1 from Spec 2; lowest IoU/F1 from Spec 2; most false
   positives; most false negatives; and manually flagged artifact cases.
3. **Given** several segmentation methods and both algorithms, **When** boards are
   generated, **Then** the method, algorithm, mask-processing configuration and image ID
   are visible as labels on or beside each board.

---

### User Story 5 - Qualitative Report Against the Defined Criteria (Priority: P3)

As a researcher, I want a qualitative report scaffold for assessing the inpainting results per
sample against the seven defined criteria (text fully removed; text missed; background
over-erased; naturalness of restoration; artifacts/noise; halo or border effects; damage
to linework, texture and panel borders), with each entry clearly labeled and the ratings
left for human completion — so that the
comparison between methods rests on documented observations instead of a single
quantitative score.

**Why this priority**: This is the feature's evaluation deliverable; depends on US 2–4.

**Independent Test**: Generate the report scaffold over a set of samples and assert every
entry carries the segmentation method, inpainting algorithm, mask-processing configuration,
image ID, an empty rating field per the seven criteria, and a comment field — with no
rating auto-filled by the system (a human reviewer completes them).

**Acceptance Scenarios**:

1. **Given** processed outputs, **When** the qualitative report is generated, **Then**
   each entry labels the segmentation method, inpainting algorithm, mask-processing
   configuration and image ID.
2. **Given** the report scaffold, **When** a human reviewer completes it, **Then** each
   sample is assessable against all seven criteria with a consistent, documented rating
   scale and a free-text observation — with no criterion pre-scored by the system.
3. **Given** the report, **When** read, **Then** no method ranking is presented as a
   single quantitative score, and no PSNR/SSIM figure appears as a primary metric.

---

### User Story 6 - Ablation Over Dilation and Radius, Separated from the Benchmark (Priority: P4)

As a researcher, I want to run supplementary sweeps over dilation configurations and
inpaint radii, stored clearly apart from the main benchmark outputs — so that the main
benchmark stays strictly uniform while parameter sensitivity can still be studied.

**Why this priority**: Valuable for tuning guidance but explicitly secondary; the main
benchmark must not be affected by it.

**Independent Test**: Run an ablation with several dilation/radius configurations and
assert each configuration's outputs live in a clearly separated namespace outside the
main benchmark results, each records its own configuration, and the main benchmark
outputs are untouched.

**Acceptance Scenarios**:

1. **Given** multiple dilation or radius configurations, **When** an ablation runs,
   **Then** each configuration produces its own outputs in a clearly separated location,
   labeled as ablation, distinct from the main benchmark tree.
2. **Given** an ablation run, **When** its metadata is inspected, **Then** the exact
   dilation and radius configuration of each output is recorded.
3. **Given** an ablation run, **When** the main benchmark comparison is produced,
   **Then** ablation outputs are excluded from it.

---

### Edge Cases

- **Mask maps to no manifest page** (unknown manga folder or page stem): recorded as
  image-ID mismatch / unmappable; never silently re-matched by similarity.
- **Mask maps to more than one manifest page**: recorded as ambiguous mapping; not
  processed.
- **Source image missing or undecodable**: recorded under its category; batch continues.
- **Prediction mask missing for a manifest page**: recorded (missing prediction mask);
  the page is simply absent from that method's outputs, not fabricated.
- **Mask size neither the aligned page size nor the known padded ground-truth size**:
  recorded as size mismatch; never silently resized.
- **Empty mask (zero text pixels)**: recorded under its category; handled consistently
  across methods; the batch continues.
- **Non-binary mask** (values outside {0, 255} after normalization): recorded; never
  silently thresholded or clipped.
- **Corrupt or truncated source image or mask file**: recorded; batch continues.
- **Inpainting failure on one sample** (e.g. an internal error): recorded with reason;
  batch continues.
- **Output write failure** (unwritable path, disk full): recorded; batch continues where
  possible; the error report states which outputs are missing.
- **Run-ID collision**: existing runs are never overwritten unless explicitly requested.
- **Ablation leakage**: ablation configurations must never end up inside the main
  benchmark outputs or be presented as benchmark results.
- **Spec 2 mask at the ground-truth padded size** (multiple-of-8 canvas): aligned by the
  agreed lossless top-left crop to the aligned page size, with pre/post sizes recorded —
  never by resampling.

## Clarifications

### Session 2026-09-10

- Q: Who fills in the seven-criteria ratings and the written observations in the
  qualitative report — the system itself, or a human reviewer working from a pre-labelled
  scaffold the system generates? → A: A human reviewer. The system generates the fully
  labelled scaffold (every entry pre-labelled with segmentation method, inpainting
  algorithm, mask-processing configuration, image ID and board reference, with empty rating
  and observation fields) and MUST NOT auto-score any criterion; the reviewer completes the
  ratings and observations. Bound by amending FR-032 and FR-035.
- Q: How many samples should each FR-034 selection rule pick, and should that count apply
  per segmentation method or across all methods combined? → A: A fixed count N per selection
  rule per segmentation method — every method receives the same number of qualitative
  samples. N is finalized at planning; the initial default is 5. Bound by amending FR-034.
- Q: Should the recorded per-sample inpainting timings be aggregated into a mean-per-image
  performance summary, or remain per-sample records only? → A: Keep per-sample timing and
  add a performance summary grouped by segmentation method × inpainting algorithm, including
  mean processing time per image, sample count, failures, and runtime/device metadata.
  Bound by amending FR-022.

## Requirements *(mandatory)*

### Functional Requirements

#### Reuse of the Spec 1 and Spec 2 foundation

- **FR-001**: The system MUST consume the existing Spec 1 benchmark manifest as the single
  source of truth for the evaluated page list and the image–mask mapping, and MUST NOT
  construct a new manifest, a new benchmark subset, or a different page list.
- **FR-002**: The system MUST reuse the Spec 1 binary mask convention — single channel,
  unsigned 8-bit, background = 0, text = 255 — for every mask it reads, processes or writes.
- **FR-003**: The system MUST reuse the Spec 1 alignment convention: all processing happens
  in the aligned page space (the raw native page size, uniform at 1654×1170 for this
  benchmark), and MUST NOT resize or otherwise alter the original page image to fit a mask.
- **FR-004**: The system MUST consume Spec 2 prediction masks according to the Spec 2 output
  contract: each mask locatable by manifest page identifier plus segmentation method name,
  at full aligned resolution under the binary convention, from a deterministic per-page
  path, interpretable without run-specific metadata.
- **FR-005**: The system MUST NOT execute, re-run, train or fine-tune any segmentation model,
  and MUST NOT contain any segmentation-model runtime. Its only inputs are stored prediction
  masks and their metadata.
- **FR-006**: The system MUST treat all of `data/raw/…`, `data/groundtruth/…`, the Spec 1
  manifest, and Spec 2's prediction-mask outputs as read-only, and MUST NOT modify, move,
  rename or delete any of them. Where a raw prediction mask is needed inside an output
  structure, it MUST be copied, never moved.
- **FR-007**: The system MUST NOT read any data from `data/no-need-to-read/` unless
  explicitly requested.

#### Intake, mapping and validation

- **FR-008**: The system MUST map every prediction mask to exactly one manifest page by
  manga folder name plus zero-padded page stem, with `image_id` defined as the composite
  `<manga>/<NNN>`. A mask mapping to zero pages is invalid (image-ID mismatch); a mask
  mapping to more than one page is invalid (ambiguous mapping). Neither case MAY be resolved
  by similarity or guesswork.
- **FR-009**: Each accepted prediction-mask input MUST carry, or be joinable to, the following
  information: image_id; manga/book name; page name; original image path; segmentation
  method; model/repository identity; experiment/run ID; mask size; and any preprocessing and
  alignment metadata recorded when the mask was produced. The hand-off physical form is
  **resolved**: Spec 2 delivers each prediction mask as a PNG accompanied by a per-page JSON
  metadata sidecar (provenance, geometry, experiment/run ID, preprocessing and alignment
  metadata) located next to the mask; a mask without its sidecar is an invalid input
  (missing metadata), recorded per FR-010/FR-012 and never silently joined from elsewhere.
- **FR-010**: The system MUST validate every input on receipt and MUST reject — never silently
  correct — any input failing: source image existence and decodability; prediction-mask
  existence and decodability; page-identity consistency between the mask's recorded identity
  and the derived mapping; mask size; binary value set after normalization; and emptiness.
- **FR-011**: A failing sample MUST NOT stop the batch: the error MUST be recorded and
  processing MUST continue with the remaining valid samples.
- **FR-012**: The error report MUST enumerate occurrences by category, covering at minimum:
  missing source image; missing prediction mask; wrong image ID / unmappable or ambiguous
  mapping; wrong size; empty mask; non-binary mask; corrupt/undecodable image; inpainting
  failure; output write failure — each with the sample identifier, the category and the
  specific reason.
- **FR-013**: The system MUST NOT silently resize, re-threshold, re-crop, rename or re-map a
  rejected input to make it pass.

#### Mask processing

- **FR-014**: The system MUST read prediction masks in a mode that preserves all channel
  information needed to verify the binary convention, and MUST normalize every accepted mask
  to the binary convention (background = 0, text = 255). If any pixel value falls outside the
  binary set after normalization, the mask is invalid (non-binary) and MUST be rejected.
- **FR-015**: The system MUST check each mask's size against the original image and apply the
  agreed alignment: a mask already at the aligned page size passes through unchanged; a mask
  at the known ground-truth padded size (the multiple-of-8 canvas) is cropped top-left to the
  aligned page size — a lossless, resampling-free operation per the verified Spec 1 geometry;
  any other size is a validation error. Pre- and post-alignment sizes MUST be recorded.
- **FR-016**: The system MUST support optional dilation of the text mask to cover text borders
  before inpainting, with configurable kernel shape, kernel size and iteration count, and
  MUST record the effective configuration in every affected sample's metadata.
- **FR-017**: In the main benchmark, the mask-processing configuration (including dilation)
  MUST be identical for every page and every segmentation method. Any differing configuration
  MUST be run only as a separately labelled ablation or supplementary experiment (FR-024).
- **FR-018**: All mask-processing steps applied to a sample — normalization, alignment
  (with pre/post sizes), dilation configuration — MUST be recorded in that sample's metadata.

#### Inpainting

- **FR-019**: The system MUST support OpenCV `INPAINT_TELEA` and `INPAINT_NS`, and in the
  main benchmark MUST run both algorithms over the same input set.
- **FR-020**: The inpainting parameters MUST be configurable: inpaint radius, algorithm,
  mask dilation, and output format.
- **FR-021**: In the main benchmark, the inpaint radius and the mask-processing configuration
  MUST be the same for every segmentation method. Per-method tuning of the main benchmark is
  prohibited; any per-method or per-configuration variation MUST be run as a labelled
  ablation or supplementary experiment.
- **FR-022**: The system MUST record the processing time per sample per inpainting algorithm
  in that sample's metadata, and MUST aggregate the recorded timings into a performance
  summary grouped by segmentation method × inpainting algorithm (Clarifications,
  2026-09-10; aligns with the course outline's mean-processing-time-per-image comparison
  criterion), reporting for each group: mean processing time per image, sample count,
  number of failures, and runtime/device metadata.
- **FR-023**: Given identical inputs and identical configuration, processing MUST be
  deterministic and reproducible.

#### Outputs and run management

- **FR-024**: The system MUST support ablation runs over multiple dilation configurations and
  inpaint radii. Ablation outputs MUST live in a clearly separated namespace (separate from
  the main benchmark output tree), MUST be labelled as ablation, MUST record their own
  configuration, and MUST be excluded from the main benchmark comparison.
- **FR-025**: For every processed sample the system MUST produce: the original page image; the
  raw prediction mask (copied); the mask after preprocessing/dilation; the `INPAINT_TELEA`
  result; the `INPAINT_NS` result; and a metadata JSON record containing at minimum:
  image_id; segmentation method; model/repository; experiment ID; input image path; prediction
  mask path; mask size; alignment (with pre/post sizes); dilation configuration; inpaint
  algorithm; inpaint radius; processing time; status; and error message when failed.
- **FR-026**: Output MUST be separated by run ID, segmentation method and inpainting
  algorithm, following the structure:

  ```
  outputs/inpainting/
    <run_id>/
      classical_baseline/
        telea/
        ns/
      manga_text_segmentation/
        telea/
        ns/
      comic_text_detector/
        telea/
        ns/
      unetpp_efficientnetv2/
        telea/
        ns/
  ```

  with masks, metadata and the error report stored per run so every output is attributable.
- **FR-027**: The system MUST identify each run by a run ID and MUST NOT overwrite the outputs
  of a previous run unless overwrite is explicitly requested; runs with different run IDs MUST
  coexist on disk.
- **FR-028**: The output image format MUST be configurable; the default MUST be a lossless
  format.
- **FR-029**: A downstream consumer (Spec 4) that knows only the run ID, the segmentation
  method and the image_id MUST be able to locate and correctly interpret a page's inpainted
  image and its metadata, without depending on run-specific knowledge beyond that.
- **FR-030**: The system MUST NOT use inpainting outputs to re-evaluate segmentation metrics,
  and MUST NOT use them as segmentation predictions anywhere.

#### Evaluation and reporting

- **FR-031**: The system MUST NOT use PSNR or SSIM as primary metrics, MUST NOT fabricate
  synthetic clean-background ground truth, and MUST NOT rank segmentation methods by a single
  quantitative score.
- **FR-032**: The system MUST support qualitative assessment against these seven criteria, using
  one documented ordinal scale applied consistently: (1) completeness of text removal;
  (2) amount of missed text; (3) amount of background over-erased; (4) naturalness of the
  restored region; (5) artifacts or noise; (6) halo or border effects around removed regions;
  (7) damage to linework, texture and manga panel borders. The assessment is performed by a
  **human reviewer** (Clarifications, 2026-09-10): the system generates a fully labelled
  scaffold — one entry per assessed sample with an empty rating field per criterion and an
  empty observation field — and MUST NOT auto-score any criterion; the reviewer completes the
  ratings and the written observations.
- **FR-033**: The system MUST produce a comparison board per selected sample showing: the
  original page; the raw prediction mask; the mask after dilation; the TELEA result; the NS
  result — each panel labeled.
- **FR-034**: The system MUST support selecting representative samples by: highest IoU/F1 from
  the Spec 2 per-page metrics; lowest IoU/F1 from Spec 2; prediction masks with the most false
  positives; prediction masks with the most false negatives; and manually flagged artifact
  cases. Each rule selects a fixed count N of samples **per segmentation method** — every
  method receives the same number of qualitative samples (Clarifications, 2026-09-10). N is
  finalized at planning; the initial default is N=5.
- **FR-035**: The system MUST produce the qualitative report as a pre-labelled scaffold in
  which every entry is labeled with: segmentation method; inpainting algorithm;
  mask-processing configuration; image ID — and carries an empty rating field per FR-032
  criterion plus an empty written-observation field, for a human reviewer to complete.

#### Command-line interface

- **FR-036**: The system MUST expose command-line operations equivalent to: processing a
  single prediction mask; processing one segmentation method; processing all prediction
  masks; running TELEA only; running NS only; generating visualizations; generating the
  qualitative report; and running a dilation/radius ablation.

#### Testing

- **FR-037**: The system MUST include tests covering at minimum: image↔prediction-mask
  mapping; mask normalization; mask alignment; dilation; output size after inpainting; both
  TELEA and NS on a small fixture; metadata completeness; one sample failing without aborting
  the batch; and non-overwrite of a previous experiment's outputs. The default test suite
  MUST run on small fixtures without requiring the full dataset.

#### Operability

- **FR-038**: All data paths, output paths and processing parameters MUST live in
  configuration outside source code and MUST NOT be hard-coded in logic.
- **FR-039**: The system MUST run on CPU; no GPU is required for any part of this feature.

### Out of Scope

- Re-running, training or fine-tuning any segmentation model.
- OCR, machine translation, text rendering, and any end-to-end translation demo.
- Creating ground-truth clean images or synthetic background ground truth.
- Modifying Spec 2's prediction masks in place.
- Re-using inpainting outputs to evaluate segmentation metrics.
- Creating a new benchmark or benchmark subset.
- Reading `data/no-need-to-read/` (unless explicitly requested).
- Building any part of Spec 1 or Spec 2; their implementations are preconditions of this
  feature, not parts of it.

### Key Entities *(include if feature involves data)*

- **InpaintRun**: One identified execution of the feature. Attributes: run ID, the set of
  segmentation methods processed, the shared mask-processing configuration, the shared
  inpainting configuration (algorithms, radius, output format), start time, and per-sample
  statuses.
- **SegmentationMethodInput**: One method's stored prediction-mask collection handed over
  from Spec 2. Attributes: method identifier (classical baseline, manga-text-segmentation,
  comic-text-detector, UNet++/EfficientNetV2), model/repository identity, experiment/run ID,
  and the per-page masks with their metadata.
- **MaskRecord**: One prediction mask for one page. Attributes: image_id (`<manga>/<NNN>`),
  mask path, raw size, aligned size, binary validity, emptiness flag, validation status.
- **MaskProcessingConfig**: The shared preprocessing configuration. Attributes: alignment
  rule (pass-through or lossless top-left crop), dilation enabled flag, kernel shape, kernel
  size, iteration count. Identical across all methods in the main benchmark.
- **InpaintingConfig**: Attributes: algorithm (TELEA, NS), inpaint radius, output format.
  Identical across all methods in the main benchmark.
- **ProcessedSample**: One page through the pipeline. Attributes: image_id, segmentation
  method, raw prediction mask, processed (dilated) mask, TELEA result, NS result, per-algorithm
  processing time, status, error message.
- **SampleMetadata**: The per-sample JSON record defined by FR-025.
- **ErrorReport**: All failures of one run, grouped by the nine categories of FR-012, with
  sample identifiers and reasons.
- **ComparisonBoard**: One selected sample's five-panel visualization (FR-033) with labels.
- **SampleSelectionRule**: One selection strategy from FR-034, backed by Spec 2 per-page
  metrics or a manual flag list.
- **QualitativeReport**: The pre-labelled scaffold defined by FR-035, with one entry per
  assessed sample; its ratings and observations are completed by a human reviewer
  (FR-032), never auto-scored by the system.
- **AblationRun**: One supplementary sweep (FR-024) with its own configurations and separated
  outputs.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of accepted prediction masks map to exactly one manifest page by the
  composite `image_id` rule; every unmappable or ambiguous mask is recorded in the error
  report with its category, and zero silent re-mappings occur.
- **SC-002**: For 100% of successfully processed samples, both a TELEA result and an NS
  result exist at exactly the aligned page size (1654×1170), in the configured output
  format, with the pre/post-alignment and dilation metadata recorded.
- **SC-003**: For 100% of samples in the main benchmark, the recorded mask-processing
  configuration and inpaint radius are identical across all four segmentation methods —
  verified by an automated completeness check over the metadata.
- **SC-004**: 100% of produced outputs carry a metadata record containing every field
  required by FR-025, verified by an automated completeness check over the output tree.
- **SC-005**: A batch in which one sample fails per each of the nine error categories
  completes over all remaining valid samples, and the error report records every induced
  failure with category, sample identifier and reason, with zero unrecorded failures.
- **SC-006**: Two runs with different run IDs both remain on disk afterwards, byte-identical
  to their state at completion; a repeated run ID refuses to overwrite prior outputs unless
  overwrite was explicitly requested.
- **SC-007**: Comparison boards are produced for samples selected under every selection rule
  of FR-034, each board showing all five labeled panels with correct method, algorithm,
  configuration and image-ID labels.
- **SC-008**: The qualitative report scaffold covers the seven criteria with a documented,
  consistently applied scale, and no system-generated rating appears in it — every rating
  and observation is entered by a human reviewer — and no PSNR/SSIM figure, synthetic
  ground truth, or single-score method ranking appears in the primary report.
- **SC-009**: A consumer knowing only (run ID, segmentation method, image_id) locates and
  correctly interprets the page's inpainted image and metadata — verified by a consumption
  test against a completed run.
- **SC-010**: A full run leaves every source tree unmodified — verified by checksum or
  modification-time invariance over `data/raw/…`, `data/groundtruth/…`, the manifest and the
  Spec 2 prediction-mask tree — and no file under `data/no-need-to-read/` is opened,
  verified by an access audit.
- **SC-011**: Ablation outputs live outside the main benchmark tree, carry their own recorded
  configurations, and are absent from the main benchmark comparison — verified by inspection
  of the output tree and the comparison artefacts.
- **SC-012**: Identical inputs with identical configuration produce byte-identical outputs
  across two consecutive runs (determinism), and the default test suite completes on small
  fixtures without the full dataset or any segmentation model.

## Assumptions

- **Dependency on Spec 1 and Spec 2 (verified state)**: As of 2026-09-10, Specs 1 and 2 exist
  as specifications only — no implementation, no manifest file, no prediction masks and no
  output tree exist yet. Their implementations are blocking preconditions of this feature;
  this specification is finalised against their written contracts and its first executable
  task cannot start until they exist. This mirrors the stance Spec 2 took toward Spec 1.
- **Prediction-mask hand-off (inherited)**: Prediction masks are assumed to arrive per the
  Spec 2 output contract (FR-043, SC-010): at full aligned resolution (1654×1170), single
  channel with background = 0 / text = 255, from a deterministic per-page path addressable by
  page identifier plus method name. The Colab-produced masks are expected to be thresholded
  and geometry-corrected by Spec 2's receipt validation before this feature sees them; this
  feature re-verifies rather than trusts.
- **Per-page Spec 2 metrics (assumed available)**: Sample selection by IoU/F1 and FP/FN
  profiles (FR-034) assumes Spec 2's per-page metric records are accessible to this feature
  by the same page identifier and method name. If they are not, only the manual artifact-flag
  selection remains, and that limitation would be reported rather than worked around.
- **Default benchmark configuration (resolved)**: The main benchmark's default inpaint
  radius is **3** (the OpenCV default), and the default mask processing is no resize with a
  small dilation of **3×3 ellipse kernel, 1 iteration** to cover text anti-aliasing borders.
  Both remain configurable regardless of the default; any other configuration in a run MUST
  be namespaced as ablation/supplementary per FR-024.
- **Default output format**: lossless PNG for images, JSON for metadata, unless the
  clarification or planning decides otherwise.
- **Binary convention and geometry (verified, inherited)**: masks use background = 0 /
  text = 255; the ground truth's multiple-of-8 canvas (1656×1176) crops losslessly top-left
  to the aligned space (1654×1170); prediction masks from Spec 2 arrive already in that
  aligned space, so alignment is expected to be pass-through with the crop path retained for
  robustness.
- **image_id rule**: `image_id` is the composite `<manga>/<NNN>` (manga folder name plus
  zero-padded page stem), exactly the manifest's mapping rule; a bare stem or bare manga name
  is not an image_id.
- **Execution**: CPU-only, in-project; no model runtime of any kind is present.
- **Spec language**: this specification is written in English for tooling consistency; the
  user communicates in Vietnamese.
