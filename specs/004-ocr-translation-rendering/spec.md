# Feature Specification: OCR, Translation & Text Rendering

**Feature Branch**: `004-ocr-translation-rendering`

**Created**: 2026-09-10

**Status**: Draft — clarifications resolved (translation provider: cloud LLM API behind the
adapter with environment-variable key and result caching; font: bundled open-licence
default with Vietnamese glyphs + configurable fallback; per-region translation requests;
shared-by-default configurable cache bypass; default translation-failure policy:
keep-OCR-text with region flagged translation-failed)

**Input**: User description: "Xây dựng feature 'OCR, Translation và Text Rendering' cho pipeline dịch ảnh Manga. Spec 1 cung cấp dataset, manifest, image ID và quy ước dữ liệu; Spec 2 cung cấp prediction mask của các phương pháp segmentation; Spec 3 cung cấp ảnh sau xóa văn bản và inpainting. Feature không chạy lại segmentation model. Mục tiêu: nhận ảnh gốc, prediction mask và ảnh sau inpainting; xác định vùng văn bản; OCR tiếng Nhật bằng manga-ocr hoặc công cụ tương đương; dịch sang tiếng Việt hoặc tiếng Anh; chèn bản dịch lên ảnh sau inpainting giữ vị trí và bố cục tương đối; xuất đầy đủ intermediate output. OCR chạy trên crop từ ảnh gốc trước inpainting. Translation qua adapter, API key từ environment variable. Rendering với font/size/color/outline/wrap/alignment cấu hình được, tự động điều chỉnh font size, không cắt text. Output tách theo run ID, image ID, segmentation method, inpainting method, target language. CLI hỗ trợ chạy từng bước riêng lẻ và chạy lại region lỗi. Không huấn luyện model, không đánh giá segmentation bằng OCR/translation, không thay đổi mask gốc. Chỉ tạo specification, chưa viết code."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Receive and Validate Inputs from Specs 1–3 (Priority: P1)

As a researcher, I want the system to assemble, for each page it is asked to process, the
original image, the chosen method's prediction mask with its metadata sidecar, and the
chosen inpainting run's restored image with its metadata — mapped by the agreed image ID
rule and validated for consistency — so that every later stage works on a coherent,
attributable set of inputs and every invalid set is rejected with a named reason.

**Why this priority**: Every downstream stage (regions, OCR, translation, rendering)
depends on a correctly assembled and validated input bundle. This is the intake gate and
the first MVP slice.

**Independent Test**: Assemble input bundles including one fully valid, one with a missing
original image, one with a missing prediction mask, one with a missing sidecar, one with a
missing inpainted image, and one whose inpainted image size disagrees with the page — and
assert exactly the intended ones pass, every failure is recorded under its named category,
and nothing is silently substituted or corrected.

**Acceptance Scenarios**:

1. **Given** a page identifier and a configured segmentation method, **When** intake runs,
   **Then** the original image, that method's prediction mask and metadata sidecar, and the
   configured inpainting run's restored image and metadata are located and joined by
   `image_id` (`<manga>/<NNN>`) without any manual path work.
2. **Given** a bundle whose inpainted image size disagrees with the aligned page size,
   **When** intake runs, **Then** the sample is recorded as failed (size inconsistency) and
   is not processed.
3. **Given** a bundle missing any required artifact, **When** intake runs, **Then** the
   failure is recorded under its category (missing original / missing mask / missing
   metadata / missing inpainted image) and the remaining pages continue.
4. **Given** ground-truth masks exist for a page, **When** the main demo runs, **Then**
   they are never used in place of the prediction mask.

---

### User Story 2 - Extract Stable Text Regions from the Prediction Mask (Priority: P1)

As a researcher, I want the system to turn the binary prediction mask into a list of text
regions — connected components merged into bubbles/lines where they belong, each with a
bounding box, a polygon, a determined orientation, a reading-order position and a stable
`region_id` — so that OCR, translation and rendering all address the same identifiable
regions.

**Why this priority**: Regions are the spine of the whole feature: every downstream record
is keyed by `region_id`. Depends only on US 1.

**Independent Test**: Run extraction on synthetic masks (single block, several separated
blocks, two components close enough to merge, one oversized component, an empty mask) and
assert the produced regions, their merge behavior, their ordering and their IDs match the
configured rules, IDs are identical across repeated runs, and edge cases are recorded.

**Acceptance Scenarios**:

1. **Given** a validated binary mask, **When** extraction runs, **Then** text regions are
   produced via connected components or contours, each recorded with a bounding box and a
   polygon.
2. **Given** two components closer than the configured merge distance, **When** extraction
   runs, **Then** they are merged into one region (same speech bubble / line); given the
   merge distance of zero or disabled merging, they stay separate.
3. **Given** the configured minimum and maximum area, **When** extraction runs, **Then**
   components outside the bounds are excluded and the exclusion is recorded.
4. **Given** any valid mask and configuration, **When** extraction runs twice, **Then** the
   region list and every `region_id` are identical (stable, deterministic IDs).
5. **Given** a crop is taken with configured padding, **When** the crop touches the page
   border, **Then** the crop is clipped to the page bounds and no character inside the
   region is cut by the crop border; the adjustment is recorded.

---

### User Story 3 - OCR Japanese Text per Region (Priority: P1)

As a researcher, I want each text region's crop — taken from the original pre-inpaint
image — recognized as Japanese text by manga-ocr (or an equivalent tool behind the same
adapter), with the text, confidence when available, timing and status stored per region —
so that translation has a reliable, inspectable input and a single region's failure never
disturbs the rest of the page.

**Why this priority**: OCR output is the raw material of translation; it depends only on
US 2 and is independent of translation and rendering.

**Independent Test**: Run the OCR adapter against a fixture crop set with a stubbed
recognizer returning known outputs and one induced failure; assert per-region records carry
crop path, region_id, text, status and timing, the failure is isolated to its region, and
the page completes.

**Acceptance Scenarios**:

1. **Given** extracted regions, **When** OCR runs, **Then** each region's crop is taken
   from the original image (before inpainting), not from the inpainted image, and the crop
   image is stored as an intermediate output.
2. **Given** a region whose OCR fails or returns empty text, **When** OCR runs, **Then**
   the region is recorded with its failure/empty status and every other region on the page
   is still processed.
3. **Given** an OCR result file that a human has edited before translation, **When**
   translation runs, **Then** the edited text is used and the region is flagged as manually
   edited.
4. **Given** the same crops and configuration, **When** OCR runs twice, **Then** the
   recorded texts are identical (deterministic for the same recognizer and weights).

---

### User Story 4 - Translate Region Text to the Configured Target Language (Priority: P2)

As a researcher, I want each region's Japanese text translated into the configured target
language (Vietnamese or English) through a pluggable translation provider adapter — with
retry, timeout and rate-limit handling, secrets taken only from the environment, and the
region_id → Japanese → translated mapping persisted — so that rendering can proceed and
no secret material ever reaches the repository or the outputs.

**Why this priority**: The feature's language bridge; depends on US 3 and is independent
of rendering.

**Independent Test**: Run translation against a mock provider (successes, a timeout, a
rate-limit response, a hard failure) and assert the mapping records are complete, retries
and back-off follow the configured policy, the failure policy (keep OCR text or skip the
region) is honored, and no secret material appears in any record or log.

**Acceptance Scenarios**:

1. **Given** OCR results and a configured target language, **When** translation runs,
   **Then** every region has a persisted mapping of region_id, Japanese text and translated
   text, including regions whose translation failed (with status).
2. **Given** a provider response that times out or is rate-limited, **When** translation
   runs, **Then** the configured retry and back-off policy is applied before the region is
   declared failed.
3. **Given** a translation failure with the "keep OCR text" policy, **When** rendering
   runs, **Then** the region's Japanese OCR text is carried forward and the region is
   flagged translation-failed; with the "skip" policy the region is excluded from rendering;
   either choice is visible in the metadata.
4. **Given** any completed run, **When** outputs and logs are audited, **Then** the
   provider/model/version is recorded but no API key or secret appears anywhere.

---

### User Story 5 - Render Translations onto the Inpainted Page (Priority: P2)

As a researcher, I want each region's translated text rendered onto the inpainted page
inside the region's stored display area — with configurable typography, automatic font
sizing, word wrap and a documented overflow strategy — so that the final page reads
naturally, keeps the original layout, and never clips text in normal cases.

**Why this priority**: The feature's visible deliverable; depends on US 2 and US 4 (and
US 1's inpainted image).

**Independent Test**: Render fixture pages with short text (fits), long text (needs
shrink/wrap), text exceeding even the minimum font (expects expansion-then-warning), a
missing font, and a region at the page border — and assert placement, fitting behavior,
overflow strategy, warning records and error isolation all follow the specification.

**Acceptance Scenarios**:

1. **Given** translated regions and the inpainted page, **When** rendering runs, **Then**
   each text is placed within its region's stored bounding box/polygon with the configured
   font family, size, color, outline/stroke, background, word wrap, line spacing and
   alignment.
2. **Given** a translation longer than its display area, **When** rendering runs, **Then**
   the overflow strategy applies in order — reduce font size down to the configured
   minimum, increase line count, modestly expand the display area — and a warning is
   recorded if the text still does not fit; text is never silently clipped.
3. **Given** a missing font file, **When** rendering runs, **Then** the region is recorded
   as failed (font missing) and the rest of the page continues.
4. **Given** a region whose original text was vertical Japanese, **When** rendering runs,
   **Then** the region's detected orientation is recorded and the rendered text direction
   follows the configuration.

---

### User Story 6 - Batch Runs with Region- and Image-Level Error Isolation (Priority: P2)

As a researcher, I want to run the pipeline over a single page or the whole manifest, with
a failure in one region never stopping its page and a failure in one page never stopping
the batch — every failure recorded in a per-run error report — and with each run identified
by a run ID that never overwrites a previous run.

**Why this priority**: Turns per-page capability into manifest-scale operation; depends on
US 1–5.

**Independent Test**: Run a batch with one induced failure of each of the eleven error
categories at region level and page level; assert the batch completes, the error report
enumerates every induced failure by category with identifiers, a second run with a
different run ID coexists with the first, and reusing an existing run ID is refused unless
overwrite was explicitly requested.

**Acceptance Scenarios**:

1. **Given** a page where one region fails at any stage, **When** the page runs, **Then**
   all other regions of that page complete and the failure is recorded.
2. **Given** a batch where one page fails, **When** the batch runs, **Then** all other
   pages complete and the failure is recorded in the error report.
3. **Given** the eleven defined error categories, **When** each is induced, **Then** the
   error report records each occurrence with category, page/region identifier and reason.
4. **Given** a run ID that already exists on disk, **When** a new run is started, **Then**
   the system refuses to overwrite the prior run's outputs unless explicitly requested.

---

### User Story 7 - Inspectable Intermediate Outputs and Pipeline Metadata (Priority: P3)

As a researcher, I want every stage's output stored in a predictable per-page location —
crops, region list, OCR results, translations, rendered image and a complete pipeline
metadata JSON — so that each step can be inspected, audited and corrected (including
manual OCR edits) without re-running the stages before it.

**Why this priority**: The user's explicit inspection and correction workflow; depends on
US 1–5.

**Independent Test**: Process a page end-to-end and assert the full artifact set exists at
the specified locations, the metadata JSON contains every required field, and each stage
can be re-executed alone against the stored outputs of the previous stage.

**Acceptance Scenarios**:

1. **Given** a successfully processed page, **When** its output directory is inspected,
   **Then** it contains the region crops, `ocr.json`, `translations.json`, the rendered
   image and `metadata.json`, plus copies of the prediction mask used and references to the
   original and inpainted images.
2. **Given** the metadata JSON, **When** it is validated, **Then** it contains image_id,
   segmentation method, inpainting method, target language, and per-region records with
   region_id, bbox, OCR text, translation and per-stage statuses.
3. **Given** stored intermediate outputs, **When** any single stage is re-run alone
   against them, **Then** it produces the same results as within a full pipeline run.

---

### User Story 8 - Selective Re-runs (Priority: P4)

As a researcher, I want to re-run only a failed region, only one stage, or only one page —
against already-stored upstream outputs — so that fixing a handful of failures after a
large batch does not require re-processing everything.

**Why this priority**: An efficiency convenience on top of US 6–7; not needed for the
first end-to-end capability.

**Independent Test**: Induce a region failure and a page failure, re-run exactly those
scopes against the stored intermediates, and assert only the targeted scope is re-processed,
results are consistent with a full run, and no unrelated output is modified.

**Acceptance Scenarios**:

1. **Given** a failed region with all upstream artifacts present, **When** a region re-run
   is requested, **Then** only that region is re-processed and its records are updated.
2. **Given** a completed stage's outputs, **When** a later stage is re-run for one page,
   **Then** the earlier stages' stored outputs are consumed as-is and not recomputed.
3. **Given** any re-run, **When** it completes, **Then** outputs of unrelated regions,
   pages and runs are unchanged.

---

### Edge Cases

- **Missing original image, prediction mask, metadata sidecar or inpainted image**: each
  recorded under its category; the page is skipped; the batch continues.
- **Inpainted image size inconsistent with the aligned page size**: recorded as size
  inconsistency; never silently resized.
- **Empty mask (zero text pixels)**: page recorded with no regions detected; not a run
  failure.
- **No text region detected after filtering** (all components outside min/max area):
  recorded; page completes with zero regions.
- **Component larger than the maximum area**: excluded and recorded, never auto-split.
- **Region touching the page border**: crop clipped to page bounds; adjustment recorded;
  characters inside the region are not cut.
- **OCR returns empty text or fails**: region recorded with status; page continues.
- **Translation returns empty text, times out, is rate-limited, or fails**: retry policy
  applies; then the configured failure policy (keep OCR text / skip region, default
  keep-OCR-text) is applied and recorded.
- **Translation much longer than the original text** (overflow): shrink → more lines →
  modest expansion → recorded warning; never silent clipping.
- **Font file missing or unreadable**: region recorded as failed (font missing); page
  continues.
- **Furigana or small annotation glyphs**: merged or split per the configured merge
  distance; the outcome is inspectable in the region list.
- **Output write failure** (unwritable path, disk full): recorded; the error report states
  which outputs are missing; the batch continues where possible.
- **Run-ID collision**: existing runs are never overwritten unless explicitly requested.
- **Manual OCR edit conflicting with re-extraction**: the edited file is honored only
  until regions are re-extracted with a different configuration; a changed region set
  invalidates edited records and that invalidation is reported, not silent.
- **Provider nondeterminism**: a translation provider may return different text for
  identical requests; this is a recorded property of the provider, and all pre-translation
  stages remain deterministic.

## Requirements *(mandatory)*

### Functional Requirements

#### Reuse of the Spec 1–3 foundation

- **FR-001**: The system MUST consume the existing Spec 1 benchmark manifest as the single
  source of truth for the page list and the image–mask mapping, and MUST NOT construct a
  new manifest, page list or benchmark subset.
- **FR-002**: The system MUST reuse the Spec 1 binary mask convention — single channel,
  unsigned 8-bit, background = 0, text = 255 — for every mask it reads or processes, and
  MUST reuse the Spec 1 alignment convention: all processing in the aligned page space
  (1654×1170 for this benchmark), with no image resizing anywhere in the pipeline.
- **FR-003**: The system MUST consume Spec 2 prediction masks per the resolved hand-off:
  each mask as a PNG accompanied by its per-page JSON metadata sidecar, locatable by page
  identifier plus segmentation method name.
- **FR-004**: The system MUST consume Spec 3 inpainting outputs per the Spec 3 downstream
  contract: knowing the inpainting run ID, the segmentation method and the `image_id` MUST
  suffice to locate a page's inpainted image and its metadata; the inpainting metadata
  (algorithm, radius, dilation configuration) MUST be consumed as provenance for every
  output of this feature.
- **FR-005**: The system MUST NOT execute, re-run, train or fine-tune any segmentation
  model and MUST NOT contain any segmentation-model runtime.
- **FR-006**: The system MUST treat `data/raw/…`, `data/groundtruth/…`, the Spec 1
  manifest, Spec 2's prediction-mask outputs and Spec 3's inpainting outputs as read-only,
  and MUST NOT modify, move, rename or delete any of them; where an upstream artifact is
  needed inside this feature's output tree it MUST be copied, never moved.
- **FR-007**: The system MUST NOT read any data from `data/no-need-to-read/` unless
  explicitly requested.
- **FR-008**: Ground-truth masks MUST NOT be used in place of prediction masks in the main
  demo path; if a GT-assisted diagnostic mode exists it MUST be a separately labelled
  configuration, never the default.

#### Segmentation and inpainting method selection

- **FR-009**: The segmentation method MUST be selectable by configuration from the Spec 2
  method set (classical_baseline, manga_text_segmentation, comic_text_detector,
  unetpp_efficientnetv2).
- **FR-010**: The default segmentation method MUST be the method selected by Spec 2's
  benchmark results when such a selection exists in the Spec 2 outputs; if no benchmark
  selection exists and no method is configured, the system MUST refuse to run with a clear
  error rather than choose a method arbitrarily.
- **FR-011**: Running multiple segmentation methods for comparison MUST be supported; each
  page's metadata MUST record which method produced its regions, and different methods MUST
  NOT be mixed within one run's per-page results.
- **FR-012**: The inpainting source MUST be selectable by configuration (a Spec 3 run ID
  and its algorithm); if it is ambiguous or absent, the system MUST refuse to run with a
  clear error rather than pick arbitrarily.

#### Region extraction

- **FR-013**: Before extraction, the prediction mask MUST be normalized to the binary
  convention and checked for size and validity; a failing mask is recorded under its
  category and never silently corrected.
- **FR-014**: Text regions MUST be extracted from the binary mask using connected
  components or contours.
- **FR-015**: Nearby components belonging to the same speech bubble or text line MUST be
  mergeable, with the merge distance configurable (including a value that disables
  merging).
- **FR-016**: Region extraction MUST expose configurable: minimum area, maximum area, merge
  distance, crop padding, and reading order (default: manga reading order — right-to-left,
  top-to-bottom).
- **FR-017**: A region's crop MUST include enough padding that no character of the region
  is cut at the crop border; when the padded crop exceeds the page bounds it MUST be
  clipped to the page and the clipping recorded.
- **FR-018**: For every region the system MUST store both a bounding box and a polygon of
  the text area.
- **FR-019**: `region_id` MUST be stable and deterministic — derived from the region's
  reading-order position under the active configuration — so the same mask and
  configuration always produce the same region IDs.
- **FR-020**: Each region's orientation (horizontal or vertical text) MUST be determined
  from the mask geometry and recorded.
- **FR-021**: A page yielding no regions (empty mask or everything filtered) MUST be
  recorded with zero regions and status indicating so — not treated as a run failure.

#### OCR

- **FR-022**: Japanese OCR MUST use manga-ocr as the preferred engine, wrapped behind an
  OCR adapter so an equivalent tool can substitute; the engine identity and version MUST be
  recorded for every run.
- **FR-023**: OCR MUST run on crops taken from the original pre-inpaint image; optional
  preprocessing of the crop MUST be configurable and, when enabled, recorded.
- **FR-024**: For every region the system MUST store: the crop image; the region_id; the
  OCR text; the OCR confidence when the engine provides one; the OCR processing time; and
  the error message when OCR failed.
- **FR-025**: Japanese capabilities — kanji/kana text, furigana and vertical text — MUST be
  supported to the extent the active OCR engine provides, and the engine's capability
  boundaries MUST be documented rather than worked around silently.
- **FR-026**: A single region's OCR failure MUST be recorded and isolated to that region;
  the rest of the page MUST continue.
- **FR-027**: The intermediate OCR result file MUST be human-editable before translation
  and rendering; downstream stages MUST honor manual edits and MUST flag manually edited
  regions as such; re-extraction that changes the region set MUST invalidate stale edited
  records and report the invalidation.
- **FR-028**: The OCR engine MUST NOT be trained or fine-tuned by this feature; only stored
  or first-run-downloaded inference weights are used.

#### Translation

- **FR-029**: Translation MUST go through a provider adapter so the concrete provider is
  pluggable; the provider, model and version MUST be recorded for every run. The first
  concrete provider MUST be a cloud LLM API whose API key is supplied exclusively through
  an environment variable (never hard-coded, never logged), and the adapter MUST cache
  translation results so an identical request — same text, target language, provider and
  model — reuses the stored result instead of re-calling the API. Each text region MUST
  be translated with its own independent provider request (per-region request
  granularity, resolved 2026-09-10): regions MUST NOT be batched into a single combined
  request, so that one region's failure or malformed response cannot affect other
  regions on the page and per-region error isolation (FR-031, FR-054) is preserved.
  The cache MUST be shared across runs by default — content-addressed on
  (text, target language, provider, model) and stored in a dedicated cache location
  outside any run directory — with a configuration option to bypass the cache and
  force fresh provider calls; each page's metadata MUST record whether a region's
  translation was served from cache or fetched live (resolved 2026-09-10,
  configurable with shared-cache default).
- **FR-030**: The target language MUST be configurable, supporting at minimum Vietnamese
  and English.
- **FR-031**: The system MUST persist, for every region, the mapping between region_id,
  Japanese OCR text and translated text — including regions whose translation failed.
- **FR-032**: Translation MUST have configurable retry, timeout and rate-limit handling
  with a documented policy applied before a region is declared failed.
- **FR-033**: On translation failure the configured failure policy MUST apply: either keep
  the Japanese OCR text carried forward (region flagged translation-failed) or exclude the
  region from rendering; the choice and the failure MUST be recorded either way. The
  DEFAULT policy (when none is configured) MUST be keep-OCR-text with the region flagged
  translation-failed, so a failed translation stays visible on the rendered page instead
  of silently blanking the region (resolved 2026-09-10).
- **FR-034**: API keys and secrets MUST come exclusively from environment variables or a
  secret configuration outside the repository; no secret MAY be hard-coded in source code,
  and the system MUST verify required secrets are present at startup when the configured
  provider needs them.
- **FR-035**: Outputs, metadata and logs MUST record provider/model/version but MUST NEVER
  contain API keys or any secret material.
- **FR-036**: Translation results MUST NOT be used to evaluate or rank segmentation
  quality, and translation quality MUST NOT be benchmarked in this feature.

#### Text rendering

- **FR-037**: Translated text MUST be rendered onto the inpainted page image; the original
  image, prediction masks and inpainting outputs remain untouched.
- **FR-038**: The display area for each region's text MUST come from that region's stored
  bounding box/polygon.
- **FR-039**: Rendering MUST support configurable: font family; font size; font color;
  outline/stroke; background or box behind the text; word wrap; line spacing; horizontal
  and vertical alignment; and text direction.
- **FR-040**: Font size MUST be automatically adjusted so the translation fits its display
  area, and in normal cases rendered text MUST NOT be clipped by or extend outside the
  display area.
- **FR-041**: When a translation does not fit, the overflow strategy MUST apply in order:
  reduce font size down to a configured minimum; increase the line count; modestly expand
  the display area; and record a warning if the text still does not fit. Silent clipping is
  prohibited.
- **FR-042**: The original page layout MUST be preserved as far as possible; rendering
  MUST NOT paint outside the configured display areas unless display-area expansion is
  configured.
- **FR-043**: Both horizontal and vertical rendering MUST be supported as determined by
  the region orientation and the configured text direction; the effective choice is
  recorded per region.
- **FR-044**: An external rendering module MAY be used but MUST be wrapped behind a
  rendering adapter with the dependency and its version recorded.
- **FR-045**: Rendering MUST be deterministic given identical inputs and configuration.

#### Outputs, runs and metadata

- **FR-046**: For every processed page the system MUST produce: the original image
  (copied or referenced); the inpainted image (copied or referenced); the per-region crops;
  the prediction mask used (copied); the text-region list; the OCR results; the translation
  results; the final rendered image; and the pipeline metadata JSON.
- **FR-047**: The per-page metadata JSON MUST contain at minimum: image_id;
  segmentation_method; inpainting_method; target_language; and a `regions` array where each
  entry carries region_id, bbox, ocr_text, translation, ocr_status, translation_status and
  render_status — extended with run provenance (run ID, OCR engine/version, translation
  provider/model, rendering font configuration) and per-stage timings and error messages.
- **FR-048**: Output MUST be separated by run ID, segmentation method and image ID,
  following the structure:

  ```
  outputs/translation/
    <run_id>/
      <segmentation_method>/
        <manga>/
          <NNN>/
            crops/
            ocr.json
            translations.json
            rendered.png
            metadata.json
  ```

- **FR-049**: One run fixes exactly one combination of segmentation method, inpainting
  source and target language; those two remaining separation axes (inpainting method,
  target language) are captured by the run configuration and repeated in every page's
  metadata — a different combination requires a new run ID. The system MUST identify each
  run by a run ID and MUST NOT overwrite a previous run's outputs unless overwrite is
  explicitly requested; runs with different run IDs MUST coexist on disk.
- **FR-050**: Every stage's outputs MUST be written as inspectable intermediates so each
  stage can be re-run alone against the previous stage's stored outputs (US 8).
- **FR-051**: Each run MUST produce an error report enumerating failures by category —
  covering at minimum: missing original image; missing prediction mask; missing inpainted
  image; empty mask; no text region detected; OCR failure; translation failure; font
  missing; text overflow (warning); API timeout/rate limit; output write failure — each
  with the page and region identifiers and the specific reason.
- **FR-052**: A downstream consumer (Spec 5) that knows only the run ID, the segmentation
  method and the image_id MUST be able to locate and correctly interpret a page's full
  artifact set and metadata, without run-specific knowledge beyond that.

#### Command-line interface

- **FR-053**: The system MUST expose command-line operations equivalent to: processing a
  single image; processing the whole manifest; running region extraction only; running OCR
  only; running translation only; running rendering only; running the full pipeline;
  selecting the segmentation method; selecting the inpainting source; selecting the target
  language; re-running a failed region; and exporting intermediate outputs.

#### Error handling and isolation

- **FR-054**: A failure in one region MUST NOT stop the processing of its page's remaining
  regions; a failure in one page MUST NOT stop the batch; every failure MUST be recorded in
  the run's error report.
- **FR-055**: The system MUST NOT silently correct, substitute or fabricate any failed
  input, region, OCR text, translation or rendering — failures are recorded, never hidden.

#### Reproducibility and secrets

- **FR-056**: Every run MUST record: segmentation method; inpainting method (Spec 3 run ID
  and algorithm); OCR engine and version; translation provider, model and version; target
  language; rendering font and configuration; input image IDs; run ID; and the elapsed time
  of each stage per page.
- **FR-057**: No API key, token or secret MAY appear in source code, configuration files in
  the repository, logs, metadata or any output.
- **FR-058**: Given identical inputs and configuration, all local stages (extraction, OCR,
  rendering) MUST be deterministic; translation determinism depends on the provider and
  MUST be recorded as a provider property rather than guaranteed.
- **FR-059**: A run's configuration MUST be stored with the run so the same configuration
  can be re-applied to the same inputs.

#### Testing

- **FR-060**: The system MUST include tests covering at minimum: region extraction from a
  binary mask; bounding box and padding; region ordering; the OCR adapter against a
  fixture/mock; the translation adapter against a mock provider; translation result
  caching; no API-key leakage; text
  wrapping; font-size fallback; text overflow; rendering position; one region failing while
  others complete; and the metadata mapping between image_id, region_id, OCR and
  translation. The default test suite MUST run on small fixtures without the full dataset,
  without any segmentation model, and without a real translation provider.

#### Operability

- **FR-061**: All data paths, output paths, model/provider settings and processing
  parameters MUST live in configuration outside source code and MUST NOT be hard-coded in
  logic.
- **FR-062**: The system MUST run on CPU; no GPU is required for any part of this feature
  (including local OCR inference).

### Out of Scope

- Training or fine-tuning any segmentation, OCR, translation model or LLM.
- Building a quantitative benchmark for OCR or translation quality.
- Evaluating or ranking segmentation quality using OCR or translation results.
- Summarizing content across multiple pages.
- Processing a whole chapter as a complete batch workflow (single-image and manifest
  scopes only, as specified).
- Automatically selecting a segmentation model based on OCR quality.
- Modifying ground-truth or prediction masks in place; using GT masks in the main demo.
- Re-running segmentation models.
- Building any part of Specs 1–3; their implementations are preconditions of this feature,
  not parts of it.
- Reading `data/no-need-to-read/` (unless explicitly requested).

### Key Entities *(include if feature involves data)*

- **TranslationRun**: One identified execution. Attributes: run ID; the fixed segmentation
  method; the inpainting source (Spec 3 run ID + algorithm); the target language; the full
  extraction/OCR/translation/rendering configuration; start time; per-page statuses.
- **PageBundle**: One page's assembled inputs. Attributes: image_id (`<manga>/<NNN>`),
  original image path, prediction mask path + sidecar path, inpainted image path +
  inpainting metadata, validation status.
- **TextRegion**: One extracted text area. Attributes: region_id, bbox, polygon,
  orientation, reading-order index, merge provenance (components merged), crop path,
  padding/clipping adjustments.
- **RegionExtractionConfig**: Minimum area, maximum area, merge distance, crop padding,
  reading order.
- **OcrResult**: Attributes: region_id, crop path, text, confidence (when available),
  engine/version, processing time, status, error message, manually-edited flag.
- **TranslationResult**: Attributes: region_id, Japanese text, translated text, target
  language, provider/model/version, processing time, status, error message, failure-policy
  outcome.
- **TranslationProviderAdapter**: The pluggable provider interface; attributes: provider
  identity, model, version, retry/timeout/rate-limit policy, translation result cache,
  secret requirements (API key via environment variable).
- **RenderingConfig**: Font family, size, color, outline/stroke, background, word wrap,
  line spacing, alignment, text direction, minimum font size, expansion allowance,
  bundled default font, configurable fallback font.
- **RenderedPage**: One page's final image plus per-region render statuses and warnings.
- **PipelineMetadata**: The per-page JSON record defined by FR-047.
- **ErrorReport**: All failures of one run, grouped by the eleven categories of FR-051.
- **ManualEditRecord**: A human edit to an intermediate OCR file: region_id, original text,
  edited text, edit time; governs downstream behavior per FR-027.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of accepted page bundles map to exactly one manifest page by the
  composite `image_id` rule; every unmappable, ambiguous or incomplete bundle is recorded
  in the error report with its category, and zero silent substitutions occur (including
  zero uses of GT masks in the main demo path).
- **SC-002**: For 100% of pages with a valid non-empty mask, extraction produces a region
  list in which every region carries bbox, polygon, orientation, reading-order index and a
  stable `region_id`; re-running extraction with the same mask and configuration yields
  identical regions and IDs.
- **SC-003**: For 100% of extracted regions, an OCR record exists with crop image, text,
  status and timing; every OCR failure is recorded with its reason and isolated to its
  region.
- **SC-004**: For 100% of OCR'd regions, a translation record exists containing the
  region_id → Japanese → translated mapping with its status; the provider/model/version is
  recorded for every run.
- **SC-005**: An automated audit over a completed run's outputs and logs finds zero
  occurrences of API keys or secret material, while provider/model/version fields are
  present.
- **SC-006**: For 100% of rendered pages, an automated bounds check confirms rendered text
  lies within its display area in normal cases; every overflow case follows the documented
  strategy and carries a record (adjustment or warning), with zero silent clipping.
- **SC-007**: For 100% of successfully processed pages, the full artifact set of FR-046
  exists at the locations of FR-048, and the metadata JSON validates against FR-047's
  required fields.
- **SC-008**: A batch containing one induced failure of each of the eleven error
  categories — at region and page level — completes over all remaining work, and the error
  report records every induced failure with category, identifiers and reason, with zero
  unrecorded failures.
- **SC-009**: Each stage (extraction, OCR, translation, rendering) re-run alone against
  the previous stage's stored outputs produces the same results as within a full pipeline
  run for the same inputs and configuration.
- **SC-010**: A consumer knowing only (run ID, segmentation method, image_id) locates and
  correctly interprets a page's full artifact set and metadata — verified by a consumption
  test against a completed run.
- **SC-011**: A full run leaves every upstream tree unmodified — verified by checksum or
  modification-time invariance over `data/raw/…`, `data/groundtruth/…`, the manifest, the
  Spec 2 prediction-mask tree and the Spec 3 inpainting tree — and no file under
  `data/no-need-to-read/` is opened.
- **SC-012**: Two runs with different run IDs both remain on disk afterwards; a repeated
  run ID refuses to overwrite prior outputs unless explicitly requested; and the default
  test suite completes on small fixtures without the full dataset, any segmentation model,
  or a real translation provider.

## Assumptions

- **Dependency on Specs 1–3 (verified state)**: As of 2026-09-10, Specs 1–3 exist as
  specifications only — no manifest, no prediction masks, no inpainting outputs exist yet.
  Their implementations are blocking preconditions of this feature; this specification is
  finalised against their written contracts. This mirrors the stance Specs 2 and 3 took
  toward their predecessors.
- **Inherited geometry and identity**: aligned page space 1654×1170 (W×H); binary mask
  convention background = 0 / text = 255; `image_id` = `<manga>/<NNN>`; prediction masks
  arrive with per-page JSON metadata sidecars (Spec 3's resolved FR-009); inpainting
  outputs are locatable per Spec 3's FR-029.
- **Translation request granularity**: one provider request per text region (resolved
  2026-09-10), never page-level batching — see FR-029.
- **Translation cache scope**: shared across runs by default (content-addressed cache in
  a location outside the run tree), with a configuration switch to bypass it and force
  fresh provider calls; cache hit vs live fetch is recorded in page metadata — see
  FR-029 (resolved 2026-09-10).
- **Default translation-failure policy**: when no policy is configured, a region whose
  translation failed is rendered with its Japanese OCR text and flagged
  translation-failed, rather than being blanked out — see FR-033 (resolved 2026-09-10).
- **Run-level configuration axes**: the output tree separates run ID, segmentation method
  and image ID (the user's example structure); inpainting method and target language are
  fixed per run and recorded in the run configuration and every page's metadata, since a
  run mixing them would make per-page results incomparable.
- **Default target language**: Vietnamese (`vi`), configurable to English (`en`); the
  source language is Japanese for all pages.
- **Reading order default**: manga reading order (right-to-left, top-to-bottom),
  configurable.
- **Vertical text rendering policy**: the original Japanese orientation is detected and
  recorded per region; Vietnamese/English translations are rendered horizontally within the
  original region geometry by default (the target languages do not typeset vertically);
  vertical rendering remains available via configuration.
- **OCR dependency**: manga-ocr is a local, inference-only dependency (adds a
  deep-learning runtime for OCR inference; weights are downloaded at setup, never trained);
  CPU inference is acceptable and no GPU is required.
- **Translation provider nondeterminism**: an external provider may not return identical
  text for identical requests; this is recorded as a provider property. All
  pre-translation stages are deterministic.
- **Secrets handling**: all provider credentials come from environment variables or a
  secret configuration outside the repository; nothing secret is stored in the repo, logs
  or outputs.
- **Default output format**: lossless PNG for images, JSON for metadata and records.
- **Default font source (resolved)**: the project bundles a default open-licence font with
  Vietnamese glyph support; a fallback font is configurable; the default rendering path
  MUST NOT depend on system-installed fonts.
- **Spec language**: this specification is written in English for tooling consistency; the
  user communicates in Vietnamese.

## Clarifications

### Session 2026-09-10

- Q: Should the translation adapter send each region's text to the provider as its own request, or batch all of a page's regions into one request? → A: **Option A — per-region: one independent provider request per text region; regions MUST NOT be batched** (protects per-region error isolation and the per-text cache key).
- Q: Should the translation cache be shared across runs or scoped per run? → A: **Option C — configurable: shared content-addressed cache (outside the run tree) is the default, with a configuration option to bypass it and force fresh provider calls; cache hit vs live fetch is recorded in page metadata.**
- Q: When a region's translation fails after the configured retries, what is the default failure policy — keep the Japanese OCR text in the rendered page or exclude the region? → A: **Option A — default keep-OCR-text with the region flagged translation-failed (policy stays configurable; skip-region remains selectable).**
