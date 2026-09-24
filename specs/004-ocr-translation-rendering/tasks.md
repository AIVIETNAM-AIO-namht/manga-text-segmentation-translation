---

description: "Task list for OCR, Translation & Text Rendering"
---

# Tasks: OCR, Translation & Text Rendering

**Input**: Design documents from `/specs/004-ocr-translation-rendering/`

**Prerequisites**: [plan.md](plan.md) (required), [spec.md](spec.md) (required for user stories),
[research.md](research.md), [data-model.md](data-model.md), [contracts/](contracts/), [quickstart.md](quickstart.md)

**Tests**: **Required for this feature.** FR-060 enumerates thirteen test areas verbatim and the
project constitution's Principle II is *Spec-Scoped Test-First (NON-NEGOTIABLE)*. Every test task
below is written before the implementation it covers and MUST fail first. Run the suite with
`pytest -o addopts='' -q` — `pyproject.toml` sets `--cov=` and `pytest-cov` is not installed in the
active environment. FR-060's closing sentence is a hard constraint on every test here: the default
suite runs **without the full dataset, without any segmentation model, and without a real
translation provider**.

**Organization**: Tasks are grouped by user story so each story is independently implementable and
testable. Story numbering follows **spec.md** — US1 intake, US2 regions, US3 OCR, US4 translation,
US5 rendering, US6 batch, US7 metadata, US8 selective re-runs. `quickstart.md`'s scenario headers use
the same mapping; spec.md wins if they ever diverge.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

Single project with `src/` layout (plan.md "Source Code"): source in `src/manga_text_seg/`, the new
subpackage in `src/manga_text_seg/translation/`, tests in `tests/{contract,unit,integration}/`,
configuration in `configs/`, fonts in `assets/fonts/`, gitignored product in `outputs/translation/`
and `cache/translation/`.

**Two collisions to respect throughout**:

- `src/manga_text_seg/run.py` (singular) is **Spec 2's existing run-record module**, covered by
  `tests/unit/test_run_record.py`. It must not be touched. This feature's module is
  `src/manga_text_seg/translation/runs.py`. Every task below spells the path in full.
- Spec 3 plans `intake.py`, `maskproc.py`, `inpaint.py` and `runs.py` at the package top level. That
  is why this feature's seven modules live inside `translation/` — no top-level name is claimed here.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: The configuration file, the bundled font, the package skeleton and the two dependency
declarations.

- [ ] T001 Create `configs/translation.json` (FR-061): the Spec 1 manifest path, the selected segmentation method, the inpainting source (`run_id` + `algorithm`), `target_language`, the region-extraction block (min/max area, merge distance, crop padding, reading order), the OCR block (engine, optional preprocessing), the translation provider block (name, model, version, base URL, `secret_env_var`, timeout, retries, backoff, `failure_policy`, cache settings incl. `bypass`), the rendering block (font path, fallback font path, size, min size, colour, outline, background, wrap, line spacing, alignment, `text_direction`, `expansion_allowance`) and the output roots. No path or parameter may live in code (FR-061).
- [ ] T002 [P] Add `assets/fonts/DejaVuSans.ttf` — copied from the file matplotlib already bundles — plus `assets/fonts/DejaVuSans.LICENSE.txt` carrying its licence text (FR-039, research R6). The default rendering path resolves the font by file path and never by system-font lookup, so the result does not depend on the machine.
- [ ] T003 [P] Create `src/manga_text_seg/translation/__init__.py` — the new subpackage's docstring naming the seven modules and the FR ranges each covers. No imports yet; the modules land in their own phases.
- [ ] T004 [P] Confirm `outputs/translation/` and `cache/translation/` are covered by `.gitignore` (`.gitignore` already lists `outputs/` and `cache/`) and that `configs/translation.json` and `assets/fonts/` are tracked. Record the result; no `.gitignore` change is expected.
- [ ] T005 [P] Extend `pyproject.toml`: add `Pillow>=10.0.0` to `[project.dependencies]` and a new `[project.optional-dependencies]` group `ocr = ["manga-ocr"]` (research R3, plan.md Complexity Tracking). `manga-ocr` stays optional so the default install and the default test suite need no OCR engine (FR-060).

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Configuration loading, the shared test fixture, the run scaffold and the error report —
everything every user story writes through.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T006 [P] Write `tests/unit/test_config_translation.py` (FR-061): `load_translation_config()` returns the extraction, OCR, provider, cache and rendering blocks with their resolved paths; a missing required key raises `ConfigError`; no path is hard-coded. MUST fail before T008.
- [ ] T007 [P] Extend `tests/conftest.py` with a `translation_bundle` fixture (plan.md "Testing"): given a method name and an `image_id`, it writes — under a tmp root — the original page at `ALIGNED_SHAPE` (`(1170, 1654)`, `tests/conftest.py:35`), a prediction mask plus its JSON sidecar in the Spec 2 hand-off layout, an inpainted page plus its Spec 3 sidecar, and the manifest entry joining them. Parameterise it so a caller can perturb any single artifact's presence, size, pixel values and emptiness. This is what lets all thirteen FR-060 areas run without the dataset (FR-060).
- [ ] T008 Implement `load_translation_config()` in `src/manga_text_seg/config.py` (FR-061), following the existing `load_config()` / `load_dl_config()` conventions in that file (a dataclass, `_require_mapping`-style validation, `_resolve` against the repo root). Expose `TranslationConfig` and its sub-configs (data-model.md `RegionExtractionConfig`, `RenderingConfig`).
- [ ] T009 Implement run-ID resolution and output-tree addressing in `src/manga_text_seg/translation/runs.py` (FR-047, FR-052): the paths from contracts/rendered-output.md "Addressing" — `outputs/translation/<run_id>/<segmentation_method>/<manga>/<NNN>/{crops/,mask.png,ocr.json,translations.json,rendered.png,metadata.json}` plus `run.json` and `errors.json` at the run root. `<segmentation_method>` is one of the four FR-009 identities; reject anything else. A consumer knowing only (run ID, segmentation method, `image_id`) resolves the page with no index file (SC-010).
- [ ] T010 Implement run scaffolding and the non-overwrite rule in `src/manga_text_seg/translation/runs.py` (FR-049): create the run tree, refuse to reuse an existing run ID unless overwrite was explicitly requested, and never touch another run's directory. A refusal leaves the previous run's tree untouched.
- [ ] T011 Implement the `ErrorReport` accumulator and `errors.json` writer in `src/manga_text_seg/translation/runs.py` (FR-051, FR-055): all **eleven** categories present in `counts` and in `categories` **even at zero occurrences** — `missing_original_image`, `missing_prediction_mask`, `missing_inpainted_image`, `empty_mask`, `no_text_region_detected`, `ocr_failure`, `translation_failure`, `font_missing`, `text_overflow`, `api_timeout_or_rate_limit`, `output_write_failure` — each entry carrying `image_id`, `region_id` (`null` at page level), `stage` (one of `intake`, `regions`, `ocr`, `translate`, `render`, `write`) and the specific reason (contracts/error-report.md).
- [ ] T012 Implement the `run.json` (`TranslationRun`) writer in `src/manga_text_seg/translation/runs.py` (FR-056, FR-059): run ID; the fixed segmentation method; the inpainting source (run ID + algorithm); the target language; the OCR engine and version; the translation provider, model and version; the rendering font and configuration; the input image IDs; the full resolved configuration stored with the run so it can be re-applied (FR-059); the provider's declared determinism as a **provider property**, not a guarantee (FR-058); and `page_statuses`. `page_list_identity` / `input_image_identity` are quoted **verbatim** from `benchmark/`, never recomputed (FR-054a, ONBOARDING §8).

**Checkpoint**: Configuration, the fixture, the run scaffold and the error report exist — user story implementation can now begin.

---

## Phase 3: User Story 1 - Receive and Validate Inputs from Specs 1–3 (Priority: P1) 🎯 MVP

**Goal**: Assemble, per page, the original image, the selected method's prediction mask and sidecar, and the configured inpainting run's restored image and metadata — joined by `image_id` and validated for consistency, with every invalid set rejected under a named reason.

**Independent Test**: Assemble input bundles including one fully valid, one with a missing original image, one with a missing prediction mask, one with a missing sidecar, one with a missing inpainted image, and one whose inpainted image size disagrees with the page — and assert exactly the intended ones pass, every failure is recorded under its named category, and nothing is silently substituted or corrected.

### Tests for User Story 1

- [ ] T013 [P] [US1] Write `tests/contract/translation/test_intake_addressing.py` (FR-001, FR-004, FR-052): given (run ID, segmentation method, `image_id`) the bundle resolves with no index file and no manifest rebuild; the directory path is the `image_id` used verbatim, not a transformation of it. MUST fail before T015.
- [ ] T014 [P] [US1] Write `tests/integration/test_intake_rejections.py` (FR-054, SC-001): the six bundles of the Independent Test above, asserting the intended ones pass, each rejection lands under its own category with a specific reason, the remaining pages continue, and nothing is substituted or corrected. MUST fail before T015–T019.

### Implementation for User Story 1

- [ ] T015 [US1] Implement bundle assembly in `src/manga_text_seg/translation/intake.py` (FR-001–FR-004): locate the original page, the selected method's prediction mask **and its JSON sidecar**, and the configured inpainting run's `<NNN>.png` and `<NNN>.json`, joined by `image_id` (`<manga>/<NNN>`) alone — no manual path work, no new manifest, no benchmark subset.
- [ ] T016 [US1] Implement bundle validation in `src/manga_text_seg/translation/intake.py` (FR-002, FR-013): normalize the mask to the Spec 1 binary convention (single channel, uint8, background 0, text 255) and check the inpainted image's size equals the aligned page size, 1654×1170. A size disagreement is recorded as an intake failure and the image is **never silently resized**. Reuse `normalize.py` and `imaging.py` rather than reimplementing them (research R1).
- [ ] T017 [US1] Wire the intake rejections into the T011 accumulator in `src/manga_text_seg/translation/intake.py` (FR-054): `missing_original_image`, `missing_prediction_mask`, `missing_inpainted_image`, `empty_mask` and the size-inconsistency reason, each with the page identifier and the specific reason, and each leaving the batch running.
- [ ] T018 [US1] Implement the read-only upstream guarantee in `src/manga_text_seg/translation/intake.py` (FR-006): every upstream path is opened for reading only; `mask.png` inside the run tree is a **copy** of the prediction mask, byte-identical to the admitted mask it came from. Nothing under `data/raw/`, `data/groundtruth/`, the Spec 1 manifest, Spec 2's masks or Spec 3's outputs is modified, moved, renamed or deleted.
- [ ] T019 [US1] Implement method and source resolution with refusal in `src/manga_text_seg/translation/intake.py` (FR-010, FR-012, research R2): resolve the default segmentation method by reading Spec 2's **recorded** selection (never importing `availability.py`); refuse with a clear error when no selection exists and none is configured, and when the inpainting source is ambiguous or absent — never choose arbitrarily. **Ground truth is never consulted**: the prediction mask is the only mask this feature opens, and no GT-assisted mode is the default (FR-008, US1 scenario 4).

**Checkpoint**: The intake gate stands alone — `tests/integration/test_intake_rejections.py` passes and every rejection is named.

---

## Phase 4: User Story 2 - Extract Stable Text Regions from the Prediction Mask (Priority: P1)

**Goal**: Turn the binary prediction mask into a list of text regions — components merged into bubbles/lines where they belong, each with a bounding box, a polygon, a determined orientation, a reading-order position and a stable `region_id`.

**Independent Test**: Run extraction on synthetic masks (single block, several separated blocks, two components close enough to merge, one oversized component, an empty mask) and assert the produced regions, their merge behaviour, their ordering and their IDs match the configured rules, IDs are identical across repeated runs, and edge cases are recorded.

### Tests for User Story 2

- [ ] T020 [P] [US2] Write `tests/unit/translation/test_regions.py` (FR-014, FR-017, FR-018, FR-060): a single block, several separated blocks, a merge pair and an oversized component against the `aligned_mask` fixture; each region carries both a `bbox` and a `polygon`; the padded crop touches the page border, is clipped to page bounds, records `clipped: true`, and cuts no character inside the region. MUST fail before T022–T025.
- [ ] T021 [P] [US2] Write `tests/unit/translation/test_region_ids.py` (FR-015, FR-019, FR-020, FR-060): `region_id` is the 1-based reading-order position under the active configuration and is **identical across two runs over the same mask**; `merge_distance: 0` leaves components separate and a non-zero distance merges them with `merged_from` recording how many went in; orientation is recorded per region. MUST fail before T023 and T026–T027.

### Implementation for User Story 2

- [ ] T022 [US2] Implement component extraction and area filtering in `src/manga_text_seg/translation/regions.py` (FR-014, FR-016): `connectedComponentsWithStats` over the binary mask with `min_area` / `max_area` from `configs/translation.json`; components outside the bounds are excluded and the exclusion is recorded — an oversized component is **never auto-split** (FR-016, edge case).
- [ ] T023 [US2] Implement the configurable merge in `src/manga_text_seg/translation/regions.py` (FR-015): a union-find pass merging components closer than `merge_distance` into one region (same speech bubble / text line), recording how many components went in; a value of `0` disables merging. Merge distance is config-driven, never a literal.
- [ ] T024 [US2] Implement bbox and polygon output in `src/manga_text_seg/translation/regions.py` (FR-018): `findContours` + `approxPolyDP` over each merged region, storing both the axis-aligned bounding box and the polygon of the text area in aligned page space.
- [ ] T025 [US2] Implement crop padding and border clipping in `src/manga_text_seg/translation/regions.py` (FR-017): pad by the configured amount so no character of the region is cut at the crop border; when the padded crop exceeds the page bounds, clip it to the page and record `clipped: true`.
- [ ] T026 [US2] Implement reading order and stable `region_id` in `src/manga_text_seg/translation/regions.py` (FR-019, research R9): the total order — horizontal centre descending, then vertical centre ascending, then bbox area descending, then top-left ascending — with `region_id` derived from the reading-order position so the same mask and configuration always produce the same IDs. Explicit sort keys only; no reliance on dict or set iteration order (research R8).
- [ ] T027 [US2] Implement orientation detection in `src/manga_text_seg/translation/regions.py` (FR-020): horizontal or vertical, determined from the mask geometry and recorded per region.
- [ ] T028 [US2] Implement the zero-region outcome in `src/manga_text_seg/translation/regions.py` (FR-021): a page yielding no regions — an empty mask, or every component filtered out — is recorded with zero regions and a status indicating so, and is **not** a run failure.
- [ ] T029 [US2] Wire the two region-level page outcomes into the T011 accumulator in `src/manga_text_seg/translation/regions.py` (FR-051): `empty_mask` when the mask has no text pixels, `no_text_region_detected` when it has text pixels but no region survives the FR-016 filters. Both are outcomes, not errors, and neither makes the run fail.

**Checkpoint**: Region extraction is deterministic and independently testable — `region_id` is stable across runs and the edge cases are recorded.

---

## Phase 5: User Story 3 - OCR Japanese Text per Region (Priority: P1)

**Goal**: Recognize each text region's crop — taken from the original pre-inpaint image — with manga-ocr behind an adapter, storing text, confidence when available, timing and status per region, so a single region's failure never disturbs the rest of the page.

**Independent Test**: Run the OCR adapter against a fixture crop set with a stubbed recognizer returning known outputs and one induced failure; assert per-region records carry crop path, `region_id`, text, status and timing, the failure is isolated to its region, and the page completes.

### Tests for User Story 3

- [ ] T030 [P] [US3] Write `tests/unit/translation/test_ocr_adapter.py` (FR-022, FR-024, FR-026, FR-060): the OCR adapter against a fixture recognizer returning known outputs plus one induced failure; assert per-region records carry `region_id`, `crop_path`, `text`, `confidence`, `processing_time_seconds` and `status`, the failure is isolated to its region, and the page completes. `confidence` is `null` when the engine reports none — **never a fabricated `0.0`**. MUST fail before T031–T038. Runs with no OCR engine installed (FR-060).

### Implementation for User Story 3

- [ ] T031 [US3] Implement crop extraction in `src/manga_text_seg/translation/ocr.py` (FR-023): each region's crop is taken from the **original pre-inpaint image**, never from the inpainted page, and the crop image is stored under `crops/<region_id>.png` (FR-046).
- [ ] T032 [US3] Implement the OCR adapter in `src/manga_text_seg/translation/ocr.py` (FR-022, research R3): `manga-ocr` as the preferred engine behind an adapter so an equivalent tool can substitute; the import is **lazy and inside the adapter**, so the module imports and the suite runs with the optional extra absent. The engine identity and version are recorded for every run (FR-056).
- [ ] T033 [US3] Implement the per-region OCR record in `src/manga_text_seg/translation/ocr.py` (FR-024, data-model.md `OcrResult`): `region_id`, `crop_path`, `text`, `confidence` (or `null`), `processing_time_seconds`, `status` and, on failure, `error` — written as `ocr.json`, one entry per region.
- [ ] T034 [US3] Implement region-level failure isolation in `src/manga_text_seg/translation/ocr.py` (FR-026): one region's failure or empty result is recorded and **every other region on the page is still processed**.
- [ ] T035 [US3] Implement manual-edit consumption in `src/manga_text_seg/translation/ocr.py` (FR-027, data-model.md `ManualEditRecord`): a hand-edited `ocr.json` is honoured by the next stage and the region is flagged `manually_edited`, so the edit survives to translation.
- [ ] T036 [US3] Implement stale-edit invalidation in `src/manga_text_seg/translation/ocr.py` (FR-027, edge case): a re-extraction that changes the region set invalidates the edited records that no longer match **and reports the invalidation** — the change is never reverted quietly.
- [ ] T037 [US3] Implement optional crop preprocessing in `src/manga_text_seg/translation/ocr.py` (FR-023): the preprocessing step is configurable and, when enabled, recorded in the run so the OCR input is reproducible.
- [ ] T038 [US3] Implement the absent-engine path in `src/manga_text_seg/translation/ocr.py` (FR-055): with the engine missing, the adapter records `ocr_failure` naming the missing engine — it never substitutes empty text for a failure, and never fabricates a result.

**Checkpoint**: OCR runs against a fixture with no engine installed, and one region's failure leaves the page's other regions intact.

---

## Phase 6: User Story 4 - Translate Region Text to the Configured Target Language (Priority: P2)

**Goal**: Translate each region's Japanese text into the configured target language through a pluggable provider adapter — with retry, timeout and rate-limit handling, secrets taken only from the environment, and the `region_id` → Japanese → translated mapping persisted.

**Independent Test**: Run translation against a mock provider (successes, a timeout, a rate-limit response, a hard failure) and assert the mapping records are complete, retries and back-off follow the configured policy, the failure policy is honoured, and no secret material appears in any record or log.

### Tests for User Story 4

- [ ] T039 [P] [US4] Write `tests/unit/translation/test_translate_adapter.py` (FR-029, FR-031, FR-032, FR-033, FR-060): the translation adapter against a mock provider covering success, a timeout, a rate-limit response and a hard failure; assert the mapping is complete **including failed regions**, retries and back-off follow the configured policy, `api_timeout_or_rate_limit` is recorded alongside the retry that followed, and each failure's `failure_policy` branch is applied and recorded. MUST fail before T042–T045.
- [ ] T040 [P] [US4] Write `tests/unit/translation/test_translation_cache.py` (FR-029, FR-060): a second identical run over the same text serves every region from cache with **zero provider calls**; `cache_status` is `hit` or `miss` per region; `bypass: true` forces fresh calls **without deleting anything** from `cache/translation/` (research R10). MUST fail before T046.
- [ ] T041 [P] [US4] Write `tests/unit/translation/test_no_secret_leakage.py` (FR-034, FR-035, FR-057, FR-060): with a sentinel key value present in the environment, assert `provider`/`model`/`version` appear in `run.json`, every `translations.json` entry and `metadata.json`, while the key's **value** appears nowhere — not in a record, a log, an exception message or a `reason` string. The key's **name** may appear. A missing variable refuses the run **at startup**, before any region is attempted. MUST fail before T047.

### Implementation for User Story 4

- [ ] T042 [US4] Implement the provider adapter in `src/manga_text_seg/translation/translate.py` (FR-029, research R4, data-model.md `TranslationProviderAdapter`): a pluggable adapter with a fixture implementation and a real one, using the standard library's `urllib.request` for the HTTP call — no new HTTP dependency. The adapter declares its own determinism (FR-058).
- [ ] T043 [US4] Implement per-region requests in `src/manga_text_seg/translation/translate.py` (FR-029, FR-032): **one provider request per text region**, with no batched call, so one region's timeout cannot affect another. The region→Japanese→translated mapping is persisted to `translations.json` including regions whose translation failed.
- [ ] T044 [US4] Implement the retry, timeout and rate-limit policy in `src/manga_text_seg/translation/translate.py` (FR-032): timeout, retry count and back-off come from `configs/translation.json`; a timeout or rate-limit response is retried per that policy before the region is declared failed, and each occurrence is recorded as `api_timeout_or_rate_limit` alongside the retry that followed.
- [ ] T045 [US4] Implement the failure policy in `src/manga_text_seg/translation/translate.py` (FR-033): on final failure the configured branch applies — `keep_ocr_text` (the default) carries the Japanese OCR text forward with the region flagged translation-failed; `exclude_from_rendering` drops it. `failure_policy` is recorded either way, so the choice is visible in the metadata.
- [ ] T046 [US4] Implement the content-addressed cache in `src/manga_text_seg/translation/translate.py` (FR-029, research R10, data-model.md `TranslationCacheRecord`): entries at `cache/translation/<cache_key>.json` keyed by a SHA-256 over `(text, target_language, provider, model)`; `cache_status` reported per region; the `bypass` switch forces fresh calls without deleting anything.
- [ ] T047 [US4] Implement secret handling in `src/manga_text_seg/translation/translate.py` (FR-034, FR-057): the API key comes **exclusively** from the environment variable named by `configs/translation.json`; nothing is hard-coded, and a required-but-missing secret is detected **at startup** with a clear error rather than failing region by region. No key, token or secret is written to source, config, logs, metadata or any output.
- [ ] T048 [US4] Implement provider provenance recording in `src/manga_text_seg/translation/translate.py` (FR-035): `provider`, `model` and `version` are written into `run.json`, every `translations.json` entry and `metadata.json`, as plain strings, with no secret material beside them (SC-004, SC-005).

**Checkpoint**: Translation is complete against a mock provider with no network, and the SC-005 audit finds zero secret material.

---

## Phase 7: User Story 5 - Render Translations onto the Inpainted Page (Priority: P2)

**Goal**: Render each region's translated text onto the inpainted page inside the region's stored display area — with configurable typography, automatic font sizing, word wrap and a documented overflow strategy.

**Independent Test**: Render fixture pages with short text (fits), long text (needs shrink/wrap), text exceeding even the minimum font (expects expansion-then-warning), a missing font, and a region at the page border — and assert placement, fitting behaviour, overflow strategy, warning records and error isolation all follow the specification.

### Tests for User Story 5

- [ ] T049 [P] [US5] Write `tests/unit/translation/test_render_fit.py` (FR-040, FR-041, FR-060): short text fits at the configured size; long text wraps and shrinks; text exceeding even `min_font_size` walks FR-041's ladder in order and ends as a recorded `text_overflow` warning — **silent clipping is prohibited**, so assert no glyph pixel lands outside the region's display area. Also covers the font-size fallback: an unreadable font path falls back to `fallback_font_path`, and with no fallback the region is `font_missing` while the rest of the page continues. MUST fail before T051–T054.
- [ ] T050 [P] [US5] Write `tests/unit/translation/test_render_position.py` (FR-038, FR-042, FR-043, FR-060): text is placed inside the region's stored bounding box/polygon and never outside it unless `expansion_allowance` permits FR-041's third rung; a region whose original text was vertical records its detected orientation and renders in the direction the configuration selects, with `direction_used` recorded per region. MUST fail before T051 and T055.

### Implementation for User Story 5

- [ ] T051 [US5] Implement rendering in `src/manga_text_seg/translation/render.py` (FR-037, FR-038, research R7): Pillow only, inside this module — font family, size, colour, outline/stroke, background, word wrap, line spacing and alignment all read from `configs/translation.json`, drawing into the region's stored display area. Pillow is declared in T005 and used nowhere else (plan.md Complexity Tracking).
- [ ] T052 [US5] Implement the fit-then-overflow ladder in `src/manga_text_seg/translation/render.py` (FR-040, FR-041): apply the strategy **in order** — reduce font size down to `min_font_size`, increase line count, then modestly expand the display area within `expansion_allowance`.
- [ ] T053 [US5] Implement the overflow warning in `src/manga_text_seg/translation/render.py` (FR-041, SC-006): a region still not fitting after the ladder's last rung is recorded as `text_overflow` with its reason, its `render_status` is `overflow_warning`, and it carries a `render_warning` — the text is **never silently clipped**.
- [ ] T054 [US5] Implement font resolution in `src/manga_text_seg/translation/render.py` (FR-039, research R6): the font is resolved by **file path** from `assets/fonts/DejaVuSans.ttf`, with `fallback_font_path` as the second attempt; no system-font lookup is involved. An unreadable path with no fallback records `font_missing` and the rest of the page continues.
- [ ] T055 [US5] Implement orientation-aware rendering in `src/manga_text_seg/translation/render.py` (FR-043): consume the region's detected orientation and render in the direction the configuration selects, recording `direction_used` per region.
- [ ] T056 [US5] Implement the rendered page writer in `src/manga_text_seg/translation/render.py` (FR-042): `rendered.png` is 1654×1170 lossless PNG — the same size as the inpainted page it was rendered onto.
- [ ] T057 [US5] Assert the no-resize, read-only guarantee in `src/manga_text_seg/translation/render.py` (FR-006, FR-042): nothing in the rendering path resizes an image, and the original image, the prediction mask, the inpainted page and every Spec 1–3 artifact are untouched by the write.

**Checkpoint**: A fixture page renders end-to-end and the overflow ladder is observable in `render_status`.

---

## Phase 8: User Story 6 - Batch Runs with Region- and Image-Level Error Isolation (Priority: P2)

**Goal**: Run the pipeline over a single page or the whole manifest, with a failure in one region never stopping its page and a failure in one page never stopping the batch — every failure recorded in a per-run error report, and every run identified by a run ID that never overwrites a previous run.

**Independent Test**: Run a batch with one induced failure of each of the eleven error categories at region level and page level; assert the batch completes, the error report enumerates every induced failure by category with identifiers, a second run with a different run ID coexists with the first, and reusing an existing run ID is refused unless overwrite was explicitly requested.

### Tests for User Story 6

- [ ] T058 [P] [US6] Write `tests/integration/test_batch_error_isolation.py` (FR-051, FR-054, FR-055): induce each of the eleven categories on one page or one region — missing original, missing mask, missing inpainted image, all-zero mask, every component below `min_area`, an unreadable crop, an always-failing provider, an unreadable font with no fallback, text longer than the minimum-size display area, a provider that times out then rate-limits, and a read-only output directory — and assert the batch finishes, a region failure leaves its page's other regions to complete, a page failure leaves the batch running, and all eleven keys are present in `counts` **even at zero** so an empty category is distinguishable from an unhandled one. MUST fail before T060–T063.
- [ ] T059 [P] [US6] Write `tests/integration/test_run_id_separation.py` (FR-049, FR-055): a second run under a different run ID coexists with the first untouched; re-running into an existing run ID is **refused** unless `--overwrite` was explicitly requested; a refusal leaves the previous run's tree byte-identical; a failing page's directory holds what the run produced up to the failure, with nothing corrected, substituted or fabricated. MUST fail before T064.

### Implementation for User Story 6

- [ ] T060 [US6] Implement stage orchestration in `src/manga_text_seg/translation/pipeline.py` (FR-046, FR-053): run intake → regions → OCR → translate → render for one page, consuming each stage's stored output as the next stage's input, and expose each stage as an entry point so a single stage can be invoked alone (FR-050).
- [ ] T061 [US6] Implement region-level isolation in `src/manga_text_seg/translation/pipeline.py` (FR-054): a region failure at any stage is recorded under its category and the page's remaining regions still complete.
- [ ] T062 [US6] Implement page-level isolation in `src/manga_text_seg/translation/pipeline.py` (FR-054): a page failure is recorded under its category and the batch continues, with `run.json`'s `page_statuses` carrying the same fact in summary form.
- [ ] T063 [US6] Implement manifest-scale batch execution in `src/manga_text_seg/translation/pipeline.py` (FR-001): iterate the pages of the existing Spec 1 manifest — the single source of truth for the page list and the image–mask mapping — constructing no new manifest, page list or benchmark subset.
- [ ] T064 [US6] Wire the non-overwrite refusal into the batch path in `src/manga_text_seg/translation/pipeline.py` (FR-049): an existing run ID refuses unless overwrite was explicitly requested, and the refusal is reported clearly rather than silently writing elsewhere.

**Checkpoint**: A batch with eleven induced failures finishes and accounts for every one of them.

---

## Phase 9: User Story 7 - Inspectable Intermediate Outputs and Pipeline Metadata (Priority: P3)

**Goal**: Store every stage's output in a predictable per-page location — crops, region list, OCR results, translations, rendered image and a complete pipeline metadata JSON — so each step can be inspected, audited and corrected without re-running the stages before it.

**Independent Test**: Process a page end-to-end and assert the full artifact set exists at the specified locations, the metadata JSON contains every required field, and each stage can be re-executed alone against the stored outputs of the previous stage.

### Tests for User Story 7

- [ ] T065 [P] [US7] Write `tests/contract/translation/test_pipeline_metadata_schema.py` (FR-047, FR-052): `metadata.json` validates against [contracts/pipeline-metadata.schema.json](contracts/pipeline-metadata.schema.json) with `additionalProperties: false` respected and the `allOf` conditional satisfied (`error_message` present when `status` is `failed`); every required field is present including `image_id`, `segmentation_method`, `inpainting_method`, `target_language`, the OCR engine, the translation provider and the rendering configuration; and the `regions` array carries one entry per extracted region **including failed ones** (FR-031). MUST fail before T066–T070.

### Implementation for User Story 7

- [ ] T066 [US7] Implement the `metadata.json` writer in `src/manga_text_seg/translation/runs.py` (FR-047, data-model.md `PipelineMetadata`): `image_id`, `run_id`, `segmentation_method`, `inpainting_method`, `target_language`, `ocr_engine`, `translation_provider`, `rendering`, `regions`, `stage_timings` and `status` — each region entry carrying `region_id`, `bbox`, `polygon`, `orientation`, `ocr_text`, `translation`, `ocr_status`, `translation_status`, `render_status`, `cache_status`, `failure_policy`, `direction_used`, `manually_edited` and the warnings.
- [ ] T067 [US7] Implement the intermediate artifact writers in `src/manga_text_seg/translation/runs.py` (FR-046, FR-048): `crops/<region_id>.png`, `mask.png`, `ocr.json`, `translations.json` and `rendered.png` are each written as their own artifact, so any stage can be re-run alone against the previous stage's stored outputs and produce the same result as inside a full run (FR-050, US7 scenario 3).
- [ ] T068 [US7] Implement the verbatim provenance quote in `src/manga_text_seg/translation/runs.py` (FR-004): `inpainting_method` is quoted **verbatim** from Spec 3's sidecar — algorithm, radius and dilation configuration — never re-derived and never defaulted.
- [ ] T069 [US7] Implement the FR-052 addressing guarantee in `src/manga_text_seg/translation/runs.py` (FR-052, SC-010): a consumer knowing only (run ID, segmentation method, `image_id`) resolves the page directory and interprets it from `metadata.json` alone — no index file, no manifest lookup, no run-specific knowledge.
- [ ] T070 [US7] Implement per-stage timing and status recording in `src/manga_text_seg/translation/runs.py` (FR-056): the elapsed time of each stage per page goes into `stage_timings`, and the page `status` is one of `ok`, `no_regions` or `failed`. `stage_timings` is the only non-deterministic field in this file (research R8).

**Checkpoint**: A completed page's directory is self-describing, and its metadata validates against the schema.

---

## Phase 10: User Story 8 - Selective Re-runs (Priority: P4)

**Goal**: Re-run only a failed region, only one stage, or only one page — against already-stored upstream outputs — so that fixing a handful of failures after a large batch does not require re-processing everything.

**Independent Test**: Induce a region failure and a page failure, re-run exactly those scopes against the stored intermediates, and assert only the targeted scope is re-processed, results are consistent with a full run, and no unrelated output is modified.

### Tests for User Story 8

- [ ] T071 [P] [US8] Write `tests/integration/test_selective_rerun.py` (FR-050, FR-053): with a region failure and a page failure induced, re-run exactly those scopes against the stored intermediates; assert only the targeted scope is re-processed, its records are updated, the earlier stages' stored outputs are consumed as-is and not recomputed, and the results match a full run's. MUST fail before T072–T074.

### Implementation for User Story 8

- [ ] T072 [US8] Implement single-stage re-execution in `src/manga_text_seg/translation/pipeline.py` (FR-050): a stage invoked alone consumes the previous stage's stored outputs without re-running it, and produces the same result as inside a full pipeline run.
- [ ] T073 [US8] Implement single-region re-execution in `src/manga_text_seg/translation/pipeline.py` (FR-053): a region re-run re-processes only the named region and updates its records, leaving the page's other regions alone.
- [ ] T074 [US8] Assert the blast-radius guarantee in `src/manga_text_seg/translation/pipeline.py` (US8 scenario 3): a re-run leaves the outputs of unrelated regions, pages and runs **byte-unchanged**.

**Checkpoint**: A failed region can be fixed without touching anything else on disk.

---

## Phase 11: Polish & Cross-Cutting Concerns

**Purpose**: The CLI surface, the two audits the success criteria demand, and the end-to-end validation.

- [ ] T075 [P] Write `tests/integration/test_determinism.py` (FR-058, SC-012): two runs over identical inputs and configuration produce **byte-identical** extraction, OCR and rendering artifacts, with exactly two documented exceptions — the elapsed-time fields (`stage_timings`, `processing_time_seconds`) and `translations.json`, whose determinism is the provider's declared property and is recorded as such in `run.json` rather than guaranteed. Assert no wall-clock timestamp appears in any per-page metadata field and that the run ID is the only time-identifying value (research R8).
- [ ] T076 [P] Write `tests/integration/test_secret_audit.py` (FR-035, FR-057, SC-005): an automated audit over a completed run's outputs and logs finds **zero** occurrences of API keys or secret material, while `provider`/`model`/`version` fields are present; a `reason` string names the environment variable, never its value.
- [ ] T077 [P] Write `tests/integration/test_no_vendoring_and_no_gt.py` (FR-005, FR-007, FR-008, SC-013): assert no model source file and no weight file of any size or form exists in the stored tree, no segmentation-model runtime is imported anywhere in `src/`, no artifact under `outputs/translation/` is derived from a ground-truth mask or copies one, and no file beneath `data/no-need-to-read/` is opened (SC-011).
- [ ] T078 Extend `src/manga_text_seg/cli.py` with the `translate` subcommand and its argument surface (FR-053): `--config`, `--run`, `--image-id`, `--stage`, `--region`, `--segmentation-method`, `--inpainting-run`, `--target-lang`, `--overwrite`. Follow the existing `cmd_*` / subparser conventions at `src/manga_text_seg/cli.py:341-386`; do not disturb the existing subcommands.
- [ ] T079 Wire the FR-053 operations to `pipeline.py` in `src/manga_text_seg/cli.py` (FR-053): process a single image; process the whole manifest; region extraction only; OCR only; translation only; rendering only; the full pipeline; re-run a failed region; and export intermediate outputs. Each maps to a `pipeline.py` entry point, and the stage names match the error report's `stage` values.
- [ ] T080 Implement the intermediate-output export in `src/manga_text_seg/translation/pipeline.py` (FR-053): copy the requested page's stored intermediates to a caller-named destination outside the run tree, **copying and never moving** — the run tree stays intact (FR-006).
- [ ] T081 Verify FR-060's thirteen areas each have a test and that the whole suite passes with **no GPU, no checkpoint, no OCR engine, no provider key and no network** — `pytest -o addopts='' -q`. Record which test file covers each of the thirteen: region extraction from a binary mask; bounding box and padding; region ordering; the OCR adapter against a fixture; the translation adapter against a mock provider; translation-result caching; no API-key leakage; text wrapping; font-size fallback; text overflow; rendering position; one region failing while others complete; and the `image_id`/`region_id`/OCR/translation metadata mapping.
- [ ] T082 Run [quickstart.md](quickstart.md)'s scenarios 1–3 and 5–10 end-to-end against the fixture bundle (scenario 4's live provider half stays optional and is not run in CI), and confirm `ruff check` and `black --check` are clean on the new files.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately.
- **Foundational (Phase 2)**: Depends on Setup — **blocks every user story**.
- **User Stories (Phases 3–10)**: All depend on Foundational. US1 and US2 are both P1 and independent of each other; US3 depends on US2; US4 depends on US3; US5 depends on US2 and US4; US6 depends on US1–US5; US7 depends on US1–US5; US8 depends on US6–US7.
- **Polish (Phase 11)**: Depends on all stories being complete.

### User Story Dependencies

- **US1 (P1)**: Starts after Foundational — no story dependencies. 🎯 MVP.
- **US2 (P1)**: Starts after Foundational; its Independent Test runs on synthetic masks, so it is testable without US1's intake.
- **US3 (P1)**: Consumes US2's regions. Independent of US4 and US5.
- **US4 (P2)**: Consumes US3's OCR records. Independent of US5 — it is tested against a mock provider alone.
- **US5 (P2)**: Consumes US2's regions and US4's translations, plus US1's inpainted image.
- **US6 (P2)**: Consumes US1–US5's per-page pipeline; adds batch behaviour, error isolation and run separation.
- **US7 (P3)**: Consumes US1–US5's artifacts; adds the metadata contract and the addressing guarantee.
- **US8 (P4)**: Consumes US6's run tree and US7's stored intermediates.

### Within Each User Story

- Tests MUST be written and FAIL before implementation (Principle II, NON-NEGOTIABLE).
- Contracts and entities before services; services before the CLI wiring.
- Core implementation before integration.
- Story complete before moving to the next priority.

### Parallel Opportunities

- Setup: T002 ∥ T003 ∥ T004 ∥ T005 (T001 first — the loader and the fixture both read it).
- Foundational: T006 ∥ T007 (test files), then T008 ∥ T009, then T010 ∥ T011 ∥ T012 (all three in `translation/runs.py` — serialize if one author holds the file).
- US1: T013 ∥ T014. US2: T020 ∥ T021, then T022 ∥ T023 ∥ T024, then T026 ∥ T027.
- US3: T030 alone. US4: T039 ∥ T040 ∥ T041. US5: T049 ∥ T050.
- US6: T058 ∥ T059. US8: T071 alone.
- Polish: T075 ∥ T076 ∥ T077, then T078 → T079 → T080, then T081 ∥ T082.

---

## Parallel Example: User Story 4

```bash
# Launch all US4 tests together (they MUST fail first):
Task: "Write the mock-provider adapter test in tests/unit/translation/test_translate_adapter.py"
Task: "Write the caching test in tests/unit/translation/test_translation_cache.py"
Task: "Write the no-secret-leakage test in tests/unit/translation/test_no_secret_leakage.py"

# Then the two independent implementation halves:
Task: "Implement the provider adapter and per-region requests in src/manga_text_seg/translation/translate.py"
Task: "Implement the content-addressed cache in src/manga_text_seg/translation/translate.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup.
2. Complete Phase 2: Foundational (CRITICAL — blocks all stories).
3. Complete Phase 3: User Story 1.
4. **STOP and VALIDATE**: run `tests/integration/test_intake_rejections.py` — the intake gate stands alone.
5. Demo the named-rejection behaviour if ready.

### Incremental Delivery

1. Setup + Foundational → foundation ready.
2. US1 → the intake gate (MVP) → validate independently.
3. US2 → stable regions with deterministic IDs → validate independently.
4. US3 → OCR on pre-inpaint crops → validate independently.
5. US4 → translation through the adapter, no secrets → validate independently.
6. US5 → rendered pages → validate independently.
7. US6 → batch scale and error isolation → US7 → inspectable metadata → US8 → selective re-runs, each validated at its checkpoint.

### Parallel Team Strategy

1. The team completes Setup + Foundational together — T010–T012 all touch
   `src/manga_text_seg/translation/runs.py`, so one author owns that file or the tasks serialize.
2. Once Foundational is done:
   - Developer A: US1 → US2 (the critical path; nothing else can produce regions without them).
   - Developer B: prepares US4's `translate.py` against a mock provider, which needs no OCR output.
   - Developer C: prepares US5's `render.py` against fixture translations and `assets/fonts/`.
3. US3 lands the OCR records that US4's live path consumes, so B integrates after it; US6–US8 land last, on top of everything.

---

## Notes

- [P] tasks = different files, no dependencies.
- **[Story] label maps every task to exactly one spec.md user story**; Setup, Foundational and Polish
  phases carry no story label, as required by the format.
- `src/manga_text_seg/run.py` (singular, Spec 2) is untouched; this feature's module is
  `src/manga_text_seg/translation/runs.py`. Keep them distinct in every import and every test.
- The seven `translation/` modules exist so this feature claims no top-level filename Spec 3 also
  plans (`intake.py`, `maskproc.py`, `inpaint.py`, `runs.py`).
- **No ground truth may reach any output.** The prediction mask is the only mask this feature opens,
  and no GT-assisted mode is the default (FR-008, US1 scenario 4).
- **Nothing third-party is vendored** — no model source, no weights, at any size, in any form
  (FR-005, SC-013). `manga-ocr` is an optional extra behind a lazy import (FR-028).
- **No secret reaches a file** — env vars only, verified at startup (FR-034, FR-035, FR-057).
- **Upstreams are read-only**; where one is needed inside the run tree it is copied, never moved
  (FR-006). `data/no-need-to-read/` is never read (FR-007).
- Verify tests fail before implementing; commit after each task or logical group; stop at any
  checkpoint to validate a story independently.
- Avoid: vague tasks, same-file conflicts, cross-story dependencies that break independence.
