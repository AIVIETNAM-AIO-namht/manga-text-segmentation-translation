# Data Model: OCR, Translation & Text Rendering

**Date**: 2026-09-24 | **Spec**: [spec.md](spec.md) | **Research**: [research.md](research.md)

Entities reused from Spec 1 — `RawPage`, `NormalizedMask`, `AlignedPair`, `FailureReport` — are
defined in [Spec 1's data model](../001-data-foundation-classical-baseline/data-model.md) and are
**not restated here**. Spec 2's `DistributablePageList`, `ImageIdentityRecord`, `Provenance` and its
per-page mask sidecar are defined in
[Spec 2's data model](../002-deep-learning-segmentation-benchmark/data-model.md). Spec 3's
`InpaintRun`, `SampleMetadata` and `MaskProcessingConfig` are defined in
[Spec 3's data model](../003-text-removal-inpainting/data-model.md) and its
[inpainted-output contract](../003-text-removal-inpainting/contracts/inpainted-output.md). All of
them are read by this feature and **none is written by it** (FR-006). Ground truth is read by no
entity here at all (FR-008).

## Entities

### TranslationRun (FR-046–FR-049)

One identified execution. Persisted as `outputs/translation/<run_id>/run.json`.

- `run_id` — the directory name; the only time-identifying field in the run (research R8).
- `segmentation_method` — one of the four FR-009 identities, fixed for the whole run (FR-049).
- `inpainting_source` — the Spec 3 run ID and the algorithm (`telea` or `ns`) this run reads
  (FR-012, FR-049).
- `target_language` — `vi` or `en` (FR-030), fixed for the whole run (FR-049).
- `region_extraction_config` — the `RegionExtractionConfig` used, recorded once (SC-002).
- `rendering_config` — the `RenderingConfig` used, recorded once (FR-039).
- `ocr_engine` — engine identity and version, recorded once (FR-022).
- `translation_provider` — provider, model and version, recorded once (FR-035).
- `counts` — pages attempted, succeeded, failed, and regions by terminal status.
- `page_statuses` — per-page terminal status, so a batch's shape is legible without reading every
  page's metadata.

*Identity is the run ID, not a timestamp (research R8). One run fixes exactly one (method, inpainting
source, language) triple, so a different combination is a different run ID (FR-049).*

### PageBundle (FR-001–FR-004, FR-046)

One page's assembled inputs and the outcome of validating them. Not a new file — an in-memory
resolution built by `intake.py`, recorded per page inside `metadata.json`.

- `image_id` — `<manga>/<NNN>`, from the Spec 1 manifest. Never reconstructed and never guessed
  (FR-001).
- `original_image_path` — the raw page, via the manifest (FR-046).
- `prediction_mask_path` / `prediction_mask_sidecar_path` — Spec 2's PNG and its JSON sidecar,
  located by page identifier plus method name (FR-003).
- `inpainted_image_path` / `inpainting_metadata_path` — Spec 3's `<NNN>.png` and `<NNN>.json`,
  resolved by (inpainting run ID, method, `image_id`) alone (FR-004, SC-010).
- `inpainting_provenance` — the algorithm, radius and dilation configuration quoted from Spec 3's
  sidecar, carried into this feature's outputs as provenance (FR-004).
- `mask_copy_path` — where the prediction mask was copied into this run's tree (FR-046). **Copied,
  never moved** (FR-006).
- `status` — `accepted`, or the FR-051 category that rejected it.
- `reason` — the specific reason, populated on rejection.

*Every path here is a read-only upstream. Nothing in this record is a path this feature may write to
(FR-006), and the record is what SC-001's "exactly one manifest page per accepted bundle" is checked
against.*

### TextRegion (FR-013–FR-021)

One extracted text area. Persisted inside `metadata.json`'s `regions` array, with its crop written to
`crops/`.

- `region_id` — the 1-based index in the run's total reading order under the active configuration
  (FR-019, research R9).
- `bbox` — `[x, y, w, h]` in aligned page space (FR-018).
- `polygon` — the text area's outline, `approxPolyDP`-simplified, in the same space (FR-018).
- `orientation` — `horizontal` or `vertical`, from the mask geometry (FR-020, FR-043).
- `reading_order_index` — the position the `region_id` was derived from, kept explicitly so the
  derivation is auditable rather than implied (FR-019).
- `merged_from` — the component count that went into this region, `1` when nothing merged, so the
  FR-015 merge is visible in the output rather than invisible (FR-015).
- `crop_path` — the crop taken from the **original pre-inpaint image** (FR-023).
- `crop_box` — the padded box actually cropped.
- `clipped` — `true` when the padded crop exceeded the page bounds and was clipped to them (FR-017).
  Recorded, never silent.
- `area` — the component area, the quantity FR-016's min/max filters act on.

*A region with zero area or one filtered out does not appear here; a page whose regions all filter
out yields an empty array and a page status, not a run failure (FR-021).*

### RegionExtractionConfig (FR-015, FR-016)

The shared extraction configuration. Held in `configs/translation.json`, recorded in `run.json`, and
identical for every page in a run (SC-002).

- `min_area` / `max_area` — the area filter bounds. A component above `max_area` is **excluded and
  recorded, never auto-split**.
- `merge_distance` — pixels; `0` disables merging (FR-015).
- `crop_padding` — pixels of padding added around each region before cropping (FR-017).
- `reading_order` — `manga` (right-to-left, top-to-bottom) by default, or `western` (FR-016).
- `polygon_epsilon` — the `approxPolyDP` tolerance for FR-018's polygon.

*Every field is configuration rather than a code default because FR-016 names all but the last as
configurable, and because a value computed from the input would break FR-058's determinism (research
R8).*

### OcrResult (FR-022–FR-027)

One region's OCR outcome. Persisted in `ocr.json`, one entry per region.

- `region_id` — joins back to `TextRegion`.
- `crop_path` — the crop that was recognised (FR-024).
- `text` — the recognised Japanese text, or empty on failure.
- `confidence` — the engine's confidence, or `null` when the engine provides none (FR-024). `null`
  means "not reported", never "zero confidence".
- `engine` / `engine_version` — recorded per result as well as per run (FR-022).
- `preprocessing` — the crop preprocessing applied, or `null` when disabled (FR-023).
- `processing_time_seconds` — excluded from byte-comparison (research R8).
- `status` — `ok`, `empty` or `failed` (FR-026).
- `error` — the failure reason, populated on failure.
- `manually_edited` — set by `ManualEditRecord` when a human edits this entry (FR-027).

*One region's failure sets this record's status and stops nothing else on the page (FR-026).*

### ManualEditRecord (FR-027)

A human edit to `ocr.json`, persisted beside it as an audit trail.

- `region_id`, `original_text`, `edited_text`, `edited_at`.
- `invalidated_by_reextraction` — set when a later extraction run produces a different region set, so
  the edit no longer applies. **Reported, never silently discarded.**

*Downstream stages honour the edit; a re-extraction that invalidates it is a recorded event rather
than a quiet revert (FR-027).*

### TranslationResult (FR-029–FR-033)

One region's translation outcome. Persisted in `translations.json`, one entry per region.

- `region_id` — joins back to `TextRegion`.
- `source_text` — the Japanese text actually sent (the edited text when one exists) (FR-031).
- `translated_text` — the translation, or the carried-forward Japanese on the default failure policy
  (FR-033).
- `target_language` — `vi` or `en` (FR-030).
- `provider` / `model` / `version` — recorded per result (FR-035).
- `cache_status` — `hit` or `miss`, so a run's cost is legible after the fact (FR-029, research R10).
- `processing_time_seconds` — excluded from byte-comparison (research R8).
- `status` — `ok`, `failed` or `skipped`.
- `error` — the failure reason, populated on failure.
- `failure_policy` — which FR-033 branch was applied, `keep_ocr_text` or `exclude_from_rendering`,
  recorded even on success so the run's policy is never inferred.

*Every region gets a record, including failures (FR-031). A failed translation is never a missing
record.*

### TranslationProviderAdapter (FR-029, FR-032, FR-034)

The pluggable provider interface. Not a persisted file — a protocol with one real implementation and
one fixture implementation.

- `provider`, `model`, `version` — identity, recorded into every result and into `run.json`.
- `required_secret_env` — the environment variable name the provider needs, or `null`. Read at
  startup and checked **present** when the provider needs it (FR-034); the value is never stored on
  this object, never logged and never written to any output (FR-035, SC-005).
- `retry_policy` — attempts, backoff and which HTTP statuses are retried, with rate-limit handling
  (FR-032).
- `timeout_seconds` — the per-request socket timeout.
- `deterministic` — the provider's determinism declared as a **provider property** (FR-058), so the
  determinism check knows whether it may span this stage (research R8).
- `translate(text, target_language) -> TranslationResult` — one call, **one region** (FR-029).

*Per-region granularity is the interface's shape, not a caller's convention: there is no method that
takes a list, so no caller can batch regions and break FR-031's isolation (FR-029).*

### TranslationCacheRecord (FR-029)

One cached provider response. Persisted as one JSON file under `cache/translation/`, keyed by the
SHA-256 of `(text, target_language, provider, model)` (research R10).

- `cache_key` — the hash; also the filename.
- `translated_text` — the stored response.
- `provider` / `model` / `version` — the identity that produced it.
- `fetched_at` — when the provider was actually called, kept **here** rather than in any page
  metadata so the run stays deterministic (research R8, R10).

*Outside every run directory (FR-029) and under a gitignored path, so no cache file can reach the
repository. `bypass: true` in the configuration forces fresh calls without deleting anything.*

### RenderingConfig (FR-039, FR-040)

The shared rendering configuration. Held in `configs/translation.json`, recorded in `run.json`.

- `font_path` — resolved **by file path**, defaulting to the bundled
  `assets/fonts/DejaVuSans.ttf` (research R6). Never a system font lookup (FR-039).
- `font_size` — the starting size, before FR-040's fitting.
- `min_font_size` — the floor of FR-041's ladder.
- `color`, `stroke_width`, `stroke_fill` — text colour and outline (FR-039).
- `background` — the box behind the text, or `null` for none.
- `wrap`, `line_spacing` — word wrap and leading (FR-039).
- `align_h`, `align_v` — horizontal and vertical alignment (FR-039).
- `text_direction` — `auto`, `horizontal` or `vertical`; `auto` follows the region's orientation
  (FR-043).
- `expansion_allowance` — how far FR-041's third rung may expand the display area, in pixels.
- `fallback_font_path` — the configured fallback, used when the primary font is missing or unreadable.

*One font file, resolved by path, is what makes FR-039's "no dependency on system-installed fonts"
true rather than aspirational (research R6).*

### RenderedPage (FR-037, FR-040–FR-043, FR-045)

One page's final image plus its per-region render outcomes. The image is `rendered.png`; the
outcomes live inside `metadata.json`'s `regions` array.

- `image_path` — `rendered.png`, always in aligned page space, the same size as the inpainted page
  (FR-042).
- Per region: `render_status` — `rendered`, `overflow_warning` or `skipped`.
- Per region: `font_size_used` — the size after FR-040's fitting, so a shrunken region is visible
  rather than invisible.
- Per region: `line_count`, `display_box` — the box actually painted into, including any FR-041
  expansion.
- Per region: `direction_used` — the effective horizontal/vertical choice (FR-043).
- Per region: `warning` — populated when FR-041's ladder reached its last rung.

*Silent clipping is prohibited (FR-041), so a region that does not fit ends as a warning record, not
as pixels outside its box.*

### PipelineMetadata (FR-047)

The per-page JSON record. Persisted as `metadata.json`, one per page, and the artifact Spec 5 reads
(FR-052).

- `image_id`, `segmentation_method`, `inpainting_method`, `target_language` — the four fields FR-047
  names at minimum.
- `run_id`, `ocr_engine` + version, `translation_provider` + model, `rendering_config` — run
  provenance (FR-047).
- `inpainting_provenance` — quoted from Spec 3's sidecar (FR-004).
- `regions` — an array, one entry per region, each carrying `region_id`, `bbox`, `ocr_text`,
  `translation`, `ocr_status`, `translation_status`, `render_status`, and the per-stage timings and
  error messages FR-047 requires.
- `stage_timings` — per-stage elapsed seconds, the only non-deterministic fields in the file
  (research R8).
- `status` — the page's terminal status, including the FR-021 zero-region case.

*This is the contract Spec 5 consumes: run ID plus method plus `image_id` must locate and interpret
it with nothing else (FR-052, SC-010).*

### ErrorReport (FR-051)

All failures of one run, grouped by category. Persisted as `errors.json` at the run root.

- `categories` — the eleven FR-051 categories, each present even at zero occurrences so an empty
  category is distinguishable from an unhandled one: *missing original image; missing prediction
  mask; missing inpainted image; empty mask; no text region detected; OCR failure; translation
  failure; font missing; text overflow (warning); API timeout/rate limit; output write failure*.
- Per occurrence: `image_id`, `region_id` (`null` at page level), `category`, `reason`, `stage`.
- `counts` — occurrences per category.

*"No text region detected" and "text overflow" are in this report rather than being swallowed: FR-021
and FR-041 both make them recordable outcomes, not errors to hide (FR-055).*
