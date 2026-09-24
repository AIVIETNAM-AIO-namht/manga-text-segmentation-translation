# Quickstart: OCR, Translation & Text Rendering

**Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md) | **Config**: `configs/translation.json`

Every scenario below is runnable on a **CPU-only** machine with **no checkpoint, no GPU and no
segmentation runtime of any kind** (FR-005, FR-062). The inputs are masks and inpainted pages that
were produced elsewhere and handed back — this project never runs a segmentation model. Scenarios 1–3
and 5–9 need **no network and no OCR engine installed**: the OCR and translation adapters are driven
against fixtures. Only scenario 4's second half reaches a real provider, and it is optional.

## 0. Prerequisites

- Spec 1 complete: `outputs/segmentation/default/manifest.json` exists with 390 pairs (FR-001).
- Spec 2 complete: at least one admitted run under `deliverables/<run_id>/<method>/`, carrying
  `masks/`, `metadata/` and `provenance.json` (FR-003). `deliverables/drill/standin/` is enough.
- Spec 3 complete: an inpainting run under `outputs/inpainting/<run_id>/<method>/<algorithm>/`,
  carrying `<NNN>.png` and `<NNN>.json` per page (FR-004). Nothing here reads
  `data/groundtruth/` (FR-008), and nothing here reads `data/no-need-to-read/` (FR-007).
- `configs/translation.json` — paths, the segmentation method, the inpainting source and algorithm,
  the target language, region-extraction parameters, OCR and translation provider settings, cache
  settings, and the rendering configuration (FR-061).

```bash
python --version && pip list     # versions recorded, not assumed
ls outputs/inpainting/           # an admitted Spec 3 run must be here before anything below runs
```

## 1. The intake gate (US1, FR-001–FR-008, FR-013)

```bash
manga-text-seg translate --config configs/translation.json --run quickstart \
    --image-id ARMS/001 --stage intake
```

Expected: for `ARMS/001` the original image, the selected method's prediction mask **and its JSON
sidecar**, and the configured inpainting run's `<NNN>.png` and `<NNN>.json` are located and joined by
`image_id` (`<manga>/<NNN>`) alone — no manual path work, no manifest rebuild (FR-001, FR-004).

Checks that must hold:

- The bundle is accepted only when every path resolves and the inpainted image's size equals the
  aligned page size, 1654×1170. A size disagreement is recorded as an intake failure; the image is
  **never silently resized** (FR-002, edge case).
- Each rejection lands in `errors.json` under its own FR-051 category — `missing_original_image`,
  `missing_prediction_mask`, `missing_inpainted_image` — with the specific reason, and the remaining
  pages continue (FR-054).
- **No ground truth is read.** The prediction mask is the only mask this feature ever opens, and no
  GT-assisted mode is the default (FR-008, US1 scenario 4).
- Every upstream path is read-only. `mask.png` inside the run tree is a **copy** of the prediction
  mask, byte-identical to the admitted mask it came from (FR-006).

## 2. Region extraction (US2, FR-014–FR-021)

```bash
manga-text-seg translate --config configs/translation.json --run quickstart \
    --image-id ARMS/001 --stage regions
```

Expected: `outputs/translation/quickstart/<method>/ARMS/001/` with `crops/<region_id>.png` per region
and the region list inside `metadata.json`.

Checks that must hold:

- Each region carries both a `bbox` and a `polygon` in aligned page space (FR-018).
- Two components closer than `merge_distance` become one region with `merged_from` recording how many
  went in; `merge_distance: 0` leaves them separate (FR-015, US2 scenario 2).
- Components outside `min_area`/`max_area` are excluded and the exclusion is recorded — an oversized
  component is **never auto-split** (FR-016).
- `region_id` is the 1-based reading-order position under the active configuration, and running the
  stage twice over the same mask produces identical IDs (FR-019, US2 scenario 4).
- A crop whose padding would cross the page border is clipped to the page with `clipped: true`
  recorded, and no character inside the region is cut (FR-017).
- An empty mask, or one whose components all filter out, yields zero regions and a page status —
  **not** a run failure (FR-021).

## 3. OCR on pre-inpaint crops (US3, FR-022–FR-028)

```bash
manga-text-seg translate --config configs/translation.json --run quickstart \
    --image-id ARMS/001 --stage ocr
```

Expected: `ocr.json`, one entry per region.

Checks that must hold:

- Every crop is taken from the **original pre-inpaint image**, not from the inpainted page, and the
  crop image is stored under `crops/` (FR-023, US3 scenario 1).
- Each entry carries `region_id`, `crop_path`, `text`, `confidence` (or `null` when the engine
  reports none — never a fabricated `0.0`), `processing_time_seconds`, `status` and, on failure,
  `error` (FR-024).
- One region's OCR failure or empty result is recorded and **every other region on the page is still
  processed** (FR-026).
- Editing `ocr.json` by hand before the next stage is honoured downstream, the region is flagged
  `manually_edited`, and a re-extraction that changes the region set invalidates the stale edit
  **and reports the invalidation** rather than reverting it quietly (FR-027).
- With the engine absent, the adapter records `ocr_failure` naming the missing engine; it never
  substitutes empty text for a failure (FR-055).

## 4. Translation through the provider adapter (US4, FR-029–FR-036)

Against the fixture provider — no network, no key, no cost:

```bash
manga-text-seg translate --config configs/translation.json --run quickstart \
    --image-id ARMS/001 --stage translate
```

Against the real provider — the only step in this file that touches the network:

```bash
export MANGA_TRANSLATE_API_KEY=…        # the value lives here and nowhere else (FR-034)
manga-text-seg translate --config configs/translation.json --run live --target-lang vi
```

Expected: `translations.json`, one entry per region, and `cache/translation/<cache_key>.json` per
distinct request.

Checks that must hold:

- Every region has a persisted `region_id` → Japanese → translated mapping **including regions whose
  translation failed** (FR-031). A failed translation is never a missing record.
- Each region is its own provider request. There is no batched call, so one region's timeout cannot
  affect another (FR-029, FR-032).
- A timeout or rate-limit response is retried per the configured policy before the region is
  declared failed, and the occurrence is recorded as `api_timeout_or_rate_limit` alongside the retry
  that followed (FR-032).
- On final failure the configured policy applies — `keep_ocr_text` (the default) carries the Japanese
  forward with the region flagged translation-failed, `exclude_from_rendering` drops it — and
  `failure_policy` is recorded either way (FR-033).
- `cache_status` is `hit` or `miss` per region. A second identical run over the same text serves
  every region from cache with zero provider calls; `bypass: true` forces fresh calls without
  deleting anything (FR-029, research R10).
- The provider needs `MANGA_TRANSLATE_API_KEY`; a missing variable refuses the run **at startup**
  with a clear error rather than failing region by region (FR-034).
- `provider`, `model` and `version` appear in `run.json`, in every `translations.json` entry and in
  `metadata.json`. The key's **name** may appear; its **value** appears nowhere (FR-035).
- Translation results are never used to rank segmentation, and translation quality is not scored
  anywhere in this feature (FR-036).

## 5. Rendering onto the inpainted page (US5, FR-037–FR-045)

```bash
manga-text-seg translate --config configs/translation.json --run quickstart \
    --image-id ARMS/001 --stage render
```

Expected: `rendered.png` — the inpainted page with the translations drawn into their regions.

Checks that must hold:

- Text is placed inside the region's stored bounding box/polygon, never outside it unless
  `expansion_allowance` permits the FR-041 third rung (FR-038, FR-042).
- Font size shrinks automatically to fit. Long text wraps; text that still does not fit walks
  FR-041's ladder in order — shrink to `min_font_size`, add lines, modestly expand the display area —
  and ends as a recorded `text_overflow` warning. **Silent clipping is prohibited** (FR-040, FR-041).
- An unreadable font path falls back to `fallback_font_path`; when no fallback resolves the region is
  recorded `font_missing` and the rest of the page continues (FR-039, US5 scenario 3).
- A region whose original text was vertical records its detected orientation and renders in the
  direction the configuration selects, with `direction_used` recorded per region (FR-043).
- `rendered.png` is 1654×1170 lossless PNG — the same size as the inpainted page. The original image,
  the prediction mask and every Spec 1–3 artifact are untouched (FR-037, FR-042).
- The font is resolved by file path from `assets/fonts/DejaVuSans.ttf`; no system font lookup is
  involved, so the result does not depend on the machine (FR-039, research R6).

## 6. A batch where things fail (US6, FR-051, FR-054, FR-055)

```bash
manga-text-seg translate --config configs/translation.json --run batch-a
```

Each row below is induced on one page or one region, and the batch must still finish:

| Induced condition | Expected category | Level |
|---|---|---|
| Delete the manifest's raw page | `missing_original_image` | page |
| Point the method at a mask that is not there | `missing_prediction_mask` | page |
| Point the inpainting source at a page that is not there | `missing_inpainted_image` | page |
| A mask of all zeros | `empty_mask` | page |
| Every component below `min_area` | `no_text_region_detected` | page |
| An unreadable or empty crop | `ocr_failure` | region |
| A provider that always fails | `translation_failure` | region |
| An unreadable `font_path` with no fallback | `font_missing` | page |
| Text longer than the minimum-size display area | `text_overflow` | region |
| A provider that times out then rate-limits | `api_timeout_or_rate_limit` | region |
| A read-only output directory | `output_write_failure` | page |

Checks that must hold:

- All eleven categories appear in `errors.json`'s `counts` **even at zero occurrences**, so an empty
  category is distinguishable from an unhandled one (FR-051).
- A region failure leaves that page's other regions to complete; a page failure leaves the batch
  running; `run.json`'s `page_statuses` carries the same facts in summary form (FR-054).
- Nothing is corrected, substituted or fabricated — the failing page's directory holds what the run
  produced up to the failure, and its category and reason are in the report (FR-055).
- Re-running into `batch-a` **refuses** unless `--overwrite` was explicitly requested, and `batch-b`
  coexists with `batch-a` on disk untouched (FR-049, US6 scenario 4).

## 7. Inspectable intermediates and metadata (US7, FR-046–FR-048, FR-052)

```bash
ls outputs/translation/quickstart/<method>/ARMS/001/
```

Expected: `crops/`, `mask.png`, `ocr.json`, `translations.json`, `rendered.png`, `metadata.json` —
plus `run.json` and `errors.json` at the run root (FR-046, FR-048).

Checks that must hold:

- `metadata.json` validates against
  [pipeline-metadata.schema.json](contracts/pipeline-metadata.schema.json) and carries `image_id`,
  `segmentation_method`, `inpainting_method`, `target_language`, the run provenance, `stage_timings`
  and a `regions` array whose entries each carry `region_id`, `bbox`, `ocr_text`, `translation`,
  `ocr_status`, `translation_status` and `render_status` (FR-047).
- `inpainting_method` is quoted **verbatim** from Spec 3's sidecar — algorithm, radius and dilation —
  never re-derived (FR-004).
- **Addressing (FR-052)**: a consumer knowing only (run ID, segmentation method, `image_id`) resolves
  this directory and interprets it with no index file, no manifest lookup and no run-specific
  knowledge (SC-010, [rendered-output.md](contracts/rendered-output.md)).
- Every stage's output is stored, so each stage can be re-run alone against the previous stage's
  stored outputs and produce the same result as inside a full run (FR-050, US7 scenario 3).

## 8. Selective re-runs (US8, FR-050, FR-053)

```bash
manga-text-seg translate --config configs/translation.json --run quickstart \
    --image-id ARMS/001 --stage ocr --region 7
manga-text-seg translate --config configs/translation.json --run quickstart \
    --image-id ARMS/001 --stage render
```

Expected: only region 7 is re-OCR'd; the render re-run consumes the stored crops, `ocr.json` and
`translations.json` as-is without recomputing them.

Checks that must hold:

- Unrelated regions, pages and runs are byte-unchanged (US8 scenario 3).
- A re-run does not require the stages before it to run again (FR-050).

## 9. Determinism, and the two documented exceptions (FR-045, FR-058)

```bash
manga-text-seg translate --config configs/translation.json --run det-a --stage render
manga-text-seg translate --config configs/translation.json --run det-b --stage render
```

Expected: extraction, OCR and rendering artifacts are **byte-identical** across the two runs.

Checks that must hold:

- The only excluded fields are elapsed-time fields (`stage_timings`, `processing_time_seconds`) and
  `translations.json`, whose determinism is the provider's property and is recorded as such in
  `run.json` rather than guaranteed (FR-058).
- No wall-clock timestamp appears in any per-page metadata field; the run ID is the only
  time-identifying value (research R8).
- `translations.json` is excluded only because the provider is nondeterministic — a provider that
  declares itself deterministic is compared across the whole tree.

## 10. Secrets, ground truth and no-vendoring (FR-035, FR-008, SC-005)

```bash
grep -riE "api[_-]?key|secret|token|bearer" outputs/translation/ | grep -v secret_env_var
grep -rl "groundtruth" outputs/translation/ || echo "no ground truth in the output tree"
git status --porcelain | grep -E "\.(pth|pt|onnx|ckpt|safetensors|bin)$" || echo "no weights"
pytest -o addopts='' -q
```

Checks that must hold:

- Zero occurrences of secret material anywhere in the run tree, while `provider`/`model`/`version`
  are present — the SC-005 audit. A `reason` string names the environment variable, never its value.
- No artifact under `outputs/translation/` is derived from a ground-truth mask and none is copied
  into it (FR-008).
- `git status` shows zero model source files and zero weight files at any size (SC-013), and
  `data/no-need-to-read/` is untouched (FR-007).
- The whole suite passes with no GPU, no checkpoint, no OCR engine and no provider key present —
  FR-060's thirteen required test areas run on small fixtures: region extraction from a binary mask;
  bounding box and padding; region ordering; the OCR adapter against a fixture; the translation
  adapter against a mock provider; translation-result caching; no API-key leakage; text wrapping;
  font-size fallback; text overflow; rendering position; one region failing while others complete;
  and the `image_id`/`region_id`/OCR/translation metadata mapping (FR-060).
