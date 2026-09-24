# Contract: Error Report

**Spec**: [../spec.md](../spec.md) | **Data model**: [../data-model.md](../data-model.md) |
**Covers**: FR-051, FR-055, FR-021, FR-026, FR-033, FR-041 | **Consumed by**: Spec 5's reporting
surface, and anyone diagnosing a run.

## Location and shape

Written once per run at `outputs/translation/<run_id>/errors.json`.

```json
{
  "run_id": "…",
  "counts": { "<category>": 0, "…": 0 },
  "categories": {
    "<category>": [
      {
        "image_id": "<manga>/<NNN>",
        "region_id": 7,
        "stage": "ocr",
        "reason": "the specific reason, not a generic message"
      }
    ]
  }
}
```

`region_id` is `null` for a page-level occurrence. `stage` is one of `intake`, `regions`, `ocr`,
`translate`, `render`, `write`.

## The eleven categories (FR-051)

Every category is present in `counts` and in `categories` **even at zero occurrences**, so an empty
category is distinguishable from an unhandled one. The keys are these, verbatim:

| Key | Level | Raised when |
|-----|-------|-------------|
| `missing_original_image` | page | the manifest's raw page is absent or unreadable |
| `missing_prediction_mask` | page | the Spec 2 mask for the selected method is absent |
| `missing_inpainted_image` | page | the Spec 3 `<NNN>.png` for the selected run and algorithm is absent |
| `empty_mask` | page | the prediction mask is present but has no text pixels |
| `no_text_region_detected` | page | the mask has text pixels but no region survives the FR-016 filters |
| `ocr_failure` | region | the OCR engine failed, returned nothing usable, or is not installed |
| `translation_failure` | region | the provider failed after the FR-032 policy was exhausted |
| `font_missing` | page | the configured font path is unreadable and no fallback resolves |
| `text_overflow` | region | FR-041's ladder reached its last rung and the text still does not fit |
| `api_timeout_or_rate_limit` | region | the provider timed out or rate-limited, recorded alongside the retry that followed |
| `output_write_failure` | page | a file of the output tree could not be written |

## What each category means for the run

- **A page-level category does not stop the batch.** One page's intake failure is recorded and the
  remaining pages continue (US6). `run.json`'s `page_statuses` carries the same fact in summary form.
- **A region-level category does not stop the page.** One region's OCR failure leaves the page's
  other regions to be OCR'd, translated and rendered (FR-026).
- **`no_text_region_detected` and `empty_mask` are outcomes, not errors.** Both are recorded so the
  page is accounted for, and neither makes the run fail (FR-021). They are in this report precisely
  so that "the page produced nothing" is a recorded fact rather than a silence.
- **`text_overflow` is a warning.** FR-041 prohibits silent clipping, so a region that does not fit
  ends here with its reason rather than as pixels outside its display area (FR-041, SC-006). The
  region's `render_status` in `metadata.json` is `overflow_warning` and it also carries a
  `render_warning`.
- **`translation_failure` records the policy that was applied.** The region's `failure_policy` in
  `metadata.json` says which FR-033 branch ran — the Japanese text carried forward with the region
  flagged, or the region excluded from rendering. The failure is recorded either way (FR-033).
- **Nothing here is a silent correction.** No failed input, region, OCR text, translation or
  rendering is corrected, substituted or fabricated anywhere in this feature; a failure is recorded
  and the pipeline moves on (FR-055).

## Audit obligations

- SC-005: an automated audit over a completed run's outputs and logs finds **zero** occurrences of
  API keys or secret material, while provider/model/version fields are present. This file is part of
  that audit's scope — `reason` strings name the environment variable, never its value.
- SC-001: every unmappable, ambiguous or incomplete page bundle is recorded here with its category.
  Zero silent substitutions.
