# Contract: Rendered Output

**Spec**: [../spec.md](../spec.md) | **Data model**: [../data-model.md](../data-model.md) |
**Covers**: FR-046, FR-048, FR-049, FR-052 | **Consumed by**: Spec 5, and any later reader of a
completed translation run.

## Layout

```
outputs/translation/
  <run_id>/
    run.json
    errors.json
    <segmentation_method>/
      <manga>/
        <NNN>/
          crops/
            <region_id>.png
          mask.png
          ocr.json
          translations.json
          rendered.png
          metadata.json
```

`<segmentation_method>` ∈ exactly four, the FR-009 identities: `classical_baseline`,
`manga_text_segmentation`, `comic_text_detector`, `unetpp_efficientnetv2`. `<manga>` and `<NNN>` are
the two halves of the manifest's `image_id` (`<manga>/<NNN>`), used verbatim — the directory path is
the page identifier, not a transformation of it (FR-001). `<run_id>` is the run's only
time-identifying value (research R8).

## Addressing (FR-052)

A consumer knowing only **(run ID, segmentation method, `image_id`)** resolves
`outputs/translation/<run_id>/<segmentation_method>/<manga>/<NNN>/metadata.json` and its siblings.
No index file, no manifest lookup, no run-specific knowledge beyond those three values. This is the
same addressing rule Spec 3's contract fixes, and it is what makes SC-010 hold.

`metadata.json` alone is sufficient to interpret the page: `rendered.png` is the image,
`ocr.json` and `translations.json` are the stage-level detail, `crops/` holds the region crops and
`mask.png` is the prediction mask this run used (FR-046, FR-047).

## Guarantees

- **One run, one triple.** A run fixes exactly one segmentation method, one inpainting source and one
  target language (FR-049). Both of the other two axes are recorded in `run.json` and repeated in
  every page's `metadata.json`, so no page's provenance is inferable only from its path.
- **A different triple is a different run ID.** Changing any of the three never writes into an
  existing run directory.
- **No overwrite.** An existing run ID refuses to run unless overwrite was explicitly requested; a
  refusal leaves the previous run's tree untouched. Different run IDs coexist on disk.
- **Aligned space.** `rendered.png` is 1654×1170, lossless PNG, the same size as the inpainted page
  it was rendered onto. Nothing in this feature resizes an image (FR-002, FR-042).
- **Read-only upstreams.** The original image, the prediction mask, the inpainted image and every
  Spec 1–3 artifact are read and never modified, moved, renamed or deleted (FR-006). Where one is
  needed inside this tree it is **copied** — `mask.png` is the copy.
- **No ground truth.** No artifact of this tree is derived from a ground-truth mask, and no
  ground-truth mask is copied into it. The pipeline reads prediction masks only (FR-008).
- **No secrets.** No file of this tree contains an API key or any secret material; provider, model
  and version appear as plain strings (FR-035, SC-005).
- **Inspectable stages.** `ocr.json`, `translations.json` and the crops are written as their own
  artifacts, so any stage can be re-run alone against the previous stage's stored outputs (FR-050,
  US8). A re-run of one stage does not require re-running the ones before it.
- **Failures are recorded, not hidden.** A page that fails intake, OCR, translation or rendering
  still has its directory and its `metadata.json` where the run got that far, and its failure appears
  in `errors.json` under its FR-051 category. No failed input is silently corrected, substituted or
  fabricated (FR-055).

## Determinism

Given identical inputs and configuration, extraction, OCR and rendering produce byte-identical
outputs across runs, with two documented exceptions:

- `stage_timings` and `processing_time_seconds` — elapsed-time fields, excluded from byte-comparison.
- `translations.json` — the translation stage is outside the determinism guarantee, because provider
  nondeterminism is a property of the provider (FR-058). `run.json` records the provider's declared
  determinism so a comparison knows whether it may span this stage.

No wall-clock timestamp appears in any per-page metadata field (research R8). The run ID is the only
time-identifying value.
