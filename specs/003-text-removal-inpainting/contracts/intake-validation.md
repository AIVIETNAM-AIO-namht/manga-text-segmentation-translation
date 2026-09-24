# Contract: Intake Validation + Rejection

**Normative for**: FR-008–FR-015. Also fixes what this feature does **not** consult (FR-029's
boundary with Spec 2).

This project **re-verifies** a hand-off rather than trusting it. Every mask is re-checked for the
`{0, 255}` convention, re-aligned to the page and re-measured, so a hand-off admitted under a
different rule cannot silently propagate into an inpainted image. Nothing here replaces or re-runs
`receipt.validate_handoff` (Spec 2, FR-058) — that gate has already run. This contract describes the
checks this feature performs on what came through it.

## Input roots

| Identity | Root |
|---|---|
| `classical_baseline` | `outputs/segmentation/default/<spec1_method>/` — Spec 1's own masks |
| `manga_text_segmentation` | `deliverables/<run_id>/manga_text_segmentation/` |
| `comic_text_detector` | `deliverables/<run_id>/comic_text_detector/` |
| `unetpp_efficientnetv2` | `deliverables/<run_id>/unetpp_efficientnetv2/` |

Masks are read from `<root>/masks/<manga>/<stem>.png` and their sidecars from
`<root>/metadata/<manga>/<stem>.json`. One intake code path serves all four identities; only the root
differs, and it is resolved by configuration (research R1).

**`notebooks/deliverables/` is not an intake path.** It is pre-admission and may hold refused results
— REISSUE.md's three hand-offs are there now. Reading it would process material the admission gate
rejected (research R5).

**`benchmark/availability.json` is not consulted.** Availability answers "can *this* machine run the
model". This feature never runs a model: inference happened on the runner's machine and the result is
already on disk. A feature that refused to inpaint a mask because its own machine lacks a checkpoint
would refuse exactly the results that are admissible (research R5, REISSUE §7).

## Checks

Performed on receipt. A failure rejects the sample with its category and reason (FR-010, FR-012) and
**never stops the batch** (FR-011).

| # | Check | On failure |
|---|---|---|
| 1 | The source page image exists and decodes | missing source image / corrupt or undecodable image |
| 2 | The prediction mask exists and decodes | missing prediction mask / corrupt or undecodable image |
| 3 | The mask maps to exactly one manifest page: manga folder plus zero-padded stem, `image_id` = `<manga>/<NNN>` | zero matches → image-ID mismatch; more than one → ambiguous mapping |
| 4 | The raw size is the aligned size, or the known GT padded canvas size | wrong size |
| 5 | Normalized values are within `{0, 255}` | non-binary mask |
| 6 | The mask contains at least one text pixel | empty mask |
| 7 | The sidecar is present and joins the mask to its `image_id` | missing metadata |

Check 3 is resolved **only** by exact name. A zero-match and a multi-match are both rejections, and
neither may be resolved by similarity or guesswork (FR-008, SC-001). A mask without its sidecar is an
invalid input, recorded as missing metadata, and is **never silently joined from elsewhere** (FR-009).

## Mask normalization and alignment (FR-014, FR-015)

- Read in a mode that preserves channel information.
- Normalize to background `0`, text `255`. A value outside `{0, 255}` after normalization is a
  rejection, never a coercion.
- At the aligned size (1654×1170): **pass through unchanged**.
- At the known GT padded size (the multiple-of-8 canvas, 1656×1176): **crop top-left, losslessly**.
- Any other size: a validation error.
- Pre- and post-alignment sizes are recorded on every accepted sample (FR-018).

## What is never done to a rejected input (FR-013)

No silent resize, re-threshold, re-crop, rename or re-map. A rejected sample produces an error-report
entry and no artifacts. Correcting it is the runner's job, not this project's.

## What is never produced

- No ground truth is written, copied or drawn — not into the output tree, not into a board, not into
  a report (FR-033, REISSUE §6). Ground truth is read only by this project's own metric code.
- No segmentation metric is recomputed from inpainting outputs (FR-030).
- No PSNR, no SSIM, no synthetic clean-background ground truth, no single-score ranking (FR-031).
