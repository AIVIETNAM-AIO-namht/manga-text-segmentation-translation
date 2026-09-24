# Contract: Returned Result + Receipt Validation

**Normative for**: FR-053, FR-057, FR-058, FR-059, FR-050a. This is the procedure the project applies
to a hand-off. It is code-normative, like Spec 1's `method-interface.md`, not a file schema — the two
file schemas it consumes are [provenance.schema.json](provenance.schema.json) and
[page-list.schema.json](page-list.schema.json).

A runner returns **prediction masks and provenance, not scores** (FR-057). No `metrics_per_page.csv`
and no `metrics_summary.json` crosses this boundary; anything of that shape is ignored, not read.

## What a hand-off contains

```text
<hand-off>/
├── masks/<manga>/<stem>.png    # one binary mask per page, aligned size, FR-004 convention
├── metadata/<manga>/<stem>.json# one sidecar per page (ONBOARDING §6, PageMetadata)
├── provenance.json             # one per hand-off (provenance.schema.json)
└── errors.json                 # pages that failed inside this method, with reasons
```

Deliberately absent: any ground-truth mask, any prediction-vs-ground-truth overlay, any metric
(ONBOARDING §8, FR-057).

## Validation order (FR-058)

Six checks, in this order, **before any metric is computed**. The first failure refuses the whole
hand-off, names the field or condition that failed, and writes **nothing** into the run.

| # | Check | Fails when |
|---|---|---|
| 1 | `page_list_identity` | absent, or not equal to the identity of the export currently committed |
| 2 | `input_image_identity` per page | absent, or not equal to the value the committed identity record holds for that `image_id` |
| 3 | mask exists, single-channel, `shape == aligned_size` | any page's mask is missing, multi-channel, or not `1654×1170` |
| 4 | mask values ⊆ `{0, 255}` | any page's mask carries another value, or is not uint8 |
| 5 | provenance complete | any required field of `provenance.schema.json` is absent |
| 6 | `fold_attribution` (Method A only) | absent, or naming a fold whose training split contained the page's book (FR-013a) |

Order is not arbitrary. Checks 1 and 2 establish that the result answers **this** page list applied to
**these** images; a result produced against a stale list would otherwise be alignable by name alone
(research.md R8). Checks 3 and 4 must precede metric computation because
`metrics.compute_metrics` does not resize — it raises
`MetricError("Shape mismatch: gt … vs prediction …; align first")`, which would surface a
size defect as a crash rather than as the named refusal FR-058 requires. Check 6 is last because it
applies to one method and needs the page identity resolved by checks 1–2.

## Rules

1. **Nothing is silently corrected.** A refused result is never resized, re-thresholded, re-binarised,
   cropped, padded, or dropped page-by-page. The refusal is total and it is recorded.
2. **Refusal names the field.** `missing_field` carries the schema path for check 5, and the failing
   `image_id` for checks 2–4, so the runner can fix the exact thing.
3. **A refusal does not damage the run.** Nothing is written; already-admitted methods stay
   byte-identical; the run remains reportable over them (FR-059, SC-014).
4. **Admission is the only writer.** On a full pass: masks are copied to
   `deliverables/<run_id>/<method>/masks/<manga>/<stem>.png`, sidecars to
   `deliverables/<run_id>/<method>/metadata/`, provenance to
   `deliverables/<run_id>/<method>/provenance.json`, then — and only then —
   `metrics.compute_metrics(gt, prediction)` runs per page, in the project, on the project's machine.
5. **Timing is carried, not compared.** Each admitted page keeps its sidecar's
   `device: {type, name}`; no report ranks timing across methods (FR-060).
6. **A method that did not run produces no row with numbers.** It appears as "not run" with its reason
   (FR-036). No placeholder, no imputation.
7. **Deviations are recorded, not absorbed.** A hand-off whose provenance declares a deviation that
   `changes_output` is a **different method** and is not admissible as the pinned one (FR-055).

## Test obligations (FR-050a)

Six behaviours, all reachable with synthetic results — no checkpoint, no download:

1. A result produced against a stale or different page list is rejected, not aligned by name.
2. A result whose input-image identity does not match the distributed images is rejected.
3. A result whose provenance lacks checkpoint-load evidence (FR-018b) is refused.
4. A result of the wrong mask size or value set is refused with a named reason and is not resized.
5. Admitting a third result after two are already admitted leaves the first two byte-identical.
6. A metric computed from a synthetic result equals the same metric computed from the same mask fed
   through the adapter path — both paths meet at one evaluation procedure.
