# `deliverables/` — admitted results

Where a returned hand-off lands **after** the project has validated it and scored it. Written only by
admission — nothing here is placed by hand:

```bash
manga-text-seg admit --config configs/dl.json --run <run_id> --method <name> <hand-off>/
```

A hand-off that fails any of the six checks in
[`contracts/returned-result.md`](../specs/002-deep-learning-segmentation-benchmark/contracts/returned-result.md)
is refused whole, names the field that failed, and writes **nothing** — already-admitted methods stay
byte-identical and the run remains reportable over them (FR-058, FR-059).

## Layout

```text
deliverables/
  <run_id>/
    <method>/
      masks/<manga>/<page_id>.png     # aligned size, FR-004 convention, values {0, 255}
      metadata/<manga>/<page_id>.json # the runner's sidecar, unmodified
      provenance.json                 # the runner's provenance record, unmodified
      errors.json                     # pages that failed inside this method
      metrics/per-page.csv            # computed HERE, by this project (FR-057)
      metrics/summary.json
      visualizations/
```

The run scope matters: two runs of the same method are two results, and the run is what names which.
`<method>/<manga>/<page_id>.png` is the mapping Spec 003 consumes (FR-043).

## Rules

- **Masks are committed.** A mask left on Google Drive does not count (ONBOARDING §9). Single-channel
  binary PNG compresses well — 390 pages is a few MB.
- **Metrics are computed here, not submitted.** A runner returns prediction masks and provenance, not
  scores. `metrics_per_page.csv` / `metrics_summary.json` from a runner is ignored, not read (FR-057).
- **No ground truth in a hand-off.** No GT mask, no prediction-vs-GT overlay (ONBOARDING §8).
- **Nothing third-party is vendored here** — no model source, no weights, at any size, in any form
  (FR-055, SC-013). Per method: configuration, returned standardised results, documentation.
- **Timing is carried, not compared.** Each page keeps its producing `device.name`; no report ranks
  timing across methods (FR-060).

## Note on `notebooks/deliverables/`

The runners' own working output lives at `notebooks/deliverables/<method>/`. That is their staging area
and their notebook deliverable. This directory is the project's admitted copy, written by `admit`.
