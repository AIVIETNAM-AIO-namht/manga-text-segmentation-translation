# Quickstart: Text Removal & Image Inpainting

**Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md) | **Config**: `configs/inpainting.json`

Every scenario below is runnable on a CPU-only machine with **no checkpoint, no GPU and no model
runtime of any kind**: `cv2.inpaint` is the whole algorithm. Nothing here downloads anything. The
inputs are masks that were inferred elsewhere and handed back — this project never runs a
segmentation model (FR-039, FR-055).

## 0. Prerequisites

- Spec 1 complete: `outputs/segmentation/default/manifest.json` exists with 390 pairs, and
  `outputs/segmentation/default/<method>/masks/` holds its prediction masks (FR-001).
- Spec 2 complete: at least one admitted run under `deliverables/<run_id>/<method>/`, carrying
  `masks/`, `metadata/` and `provenance.json` (FR-002). `deliverables/drill/standin/` is enough to
  exercise every scenario here.
- `configs/inpainting.json` — paths, the four method identities and their source roots, dilation,
  radius, algorithms, output format and the selection N (FR-038).

```bash
python --version && pip list     # versions recorded, not assumed
ls deliverables/                 # an admitted run must be here before anything below runs
```

## 1. Process a single prediction mask (US1, FR-019/FR-025/FR-026)

```bash
manga-text-seg inpaint --config configs/inpainting.json --run quickstart \
    --method standin --image-id ARMS/001
manga-text-seg inpaint --config configs/inpainting.json --run telea-only \
    --method standin --algorithm telea          # TELEA only; --algorithm ns for NS only
```

Expected, all under `outputs/inpainting/quickstart/`:

- `standin/telea/ARMS/001.png` and `standin/ns/ARMS/001.png` — **both** algorithms, always, over the
  same input (FR-019).
- Beside each: `001.json` (`SampleMetadata`), `001.raw.png` (the raw prediction mask copy),
  `001.mask.png` (the post-dilation mask actually handed to `cv2.inpaint`), `001.page.png` (the
  original page) — FR-025, [inpainted-output.md](contracts/inpainted-output.md).
- `run.json` recording the method source, the shared mask-processing configuration and the shared
  inpainting configuration, with `page_list_identity` and `input_image_identity` quoted **verbatim**
  from `benchmark/` — never recomputed.

Checks that must hold:

- `001.json` validates against [sample-metadata.schema.json](contracts/sample-metadata.schema.json)
  and carries all of FR-025's fields: `image_id`, method, `model_repository`, `experiment_id`,
  input image path, prediction mask path, `mask_size`, `alignment` with pre/post sizes, `dilation`
  configuration, `inpaint_algorithm`, `inpaint_radius`, `processing_time_seconds`, `status`.
- The inpainted image is exactly **1654×1170**, lossless PNG (SC-002, FR-028).
- `001.raw.png` is **byte-identical** to the admitted mask it came from — the same object, not a
  re-derivation of it.
- `001.mask.png` differs from `001.raw.png` by exactly the configured dilation and nothing else.

## 2. Process one segmentation method (US1, FR-017/FR-021)

```bash
manga-text-seg inpaint --config configs/inpainting.json --run quickstart --method standin
```

Expected: every page in the page list processed for that method, both algorithms. The mask-processing
configuration and the inpaint radius in every sidecar are **identical** — per-page and per-method
variation is prohibited in the main benchmark (FR-017, FR-021).

Checks that must hold:

- `run.json`'s `counts` equals the number of pages attempted, succeeded and failed, per
  method × algorithm.
- `performance.json` groups the recorded timings by method × algorithm with mean processing time per
  image, sample count, failure count and runtime device (FR-022).
- No ground truth appears in any artifact: not in the tree, not in a board, not in a report
  (FR-033). `grep` the run directory for a GT path and expect zero hits.

## 3. A failing sample never stops the batch (US2, FR-010–FR-013)

Perturb the fixtures one at a time and re-run scenario 2 into a fresh run ID:

| Perturbation | Rejection |
|---|---|
| source page image deleted | missing source image |
| prediction mask deleted | missing prediction mask |
| mask renamed so it maps to no page | wrong image ID / unmappable mapping |
| mask copied so two files map to one `image_id` | ambiguous mapping |
| mask resized to 1024×1024 | wrong size, **not** silently resized |
| mask with a value of 128 | non-binary mask, **not** silently re-thresholded |
| all-zero mask | empty mask |
| mask truncated mid-file | corrupt or undecodable image |
| sidecar deleted | missing metadata |

Expected in every case: the sample is rejected with its category and reason, **no artifacts are
written for it**, every other sample still processes, and the batch exits 0. `errors.json`
enumerates occurrences by all nine FR-012 categories with sample identifier, category and reason.

Checks that must hold:

- The rejected input is untouched on disk — no resize, no re-threshold, no re-crop, no rename, no
  re-map (FR-013).
- A mask without its sidecar is never silently joined from elsewhere (FR-009).
- A mask whose stem matches no manifest page is **never** resolved by similarity or guesswork
  (FR-008, SC-001).

## 4. Determinism (SC-012, FR-023)

```bash
manga-text-seg inpaint --config configs/inpainting.json --run det-a --method standin
manga-text-seg inpaint --config configs/inpainting.json --run det-b --method standin
```

Expected: the two runs' artifacts are **byte-identical** apart from `processing_time_seconds` and the
run ID itself.

Checks that must hold:

- No wall-clock timestamp appears in any per-sample metadata — the run ID is the only
  time-identifying field.
- Re-running `inpaint` twice into the same run ID refuses unless overwrite was explicitly requested
  (FR-027, SC-006), and two different run IDs coexist on disk untouched.

## 5. Comparison boards and sample selection (US3, FR-033/FR-034)

```bash
manga-text-seg boards --config configs/inpainting.json --run quickstart --method standin
```

Expected: `boards/<method>/<rule>/<manga>_<stem>.png` for each of the five selection rules, each board
**five panels** — original page, raw prediction mask, mask after dilation, TELEA result, NS result —
labelled with method, algorithm, mask-processing configuration and `image_id`. `selection.json`
records each rule's selected `image_id`s per method, N = 5 (research R2).

Checks that must hold:

- **No panel is a ground-truth mask and no panel is a prediction-vs-GT overlay** — the board shows
  inputs and outputs, never correctness (FR-033).
- `manual_artifact_flags` selects **at most** N and records the shortfall rather than inventing flags.
- A rule with fewer than N candidates records the shortfall, never padding the board set.
- `classical_baseline`'s selection metrics come from Spec 1's `metrics.csv` for the method it
  resolved to, so selection is computed against the same masks that were inpainted (research R1).

## 6. Qualitative report scaffold (US4, FR-032/FR-035)

```bash
manga-text-seg qualitative --config configs/inpainting.json --run quickstart
```

Expected: `qualitative.md` and `qualitative.json`, one entry per assessed sample, every entry
pre-labelled with segmentation method, inpainting algorithm, mask-processing configuration and
`image_id`, with the **seven** FR-032 criteria — completeness of text removal; amount of missed text;
amount of background over-erased; naturalness of the restored region; artifacts or noise; halo or
border effects; damage to linework, texture and panel borders — each on one documented ordinal scale.

Checks that must hold:

- Every rating field and every written-observation field is **empty**. No system-generated rating
  exists to be mistaken for a reviewer's (FR-032).
- No PSNR, no SSIM, no synthetic clean-background ground truth, no single-score method ranking
  anywhere in the report (FR-031, SC-008).
- Re-running into the same run ID does not overwrite a scaffold a reviewer has filled in (FR-027).

## 7. Ablation runs live apart (FR-024)

```bash
manga-text-seg ablate --config configs/inpainting.json --run dil-3x3 \
    --vary dilation.kernel_size --values 3,5,7
```

Expected: everything under `outputs/inpainting/ablation/dil-3x3/`, never in the main tree, with its
own `run.json` recording `varied` and the `baseline_config` it is compared against, and
`ablation: true`.

Checks that must hold:

- The main benchmark's run directories are untouched by the ablation (FR-017, FR-021).
- The ablation's outputs are excluded from the main comparison — the combined report does not read
  them (FR-024).

## 8. Downstream consumption and the no-vendoring check (FR-029/FR-039/FR-055)

A consumer knowing only **(run ID, segmentation method, `image_id`)** must locate and interpret a page
without any run-specific knowledge beyond that:

```bash
ls outputs/inpainting/quickstart/standin/telea/ARMS/001.png
ls outputs/inpainting/quickstart/standin/telea/ARMS/001.json
```

Expected: both paths resolve with no index lookup and no manifest read. This is the exact addressing
Spec 4's FR-012 and Spec 5's "processed mask" assumption depend on
([inpainted-output.md](contracts/inpainted-output.md)).

Finally, `git status` must show **zero** model source files and **zero** weight files in the stored
tree (SC-013), and the whole suite must pass with no GPU and no checkpoint present (FR-037, FR-039):

```bash
pytest -o addopts='' -q
```
