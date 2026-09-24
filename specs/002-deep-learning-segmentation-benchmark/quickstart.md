# Quickstart: Deep Learning Segmentation Benchmark

**Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md) | **Config**: `configs/dl.json` (DL), `configs/default.json` (classical, unchanged)

Every scenario below is runnable on a CPU-only machine with **no checkpoint and no GPU**: the stand-in
adapter (FR-050) and synthetic hand-offs (FR-050a) are what make that true. Real inference happens in the
runners' own environments, not here.

## 0. Prerequisites

- Spec 1 complete: `outputs/segmentation/default/manifest.json` exists with 390 pairs (FR-002).
- Python 3.11+ for the project itself. The **runners'** interpreters differ per method and are recorded
  per method, not assumed (FR-044, FR-052) — Method A's pinned stack caps at Python 3.7.
- No DL runtime is required to run this quickstart. Availability verdicts for the three real methods will
  legitimately read `unavailable` on a machine without them; that verdict is true of *that machine only*
  (FR-017, FR-021).

```bash
python --version && pip list     # versions recorded, not assumed
```

## 1. Export the page list (US1, FR-054/FR-054a)

```bash
manga-text-seg export --config configs/default.json
```

Expected, all under `benchmark/`:

- `page-list.json` — 390 pages sorted `(manga, stem)`, each with `image_id`, `image_ref` and
  `input_image_identity`; plus `page_list_identity`, `aligned_size: [1654, 1170]`, `alignment` and
  `mask_convention` ([page-list.schema.json](contracts/page-list.schema.json)).
- `image-identity.json` — 390 sha256 entries, committed.
- `dist/images/` — the ~138 MiB bundle, gitignored.

Checks that must hold:

- `page-list.json` contains **no** `gt_root`, **no** `mask_path` and no ground-truth reference of any kind.
  Re-running `export` yields a byte-identical `page-list.json` and the same `page_list_identity`.
- The bundle's bytes match `image-identity.json`, and those are the same bytes Spec 1's baseline ran on.

## 2. Availability check (US2, FR-016–FR-022)

```bash
manga-text-seg status --config configs/dl.json            # pre-flight verdicts, per method
manga-text-seg status --config configs/dl.json --run drill     # admitted vs awaited for a run
```

Expected on this machine: all three `unavailable`, each naming the missing artefact, where to obtain it
and how large it is, with the remediation steps — and each verdict stamped with the environment it was
checked in. Verdicts persist, so re-running `status` retrieves them without re-checking (FR-022).

Add the checkpoint and the pinned repository outside the repo, point `configs/dl.json` at them, re-run
`status`: that one method flips to `available`. A method needing an accelerator or a legacy interpreter
this machine lacks is re-checked in a suitable environment before being called unavailable (FR-021).

## 3. Run one method end-to-end (US1, FR-050 drill)

```bash
pytest tests/contract -q          # stand-in adapter: no checkpoint, no download
manga-text-seg admit --config configs/dl.json --run <run_id> --method standin <hand-off>/
```

The stand-in returns a synthetic mask for a two-page fixture. Confirm: one binary mask per page at
`1654×1170` with values ⊆ `{0, 255}`, one sidecar per page carrying `page_list_identity` and
`input_image_identity` **verbatim from the page list**, an inference time, and a provenance record.
Confirm the metric and reporting modules were **not** modified to accept it.

`admit` is the whole DL flow: a runner infers **outside** this repository and hands the standardised
result back, so no `run` subcommand takes `configs/dl.json` — `run` drives the classical methods over
the Spec 1 manifest, and a stand-in has nothing to infer from (FR-050 keeps the stand-in to the
fixtures, FR-051 keeps real checkpoints out of the default suite).

## 4. Admit a returned result (US1/US3, FR-058/FR-059)

```bash
manga-text-seg admit --config configs/dl.json --run <run_id> --method <name> <hand-off>/
```

The six checks of [returned-result.md](contracts/returned-result.md) run in order, before any metric is
computed. Expected on a good hand-off: masks copied under `deliverables/<run_id>/<method>/`, then
`metrics.compute_metrics` scores every page **in this project**, and a `RunRecord` names the run.

Expected on a bad one — each refusal names what failed and writes nothing:

| Perturbation | Refusal |
|---|---|
| `page_list_identity` altered | stale/different list, not aligned by name (check 1) |
| one `input_image_identity` altered | that `image_id` named (check 2) |
| one mask resized to 1024×1024 | wrong size, not silently resized (check 3) |
| one mask with a value of 128 | wrong value set (check 4) |
| `checkpoint_load_evidence` removed | missing field named (check 5, SC-014) |
| Method A sidecar missing `fold_attribution` | check 6 |

Then admit a **second** method: the first method's files stay byte-identical and the run remains
reportable while the third is outstanding (FR-059).

## 5. Combined report + cases (US3/US4, FR-034/FR-037/FR-040)

```bash
manga-text-seg report    --config configs/default.json --run <run_id>   # one table, one set of charts
manga-text-seg visualize --config configs/default.json --run <run_id>   # 4-panel cases per method
```

`report`/`visualize` need the **classical** config even with `--run`: the baseline row is read from
`outputs/segmentation/metrics.csv`, which only `configs/default.json` defines. The DL config supplies
the method registry and is what `status`, `export` and `admit` take.

Expected: one row per method **including the classical baseline**, IoU/P/R/F1 with mean and std, every
timing labelled with its producing `device.name` and **no cross-device timing ranking**, contamination
disclosure beside each method's scores (SC-011), Pixel Accuracy absent as a ranking metric. A method that
did not run appears as "not run" with its reason and **no numeric placeholder**. The visualisation report
separates good from failure cases, and distinguishes "pages that failed inside a method that ran" from
"method did not run".

Finally, `git status` must show **zero** third-party model source files and **zero** weight files in the
stored tree (SC-013) — per method, only configuration, returned standardised results and documentation.

## 6. Add a new DL method (adapter drill)

1. New file `src/manga_text_seg/adapters/<name>.py` implementing `DLMethodAdapter` — it returns a mask at
   aligned size, having inverted its own padding internally (FR-024a).
2. One registry line in `adapters/__init__.py`.
3. An entry in `configs/dl.json`: repository, revision, checkpoint sha256, device, threshold, padding
   multiple, channel order, licences.
4. Contract test asserting aligned size and `{0, 255}` (the suite parametrises over the live registry, so
   it picks the new adapter up with no test edit); run `status` and `admit`.
5. Confirm `metrics.py`, `report.py` and `visualize.py` were **not touched** — no method-specific
   branching anywhere outside `adapters/` (FR-051, US1 scenario 6).
