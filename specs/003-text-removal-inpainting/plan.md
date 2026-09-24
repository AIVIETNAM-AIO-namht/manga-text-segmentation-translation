# Implementation Plan: Text Removal & Image Inpainting

**Branch**: `003-text-removal-inpainting` | **Date**: 2026-09-24 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/003-text-removal-inpainting/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command; its definition describes the execution workflow.

## Summary

Spec 3 turns Spec 2's prediction masks into inpainted page images. It is the first feature in this
project that **consumes** another feature's outputs as its input contract rather than producing
artifacts for a runner: the intake is `deliverables/<run_id>/<method>/masks/<manga>/<page_id>.png`,
the mapping Spec 2's FR-043 promised would be addressable by page identifier and method name alone.
The feature re-verifies that promise instead of trusting it — every mask is re-checked for the
`{0, 255}` convention, re-aligned to the page, and re-measured against the ground truth by this
project's own metrics code, so a hand-off that was admitted under a different rule cannot silently
propagate into an inpainted image. Two OpenCV inpainting algorithms (`INPAINT_TELEA`, `INPAINT_NS`)
run over each mask, both always, so the comparison the feature exists to enable is available for
every page rather than for a chosen subset. Four method identities are processed into a fixed
four-directory tree (FR-026); one of them, `classical_baseline`, is a **canonical identity that this
plan defines**, because Spec 1 emitted six separate classical methods where Spec 2, Spec 4 and
Spec 5 all speak of one baseline. Evaluation is deliberately qualitative — no PSNR, no SSIM, no
synthetic clean-background ground truth, no single-score ranking (FR-031) — and is scaffolded for a
human reviewer who fills in the ratings (FR-032, FR-035). Nothing third-party is vendored, no
checkpoint is opened, no ground truth leaves this project, and the whole feature runs on CPU.

## Technical Context

**Language/Version**: Python `>=3.10` (`pyproject.toml:requires-python`), developed on 3.11
(`ruff target-version = "py311"`). Spec 1's pinned runner stacks (Method A caps at Python 3.7) are
**not** this feature's concern — Spec 3 never runs a segmentation model, only reads what the runners
returned.

**Primary Dependencies**: Exactly Spec 1's four runtime dependencies, unchanged — `opencv-python>=4.8.0`
(`cv2.inpaint`, `cv2.getStructuringElement`, `cv2.dilate`), `numpy>=1.24.0`, `pandas>=2.0.0`,
`matplotlib>=3.7.0`. **No new dependency is added.** Inpainting needs no model, no framework and no
weights: `cv2.inpaint` is the whole algorithm, which is what makes FR-039's CPU-only constraint
trivially satisfiable and FR-055's no-vendoring rule trivially observable. Dev extras unchanged
(`pytest`, `pytest-cov`, `ruff`, `black`).

**Storage**: Filesystem only, no database. Inputs are read from `deliverables/<run_id>/<method>/`
(committed, Spec 2's admitted layout — [deliverables/README.md](../../deliverables/README.md));
outputs are written under `outputs/inpainting/<run_id>/` (FR-026), which is **gitignored**
(`.gitignore:23` is `outputs/`). Page images and masks are lossless PNG by default (FR-028);
metadata is JSON sidecars beside each mask (FR-025).

**Testing**: `pytest`, `testpaths = ["tests"]`, new suites under `tests/unit/` and `tests/integration/`
following the existing `tests/{contract,unit,integration}/` split. Run with `-o addopts=''` — the
project's `addopts` requests `--cov=` and `pytest-cov` is absent from the active environment.
FR-037's default suite must complete on small fixtures with neither the full dataset nor any
segmentation model; Spec 2's `tests/conftest.py` already supplies the two-page manifest, the
synthetic binary mask at `ALIGNED_SHAPE = (1170, 1654)` and the hand-off builder (FR-050a) that this
feature's fixtures build on.

**Target Platform**: CPU-only, in-project, no GPU (FR-039), no model runtime of any kind. Reproducible
on any machine that can run Spec 1's suite.

**Project Type**: Single Python package, `src/` layout, console entry point
`manga-text-seg = "manga_text_seg.cli:main"`. Spec 3 extends the existing CLI with new
subcommands rather than creating a second executable (FR-036).

**Performance Goals**: Inpainting is a per-pixel, single-pass OpenCV operation over a 1654×1170 page;
it is not a bottleneck. The real cost is the number of `(page × method × algorithm)` combinations
processed — with four method identities, two algorithms and a 390-page list that is 3,120 inpainting
operations, each sub-second. Per-sample processing time is recorded in the metadata record (FR-025)
because the spec asks for it; no throughput target is claimed, because none is required.

**Constraints**:

- **FR-001/FR-002 are blocking preconditions — MET.** Spec 1 is implemented and frozen
  (`src/manga_text_seg/`, 3,940 lines; `outputs/segmentation/default/manifest.json`, 390 pairs) and
  Spec 2 is implemented and frozen (`receipt.py`, `benchmark.py`, `adapters/`, the
  `deliverables/<run_id>/<method>/` layout). The spec's Assumptions bullet
  (spec.md line 529, "Dependency on Spec 1 and Spec 2 (verified state)") still says both "exist as
  specifications only — no implementation, no manifest file, no prediction masks and no output tree
  exist yet". **Every clause of that bullet is false now.** This plan supersedes it; no spec edit is
  required to proceed. (Same disposition Spec 2's plan took toward its own stale assumption,
  `specs/002-deep-learning-segmentation-benchmark/plan.md` Constraints.)
- **No ground truth in a hand-off, and none in an output.** Ground truth is read only by this
  project's own metric code, exactly as in Spec 2. No deliverable of this feature contains a GT mask
  or a prediction-vs-GT overlay; the comparison boards are five panels of *inputs and outputs*, not
  of correctness (FR-033, REISSUE §6).
- **Nothing third-party is vendored** (FR-055, SC-013) — no model source, no weights, at any size, in
  any form. Spec 3 needs none, so this is a property to preserve rather than a risk to manage.
- **`image_id` is not modified and ground truth is not modified.** Spec 3 reads both and writes
  neither.
- **Determinism** (SC-012): identical inputs and identical configuration produce byte-identical
  outputs across two consecutive runs. This forbids wall-clock timestamps in metadata that is
  compared byte-wise, and forbids iteration over unordered sets.
- **`outputs/inpainting/` is gitignored** (`.gitignore:23` = `outputs/`) while `deliverables/` is
  committed. This is not cosmetic and the asymmetry is deliberate: `outputs/` is the *project's own*
  regenerable product (Spec 1's masks, Spec 3's inpainted pages — large, derivable from committed
  inputs by a committed command), whereas `deliverables/` is the *runners'* submitted evidence, which
  cannot be regenerated by this project and therefore must be committed to exist at all. FR-027's
  non-overwrite rule is what makes `outputs/inpainting/` safe to leave untracked: run IDs namespace
  every output, so a fresh clone re-derives rather than collides.
- **The output contract must be legible to Spec 4 and Spec 5.** Spec 4's FR-012 selects an inpainting
  source by "a Spec 3 run ID and its algorithm"; Spec 5's Assumptions define the "processed mask" as
  "the post-dilation mask that Spec 3 exports alongside the raw prediction mask copy". Both readings
  are fixed by FR-025's artifact list and FR-026's tree, and this plan's `data-model.md` and
  `contracts/` state them normatively so those two features can be built against them.
- **`data/no-need-to-read/` is never read** (spec.md Out of Scope).

**Scale/Scope**: Four method identities (FR-026) × two algorithms × 390 pages = 3,120 inpainted pages
per full run, plus 3,120 metadata records and 3,120 processed masks. Plus per-method comparison
boards (FR-033) over a selection of **N = 5 samples per selection rule per segmentation method**
(FR-034 — see research R2), and one qualitative report scaffold (FR-035). Ablation runs (FR-024) are
namespaced separately and excluded from the main comparison.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Assessment |
|---|---|
| **I. Task List Is the Contract (Superpowers Workflow)** | PASS — this plan's `research.md`, `data-model.md`, `contracts/` and `quickstart.md` are the inputs `/speckit-tasks` consumes; the generated `tasks.md` is the execution contract, and no work begins before it exists. The plan does not pre-empt it by naming tasks. |
| **II. Spec-Scoped Test-First (NON-NEGOTIABLE)** | PASS — FR-037 enumerates nine test obligations (mapping, normalization, alignment, dilation, output size, both algorithms on a fixture, metadata completeness, one sample failing without aborting the batch, non-overwrite of a previous run), and SC-012 adds determinism and the fixture-only default suite. Each maps to a test written before its implementation, in the existing `tests/{contract,unit,integration}/` split. |
| **III. Simplicity and Reuse (Ponytail Full Mode)** | PASS — zero new dependencies, zero new executables, zero new registries. The feature reuses `align`, `normalize_mask`, `load_gt_mask`, `compute_metrics`, `load_config` and the CLI's `_repo_root`/`_output_dir` conventions verbatim. The four-directory tree in FR-026 is data, not four code paths. No abstraction is introduced that a second caller does not already need. Ponytail does **not** license weakening input validation: FR-008–FR-013's nine error categories are implemented in full, because the intake is exactly the boundary where untrusted input arrives. |
| **Development Workflow — Worktree → TDD → Subagent-driven execution → Code review → Finish-branch** | PASS — this feature is developed on its own branch/worktree, test-first, with review before merge. Nothing in the design requires bypassing a step. |
| **Quality Gates — plan-stage ponytail review** | PASS — the plan-stage review is this section and the Complexity Tracking table below; no violation is claimed, so the table stays empty. Pre-commit, test-scope and review gates are unchanged and apply to the code this plan produces. |
| **Governance — semver, Sync Impact Report, no silent amendments** | PASS — this feature adds no constitutional principle and amends none. It is a new spec under the existing v1.0.0 constitution; no Sync Impact Report is triggered. |

*Gate status: PASS. No violations requiring justification. **Re-checked after Phase 1 design below
(same verdict).** Complexity Tracking table intentionally left empty.*

## Project Structure

### Documentation (this feature)

```text
specs/003-text-removal-inpainting/
  plan.md              # This file (/speckit-plan command output)
  spec.md              # Feature specification (input to /speckit-plan)
  research.md          # Phase 0 output (/speckit-plan command)
  data-model.md        # Phase 1 output (/speckit-plan command)
  contracts/           # Phase 1 output (/speckit-plan command)
  quickstart.md        # Phase 1 output (/speckit-plan command)
  checklists/          # Requirements checklist (pre-existing)
  tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

Spec 1's contracts ([specs/001-data-foundation-classical-baseline/contracts/](../001-data-foundation-classical-baseline/contracts/))
and Spec 2's contracts ([specs/002-deep-learning-segmentation-benchmark/contracts/](../002-deep-learning-segmentation-benchmark/contracts/))
are **reused, not restated** — this feature consumes them. The one thing this feature adds is the
**inpainted-output contract**, which is new and is written here because Spec 4 and Spec 5 will read
it.

### Source Code (repository root)

```text
src/manga_text_seg/
  align.py             # unchanged (Spec 1) — reused for mask↔page alignment, FR-016
  normalize.py         # unchanged (Spec 1) — reused for mask binarisation, FR-015
  imaging.py           # unchanged (Spec 1) — load_mask / load_gt_mask / load_raw_page / save_mask
  metrics.py           # unchanged (Spec 1) — reused for FR-034's selection metrics
  manifest.py          # unchanged (Spec 1)
  pagelist.py          # unchanged (Spec 1)
  provenance.py        # unchanged (Spec 1)
  validation.py        # unchanged (Spec 1)
  discovery.py         # unchanged (Spec 1)
  preprocess.py        # unchanged (Spec 1)
  report.py            # unchanged (Spec 1)
  visualize.py         # unchanged (Spec 1) — its GT-overlay colours are NOT used by FR-033's boards
  availability.py      # unchanged (Spec 2) — NOT consulted by this feature's intake (see research R5)
  receipt.py           # unchanged (Spec 2) — the admission procedure this feature reads the result of
  benchmark.py         # unchanged (Spec 2)
  methods/             # unchanged (Spec 1) — the classical registry; this feature does not extend it
  adapters/            # unchanged (Spec 2) — the DL registry; this feature does not extend it
  config.py            # EXTENDED: load_inpaint_config() for configs/inpainting.json (FR-038)
  intake.py            # NEW  FR-008–FR-013: locate + validate + map a Spec 2 hand-off's masks
  maskproc.py          # NEW  FR-014–FR-018: binarise, align, dilate, emptiness check
  inpaint.py           # NEW  FR-019–FR-023: INPAINT_TELEA / INPAINT_NS wrappers, radius, format
  runs.py              # NEW  FR-026–FR-030: run_id namespacing, output tree, metadata records
  selection.py         # NEW  FR-034: the five selection rules over Spec 2's per-page metrics
  boards.py            # NEW  FR-033: five-panel comparison boards (no ground truth)
  qualitative.py       # NEW  FR-032/FR-035: the pre-labelled rating scaffold, never auto-scored
  ablation.py          # NEW  FR-024: dilation/radius ablation in a separated namespace
  cli.py               # EXTENDED: new subcommands (FR-036)

configs/
  default.json         # unchanged (Spec 1)
  dl.json              # unchanged (Spec 2)
  inpainting.json      # NEW  FR-038: paths, dilation, radius, algorithms, output format, N

outputs/inpainting/    # gitignored — this feature's product, regenerable from committed inputs
  <run_id>/
    classical_baseline/{telea,ns}/
    manga_text_segmentation/{telea,ns}/
    comic_text_detector/{telea,ns}/
    unetpp_efficientnetv2/{telea,ns}/
  ablation/<run_id>/   # FR-024: separated namespace, excluded from the main comparison

tests/
  conftest.py          # EXTENDED: an inpaint fixture (page + mask + sidecar) at ALIGNED_SHAPE
  contract/            # EXTENDED: the inpainted-output contract (new file)
  unit/                # EXTENDED: intake, maskproc, inpaint, runs, selection, qualitative, ablation
  integration/         # EXTENDED: a batch run over fixtures, incl. the one-sample-fails case
```

**Structure Decision**: A single package extended in place. Twelve new modules at the top level of
`src/manga_text_seg/`, each owning one requirement group, mirroring Spec 2's own choice to add
top-level modules rather than nest packages. The split is by *stage of the pipeline*, not by *type of
code*: `intake` (read someone else's output) → `maskproc` (make it a usable mask) → `inpaint` (apply
the algorithm) → `runs` (write it down attributably), with `selection`/`boards`/`qualitative`/
`ablation` as the reporting and experimental branches. The four method identities of FR-026 are
**data**, held in `configs/inpainting.json` and iterated — deliberately not four modules, because
nothing about them differs except their name and where their masks came from. `classical_baseline` is
the one identity whose input does not come from `deliverables/` (Spec 1's masks live under
`outputs/segmentation/default/`); it is resolved by configuration, not by a branch (research R1).

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|---|---|---|
| *(none)* | — | — |

Empty by design. No constitutional principle is violated, weakened or worked around; the only
non-obvious decisions are recorded as research decisions (R1–R7) and none of them trades a principle
for convenience.

---

## Phase 0: Research

See [research.md](research.md). Seven decisions, each in Decision / Rationale / Alternatives
considered form. No item is left as NEEDS CLARIFICATION.

## Phase 1: Design

- [data-model.md](data-model.md) — the entities FR-025's metadata record and FR-026's tree imply, plus
  the intake-side records.
- [contracts/](contracts/) — the inpainted-output contract that Spec 4 (FR-012) and Spec 5 (the
  "processed mask" assumption) consume, and the intake-validation contract.
- [quickstart.md](quickstart.md) — runnable scenarios, CPU-only, fixture-sized.

## Constitution Check (post-design re-evaluation)

*Re-checked after Phase 1 design — same verdict.*

| Principle | Assessment |
|---|---|
| **I. Task List Is the Contract** | PASS — unchanged; the design artifacts are complete and consistent, and `/speckit-tasks` can consume them without further planning. |
| **II. Spec-Scoped Test-First** | PASS — unchanged; the design introduced no obligation that FR-037 does not already name, and no test obligation was dropped. |
| **III. Simplicity and Reuse (Ponytail Full Mode)** | PASS — unchanged. Design review confirmed the module split is by pipeline stage, that no new dependency or registry was introduced, and that the four method identities are configuration rather than code. Intake validation (FR-008–FR-013) is implemented in full, not reduced. |

*Gate status: PASS. Complexity Tracking remains empty.*
