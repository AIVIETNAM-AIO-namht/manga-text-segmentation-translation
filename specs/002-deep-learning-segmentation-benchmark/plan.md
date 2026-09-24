# Implementation Plan: Deep Learning Segmentation Benchmark

**Branch**: `002-deep-learning-segmentation-benchmark` | **Date**: 2026-09-19 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/002-deep-learning-segmentation-benchmark/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command; its definition describes the execution workflow.

## Summary

Benchmark three pretrained deep-learning text-segmentation methods against the frozen classical baseline,
without running any of them in this repository. The project **exports** a distributable page list plus the
input images it denotes (FR-054/FR-054a); each runner takes that export into their own hosted environment,
clones the method's external repository there, runs inference over all 390 pages, and hands back
prediction masks with per-page provenance. The project **validates and admits** each hand-off into a named
run (FR-058/FR-059) and computes every metric itself with the Spec 1 evaluation procedure (FR-057) — a
runner returns masks and provenance, never scores. Adapters live in this repository and drive the external
model code at a configured path and revision; no third-party source or weight file is vendored, committed
or patched in-tree (FR-055, SC-013). The design adds five modules to the existing package —
availability check, DL adapters, page-list export, receipt validation/admission, and combined benchmark
reporting — and extends the existing CLI with three subcommands. Nothing in `metrics.py`, `normalize.py`,
`align.py`, `manifest.py` or the Spec 1 method interface changes; the DL path is a new producer of the same
binary mask that `metrics.compute_metrics` already consumes, which is what makes the four metrics
comparable across methods and across the classical baseline.

## Technical Context

**Language/Version**: Python 3.10.1 (development machine, verified 2026-09-10). Runner-side interpreters
are *not* fixed here and MUST NOT be forced to match: Method A's pinned fastai 1.0.60 / torch 1.4.0 stack
caps at Python 3.7 and its checkpoints are fastai-v1 pickles, so a runner environment may legitimately be
older or newer than this machine. Every interpreter version is recorded per method from the environment
that produced that method's result (FR-044), never once for the run.

**Primary Dependencies**: The project itself adds **no** dependency. It continues on the already-present
numpy 2.2.6, opencv-python 5.0.0.93, pandas 2.3.3, matplotlib 3.10.8, Pillow 12.1.1, pytest 9.0.2
(stdlib `hashlib`, `zipfile`, `json`, `argparse`, `importlib.metadata` cover the new work). The deep-learning
runtimes — torch/torchvision/fastai for Method A, torch + the upstream geometry deps for Method B, torch +
timm for Method C — are **runner-side only** and never enter this project's environment or dependency
manifest. This is the constitution's simplicity-and-reuse justification hook exercised exactly as the spec
authorises it: spec.md states the deep-learning runtime and the third-party model repositories "MUST be
justified against the constitution's simplicity-and-reuse principle, since this feature cannot be delivered
without them." They cannot: the feature *is* the benchmark of three external pretrained models, and
re-implementing any of them would produce a different model, not the published one (FR-015 forbids training
or fine-tuning).

**Storage**: Filesystem only. Committed: `benchmark/` (distributable page list + per-page image identity,
FR-054/FR-054a), `configs/dl.json` (DL method configuration), `deliverables/<run_id>/<method>/`
(admitted masks, per-page metadata, metrics, visualisations — ONBOARDING §9). Gitignored and never
committed: `outputs/` (Spec 1 scratch), checkpoints, and any third-party source. No database, no cache
service; the existing `.gitignore` weight block (`*.pth`, `*.pt`, `*.ckpt`, `*.onnx`, `*.bin`,
`*.safetensors`, `checkpoints/`, `weights/`) already enforces FR-046/FR-055/SC-013 at the commit boundary.

**Testing**: pytest, run as `pytest -o addopts=''` (the `addopts = "--cov=..."` in `pyproject.toml`
requires an uninstalled pytest-cov). Default unit tests MUST NOT load a real large pretrained model or
download a checkpoint (FR-051): the adapter contract is exercised against a stand-in that returns a
synthetic mask, and the receipt path against synthetic hand-offs (FR-050, FR-050a). Real-checkpoint
integration tests are separated and opt-in.

**Target Platform**: Development and evaluation on this Windows 11 machine, CPU-only, which is where the
export, the receipt validation and every metric are computed. Per-method inference runs on hosted
notebooks with an accelerator (Colab), which is an environment requirement under FR-052 rather than a scope
decision — Spec 1's CPU mandate binds what *this project* executes, not the runners' inference. Device
identity is recorded per method as a specific name (`"cuda"` alone is insufficient, FR-060).

**Project Type**: Single Python package (`src/manga_text_seg/`) with a CLI, extending the existing one.

**Performance Goals**: Not a throughput feature. The binding numbers are correctness-and-cost: the export
must be reproducible byte-for-byte from the manifest; receipt validation of 390 pages must complete in
seconds so a hand-off can be admitted immediately; inference cost is the runners' and is recorded
per page (Method A measured at roughly 0.6 s/page on GPU). The plan's own budget is setup, not compute —
which is why ONBOARDING §4 directs runners to run all 390 pages immediately rather than a trial subset.

**Constraints**:
- FR-002 is a blocking precondition: Spec 1 must be implemented and frozen. **MET** — `src/manga_text_seg/`
  is complete (2010 lines), `outputs/segmentation/default/manifest.json` holds the canonical 390-pair
  manifest, and the suite passes. The spec's Assumptions section (spec.md line 871) still says Spec 1
  "exists as a specification only" and that the project "is not a git repository"; both were true when
  written and are **false now**. This plan supersedes them; no spec edit is required to proceed.
- Ground truth MUST NOT reach a runner: the distributable page list carries page identity, raw image
  reference and aligned target size, and **no** GT reference and **no** GT pixel data (FR-054a). The Spec 1
  manifest cannot be that artefact — `Manifest.to_dict()` emits `gt_root`, `raw_root` and per-pair
  `mask_path` — so the export is a derived file, not a copy of the manifest.
- No third-party code or weights in any form, any size: no vendored tree, no partial copy, no patch series
  against upstream, no single-file extract (FR-055, SC-013). The three verified upstream defects are
  therefore worked around in the runner's own environment and recorded as deviations from the pinned
  revision (FR-056) — never patched in-tree.
- Metrics are computed once, here, by `metrics.compute_metrics` (FR-057). `compute_metrics` raises on shape
  mismatch and does not resize, so FR-058's receipt validation MUST enforce mask size *before* it is called.
- Licences differ per method and constrain repair, not integration: A MIT, B GPL-3.0, C no licence at all.
  The licence identifier is recorded for code **and** weights separately in the availability report
  (FR-016), the provenance record (FR-056) and the run record (FR-044).
- Method A additionally requires the FR-013a leave-one-fold-out attribution: all five released checkpoints,
  each page scored by the fold whose training split excluded that page's book, derived from upstream's
  published seed 42 and verified. Unmappable ⇒ `unavailable` with a reason, never approximated.
- Contamination is disclosed next to every method's scores (SC-011): A is de-contaminated by LOFO, C is
  fully contaminated (its published training set is set-identical to this GT), B's extent is unknowable.

**Scale/Scope**: 390 page pairs across 39 manga, 3 deep-learning methods plus the 6-configuration classical
baseline, 67 functional requirements, 14 success criteria, 24 edge cases. 5 new modules, 3 new CLI
subcommands, ~13 new test modules. Zero new runtime dependencies, zero third-party files in the tree.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Assessment |
|---|---|
| I. Task List Is the Contract (Superpowers Workflow) | PASS — this plan fixes scope and the task list derived from it is binding; implementation agents MUST NOT re-plan, re-scope or rewrite. One precondition is recorded rather than absorbed: the spec's Assumptions section is stale (spec.md line 871) and is superseded here explicitly, not silently. If FR-013a's fold mapping proves irreproducible for Method A, the correct action is to stop and surface it as `unavailable` — which the spec itself already mandates — not to substitute a single checkpoint. |
| II. Spec-Scoped Test-First (NON-NEGOTIABLE) | PASS — tests are written before implementation for every new module, strictly scoped to this spec's acceptance criteria: FR-050's adapter contract tests (synthetic output, no checkpoint download), FR-050a's receipt-path tests (stale page-list rejected, mismatched image identity rejected, missing FR-018b evidence refused, wrong size/value-set refused with a named reason and never silently resized, admitting a third result leaves the first two byte-identical, and a synthetic result's metric equals the same mask through the adapter path). No speculative tests, no framework beyond the existing pytest. |
| III. Simplicity and Reuse (Ponytail Full Mode) | PASS — reuse is the design: `metrics.compute_metrics` (FR-003/FR-057), `normalize.normalize_mask`, `align.align`, `imaging.save_mask`, `manifest.load_manifest`, `report` writers and the Spec 1 method interface are all consumed unchanged. The DL runtime is not a dependency this project takes on — it is deliberately kept out of the project environment entirely (FR-052), which is *less* machinery than integrating it. Five new modules map 1:1 to requirement groups; the adapter registry is the existing `methods/` dict-of-`name -> factory` pattern extended, not a new framework. No new dependency is added. Validation, licence recording and refusal paths are strengthened, never weakened. |
| Development Workflow (worktree → TDD → subagents → review → finish) | PASS — a git repository now exists (Spec 1's plan noted it did not); work proceeds on a feature branch with TDD per task, review before finish. |
| Quality gates (plan-stage ponytail; pre-commit; test-scope; review) | PASS — no gate violations at plan stage; Complexity Tracking table below is therefore empty. |

*Gate status: PASS. No violations requiring justification. **Re-checked after Phase 1 design below (same
verdict).** Complexity Tracking table intentionally left empty.*

## Project Structure

### Documentation (this feature)

```text
specs/002-deep-learning-segmentation-benchmark/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── contracts/           # Phase 1 output (/speckit-plan command)
│   ├── index.md
│   ├── page-list.schema.json      # FR-054 / FR-054a distributable export
│   ├── returned-result.md         # FR-058 receipt validation + admission
│   └── provenance.schema.json     # FR-056 / FR-018b
└── tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

Spec 1's frozen contracts are **reused, not restated**: `specs/001-data-foundation-classical-baseline/contracts/`
`method-interface.md` (the `segment(image) -> mask` interface every adapter satisfies, FR-008),
`manifest.schema.json` (the internal manifest, FR-001), `metrics.schema.json` (the metric record shape
FR-057 produces) and `output-layout.md` (the mask sidecar convention the DL path extends).

### Source Code (repository root)

```text
src/manga_text_seg/
├── __init__.py
├── align.py             # unchanged (Spec 1)
├── cli.py               # EXTENDED: + export, admit, status (FR-049)
├── config.py            # EXTENDED: DL method entries + paths outside source (FR-046)
├── discovery.py         # unchanged
├── imaging.py           # unchanged
├── manifest.py          # unchanged
├── metrics.py           # unchanged — FR-057's single evaluation procedure
├── normalize.py         # unchanged
├── preprocess.py        # unchanged
├── report.py            # EXTENDED: comparison table + charts across methods (FR-037/038)
├── run.py               # unchanged (classical path)
├── validation.py        # unchanged
├── visualize.py         # EXTENDED: 4-panel cases over admitted DL masks (FR-040/041)
├── availability.py      # NEW  FR-016–FR-022 availability check + verdict store
├── pagelist.py          # NEW  FR-054/FR-054a export page list + image identity
├── receipt.py           # NEW  FR-058/FR-059 validate, refuse, admit into a run
├── provenance.py        # NEW  FR-056/FR-018b provenance schema + completeness check
├── benchmark.py         # NEW  FR-034/FR-037/FR-059 combined run + awaited methods
├── adapters/
│   ├── __init__.py      # registry: name -> factory (same pattern as methods/)
│   ├── base.py          # DLMethodAdapter: segment(image) -> uint8 {0,255}; FR-008–FR-011
│   ├── manga_text_segmentation.py   # Method A  (LOFO, FR-013a)
│   ├── comic_text_detector.py       # Method B  (segmentation head only, FR-013)
│   └── unetpp_efficientnetv2.py     # Method C
└── methods/             # unchanged (classical registry, Spec 1)

benchmark/                            # COMMITTED — the distributable export (FR-054/054a)
├── page-list.json                    # identity, ordered page ids, raw ref, aligned size
├── image-identity.json               # per-page sha256 of the exact bytes handed out
└── dist/                             # gitignored: the image bundle produced by `export`

deliverables/<run_id>/<method>/       # COMMITTED — admitted results (ONBOARDING §9, FR-043)
├── masks/<manga>/<page_id>.png
├── metadata/<manga>/<page_id>.json
├── metrics/per-page.csv, summary.json
├── visualizations/
└── errors.json

configs/
├── default.json                      # unchanged (classical)
└── dl.json                           # NEW: per-method repo path, revision, checkpoint, device

tests/
├── conftest.py
├── contract/                         # schema + interface contracts (Spec 1 + this feature)
├── unit/                             # adapter/availability/provenance units, synthetic only
└── integration/                      # opt-in: real checkpoints (FR-051)
```

**Structure Decision**: Single-package layout, extending the Spec 1 tree in place rather than adding a
second package or a service boundary — the DL path produces the same binary mask the classical path
produces and is scored by the same module, so a package split would buy nothing and cost a distribution
problem. Five new top-level modules under `src/manga_text_seg/` map 1:1 to requirement groups
(availability, export, receipt, provenance, benchmark); `adapters/` mirrors the existing `methods/`
registry pattern. Two destinations are added to the repository root for a reason that is not cosmetic:
**`benchmark/` is committed** because the export is the artefact three external runners depend on and
cannot regenerate, and **`deliverables/` is committed** because a DL mask is *not* reproducible here —
it needs a third-party repository, a checkpoint and a GPU that this machine does not have, so unlike the
classical baseline's `outputs/` it cannot be regenerated from the repo and must be stored. `outputs/`
stays gitignored for the classical path exactly as Spec 1 left it. The image bundle is ~138 MiB for the
390 paired pages (measured: 144,364,052 bytes, largest single file 627 KB); the page list and the per-page
identity file are committed and small, while the bundle bytes go to `benchmark/dist/` (gitignored, matching
the existing `*.zip` rule) for hand-off — the committed identity record is what makes any copy of those
bytes verifiable on receipt, which is what FR-054a actually requires.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations. The five new modules each exist because a distinct requirement group cannot be satisfied
without them: `availability.py` because a verdict is per-environment and must be storable without
re-running (FR-017/FR-022), `pagelist.py` because the distributable export is provably not the manifest
(FR-054), `receipt.py` because validation must happen before any metric and must never silently repair
(FR-058), `provenance.py` because completeness is machine-checked and refusal must name the missing field
(FR-056/SC-014), and `benchmark.py` because a run must remain reportable while methods are outstanding
(FR-059). The adapter layer is the existing registry pattern with a second interface, not a new
abstraction: one base class, three concrete adapters, one dict. No new dependency, no database, no service,
no framework. Empty by design.
