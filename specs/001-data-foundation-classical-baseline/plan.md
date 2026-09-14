# Implementation Plan: Data Foundation & Classical Baseline

**Branch**: `001-data-foundation-classical-baseline` | **Date**: 2026-09-13 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/001-data-foundation-classical-baseline/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command; its definition describes the execution workflow.

## Summary

Build the data foundation and classical segmentation baseline for the manga text-segmentation
system. A deterministic discovery + validation pipeline indexes the existing ground-truth set
(450 masks, 45 manga) into a reusable, byte-identical manifest of 390 valid image–mask pairs
plus a categorized validation report (60 orphan masks, 0 corrupt files — nothing dropped
silently, source data never modified). A shared loader normalizes every mask to one binary
convention (background = 0, text = 255, single-channel uint8) regardless of text encoding
(magenta `(255,1,255)` / near-black `(1,1,1)` / mixed / all-white-empty), and aligns each mask
to the raw page with a lossless top-left crop to the raw native space 1654×1170 (the GT mask is
a uniform +2×+6 canvas pad with zero text pixels in the padding — verified across all 390 pairs).
On top of this foundation, a configurable classical baseline (preprocessing + Otsu, adaptive
threshold, MSER, edge detection, connected components, morphology, composable pipelines) produces
binary prediction masks per page, and a frozen metric/evaluation pipeline computes per-image IoU,
Precision, Recall, F1 (never Pixel Accuracy), persists per-image results, and emits per-method
summaries (mean/std/average time). Visualization composites (raw / GT / prediction / overlay) are
rendered for a configurable subset (`first-N` default; `best-N`/`worst-N` by F1). A stable common
method interface and output contract guarantee Spec 2 can add deep-learning models without touching
metric definitions or output format. Everything runs on CPU via a CLI driven by a single
configuration file; deterministic, read-only w.r.t. the dataset, and test-first per the
constitution.

## Technical Context

**Language/Version**: Python 3.11+ (local dev target; CPU-only). Exact interpreter version and
installed library versions to be verified at implementation start via `python --version` and
`pip list` — the plan does not assume anything beyond the five libraries the spec already lists
as available (OpenCV, NumPy, pandas, matplotlib; Pillow and SciPy also present but not required).

**Primary Dependencies**: `opencv-python` (decode, thresholding, MSER, edge, morphology,
connected components), `numpy` (array ops, metrics), `pandas` (manifest + summary tables),
`matplotlib` (visualization composites). Configuration parsed with the `json` stdlib module — no
new dependency over the spec's confirmed set (Ponytail III: no unvetted additions). CLI via
`argparse` (stdlib). Dev/test: `pytest` + `pytest-cov` (80%+ coverage gate, per testing rules).

**Storage**: Filesystem. Inputs are read-only: raw pages under
`data/raw/Manga109s_released_2026_05_21/images/<manga>/<NNN>.jpg`, ground-truth masks under
`data/groundtruth/post-processed/<manga>/<NNN>.png`. Generated artifacts: manifest + validation
report (JSON), per-image metrics (CSV), per-method summaries (JSON/CSV), prediction masks (PNG,
single-channel uint8 {0,255}), metadata sidecars (`<page>.json` per mask),
visualization composites (PNG) — all under `outputs/segmentation/<run_id>/...`. Never under
`data/`; `data/no-need-to-read/` is never read.

**Testing**: `pytest` with strict red-green-refactor (Constitution II). Fixtures are tiny
synthetic images/masks (encoded polygons) so unit tests are fast and deterministic — the 450-mask
dataset is only touched by a small integration smoke test subset and the quickstart run.
Contract tests pin the public method interface and the output schema (manifest, metrics,
summary, mask conventions); spec-scoped acceptance tests cover FR-001…FR-037 without speculative
coverage.

**Target Platform**: Local Windows dev machine, CPU only (FR-037). Output contract must remain
portable for later Colab-based specs (Spec 2 consumes the manifest, aligned space, metric module
and output format unchanged).

**Project Type**: Python package (`src/manga_text_seg/`) exposing a CLI (`python -m
manga_text_seg ...` or `manga-text-seg` entry point). No web service, no GPU runtime.

**Performance Goals**: Full sweep = 390 valid pairs × ~7 methods on CPU (FR-003/FR-026). One
image is processed at a time (memory stays bounded, ~tens of MB peak). No latency target; per-image
processing time is recorded (FR-025) and averaged in summaries (FR-030). MSER is the slowest
expected method; a single slow image must not abort the run (FR-031).

**Constraints**:
- Data safety: `data/raw/**` and `data/groundtruth/**` are read-only; no modify/move/delete
  (FR-009); prediction masks go to a separate output directory (FR-024).
- `data/no-need-to-read/` is never read (FR-010).
- Determinism: manifest ordered by (manga name, page stem); no timestamps, locale-dependent
  ordering, or filesystem-walk order leaks (FR-004 + US1-scenario-3: byte-for-byte identical
  across runs).
- Binary convention fixed: background = 0, text = 255, single-channel uint8, applied identically
  by loader, methods, metrics, visualization and the Spec-2 interface (FR-012/FR-014).
- Alignment: lossless top-left crop of GT mask to raw native size 1654×1170 by default; resize
  must NOT be the default (binary resampling artifacts). Pre/post sizes recorded (FR-015…FR-019).
- Metrics: IoU, Precision, Recall, F1 only as primary set (FR-027/FR-028). No Pixel Accuracy.
- Configuration: all paths and parameters in one configuration file; nothing hard-coded in logic
  (FR-036).
- Scope: exactly the existing benchmark (390 pairs); no new benchmark/subset (FR-003); no DL
  integration, inpainting, OCR, translation or rendering (Out of Scope).

**Scale/Scope**: 450 GT masks (45 manga × 10), 8519 raw JPGs across 87 manga folders, 390 valid
pairs in the manifest, 60 orphan masks (6 manga without source images) reported not evaluated,
~7 classical methods (preprocessing + 6 method families) over the full manifest, per-image
metrics for 390×N method outputs, visualizations for a configurable subset.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Assessment |
|---|---|
| I. Task List Is the Contract | PASS — the plan and subsequent `/speckit-tasks` task list become the binding contract; this plan defers all re-planning/scope decisions to explicit task-list amendments via the user. Design decisions recorded here (alignment, convention, method set) are the fixed inputs to tasks. |
| II. Spec-Scoped Test-First (NON-NEGOTIABLE) | PASS — testing strategy is red-green-refactor with tests traceable to FR-001…FR-037 acceptance criteria and user-story Independent Tests. No speculative tests (e.g., no DL models, no Pixel Accuracy tests). |
| III. Simplicity and Reuse (Ponytail Full Mode) | PASS — dependency set is exactly the spec's confirmed five libraries + stdlib; no new packages; JSON config via stdlib; reuse of OpenCV/NumPy instead of hand-rolled image code. Ponytail never weakens validation (FR-006…FR-008), data-loss handling (dimension check before crop in FR-016), or determinism guarantees. |
| Development Workflow (worktree → TDD → subagents → review → finish) | PASS — implementation will run in an isolated worktree/branch `001-data-foundation-classical-baseline` off `main`; reviewed against the constitution gates before finish-branch. |
| Quality gates (plan-stage ponytail; pre-commit; test-scope; review) | PASS — no gate violations at plan stage; Complexity Tracking table below is therefore empty. The only "system" is one small Python package with a single environment. |

*Gate status: PASS. No violations requiring justification. **Re-checked after Phase 1 design
below (same verdict).** Complexity Tracking table intentionally left empty.*

## Project Structure

### Documentation (this feature)

```text
specs/001-data-foundation-classical-baseline/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── contracts/           # Phase 1 output (/speckit-plan command)
└── tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)

```text
src/manga_text_seg/          # the metric & method modules + CLI (FR-034 modules)
├── __init__.py
├── config.py                # schema + loader for the single configuration file (FR-036)
├── discovery.py             # FR-001..FR-003: recursive GT discovery (manga, stem)
├── validation.py            # FR-006..FR-008: per-category anomaly report
├── manifest.py              # FR-004..FR-005: deterministic ordered manifest + persistence
├── imaging.py               # loaders: raw page (BGR) + GT mask (RGB, all channels)
├── normalize.py             # FR-011..FR-013: per-mask binary normalization (bg=0, text=255)
├── align.py                 # FR-015..FR-019: lossless crop default + configurable strategies
├── preprocess.py            # FR-020: grayscale + optional denoise (configurable)
├── methods/
│   ├── __init__.py          # method registry + common interface (FR-021..FR-026, FR-035)
│   ├── otsu.py              # global Otsu threshold
│   ├── adaptive.py          # adaptive threshold
│   ├── mser.py              # MSER regions → mask
│   ├── edges.py             # edge detection → text mask
│   ├── components.py        # connected components → text mask
│   ├── morphology.py        # dilation/erosion/open/close params
│   └── pipeline.py          # composable multi-stage pipelines (FR-020 last item)
├── metrics.py               # FR-027..FR-029: IoU/Precision/Recall/F1 per image
├── report.py                # FR-030..FR-031: per-method summaries + failure report
├── visualize.py             # FR-032..FR-033: 4-panel composites + selection modes
└── cli.py                   # command entry points (run one method / sweep / report / viz)

configs/
└── default.toml.json        # shipped default configuration (paths, alignment, methods, viz) -- renamed per config decision

tests/
├── conftest.py              # synthetic image/mask fixtures (fixed seeds, polygons)
├── unit/                    # discovery, pairing, normalization, alignment, metrics edge cases
├── integration/             # small real-data smoke (e.g., 2 mangas), end-to-end CLI run
└── contract/                # interface + schema contracts (method registry, manifest/output JSON)

outputs/                     # gitignored; runtime artifacts only (never committed)
└── segmentation/<run_id>/<method>/<manga>/<page>.png|.json + metrics/ + viz/

data/                        # read-only inputs (never written by this feature)
```

**Structure Decision**: Single project, `src`-layout Python package. Mirrors FR-034's module
separation 1:1 (discovery, validation, manifest, image loading, mask loading, normalization,
alignment, preprocessing, classical segmentation, metrics, visualization, configuration, CLI) so
each concern is a small cohesive file (<400 lines, per coding-style rules) and Spec 2 consumes
only `manifest`, `align`, `metrics` and the method interface — the rest stays internal. Methods
live in their own subpackage so registering a new method (Spec-2 adapters, FR-035, SC-008) is a
one-file, no-touch-elsewhere change. `outputs/` is gitignored; documentation of the contract is
under `contracts/` (see Phase 1).

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations. The single-package layout with a method registry adds no architectural overhead
beyond what FR-034/FR-035 mandate: modules map 1:1 to requirements, the registry is a dict of
`name -> factory(config)` with a two-method interface, and metrics/output formats are frozen plain
CSV/JSON/PNG artifacts. Empty by design.