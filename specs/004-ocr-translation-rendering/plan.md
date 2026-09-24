# Implementation Plan: OCR, Translation & Text Rendering

**Branch**: `004-ocr-translation-rendering` | **Date**: 2026-09-24 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/004-ocr-translation-rendering/spec.md`

## Summary

Spec 4 is the last stage of the pipeline: it turns a page that has already been segmented and had its
text removed into a page whose text has been read, translated and drawn back in. For each page it
consumes three upstream artifacts — the original page image, a prediction mask with its sidecar, and
an inpainted image with its Spec 3 metadata — extracts the text regions from the mask, crops each
region **from the original pre-inpaint image**, OCRs it, translates it through a pluggable provider,
and renders the translation onto the inpainted page. Every stage writes inspectable intermediates so
a stage can be re-run alone against the previous stage's stored output, every failure is recorded
under its category and isolated to its region, and a run is identified by a run ID that never
overwrites a previous run.

Three things are deliberately *not* built. **No segmentation model is executed, re-run, trained or
fine-tuned** — this feature contains no segmentation runtime at all (FR-005). **No ground truth is
read in the main demo path**: the prediction mask is the only mask, and ground truth exists in this
repository solely as Spec 1's evaluation input (FR-008, US1 scenario 4). **Nothing third-party is
vendored**: no model source, no weights at any size in any form, and no API key anywhere in the
repository or in any output (SC-013, FR-057). The OCR engine is a local, inference-only dependency
whose weights are downloaded at setup and never trained (FR-028); the translation provider is reached
over the network with its key supplied exclusively by an environment variable (FR-029, FR-034). The
whole feature runs on CPU (FR-062), and its default test suite runs on small fixtures with no full
dataset, no segmentation model and no real translation provider (FR-060).

## Technical Context

**Language/Version**: Python `>=3.10` (`pyproject.toml:requires-python`), developed and linted as
3.11 (`ruff target-version = "py311"`); the interpreter that produced the admitted Spec 2 drill
hand-off records `3.10.1` (`deliverables/drill/run.json`), so 3.10 is the floor this feature must
hold.

**Primary Dependencies**: Spec 1's four stay as they are — `opencv-python>=4.8.0`, `numpy>=1.24.0`,
`pandas>=2.0.0`, `matplotlib>=3.7.0` (`pyproject.toml`). Two things are added, each with a reason
that cannot be worked around:

- **`Pillow` — promoted from transitive to direct.** It is already installed (`12.1.1`) because
  `matplotlib` requires `pillow>=8`, but it is not declared. FR-039 (font family, size, colour,
  outline, word wrap, line spacing, alignment, text direction) and FR-040/FR-041 (automatic
  font-size fitting, then line count, then bounded area expansion) cannot be done with
  `cv2.putText`, which has no TrueType support, no wrapping, no fallback and no measured layout.
  Declaring it makes a dependency the code imports directly an explicit one instead of a coincidence
  of another package's dependency list.
- **`manga-ocr` — a new *optional* extra, never a base dependency.** FR-022 names it the preferred
  engine and FR-028 forbids training it. It is installed by an extra
  (`pip install -e ".[ocr]"`), imported lazily inside the OCR adapter, and **never imported by the
  default test suite** (FR-060, research R3). No base install grows.

The translation client is **stdlib `urllib.request`** — no new dependency for one HTTP POST
(research R4). `requests` and `httpx` are present in the development environment but undeclared;
adopting either would be a new dependency the task does not need.

**Storage**: Filesystem only, no database. This feature's product is
`outputs/translation/<run_id>/…` (FR-048), gitignored via `.gitignore`'s `outputs/` entry. The shared
translation cache lives at `cache/translation/`, also gitignored, and **outside any run directory**
(FR-029). The bundled font lives at `assets/fonts/`, which is **not** gitignored and is committed —
it is a build input, not a regenerable product.

**Testing**: `pytest` (`testpaths = ["tests"]`), run as `pytest -o addopts='' -q` because
`pyproject.toml` sets `--cov=` and `pytest-cov` is absent from the active environment. The suite
extends Spec 2's `tests/conftest.py` fixtures (`aligned_mask`, `two_page_manifest`,
`make_handoff`, `ALIGNED_SHAPE`) with a **translation bundle** fixture — original page + prediction
mask + sidecar + inpainted image + Spec 3 metadata at `ALIGNED_SHAPE` — so every scenario in
[quickstart.md](quickstart.md) runs with no dataset, no checkpoint, no OCR weights and no network.

**Target Platform**: CPU-only, Windows and POSIX (FR-062). No GPU is required for any part, local OCR
inference included. No part of the default suite touches the network.

**Project Type**: A single Python package extended in place — `src/` layout, console entry point
`manga-text-seg = "manga_text_seg.cli:main"`. This feature **extends the existing CLI with new
subcommands** rather than shipping a second executable (FR-053).

**Performance Goals**: No throughput target is claimed — this is a 390-page academic pipeline, not a
service. The only goal that matters operationally is that a page's cost is dominated by two things
this feature does not control: local OCR inference (the slowest local stage) and one network
round-trip **per text region** (FR-029's per-region granularity, clarified 2026-09-10). The cache
exists precisely so a repeated run does not pay the second cost again. Per-stage timings are recorded
per page (FR-047, FR-056) so the split is observable rather than assumed.

**Constraints**:

- **Blocking preconditions — and the spec's stale Assumptions text, superseded.** Spec 1 and Spec 2
  are implemented and frozen: `outputs/segmentation/default/manifest.json` holds 390 pairs, its
  per-method mask trees exist, and `deliverables/drill/` is an admitted Spec 2 hand-off. **Spec 3 is
  not implemented** — `outputs/inpainting/` does not exist. `spec.md:646–649` still reads *"As of
  2026-09-10, Specs 1–3 exist as specifications only — no manifest, no prediction masks, no
  inpainting outputs exist yet."* Two of those three clauses are false now and the third is true.
  **This plan supersedes it; no spec edit is required to proceed.** (Same disposition Spec 3's plan
  took toward its own stale assumption, and Spec 2's toward its own.) The consequence is concrete:
  FR-004's inpainting source is a *runtime* precondition, so this feature's tests must be
  fixture-only and must never require a real Spec 3 run to pass.
- **No ground truth in any hand-off or output.** The main demo path uses the prediction mask and
  nothing else (FR-008, US1 scenario 4). Ground truth is not copied into
  `outputs/translation/`, is not rendered, and is not compared against. A GT-assisted diagnostic
  mode, if one ever exists, must be a separately labelled configuration and never the default.
- **Nothing third-party is vendored.** No model source and no weights at any size in any form
  (SC-013). OCR weights arrive by first-run download into a gitignored cache (FR-028); no
  `*.pth`/`*.onnx`/`*.bin`/`*.safetensors` file is committed. The one committed binary is the
  bundled font, which is a build input under a permissive licence, not a model.
- **Upstreams are read-only.** `data/raw/…`, `data/groundtruth/…`, the Spec 1 manifest, Spec 2's
  prediction-mask tree and Spec 3's inpainting tree are opened read-only and never modified, moved,
  renamed or deleted (FR-006). Where an upstream artifact is needed inside this feature's output
  tree it is **copied, never moved**. SC-011 checks this by checksum/mtime invariance.
- **`data/no-need-to-read/` is never read** (FR-007). SC-011 asserts no file beneath it is opened.
- **`image_id`s are never changed and ground truth is never modified.** `image_id` is the composite
  `<manga>/<NNN>` the manifest already defines; this feature spells it, it does not invent it.
- **Determinism (FR-058, SC-012).** All local stages — extraction, OCR, rendering — are
  deterministic given identical inputs and configuration. That forbids a wall-clock timestamp in
  per-page metadata, forbids iterating an unordered set, and requires the region order to be derived
  from configuration rather than from a container's incidental ordering (research R8, R9).
  Translation determinism is a *provider property*, recorded rather than guaranteed (FR-058).
- **Secrets never reach a file.** The provider key comes from an environment variable, is verified
  present at startup when the configured provider needs it, and appears in no source, config, log,
  metadata or output (FR-034, FR-035, FR-057). SC-005 is an automated audit for exactly this.
- **Failures are recorded, never hidden** (FR-055). Nothing is silently corrected, substituted or
  fabricated — not an input, a region, OCR text, a translation or a rendering.
- **The output contract must be legible to Spec 5.** A consumer knowing only (run ID, segmentation
  method, `image_id`) locates and interprets the full artifact set with no index file and no
  run-specific knowledge (FR-052, SC-010) — the same addressing discipline Spec 3's
  `inpainted-output.md` fixed for this feature.
- **`outputs/` vs `deliverables/` asymmetry is deliberate.** `outputs/translation/` stays
  gitignored because it is regenerable from committed inputs and from the upstream trees; the
  bundled font under `assets/` is committed because it is an input nothing else can regenerate.

**Scale/Scope**: 390 manifest pages; 4 selectable segmentation methods (FR-009); 2 target languages
at minimum (FR-030); one run fixes exactly one (segmentation method, inpainting source, target
language) triple and a different combination requires a new run ID (FR-049). Regions per page are
typically tens, not thousands; the pipeline is per-region granular for OCR, translation and
rendering, and per-page granular for isolation.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Assessment |
|---|---|
| **I. Task List Is the Contract (Superpowers Workflow)** | PASS — this feature is planned here, then implemented worktree → TDD → subagent-driven → review → finish-branch. Nothing in the design requires re-planning after `/speckit-tasks`; the module split below is stable and each phase of the task list maps to a module. |
| **II. Spec-Scoped Test-First (NON-NEGOTIABLE)** | PASS — FR-060 enumerates thirteen test areas that are exactly this spec's acceptance criteria, so the red-green-refactor obligation is scoped by the spec rather than by speculation. FR-060's fixture-only requirement also *forbids* the tempting out-of-scope tests: nothing here may require the full dataset, a segmentation model or a real provider. |
| **III. Simplicity and Reuse (Ponytail Full Mode)** | PASS — reuse is the design: Spec 1's `imaging`, `normalize`, `align` and Spec 2's `conftest` fixtures are consumed rather than reimplemented; the four method identities and every path/parameter are configuration, not code (FR-061). Two dependencies are added, both because the requirement cannot be met without them: `Pillow` (FR-039–FR-041 need measured text layout and TrueType, which OpenCV does not provide) and `manga-ocr` as an **optional extra** behind an adapter (FR-022–FR-028 need an OCR engine; FR-060 needs the default suite to run without it). The optional external rendering module FR-044 permits is **declined** — Pillow covers it, so adopting one would be a dependency with no requirement behind it (research R7). Ponytail is not used to skip validation: the intake gate (FR-013) and the secret checks (FR-034) are built in full. |
| **Development Workflow — Worktree → TDD → Subagent-driven execution → Code review → Finish-branch** | PASS — this feature is developed on its own branch/worktree, test-first, with review before merge. Nothing in the design requires bypassing a step. |
| **Quality Gates — plan-stage ponytail review** | PASS — the plan-stage review is this section and the Complexity Tracking table below; no violation is claimed, so the table stays empty. Pre-commit, test-scope and review gates are unchanged and apply to the code this plan produces. |
| **Governance — semver, Sync Impact Report, no silent amendments** | PASS — this feature adds no constitutional principle and amends none. It is a new spec under the existing v1.0.0 constitution; no Sync Impact Report is triggered. |

*Gate status: PASS. No violations requiring justification. **Re-checked after Phase 1 design below
(same verdict).** Complexity Tracking table intentionally left empty.*

## Project Structure

### Documentation (this feature)

```text
specs/004-ocr-translation-rendering/
  plan.md              # This file (/speckit-plan command output)
  spec.md              # Feature specification (input to /speckit-plan)
  research.md          # Phase 0 output (/speckit-plan command)
  data-model.md        # Phase 1 output (/speckit-plan command)
  contracts/           # Phase 1 output (/speckit-plan command)
  quickstart.md        # Phase 1 output (/speckit-plan command)
  checklists/          # Requirements checklist (pre-existing)
  tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

Spec 1's contracts ([specs/001-data-foundation-classical-baseline/contracts/](../001-data-foundation-classical-baseline/contracts/)),
Spec 2's contracts ([specs/002-deep-learning-segmentation-benchmark/contracts/](../002-deep-learning-segmentation-benchmark/contracts/))
and Spec 3's contracts ([specs/003-text-removal-inpainting/contracts/](../003-text-removal-inpainting/contracts/))
are **reused, not restated** — this feature consumes all three. The contracts this feature adds are
the **rendered-output contract** (the tree and its addressing rule, which Spec 5 reads), the
**pipeline-metadata schema** (FR-047's record), and the **error-report contract** (FR-051's eleven
categories).

### Source Code (repository root)

```text
src/manga_text_seg/
  align.py             # unchanged (Spec 1) — reused for mask↔page alignment, FR-002
  normalize.py         # unchanged (Spec 1) — reused for mask binarisation, FR-013
  imaging.py           # unchanged (Spec 1) — load_raw_page / load_mask / save_mask, FR-046
  metrics.py           # unchanged (Spec 1) — NOT consulted: FR-036 forbids ranking segmentation by translation
  manifest.py          # unchanged (Spec 1) — the manifest FR-001 consumes as the single source of truth
  pagelist.py          # unchanged (Spec 1)
  provenance.py        # unchanged (Spec 1)
  validation.py        # unchanged (Spec 1)
  discovery.py         # unchanged (Spec 1)
  preprocess.py        # unchanged (Spec 1)
  report.py            # unchanged (Spec 1)
  visualize.py         # unchanged (Spec 1)
  availability.py      # unchanged (Spec 2) — NOT consulted (research R2)
  receipt.py           # unchanged (Spec 2) — the admission whose *result* this feature reads
  benchmark.py         # unchanged (Spec 2)
  methods/             # unchanged (Spec 1) — the classical registry; this feature never executes it (FR-005)
  adapters/            # unchanged (Spec 2) — the DL registry; this feature never executes it (FR-005)
  config.py            # EXTENDED: load_translation_config() for configs/translation.json (FR-061)
  cli.py               # EXTENDED: new subcommands (FR-053)
  translation/         # NEW PACKAGE — FR-022..FR-045, FR-046..FR-059
    __init__.py
    intake.py          # FR-001–FR-008: locate + validate a page bundle; refuse, never substitute
    regions.py         # FR-013–FR-021: components/contours, merge, area, padding, bbox+polygon,
                       #              reading order, orientation, stable region_id
    ocr.py             # FR-022–FR-028: crop from the PRE-INPAINT original; OCR adapter + lazy manga-ocr
    translate.py       # FR-029–FR-036: provider adapter, per-region request, content-addressed cache,
                       #              retry/timeout/rate-limit policy, failure policy
    render.py          # FR-037–FR-045: Pillow rendering, fit-then-overflow ladder, orientation
    pipeline.py        # FR-046, FR-050, FR-053, FR-054, FR-055: stage orchestration, per-region and
                       #              per-page isolation, single-stage re-runs
    runs.py            # FR-047–FR-049, FR-056–FR-059: run ID, output tree, metadata, config capture,
                       #              error report, no-overwrite

configs/
  default.json         # unchanged (Spec 1)
  dl.json              # unchanged (Spec 2)
  inpainting.json      # Spec 3's (not yet written) — consumed, not created here
  translation.json     # NEW  FR-061: paths, method selection, inpainting source, target language,
                       #      extraction/OCR/rendering/translation parameters, cache settings

assets/
  fonts/DejaVuSans.ttf          # NEW  FR-039: the bundled default font (Vietnamese glyph coverage)
  fonts/DejaVuSans.LICENSE.txt  # NEW  the font's licence — a committed binary needs its terms beside it

outputs/translation/   # gitignored — this feature's product, regenerable from committed inputs
  <run_id>/<segmentation_method>/<manga>/<NNN>/{crops/,ocr.json,translations.json,rendered.png,metadata.json}
cache/translation/     # gitignored — FR-029's shared content-addressed cache, outside every run tree

tests/
  conftest.py          # EXTENDED: a translation bundle fixture at ALIGNED_SHAPE (page + mask +
                       #           sidecar + inpainted image + Spec 3 metadata), no network, no weights
  unit/translation/    # EXTENDED: intake, regions, ocr, translate, render, pipeline, runs (FR-060)
  contract/translation/# EXTENDED: the rendered-output addressing contract (FR-052)
  integration/         # EXTENDED: a fixture batch incl. one region and one page failing
```

**Structure Decision**: A single package extended in place, with this feature's modules grouped in a
new `translation/` subpackage rather than as top-level modules. Specs 2 and 3 both chose top-level
modules, and Spec 3 chose that shape partly because its names did not collide. **Spec 4's do.** Spec
3 plans `intake.py`, `maskproc.py`, `inpaint.py` and `runs.py` at the top level, and this feature
needs an intake, a mask reader and a run/output module of its own; four of its seven modules would
otherwise fight Spec 3 for the same filenames in the same directory. A subpackage is the smallest
structure that lets both features land, it keeps the seven modules addressed by their stage of the
pipeline, and it matches the two subpackages this repository already has (`methods/`, `adapters/`).
`config.py` and `cli.py` are extended in place, because a second config loader and new subcommands
are the established pattern there — `load_config` and `load_dl_config` already coexist, and
`load_translation_config` is the third of the same kind.

Inside the package the split is by **stage of the pipeline**, not by type of code: `intake` (read
someone else's output and refuse it if it is wrong) → `regions` (turn a mask into addressed
regions) → `ocr` (read them) → `translate` (say them in another language) → `render` (draw them) →
`runs` (write it down attributably), with `pipeline` as the control flow that isolates failures
between those stages. The OCR and translation engines and the renderer are each reached through an
adapter (FR-022, FR-029, FR-044), so a substitute engine is a new implementation of an existing
protocol rather than a new branch in the pipeline — and so the default test suite can drive the
whole pipeline against fixtures with no engine installed at all.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|---|---|---|
| *(none)* | — | — |

Empty by design. No constitutional principle is violated, weakened or worked around. The two
dependency additions are not violations of Principle III — they are the "unless the task cannot be
completed without them" clause: text layout with TrueType, wrapping and a size-fitting ladder cannot
be written on OpenCV, and OCR cannot be written without an OCR engine. Both are contained — `Pillow`
is declared and used only inside `render.py`, and `manga-ocr` is an optional extra imported lazily
inside `ocr.py` so that no base install and no default test run ever loads it. The remaining
non-obvious decisions are recorded as research decisions (R1–R10) and none of them trades a
principle for convenience.

---

## Phase 0: Research

See [research.md](research.md). Ten decisions, each in Decision / Rationale / Alternatives considered
form. No item is left as NEEDS CLARIFICATION.

## Phase 1: Design

- [data-model.md](data-model.md) — the entities FR-047's metadata record and FR-048's tree imply, plus
  the intake-side bundle record and the cache record.
- [contracts/](contracts/) — the rendered-output contract that Spec 5 reads (FR-052), the
  pipeline-metadata schema (FR-047) and the error-report contract (FR-051).
- [quickstart.md](quickstart.md) — runnable scenarios, CPU-only, fixture-sized, no network.

## Constitution Check (post-design re-evaluation)

*Re-checked after Phase 1 design — same verdict.*

| Principle | Assessment |
|---|---|
| **I. Task List Is the Contract** | PASS — unchanged; the design artifacts are complete and consistent, and `/speckit-tasks` can consume them without further planning. |
| **II. Spec-Scoped Test-First** | PASS — unchanged; the design introduced no obligation that FR-060 does not already name, and no test obligation was dropped. The fixture bundle in `tests/conftest.py` exists so FR-060's thirteen areas can all be tested inside the spec's scope. |
| **III. Simplicity and Reuse (Ponytail Full Mode)** | PASS — unchanged. Design review confirmed the subpackage exists only to keep seven modules off Spec 3's filenames, that the adapters add no implementation with a single implementation behind them (each has a fixture implementation and a real one), that no external rendering module was adopted, and that intake validation (FR-013), the secret checks (FR-034) and the failure recording (FR-055) are implemented in full rather than reduced. |

*Gate status: PASS. Complexity Tracking remains empty.*
