---

description: "Task list for Text Removal & Image Inpainting"
---

# Tasks: Text Removal & Image Inpainting

**Input**: Design documents from `/specs/003-text-removal-inpainting/`

**Prerequisites**: [plan.md](plan.md) (required), [spec.md](spec.md) (required for user stories),
[research.md](research.md), [data-model.md](data-model.md), [contracts/](contracts/), [quickstart.md](quickstart.md)

**Tests**: **Required for this feature.** FR-037 enumerates nine test obligations and the project
constitution's Principle II is *Spec-Scoped Test-First (NON-NEGOTIABLE)*. Every test task below is
written before the implementation it covers and MUST fail first. Run the suite with
`pytest -o addopts='' -q` — `pyproject.toml` sets `--cov=` and `pytest-cov` is not installed.

**Organization**: Tasks are grouped by user story so each story is independently implementable and
testable. Story numbering follows **spec.md** (US3 = batch runs, US4 = boards, US5 = qualitative,
US6 = ablation). `quickstart.md`'s scenario headers use a different mapping — spec.md wins.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

Single project with `src/` layout (plan.md "Source Code"): source in `src/manga_text_seg/`, tests in
`tests/{contract,unit,integration}/`, configuration in `configs/`, gitignored product in
`outputs/inpainting/`.

**Name collision to respect throughout**: `src/manga_text_seg/run.py` (singular) is **Spec 2's
existing run-record module** and must not be touched. This feature's new module is
`src/manga_text_seg/runs.py` (**plural**). Every task below spells the path in full.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: The configuration file, the gitignore confirmation and the shared test fixture.

- [ ] T001 Create `configs/inpainting.json` (FR-038): the four method identities and their input roots per research R1 — `classical_baseline` → `outputs/segmentation/default/adaptive/`, the three DL identities → `deliverables/<run_id>/<method>/` — plus the shared mask-processing config (dilation enabled, `ellipse`, `[3, 3]`, `1` iteration), `inpaint.radius` = 3, `inpaint.algorithms` = `["telea", "ns"]`, `inpaint.output_format` = `"png"`, `selection.n` = 5, `selection.manual_artifact_flags` = `[]`, and the output root `outputs/inpainting/`. No path or parameter may live in code (FR-038).
- [ ] T002 [P] Confirm `outputs/inpainting/` is covered by `.gitignore` (research R6 — `.gitignore` line 23 is `outputs/`) and that `configs/inpainting.json` is tracked. Record the result; no `.gitignore` change is expected.
- [ ] T003 [P] Extend `tests/conftest.py` with an `inpaint_fixture` factory: given a method name and an `image_id`, it writes a page image, a prediction mask at `ALIGNED_SHAPE` (`(1170, 1654)`, `tests/conftest.py:35`) and the mask's JSON metadata sidecar under a tmp admitted-run root at `<root>/masks/<manga>/<stem>.png` and `<root>/metadata/<manga>/<stem>.json` (contracts/intake-validation.md "Input roots"). Parameterise it so a caller can perturb mask size, pixel values, emptiness and sidecar presence.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Configuration loading and the run/output scaffolding every user story writes through.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T004 [P] Write `tests/unit/test_config_inpaint.py` (FR-038): `load_inpaint_config()` returns the four identities with their resolved roots, the shared mask-processing and inpainting configs, and `selection.n`; a missing required key raises `ConfigError`; no path is hard-coded. MUST fail before T006.
- [ ] T005 [P] Write `tests/unit/test_runs.py` (FR-026, FR-027, FR-029, SC-006): the two addressing templates resolve from `(run_id, method, algorithm, image_id)` alone with no index lookup; a repeated run ID is refused; an explicit overwrite request is honoured; two run IDs coexist untouched. MUST fail before T007–T011.
- [ ] T006 Implement `load_inpaint_config()` in `src/manga_text_seg/config.py` (FR-038), following the existing `load_config()` / `load_dl_config()` conventions in that file (a dataclass, `_require_mapping`-style validation, `_resolve` against the repo root). Expose `MaskProcessingConfig` and `InpaintingConfig` (data-model.md) and the selection `N`.
- [ ] T007 Implement run-ID resolution and output-tree addressing in `src/manga_text_seg/runs.py` (FR-026, FR-029): the paths from contracts/inpainted-output.md "Addressing" — `outputs/inpainting/<run_id>/<method>/<algorithm>/<manga>/<page_id>.png` and its `.json` sibling — plus the ablation variant `outputs/inpainting/ablation/<run_id>/…`. `<method>` is one of the four fixed identities and `<algorithm>` one of `telea` / `ns`; reject anything else.
- [ ] T008 Implement run scaffolding and the non-overwrite rule in `src/manga_text_seg/runs.py` (FR-027, SC-006): create the run tree, refuse to reuse an existing run ID unless overwrite was explicitly requested, and never touch another run's directory.
- [ ] T009 Implement the `SampleMetadata` writer in `src/manga_text_seg/runs.py` (FR-018, FR-025) emitting every field of contracts/sample-metadata.schema.json, with `additionalProperties: false` respected and the two `allOf` conditionals satisfied (`error_message` present when `status` is `failed`; `kernel_shape` / `kernel_size` / `iterations` present when dilation is enabled).
- [ ] T010 Implement the `ErrorReport` accumulator and `errors.json` writer in `src/manga_text_seg/runs.py` (FR-011, FR-012): the nine categories — missing source image; missing prediction mask; wrong image ID / unmappable or ambiguous mapping; wrong size; empty mask; non-binary mask; corrupt or undecodable image; inpainting failure; output write failure — each entry carrying sample identifier, category and reason.
- [ ] T011 Implement the `run.json` (`InpaintRun`) writer in `src/manga_text_seg/runs.py` (FR-026): run ID, the method identities processed, the shared mask-processing configuration, the shared inpainting configuration, and `page_list_identity` / `input_image_identity` quoted **verbatim** from `benchmark/` — never recomputed (FR-054a, ONBOARDING §8). Record the resolved `classical_baseline` source method and why (research R1).

**Checkpoint**: Configuration and the run scaffold exist — user story implementation can now begin.

---

## Phase 3: User Story 1 - Receive and Validate Spec 2 Prediction Masks (Priority: P1) 🎯 MVP

**Goal**: Map every prediction mask to exactly one manifest page by the composite identity rule, and validate each input on receipt — recording every rejection under a named category, never silently correcting it.

**Independent Test**: Feed a small set of prediction masks — one correctly mapped, one whose page identity mismatches the manifest, one of the wrong size, one non-binary, one with a missing source image — and assert exactly the intended ones pass, every failure is recorded under its named category, no input is resized, re-thresholded or renamed, and valid processing continues.

### Tests for User Story 1 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T012 [P] [US1] Write `tests/unit/test_intake.py` (FR-008–FR-013, SC-001): exact-name mapping to exactly one manifest page with `image_id` = `<manga>/<NNN>`; a zero-match is an image-ID mismatch and a multi-match is an ambiguous mapping; neither is resolved by similarity; each of the seven checks in contracts/intake-validation.md "Checks" produces its documented category; a missing sidecar is `missing metadata` and is never joined from elsewhere.
- [ ] T013 [P] [US1] Write `tests/unit/test_maskproc.py` (FR-014–FR-018): normalization to background 0 / text 255 with a value of 128 rejected as non-binary; pass-through unchanged at 1654×1170; lossless top-left crop from 1656×1176; any other size a validation error; pre- and post-alignment sizes recorded; dilation applied with the configured kernel shape, size and iterations, and the effective config recorded.
- [ ] T014 [US1] Write `tests/integration/test_intake_rejections.py` (US1 Independent Test, FR-010–FR-013): drive the five inputs named in the Independent Test through the intake path and assert the exact pass set, each failure's category and reason, that no rejected input is modified on disk, and that the valid samples still pass.

### Implementation for User Story 1

- [ ] T015 [US1] Implement input location in `src/manga_text_seg/intake.py` (FR-009, research R5): one code path serves all four identities, reading `<root>/masks/<manga>/<stem>.png` and `<root>/metadata/<manga>/<stem>.json` with only the root resolved from configuration. `notebooks/deliverables/` is not an intake path and `benchmark/availability.json` is not consulted — neither may be imported.
- [ ] T016 [US1] Implement mapping in `src/manga_text_seg/intake.py` (FR-008, SC-001): resolve a mask to exactly one manifest page by manga folder name plus zero-padded stem, with `image_id` = `<manga>/<NNN>`. Zero matches and multiple matches are both rejections (contracts/intake-validation.md check 3) — no similarity, no guesswork, no renumbering.
- [ ] T017 [US1] Implement the receipt checks in `src/manga_text_seg/intake.py` (FR-010, FR-012): source image existence and decodability; mask existence and decodability; identity consistency; size; binary value set after normalization; emptiness; sidecar presence — each raising a categorized, non-fatal rejection. A rejected sample produces an error-report entry and **no artifacts** (FR-013).
- [ ] T018 [US1] Implement normalization in `src/manga_text_seg/maskproc.py` (FR-014): read preserving channel information, normalize to the binary convention, and reject any value outside `{0, 255}` after normalization — never coerce, never re-threshold.
- [ ] T019 [US1] Implement alignment in `src/manga_text_seg/maskproc.py` (FR-015, FR-018): pass-through at the aligned size, lossless top-left crop from the known GT padded canvas (1656×1176), any other size a validation error; record `rule`, `pre_size` and `post_size` on every accepted sample.
- [ ] T020 [US1] Implement the emptiness check and dilation in `src/manga_text_seg/maskproc.py` (FR-016): an all-zero mask is the `empty mask` category; dilation takes kernel shape, kernel size and iteration count from configuration and records the effective values (FR-018).
- [ ] T021 [US1] Wire US1's rejections into the run's `ErrorReport` in `src/manga_text_seg/runs.py` (FR-011, FR-012) so a failing sample is recorded and the caller continues with the remaining valid samples.

**Checkpoint**: Intake is fully functional and testable on its own — every rejection is named and nothing is silently corrected.

---

## Phase 4: User Story 2 - Inpaint Text Regions with TELEA and NS (Priority: P1)

**Goal**: Normalize, align and dilate each validated mask by the one shared configuration, then produce both `INPAINT_TELEA` and `INPAINT_NS` results at the aligned page size, with a complete metadata record per sample.

**Independent Test**: Run the core processing over a small fixture and assert that, per sample, both algorithm outputs exist at exactly the aligned page size, the processed mask differs from the raw mask only by the configured dilation, all processing parameters appear in the metadata, and identical inputs with identical configuration produce byte-identical outputs.

### Tests for User Story 2 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T022 [P] [US2] Write `tests/contract/test_inpainted_output.py` (FR-025–FR-029, SC-002): the six FR-025 artifacts exist per sample — original page, raw mask copy, post-dilation mask, TELEA result, NS result, metadata JSON — at the addresses contracts/inpainted-output.md fixes; the inpainted page is exactly 1654×1170 in the configured format; `run.json`, `errors.json` and `performance.json` sit at the run root.
- [ ] T023 [P] [US2] Write `tests/contract/test_sample_metadata_schema.py` (FR-018, FR-022, FR-025, SC-004): every produced sidecar validates against `contracts/sample-metadata.schema.json` with `additionalProperties: false`, all fifteen required properties present, and the failed-status and dilation-enabled conditionals satisfied.
- [ ] T024 [P] [US2] Write `tests/unit/test_inpaint.py` (FR-019–FR-023): both algorithms run on the fixture and both outputs are the aligned size; `--algorithm telea` and `--algorithm ns` each produce only their own; radius and output format come from configuration; per-sample timing is recorded per algorithm; `cv2.inpaint` is called with the post-dilation mask and the raw copy is byte-identical to its source.
- [ ] T025 [US2] Write `tests/integration/test_inpaint_pipeline.py` (US2 Independent Test, FR-023, SC-012): the full per-sample pipeline over the fixture, asserting both outputs at the aligned size, that `001.mask.png` differs from `001.raw.png` by exactly the configured dilation and nothing else, metadata completeness, and that two consecutive runs are byte-identical apart from `processing_time_seconds` and the run ID.

### Implementation for User Story 2

- [ ] T026 [P] [US2] Implement the inpainting wrappers in `src/manga_text_seg/inpaint.py` (FR-019, FR-020): `INPAINT_TELEA` and `INPAINT_NS` over the post-dilation mask, with the inpaint radius and output format taken from `InpaintingConfig`. Both algorithms always run over the same input set in the main benchmark.
- [ ] T027 [P] [US2] Implement the artifact writers in `src/manga_text_seg/runs.py` (FR-025, contracts/inpainted-output.md "Artifact rules"): the byte-identical raw mask copy (`<page_id>.raw.png`, copied never moved, FR-006), the post-dilation mask (`<page_id>.mask.png` — Spec 5's "processed mask"), the original page copy, and the inpainted page in the configured lossless format (FR-028).
- [ ] T028 [US2] Implement `process_sample()` in `src/manga_text_seg/inpaint.py` (FR-019–FR-025): one validated sample through normalize → align → dilate → both algorithms → all six artifacts plus the `SampleMetadata` sidecar. `processing_time_seconds` is recorded per algorithm (FR-022) and is the only non-deterministic field.
- [ ] T029 [US2] Make processing deterministic in `src/manga_text_seg/inpaint.py` and `src/manga_text_seg/runs.py` (FR-023, SC-012, research R3): no wall-clock timestamp in any per-sample metadata — the run ID is the only time-identifying field — and every iteration over pages, method identities and algorithms follows a sorted, total order rather than a `set` or incidental `dict` order.
- [ ] T030 [US2] Implement per-sample timing capture in `src/manga_text_seg/inpaint.py` (FR-022), recorded separately per inpainting algorithm so FR-022's method × algorithm aggregate has its input.
- [ ] T031 [US2] Add the `inpaint` subcommand to `src/manga_text_seg/cli.py` (FR-036), following the existing subparser conventions (`_common_parser`, `cmd_*` handlers, `src/manga_text_seg/cli.py:333`): `--config`, `--run`, `--method`, `--image-id`, `--algorithm {telea,ns}`, `--overwrite`. This covers "processing a single prediction mask", "processing one segmentation method" and "running TELEA only / NS only".

**Checkpoint**: User Stories 1 and 2 both work independently — a validated mask becomes two inpainted pages with a complete metadata record.

---

## Phase 5: User Story 3 - Batch Runs, Error Isolation and Run Separation (Priority: P2)

**Goal**: Process a whole segmentation method or every prediction mask in batch, log every failure without stopping the run, and keep each run ID's outputs separate and permanent.

**Independent Test**: Run a batch containing a deliberate failure of each error category; assert the run completes over all remaining samples, the error report enumerates every induced failure by category with sample identifier and reason, a second run with a different run ID coexists with the first, and reusing an existing run ID is refused unless overwrite was explicitly requested.

### Tests for User Story 3 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T032 [P] [US3] Write `tests/integration/test_batch_isolation.py` (FR-011, FR-012, SC-005): induce each of the nine categories — the perturbation table in quickstart.md §3 — assert the batch exits 0, every other sample completes, `errors.json` records every induced failure with category, sample identifier and reason, and zero failures go unrecorded.
- [ ] T033 [P] [US3] Write `tests/integration/test_run_separation.py` (FR-027, SC-006): two run IDs both remain on disk byte-identical to their completed state; a repeated run ID refuses without `--overwrite` and proceeds with it; a refused run writes nothing.
- [ ] T034 [P] [US3] Write `tests/unit/test_performance_summary.py` (FR-022): `performance.json` groups by method × algorithm and reports, per group, mean processing time per image, sample count, failure count and runtime/device metadata.

### Implementation for User Story 3

- [ ] T035 [US3] Implement batch orchestration in `src/manga_text_seg/inpaint.py` (FR-011, FR-036): iterate the manifest's page list in its `(manga, stem)` order for one method, or all four identities, isolating each sample's failure so the batch always runs to completion.
- [ ] T036 [US3] Implement the overwrite request path in `src/manga_text_seg/runs.py` and `src/manga_text_seg/cli.py` (FR-027, SC-006): the default refuses a repeated run ID, `--overwrite` is the only way past it, and a refusal leaves the prior run untouched.
- [ ] T037 [US3] Implement the performance summary in `src/manga_text_seg/runs.py` (FR-022): aggregate the recorded per-sample timings into `performance.json` grouped by segmentation method × inpainting algorithm, with mean processing time per image, sample count, failure count and the runtime/device metadata.
- [ ] T038 [US3] Record run counts in `run.json` in `src/manga_text_seg/runs.py`: pages attempted, succeeded and failed per method × algorithm (quickstart.md §2).
- [ ] T039 [US3] Extend the `inpaint` subcommand in `src/manga_text_seg/cli.py` (FR-036) with the all-methods form ("processing all prediction masks") and ensure a run with failures still exits 0.

**Checkpoint**: User Stories 1, 2 and 3 all work independently — a full method runs to completion over real failures and its outputs persist.

---

## Phase 6: User Story 4 - Comparative Visualization Boards (Priority: P3)

**Goal**: Produce a five-panel comparison board per selected sample, with samples chosen by the five selection rules over Spec 2's per-page metrics.

**Independent Test**: Generate boards for samples selected by each selection rule and assert each board contains all five labeled panels with the correct image ID, method and configuration labels.

### Tests for User Story 4 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T040 [P] [US4] Write `tests/unit/test_selection.py` (FR-034, SC-007, research R2): each of the five rules selects at most N = 5 samples **per segmentation method**; a rule with fewer than N candidates records the shortfall and never pads; `manual_artifact_flags` selects at most N from the configured list and records the shortfall rather than inventing flags; `classical_baseline`'s metrics come from Spec 1's `metrics.csv` for the method it resolved to (research R1).
- [ ] T041 [P] [US4] Write `tests/unit/test_boards.py` (FR-033, SC-007): a board carries exactly five panels — original page, raw prediction mask, mask after dilation, TELEA result, NS result — each labeled with the segmentation method, inpainting algorithm, mask-processing configuration and `image_id`; no panel is a ground-truth mask and no panel is a prediction-vs-GT overlay.

### Implementation for User Story 4

- [ ] T042 [US4] Implement the five selection rules in `src/manga_text_seg/selection.py` (FR-034): highest IoU/F1, lowest IoU/F1, most false positives, most false negatives, and manually flagged artifact cases — N per rule per method, shortfalls recorded, and the result written to `selection.json`.
- [ ] T043 [US4] Implement board rendering in `src/manga_text_seg/boards.py` (FR-033) at `boards/<method>/<rule>/<manga>_<stem>.png`, reading only this feature's own artifacts. Do **not** reuse `src/manga_text_seg/visualize.py`'s GT-overlay colours — no ground truth appears in a board.
- [ ] T044 [US4] Add the `boards` subcommand to `src/manga_text_seg/cli.py` (FR-036) with `--config`, `--run` and `--method`.

**Checkpoint**: User Story 4 delivers reviewable boards without depending on US5 or US6.

---

## Phase 7: User Story 5 - Qualitative Report Against the Defined Criteria (Priority: P3)

**Goal**: Produce the pre-labelled qualitative scaffold over the selected samples, with every rating and observation field left empty for a human reviewer.

**Independent Test**: Generate the report scaffold over a set of samples and assert every entry carries the segmentation method, inpainting algorithm, mask-processing configuration, image ID, an empty rating field per the seven criteria, and an empty comment field — with no rating auto-filled.

### Tests for User Story 5 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T045 [P] [US5] Write `tests/unit/test_qualitative.py` (FR-032, FR-035, SC-008): every entry is labelled with segmentation method, inpainting algorithm, mask-processing configuration and `image_id`; all **seven** criteria are present — completeness of text removal, amount of missed text, amount of background over-erased, naturalness of the restored region, artifacts or noise, halo or border effects, damage to linework/texture/panel borders — each on one documented ordinal scale; every rating and observation field is empty; no PSNR, no SSIM, no synthetic clean-background ground truth and no single-score ranking appears.

### Implementation for User Story 5

- [ ] T046 [US5] Implement the scaffold generator in `src/manga_text_seg/qualitative.py` (FR-032, FR-035) producing `qualitative.md` and `qualitative.json` with identical content and identical emptiness. The system MUST NOT auto-score any criterion.
- [ ] T047 [US5] Add the `qualitative` subcommand to `src/manga_text_seg/cli.py` (FR-036) with `--config` and `--run`, and make it refuse to clobber a scaffold a reviewer has already filled in (FR-027, quickstart.md §6).

**Checkpoint**: User Story 5 delivers the evaluation deliverable, still independently testable.

---

## Phase 8: User Story 6 - Ablation Over Dilation and Radius, Separated from the Benchmark (Priority: P4)

**Goal**: Run supplementary sweeps over dilation configurations and inpaint radii in a clearly separated namespace, leaving the main benchmark strictly uniform.

**Independent Test**: Run an ablation with several dilation/radius configurations and assert each configuration's outputs live outside the main benchmark tree, each records its own configuration, and the main benchmark outputs are untouched.

### Tests for User Story 6 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T048 [P] [US6] Write `tests/unit/test_ablation.py` (FR-017, FR-021, FR-024, SC-011): each configuration's outputs land under `outputs/inpainting/ablation/<run_id>/` and never in the main tree; the ablation's `run.json` records `varied`, the `baseline_config` it is compared against, and `ablation: true`; the main benchmark's run directories are untouched.

### Implementation for User Story 6

- [ ] T049 [US6] Implement the ablation sweep in `src/manga_text_seg/ablation.py` (FR-024) over a list of dilation/radius configurations, writing into the separated namespace via T007's ablation addressing and recording each configuration.
- [ ] T050 [US6] Add the `ablate` subcommand to `src/manga_text_seg/cli.py` (FR-036) with `--config`, `--run`, `--vary` (e.g. `dilation.kernel_size`) and `--values` (e.g. `3,5,7`).
- [ ] T051 [US6] Exclude ablation outputs from the main benchmark comparison in `src/manga_text_seg/runs.py` and `src/manga_text_seg/selection.py` (FR-024, SC-011): the main report never reads `outputs/inpainting/ablation/`.

**Checkpoint**: All six user stories are independently functional.

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: The cross-story guarantees the spec states as success criteria, plus documentation.

- [ ] T052 [P] Write `tests/integration/test_source_tree_invariance.py` (FR-006, SC-010): checksum or modification-time invariance over `data/raw/…`, `data/groundtruth/…`, the Spec 1 manifest and the Spec 2 prediction-mask tree after a full run, plus an access audit proving no file under `data/no-need-to-read/` was opened (FR-007).
- [ ] T053 [P] Write `tests/integration/test_config_uniformity.py` (FR-017, FR-021, SC-003): across all four identities in a main-benchmark run, every sample's recorded mask-processing configuration and inpaint radius are identical.
- [ ] T054 [P] Write `tests/integration/test_no_ground_truth_in_artifacts.py` (FR-033, SC-008): grep a completed run directory and its boards for any ground-truth path and expect zero hits; assert no prediction-vs-GT overlay exists in any board.
- [ ] T055 [P] Write `tests/unit/test_no_vendored_models.py` (FR-055, SC-013): scan the tracked tree and assert zero segmentation-model source files and zero weight files at any size, in any form.
- [ ] T056 [P] Write `tests/integration/test_downstream_consumption.py` (FR-029, SC-009): a consumer knowing only `(run_id, method, image_id)` locates and interprets the page's inpainted image and its metadata with no index lookup and no manifest read.
- [ ] T057 [P] Write `tests/integration/test_no_metric_recomputation.py` (FR-030, FR-031): no segmentation metric is recomputed from inpainting outputs, and no PSNR, SSIM, synthetic ground truth or single-score ranking appears in any primary artefact.
- [ ] T058 Run every scenario in [quickstart.md](quickstart.md) §0–§8 against the fixture-scale setup and fix any divergence between the documented commands and the implemented CLI (FR-036).
- [ ] T059 [P] Update `README.md` with the new subcommands, the `configs/inpainting.json` contract and the `outputs/inpainting/` gitignore rationale (research R6).
- [ ] T060 [P] Verify the full suite passes CPU-only with no checkpoint and no model runtime present: `pytest -o addopts='' -q` (FR-037, FR-039, SC-012).

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately.
- **Foundational (Phase 2)**: Depends on Setup — BLOCKS all user stories.
- **User Stories (Phases 3–8)**: All depend on Foundational. US2 additionally depends on US1's `maskproc`/`intake` output; US3 depends on US2's `process_sample`; US4–US6 depend on US3's run tree. In priority order they are therefore sequential, but each remains independently testable at its own checkpoint.
- **Polish (Phase 9)**: Depends on all stories being complete.

### User Story Dependencies

- **US1 (P1)**: Starts after Foundational — no story dependencies. 🎯 MVP.
- **US2 (P1)**: Starts after Foundational, consumes US1's validated samples. Independently testable: its Independent Test runs on a fixture.
- **US3 (P2)**: Consumes US2's per-sample pipeline; adds batch behaviour, error isolation and run separation.
- **US4 (P3)**: Consumes US2's artifacts and Spec 2's per-page metrics; independent of US5/US6.
- **US5 (P3)**: Consumes US4's selection; independent of US6.
- **US6 (P4)**: Consumes US2's pipeline with varied configuration; independent of US4/US5.

### Within Each User Story

- Tests MUST be written and FAIL before implementation (Principle II, NON-NEGOTIABLE).
- Contracts and entities before services; services before the CLI wiring.
- Core implementation before integration.
- Story complete before moving to the next priority.

### Parallel Opportunities

- Setup: T002 and T003 in parallel (T001 first — the fixture and the loader both read it).
- Foundational: T004 ∥ T005 (test files), then T006 ∥ T007, then T008 ∥ T009 ∥ T010 ∥ T011 (all in `runs.py` — serialize T008–T011 if one author holds the file).
- US1: T012 ∥ T013. US2: T022 ∥ T023 ∥ T024, then T026 ∥ T027.
- US3: T032 ∥ T033 ∥ T034.
- US4: T040 ∥ T041. US6: T048 alone.
- Polish: T052 ∥ T053 ∥ T054 ∥ T055 ∥ T056 ∥ T057, then T058 ∥ T059 ∥ T060.

---

## Parallel Example: User Story 2

```bash
# Launch all US2 tests together (they MUST fail first):
Task: "Contract test for the inpainted output tree in tests/contract/test_inpainted_output.py"
Task: "Contract test for the metadata schema in tests/contract/test_sample_metadata_schema.py"
Task: "Unit test for the inpaint wrappers in tests/unit/test_inpaint.py"

# Then the two independent implementation halves:
Task: "Implement INPAINT_TELEA / INPAINT_NS wrappers in src/manga_text_seg/inpaint.py"
Task: "Implement the artifact writers in src/manga_text_seg/runs.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup.
2. Complete Phase 2: Foundational (CRITICAL — blocks all stories).
3. Complete Phase 3: User Story 1.
4. **STOP and VALIDATE**: run `tests/integration/test_intake_rejections.py` — the intake gate stands alone.
5. Demo the named-rejection behaviour if ready.

### Incremental Delivery

1. Setup + Foundational → foundation ready.
2. US1 → the intake gate (MVP) → validate independently.
3. US2 → both algorithms producing comparable pages → validate independently.
4. US3 → batch scale, error isolation, run separation → validate independently.
5. US4 → boards → US5 → the qualitative scaffold → US6 → ablation, each validated at its checkpoint.

### Parallel Team Strategy

1. The team completes Setup + Foundational together — T006–T011 all touch
   `src/manga_text_seg/runs.py`, so one author owns that file or the tasks serialize.
2. Once Foundational is done:
   - Developer A: US1 → US2 (the critical path; nothing else can start without them).
   - Developer B: prepares US4's `selection.py` against Spec 1's `metrics.csv` fixtures.
   - Developer C: prepares US6's `ablation.py` against the configured namespace.
3. US3 lands the run tree that US4–US6 read, so B and C integrate after it.

---

## Notes

- [P] tasks = different files, no dependencies.
- **[Story] label maps every task to exactly one spec.md user story**; Setup, Foundational and Polish
  phases carry no story label, as required by the format.
- `src/manga_text_seg/run.py` (singular, Spec 2) is untouched; this feature's module is
  `src/manga_text_seg/runs.py` (plural). Keep them distinct in every import and every test.
- FR-054a / ONBOARDING §8 hold throughout: **no ground truth may reach a runner**, and
  `page_list_identity` / `input_image_identity` are quoted verbatim, never recomputed.
- Nothing third-party is vendored — no model source, no weights, at any size, in any form (FR-055).
- Verify tests fail before implementing; commit after each task or logical group; stop at any
  checkpoint to validate a story independently.
- Avoid: vague tasks, same-file conflicts, cross-story dependencies that break independence.
