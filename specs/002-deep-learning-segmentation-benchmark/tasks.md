# Tasks: Deep Learning Segmentation Benchmark

**Input**: Design documents from `specs/002-deep-learning-segmentation-benchmark/`
**Prerequisites**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/](contracts/), [quickstart.md](quickstart.md), `.specify/memory/constitution.md`
**Tests**: REQUIRED. The constitution's Principle II (Spec-Scoped Test-First) is NON-NEGOTIABLE, and FR-050, FR-050a and FR-051 request tests explicitly. Test tasks are written before their implementation tasks in every phase.
**Organization**: Tasks are grouped by user story so each can be implemented and tested independently.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

Single Python package, extending the Spec 1 tree in place. Source in `src/manga_text_seg/`, tests in `tests/{contract,unit,integration}/`, committed artefacts at the repository root in `benchmark/` and `deliverables/`, configuration in `configs/`.

**Every pytest invocation MUST be `pytest -o addopts=''`** — `pyproject.toml` sets `addopts = "--cov=..."` and `pytest-cov` is not installed.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Repository-level scaffolding the whole feature needs. No user story behaviour yet.

- [X] T001 Verify the FR-002 precondition and the frozen baseline: `outputs/segmentation/default/manifest.json` holds 390 pairs across 39 manga, and `pytest -o addopts='' -q` is green
- [X] T002 [P] Add `configs/dl.json` carrying per method the external repository URL, the 40-hex revision, checkpoint path and expected sha256, device, threshold, padding multiple, channel order, `code_license` and `weight_license` — `configs/default.json` stays untouched (research.md R11)
- [X] T003 [P] Create the `src/manga_text_seg/adapters/` package with a `name -> factory` registry dict in `adapters/__init__.py`, mirroring the existing `methods/__init__.py` pattern
- [X] T004 [P] Create `benchmark/README.md` and `deliverables/README.md` documenting what belongs in each committed destination (ONBOARDING §9, research.md R10)
- [X] T005 [P] Add `benchmark/dist/` to `.gitignore` and confirm the existing weight and third-party rules (`checkpoints/`, `weights/`, `*.pth`, `*.pt`, `*.ckpt`, `*.onnx`, `*.bin`, `*.safetensors`) already cover SC-013

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The shared pieces every user story consumes — the DL configuration loader, the provenance schema, the adapter interface and the synthetic fixtures.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T006 Extend `tests/conftest.py` with the fixtures every story's tests use: a two-page fixture drawn from the real manifest, a synthetic aligned-size binary mask, and a hand-off builder that writes `masks/`, `metadata/`, `provenance.json` and `errors.json` (FR-050, FR-050a)
- [X] T007 [P] Write `tests/unit/test_provenance.py`: a complete record passes; each required field of `contracts/provenance.schema.json` removed in turn is refused with that field's schema path named; a `revision` that is not 40 hex is refused (FR-056, SC-014)
- [X] T008 [P] Write `tests/unit/test_config_dl.py` for the DL configuration loader: a valid `configs/dl.json` loads, a missing required key raises `ConfigError`, and `configs/default.json` still loads unchanged
- [X] T009 Implement `src/manga_text_seg/provenance.py`: the required-field table from `contracts/provenance.schema.json` and `check_completeness(record)` returning the missing schema path, raising with that path named rather than accepting gaps (FR-056, FR-018b, SC-014)
- [X] T010 Implement the DL configuration loader in `src/manga_text_seg/config.py`, reusing the existing `_require_mapping` and `_resolve` helpers and returning a frozen dataclass in the existing style; the frozen `Config` dataclass and `configs/default.json` are not modified (research.md R11)
- [X] T011 Write `tests/contract/test_adapter_interface.py`: a registered adapter returns one single-channel uint8 mask at the aligned page size with values ⊆ {0, 255}, an inference time, and a metadata record — no checkpoint, no download (FR-008–FR-011, FR-050, FR-051)
- [X] T012 Implement `src/manga_text_seg/adapters/base.py` — `DLMethodAdapter` with `name` and `segment(image) -> np.ndarray` — and wire the registry lookup in `adapters/__init__.py`
- [X] T013 Implement `src/manga_text_seg/adapters/standin.py`: a deterministic synthetic adapter that satisfies the interface over the two-page fixture with no checkpoint and no network access (FR-050, FR-051)

**Checkpoint**: Foundation ready — user story implementation can now begin in parallel.

---

## Phase 3: User Story 1 - Benchmark one pretrained deep-learning method end-to-end (Priority: P1) 🎯 MVP

**Goal**: Export the distributable page list and the input images it denotes, run one method against it, and admit the returned hand-off into a named run where the project's own evaluation procedure scores it.

**Independent Test**: Register a stand-in method, run the benchmark for that one method, and confirm per page a binary mask at aligned size, the four metrics, an inference time and a metadata record — with the metric and reporting modules unmodified. Then express the same synthetic result as a hand-off produced "elsewhere", admit it through the receipt path, and confirm byte-identical metrics.

### Tests for User Story 1 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T014 [P] [US1] Write `tests/unit/test_pagelist.py`: `export` emits `page-list.json` with 390 pages sorted `(manga, stem)`; the file carries no `gt_root`, no `mask_path` and no ground-truth reference of any kind; re-running yields a byte-identical file and the same `page_list_identity` (FR-054, FR-054a, quickstart §1)
- [X] T015 [P] [US1] Write `tests/contract/test_page_list_schema.py`: the emitted `page-list.json` validates against `contracts/page-list.schema.json`, carries `aligned_size == [1654, 1170]`, `alignment == "crop-topleft"` and the FR-004 `mask_convention`
- [X] T016 [P] [US1] Write `tests/unit/test_receipt.py` covering the six FR-050a behaviours: a stale `page_list_identity` is refused rather than aligned by name; a mismatched `input_image_identity` is refused naming that `image_id`; a missing checkpoint-load evidence field is refused naming the field; a wrong-sized mask is refused and not resized; an out-of-convention value set is refused; and a refusal writes nothing into the run (FR-058, SC-014)
- [X] T017 [P] [US1] Write `tests/unit/test_admit_metrics_equality.py`: the metric computed from a synthetic hand-off admitted through `receipt.py` is byte-identical to the same mask fed through the adapter path — both paths meet at one evaluation procedure (FR-053(e), FR-057, research.md R12)

### Implementation for User Story 1

- [X] T018 [US1] Implement `src/manga_text_seg/pagelist.py`: derive the list from `manifest.load_manifest`, hash each raw image's exact bytes, compute `page_list_identity` over the document's own canonical serialisation with that field omitted, and write `benchmark/page-list.json` plus `benchmark/image-identity.json` (FR-054, FR-054a)
- [X] T019 [US1] Implement the image bundle writer in `pagelist.py`: copy the 390 paired raw images into `benchmark/dist/images/` with each page's `image_ref` relative to the page list, so the bundle can be relocated (FR-054a, research.md R2)
- [X] T020 [US1] Add the `export` subcommand to `src/manga_text_seg/cli.py`, reusing `_common_parser` and the existing `ConfigError`/`FileNotFoundError` exit-2 handling (FR-049)
- [X] T021 [US1] Implement `src/manga_text_seg/receipt.py`: the six checks in the exact order fixed by `contracts/returned-result.md`, refusing at the first failure with the failing field or `image_id` named and writing nothing into the run (FR-058)
- [X] T022 [US1] Implement admission in `receipt.py`: on a full pass copy masks, sidecars and provenance into `deliverables/<run_id>/<method>/`, then — and only then — call `metrics.compute_metrics(gt, prediction)` per page in this project; `metrics.py` is not modified (FR-057, returned-result.md rule 4)
- [X] T023 [US1] Add the `admit` subcommand to `src/manga_text_seg/cli.py`: `admit --run <run_id> --method <name> <hand-off>/` (FR-049)
- [X] T024 [US1] Write the run record from the admitted provenance: method, checkpoint identity, external repository and exact revision, code and weight licence, checkpoint-load evidence, device, input size, threshold, preprocessing, postprocessing, seed, timings and package versions, each attributed to the environment that produced it (US1 scenario 7, FR-044, FR-056)
- [X] T025 [US1] Implement `src/manga_text_seg/adapters/manga_text_segmentation.py`: Method A with the FR-013a leave-one-fold-out attribution — all five released fold checkpoints verified present, the book→fold map derived from the published seed 42 and verified against the checkpoints before any page is inferred; unmappable ⇒ `unavailable` with the reason, never approximated (research.md R4)
- [X] T026 [US1] Record Method A's per-page `fold_attribution`, padding multiple, effective threshold and preprocessing in its sidecar, and verify the FR-024b collapse per page — no GT pixel left unrepresented, no non-text class admitted as text — recording the collapse rule in run metadata (FR-024b, FR-024a)

**Checkpoint**: US1 is complete — the page list is exported, one method runs end to end, and a returned hand-off is validated, refused or admitted, and scored by this project.

---

## Phase 4: User Story 2 - Know before running whether each method can actually run (Priority: P2)

**Goal**: A per-method pre-flight verdict, produced in the environment it is run in, covering checkpoint, pinned third-party code, dependencies, licence and compute device, with remediation steps — persisted so it can be retrieved without re-running.

**Independent Test**: Run the availability check with no checkpoints installed and confirm each method reports unavailable with a named missing artefact and an install instruction. Then supply a valid checkpoint for one method and confirm it flips to available, that the verdict records the environment it was produced in, and that a run proceeds with that method while the others are listed as not yet returned.

### Tests for User Story 2 ⚠️

- [X] T027 [P] [US2] Write `tests/unit/test_availability.py`: no checkpoint ⇒ unavailable naming the artefact, where to obtain it and how large it is; an absent or incompatible dependency ⇒ the version conflict is named rather than deferred to inference time; a checkpoint that is present but truncated or of the wrong architecture ⇒ unavailable with the mismatch described (US2 scenarios 1, 2, 4; FR-016–FR-019)
- [X] T028 [P] [US2] Write `tests/unit/test_availability_store.py`: a recorded verdict, its reason and its producing environment are retrievable later without re-running the check (FR-022, US2 scenario 5)
- [X] T029 [P] [US2] Write `tests/unit/test_deviation_record.py`: a provenance record declaring `deviation.changes_output: true` is inadmissible as the pinned method (FR-055, returned-result.md rule 7, US2 scenario 6)

### Implementation for User Story 2

- [X] T030 [US2] Implement `src/manga_text_seg/availability.py`: a per-method verdict over checkpoint, third-party code at the pinned revision, dependencies, licence and compute device, each verdict stamped with the environment that produced it (FR-016, FR-017)
- [X] T031 [US2] Add verdict persistence to `availability.py`: write the verdicts as JSON and read them back so a later `status` retrieves recorded verdicts, reasons and environments without re-checking (FR-022)
- [X] T032 [US2] Add remediation steps to each failing verdict — what to obtain, from where, how large — and the FR-021 re-check path: a method needing an accelerator or a legacy interpreter this machine lacks is re-checked in a suitable environment before being declared unavailable (FR-019, FR-021)
- [X] T033 [US2] Add the `status` subcommand to `src/manga_text_seg/cli.py`: `status --config configs/dl.json [--run <run_id>]`, reporting pre-flight verdicts and, with `--run`, admitted versus awaited methods with their reasons and no synthetic values (FR-049, FR-036)

**Checkpoint**: US2 is complete — every method's runnability is known before inference starts, and the verdicts survive the process that produced them.

---

## Phase 5: User Story 3 - Compare the three deep-learning methods against the classical baseline (Priority: P3)

**Goal**: One summary table and one set of charts across the classical baseline and every admitted deep-learning method, with comparable accuracy and non-comparable timing labelled by its producing device.

**Independent Test**: Assemble a combined run on a small fixture from the classical baseline plus two stand-in methods admitted through the receipt path from two separately constructed hand-offs. Confirm one row per method with all four metrics plus mean, standard deviation and timing, that every admitted result carries the same page-list identity, and that every score was computed by this project's own evaluation procedure.

### Tests for User Story 3 ⚠️

- [X] T034 [P] [US3] Write `tests/unit/test_benchmark.py`: a run assembled from the classical baseline plus two admitted stand-ins has one row per method with IoU/Precision/Recall/F1, mean, standard deviation and timing; admitting a third result leaves the first two byte-identical; and a run with one method outstanding is still reportable (US3 scenarios 1, 2; FR-034, FR-059)
- [X] T035 [P] [US3] Write `tests/unit/test_report_dl.py`: every timing figure carries its producing `device.name` and no cross-device timing ranking is presented; a not-run method appears with its reason and no placeholder numeric value; the contamination disclosure appears beside each method's scores; Pixel Accuracy is absent as a ranking metric (US3 scenarios 4, 5, 6, 7; FR-036, FR-060, SC-011)
- [X] T036 [P] [US3] Write `tests/unit/test_adapter_b_geometry.py`: Method B's letterbox to the 1024×1024 stride-64 canvas is inverted back to the aligned page size, and the threshold actually applied is verified against the produced mask rather than read from configuration (FR-024a, FR-026/FR-030 edge cases)

### Implementation for User Story 3

- [X] T037 [US3] Implement `src/manga_text_seg/benchmark.py`: assemble the combined run from the classical baseline plus every admitted DL method, keeping it reportable while methods are outstanding and never recomputing an earlier admission (FR-034, FR-037, FR-059)
- [X] T038 [US3] Extend `src/manga_text_seg/report.py` with the cross-method comparison table and charts — one table, one chart set, mean and standard deviation per accuracy metric, per-device timing labels, contamination disclosure, no Pixel Accuracy ranking — consuming `metrics.compute_metrics` output unchanged and adding no method-specific branching (FR-037, FR-038, FR-060, SC-011)
- [X] T039 [US3] Implement `src/manga_text_seg/adapters/comic_text_detector.py`: Method B's refined segmentation head only with polygons discarded (FR-013), letterbox with its inverse mapping recorded, the effective threshold verified against the produced mask, and the channel order actually used recorded (FR-024a)
- [X] T040 [US3] Implement `src/manga_text_seg/adapters/unetpp_efficientnetv2.py`: Method C, zero-padding to a multiple of 32 recorded and inverted separately, ImageNet normalisation, flip TTA and mixed precision recorded as configuration, and the FR-018a/FR-018b positive checkpoint-load evidence produced by tensor-key identity rather than inferred from a plausible mask (FR-024a, FR-018b)

**Checkpoint**: US3 is complete — the three-way comparison exists, with accuracy comparable and timing explicitly not.

---

## Phase 6: User Story 4 - Inspect where methods succeeded and where they failed (Priority: P4)

**Goal**: Per-method success and failure lists with reasons, four-panel visualisations, and a written report of representative good and failure cases.

**Independent Test**: Run the visualisation and reporting step against an existing run directory containing one deliberately failed page. Confirm the failed page appears in the failure list with its reason, that a configurable number of cases produce four-panel visualisations, and that the report separates good from failure cases.

### Tests for User Story 4 ⚠️

- [X] T041 [P] [US4] Write `tests/unit/test_cases.py`: each method has a success list and a failure list, each failure carrying its recorded reason; a page whose inference raised an error is failed for that method only while the remaining pages of that hand-off are still scored and no other method is affected; and the report distinguishes "pages that failed inside a method that ran" from "method did not run", never presenting an absent method as having zero failures (US4 scenarios 1, 3, 5; FR-039, FR-041)

### Implementation for User Story 4

- [X] T042 [US4] Extend `src/manga_text_seg/visualize.py` to render the four-panel case over admitted DL masks — raw page, ground-truth mask, prediction mask, prediction-versus-ground-truth overlay — with a configurable case count (FR-040, FR-041)
- [X] T043 [US4] Implement the case report in `src/manga_text_seg/visualize.py`: representative good cases and representative failure cases per method, written from the admitted results and the recorded reasons (FR-041)

**Checkpoint**: US4 is complete — the failure modes behind the averages are visible and attributable.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Reproducibility, the no-third-party-files guarantee, and the opt-in boundary around real checkpoints.

- [X] T044 [P] Write the runner-facing documentation: a section sufficient for a runner with no access to this project's environment to produce an admissible result alone — page list, mask convention, sidecar schema, what must not be submitted (FR-042, ONBOARDING §7/§8/§12)
- [X] T045 [P] Add the opt-in marker and `tests/integration/` separation so the default suite never loads a real checkpoint or downloads a model (FR-051)
- [X] T046 [P] Run the SC-013 audit: `git status` and a tree walk show zero third-party model source files and zero weight files of any size, with each method contributing only configuration, returned standardised results and documentation
- [X] T047 Walk [quickstart.md](quickstart.md) end to end on this CPU-only machine and confirm each section's stated outcome holds, including all three methods legitimately reading `unavailable` here (FR-017, FR-021)
- [X] T048 Run the adapter drill of quickstart §6 for a throwaway fourth method and confirm only `adapters/` and `configs/dl.json` changed — `metrics.py`, `report.py` and `visualize.py` carry no method-specific branching (US1 scenario 6, FR-051)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately.
- **Foundational (Phase 2)**: Depends on Setup. **BLOCKS all user stories** — the DL config loader, the provenance check, the adapter interface and the synthetic fixtures are consumed by every story.
- **US1 (Phase 3)**: Depends on Foundational. No dependency on US2–US4.
- **US2 (Phase 4)**: Depends on Foundational. No dependency on US1, US3 or US4 — the availability check is a pre-flight that never touches a returned result.
- **US3 (Phase 5)**: Depends on US1 (it admits results through `receipt.py` and consumes the run record) and benefits from US2 (a not-run method's reason comes from the verdict store). It is independently *testable* with stand-ins regardless.
- **US4 (Phase 6)**: Depends on US1 (there must be admitted masks to visualise). Independent of US2 and US3.
- **Polish (Phase 7)**: Depends on all stories.

### User Story Dependencies

- **US1 (P1)**: the trunk. Everything else composes it.
- **US2 (P2)**: independent of US1 — it answers a question about an environment, not about a result.
- **US3 (P3)**: composes US1; the composition is the deliverable.
- **US4 (P4)**: composes US1; purely diagnostic.

### Within Each User Story

- Test tasks come first and must FAIL before their implementation tasks.
- `pagelist.py` before the `export` subcommand; `receipt.py` before `admit`.
- Module implementation before its CLI wiring.
- Method adapters (T025, T039, T040) are independent of each other and of the receipt path.

### Parallel Opportunities

- Setup: T002–T005 all touch different files.
- Foundational: T007 and T008 are independent test files; T009 and T010 are independent modules.
- US1: T014, T015, T016, T017 are four separate test files and can be written in one pass.
- US2: T027, T028, T029 are three separate test files.
- US3: T034, T035, T036 are three separate test files; T039 and T040 are two independent adapter files.
- Polish: T044, T045, T046 are independent.

---

## Parallel Example: User Story 1

```bash
# Write all four US1 test files at once — different files, no shared state:
Task: "tests/unit/test_pagelist.py — export emits 390 sorted pages, no GT reference, byte-identical re-export"
Task: "tests/contract/test_page_list_schema.py — validates against contracts/page-list.schema.json"
Task: "tests/unit/test_receipt.py — the six FR-050a refusal behaviours"
Task: "tests/unit/test_admit_metrics_equality.py — hand-off metric byte-identical to adapter metric"

# Then the independent adapters in parallel:
Task: "adapters/manga_text_segmentation.py — Method A with FR-013a LOFO attribution"
Task: "adapters/comic_text_detector.py — Method B, segmentation head only"
Task: "adapters/unetpp_efficientnetv2.py — Method C with FR-018b load evidence"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 (Setup) and Phase 2 (Foundational).
2. Complete Phase 3 (US1) — T014 through T026.
3. **STOP and VALIDATE**: run T014–T017, then `manga-text-seg export`, then `manga-text-seg admit` on a synthetic hand-off, then `pytest -o addopts='' -q`.
4. **Ship the export**: `benchmark/page-list.json` + `benchmark/image-identity.json` + `benchmark/dist/images/` are what the three external runners are blocked on. Producing them is the point at which the teammates can start.

### Incremental Delivery

1. Setup + Foundational → foundation ready.
2. US1 → the export and the receipt path exist → **runners unblocked**.
3. US2 → availability verdicts → a runner knows before spending GPU hours.
4. US3 → the comparison table and charts → the thesis deliverable.
5. US4 → the failure report → the discussion chapter.
6. Polish → reproducibility, the no-third-party-files audit, the opt-in boundary.

### Parallel Team Strategy

After Foundational completes:

- One developer on US1 (the trunk — export, receipt, Method A).
- One developer on US2 (availability — fully independent, different modules).
- One developer on US3's Method B and Method C adapters (independent files, blocked only on `adapters/base.py`).
- US4 waits on US1's admitted masks; the stand-in fixture makes it startable early.

---

## Notes

- `metrics.py` MUST NOT be modified (FR-057). If any task appears to require it, the task is wrong, not the module.
- `src/manga_text_seg/adapters/` is the only place method-specific logic may live (US1 scenario 6).
- Nothing from a third-party repository is vendored, copied, patched or committed in any form, any size (FR-055, SC-013). Workarounds for upstream defects are applied in the runner's own environment and recorded as deviations from the pinned revision.
- Ground truth never reaches a runner: `data/groundtruth/` and `data/no-need-to-read/` are not part of the export and `page-list.json` carries no GT reference (FR-054a).
- Every pytest invocation uses `-o addopts=''`.
- Commit after each task or logical group. Do not commit unless asked.
