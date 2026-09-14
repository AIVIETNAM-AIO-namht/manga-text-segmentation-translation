# Tasks: Data Foundation & Classical Baseline

**Input**: Design documents from `/specs/001-data-foundation-classical-baseline/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create project structure per implementation plan (`src/manga_text_seg/`, `tests/`, `configs/`)
- [X] T002 Initialize Python project with `pyproject.toml` and dependencies (opencv-python, numpy, pandas, matplotlib, pytest)
- [X] T003 [P] Configure linting and formatting tools in `pyproject.toml`
- [X] T004 Create `src/manga_text_seg/__init__.py` and `src/manga_text_seg/cli.py` (stub)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T005 Implement configuration loader and schema in `src/manga_text_seg/config.py` (FR-036)
- [X] T006 Implement discovery logic (recursive GT scan) in `src/manga_text_seg/discovery.py` (FR-001, FR-002)
- [X] T007 Implement deterministic manifest logic and persistence in `src/manga_text_seg/manifest.py` (FR-004, FR-005)
- [X] T008 Implement image/mask loading (read-only) in `src/manga_text_seg/imaging.py` (FR-009)
- [X] T009 Implement binary mask normalization (KMeans) in `src/manga_text_seg/normalize.py` (FR-011, FR-012, Research R1)
- [X] T010 Implement lossless alignment (crop-topleft) in `src/manga_text_seg/align.py` (FR-015, Research R2)
- [X] T011 Implement CLI discovery subcommands in `src/manga_text_seg/cli.py` (FR-036)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Single Method Execution (Priority: P1) 🎯 MVP

**Goal**: Run one classical method on one page and get a prediction mask.

**Independent Test**: Execute CLI `run --method otsu` and verify prediction mask exists and follows binary convention.

### Implementation for User Story 1

- [X] T012 [P] [US1] Implement base class and registry in `src/manga_text_seg/methods/__init__.py` (FR-035)
- [X] T013 [P] [US1] Implement Otsu threshold method in `src/manga_text_seg/methods/otsu.py`
- [X] T014 [US1] Implement preprocessing (grayscale/denoise) in `src/manga_text_seg/preprocess.py` (FR-020)
- [X] T015 [US1] Implement per-image metric calculation in `src/manga_text_seg/metrics.py` (FR-025, FR-027, Research R3)
- [X] T016 [US1] Implement CLI single-run subcommand in `src/manga_text_seg/cli.py`
- [X] T017 [US1] Create metadata sidecar writing logic in `src/manga_text_seg/run.py`

**Checkpoint**: User Story 1 complete - single page processed successfully.

---

## Phase 4: User Story 2 - Full Dataset Sweep (Priority: P2)

**Goal**: Run a full method sweep across the dataset and get a ranking.

**Independent Test**: Execute CLI `sweep` and verify `metrics.csv` and `summaries.json` are created for all pairs.

### Implementation for User Story 2

- [X] T018 [P] [US2] Implement report generation logic in `src/manga_text_seg/report.py` (FR-030, FR-031)
- [X] T019 [US2] Implement CLI sweep subcommand in `src/manga_text_seg/cli.py` (FR-003)
- [X] T020 [US2] Implement error handling for failed images (FR-031)

**Checkpoint**: User Story 2 complete - full sweep works and produces summary.

---

## Phase 5: User Story 3 - Data Validation (Priority: P3)

**Goal**: Discover orphans and mismatched pages to ensure data integrity.

**Independent Test**: Execute CLI `validate` and verify orphan report is generated correctly.

### Implementation for User Story 3

- [X] T021 [P] [US3] Implement validation report logic in `src/manga_text_seg/validation.py` (FR-006, FR-007)
- [X] T022 [US3] Implement CLI validate subcommand in `src/manga_text_seg/cli.py`
- [X] T023 [US3] Add orphan detection to discovery workflow

**Checkpoint**: User Story 3 complete - validation report generated.

---

## Phase 6: User Story 4 - Visualization (Priority: P4)

**Goal**: Visualize a subset of results to spot quality issues.

**Independent Test**: Execute CLI `visualize` and verify 4-panel composites are generated.

### Implementation for User Story 4

- [X] T024 [P] [US4] Implement visualization logic in `src/manga_text_seg/visualize.py` (FR-032, FR-033)
- [X] T025 [US4] Implement selection modes (first-N, best-N, worst-N) in `src/manga_text_seg/visualize.py`
- [X] T026 [US4] Implement CLI visualize subcommand in `src/manga_text_seg/cli.py`

**Checkpoint**: User Story 4 complete - visualization works.

---

## Phase 7: User Story 5 - Scorecard Reporting (Priority: P5)

**Goal**: Compare classical baseline results in a scorecard format.

**Independent Test**: Verify `metrics_summary.csv` is generated and readable.

### Implementation for User Story 5

- [X] T027 [P] [US5] Implement scorecard CSV export in `src/manga_text_seg/report.py`
- [X] T028 [US5] Refine CLI to support scorecard mode

**Checkpoint**: User Story 5 complete - scorecard exported.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories and final validation

- [X] T029 Write contract tests for manifest schema in `tests/contract/test_manifest.py`
- [X] T030 Write contract tests for metrics schema in `tests/contract/test_metrics.py`
- [X] T031 Write contract tests for method interface in `tests/contract/test_method_interface.py`
- [X] T032 Write unit tests for normalization and alignment in `tests/unit/test_imaging.py`
- [X] T033 Run quickstart.md validation scenarios

---

## Dependencies & Execution Order

### Phase Dependencies
- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-7)**: All depend on Foundational phase completion
  - User stories can proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3 → P4 → P5)
- **Polish (Phase 8)**: Depends on all desired user stories being complete

### Parallel Opportunities
- T005-T010 in Foundational can run in parallel
- T012-T013 in US1 can run in parallel
- T018, T021, T024, T027 in different stories can run in parallel

## Parallel Example: User Story 1

```bash
# Launch all models for User Story 1 together:
Task: "Implement base class and registry in src/manga_text_seg/methods/__init__.py"
Task: "Implement Otsu threshold method in src/manga_text_seg/methods/otsu.py"
```
