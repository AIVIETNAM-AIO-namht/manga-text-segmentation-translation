---

description: "Task list for End-to-End Manga Translation Web Demo & Reporting"
---

# Tasks: End-to-End Manga Translation Web Demo & Reporting

**Input**: Design documents from `/specs/005-web-e2e-reporting/`

**Prerequisites**: [plan.md](plan.md) (required), [spec.md](spec.md) (required for user stories).
`research.md`, `data-model.md`, `contracts/` and `quickstart.md` are **not produced for this
feature** — plan.md states why: the upstream output contracts already exist in Specs 1–4, the data
model is fixed by spec.md's Key Entities, and the API contract lives in plan.md's "Application API
Contract" section rather than in a separate `contracts/` directory. plan.md is the contract
document for every endpoint named below.

**Tests**: **Required for this feature.** FR-087 enumerates nineteen mandatory test areas and the
constitution's Principle II is *Spec-Scoped Test-First (NON-NEGOTIABLE)*. Every test task below is
written before the implementation it covers and MUST fail first. Run the suite with
`pytest -o addopts='' -q` — `pyproject.toml` sets `--cov=` and `pytest-cov` is not installed in the
active environment. Per FR-089 and FR-087's closing constraint, the default suite runs **CPU-only,
without the full Manga109 dataset, without any segmentation model, without any inpainting
checkpoint, and without a real translation provider** — all of them are replaced by the fixture
bundles created in Phase 1.

**Organization**: Tasks are grouped by user story so each story is independently implementable and
testable. Story numbering follows **spec.md** — US1 select and open a page, US2 view the visual
chain, US3 inspect regions, US4 edit and re-render, US5 summaries, US6 quantitative report, US7
qualitative report, US8 batch, US9 download, US10 CLI parity. Priorities come from spec.md
(US1–US3 P1, US4–US7 P2, US8–US10 P3).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

Single project with `src/` layout (plan.md "Source Code"): the new subpackage is
`src/manga_text_seg/web/` with routers in `src/manga_text_seg/web/routes/`; tests in
`tests/{contract,unit,integration}/`; configuration in `configs/`; demo runs written under
`outputs/end-to-end/<run_id>/` (FR-063). The frontend bundle is **not** in this repository — it is a
static directory dropped into the configured static root and served verbatim.

**Four constraints to respect throughout**:

- **`src/manga_text_seg/run.py` (singular) is Spec 2's existing run-record module.** This feature's
  run lifecycle lives in `src/manga_text_seg/web/session.py` (plan.md gives `session.py` "run
  ownership (FR-065)"). Every task below spells the path in full.
- **Specs 1–4 code paths must not change.** `manifest.py`, `runs.py`, `availability.py`,
  `selection.py`, `report.py`, `qualitative.py`, `boards.py`, `inpaint.py` and `translation/*` are
  imported, never edited. SC-011 and FR-090 verify this; T073 is the audit.
- **`/api/artifact` is the only route that returns file bytes** (plan.md). No other route may read a
  file off disk for the response body, and it never accepts a caller-supplied filesystem path.
- **`configs/web.json` does not exist yet.** plan.md assumes a Spec 005 config file via FR-086 and
  FR-013c/FR-013d need configured roots; T001 creates it. No path, host, port or root may live in
  code (FR-061's rule, applied here).

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: The configuration file, the dependency declarations, the package skeleton and the
fixture bundles every later test depends on.

- [x] T001 Create `configs/web.json` (FR-086, FR-013c, FR-013d): the bind host and port (loopback only), the static root for the frontend bundle, the allow-listed artifact roots for Spec 1–4 outputs, the demo run root (`outputs/end-to-end/`), the session cookie/header name, and the summary provider block (provider name, model, `secret_env_var`, timeout, retries, cache settings). No path, host, port or root may live in code.
- [x] T002 [P] Extend `pyproject.toml`: add `fastapi>=0.115` and `uvicorn>=0.30` to `[project.dependencies]` (plan.md Complexity Tracking — the one recorded Principle III violation), then refresh `uv.lock` in the same change so `uv lock --check` still passes.
- [x] T003 [P] Create `src/manga_text_seg/web/__init__.py` — the new subpackage's docstring naming the modules (`app`, `security`, `session`, `jobs`, `errors`, `schemas`, `routes/*`) and the FR ranges each covers. No imports yet; the modules land in their own phases.
- [x] T004 [P] Confirm `outputs/end-to-end/` is covered by `.gitignore` (`.gitignore` already lists `outputs/`) and that `configs/web.json` is tracked. Record the result; no `.gitignore` change is expected.
- [x] T005 [P] Add a fixture-bundle factory to `tests/conftest.py`: builds a throwaway Spec 1–4 artifact tree (manifest, prediction mask, processed mask, inpainted image, OCR JSON, translations JSON, rendered PNG, metadata sidecars) for one manga/page under `tmp_path`, with per-stage omission flags so a test can request a bundle missing any one stage (SC-002). Also add the `app` + `TestClient` fixtures every web test uses.
- [x] T006 [P] Add `tests/contract/`, `tests/integration/` and `tests/unit/` web test modules as empty-but-collecting files (`tests/contract/test_web_*.py` etc.) so the three-way split exists before the first red test. Follow the existing suite's naming so `pytest -o addopts='' -q` collects them.

**Checkpoint**: `pytest -o addopts='' -q` still green; `uv lock --check` passes; fixture factory builds a complete bundle and each of its omission flags builds an incomplete one.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The app, its security guards, its error taxonomy, its session/run store and its shared
schemas. Every user story below is an HTTP route; none of them can exist until this phase is done.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T007 [P] Contract test for the FR-079 category → HTTP status mapping and the error envelope in `tests/contract/test_web_error_envelope.py`: every one of FR-079's 19 categories maps [corrected: this task said 18; FR-079 lists 19] to the status in plan.md's mapping table, and every non-2xx JSON body is `{"error": {"category", "message", "detail"}}`.
- [x] T008 Implement `src/manga_text_seg/web/errors.py` (FR-079–FR-084): the category enum, the status mapping, the envelope serializer, and helpers that put `detail.candidates` on ambiguity errors (FR-082) and `detail.expected` on missing-artifact errors (FR-083).
- [x] T009 [P] Unit test for the security guards in `tests/unit/test_web_security.py`, covering **SC-015's four refusal cases**: a non-loopback `Host` header, a foreign `Origin` header, a path escaping the allow-listed artifact root (traversal), and any path under `data/no-need-to-read/` (FR-008).
- [x] T010 Implement `src/manga_text_seg/web/security.py` (FR-013c, FR-013d, FR-008): the loopback bind assertion, the `Host`/`Origin` guard, artifact-root allow-list resolution, traversal refusal, and the `data/no-need-to-read/` refusal. Resolution returns an absolute path only when it is inside an allow-listed root.
- [x] T011 [P] Unit test for the session/run store in `tests/unit/test_web_session.py`, covering **FR-022a** (a reload with the issued id resumes the same session) and FR-065 (two distinct sessions get two distinct runs, each owning its own).
- [x] T012 Implement `src/manga_text_seg/web/session.py` (FR-022, FR-022a, FR-065): issue-or-resume by id, in-process store, one run per session, run ownership.
- [x] T013 [P] Contract test for the app skeleton in `tests/contract/test_web_app_skeleton.py`: `serve` starts and binds loopback, `GET /` serves the fixture bundle from the static root, a missing/empty static root still starts and returns a `not_ready` envelope naming the expected root (never a built-in UI), and the health endpoint answers.
- [x] T014 Implement `src/manga_text_seg/web/app.py` (FR-013a, FR-013b, FR-086): the FastAPI app factory that mounts the routers and the configured static root, wires the security middleware, registers the error handlers, and holds the health endpoint.
- [x] T015 Add `cmd_serve` plus the `serve` subparser and its dispatch to `src/manga_text_seg/cli.py` (plan.md "Source Code" — `cli.py` is the only existing file this feature modifies).
- [ ] T016 [P] Implement `src/manga_text_seg/web/schemas.py`: the shared request/response models for the endpoints in plan.md's contract table (session, catalog, page bundle, edit, render, summary, reports, artifact, jobs, run status).
- [ ] T017 Extend `src/manga_text_seg/web/session.py` with run creation and materialization under `outputs/end-to-end/<run_id>/` (FR-063, FR-066): the run directory tree, the run id, and the recorded selection. Test first in `tests/integration/test_web_run_lifecycle.py`.

**Checkpoint**: `manga-text-seg serve` starts against a fixture bundle; all four SC-015 refusals return their envelopes; every FR-079 category maps to its documented status. Foundation ready — user stories can now begin.

---

## Phase 3: User Story 1 - Select and open a fully-artifacted manga page (Priority: P1) 🎯 MVP

**Goal**: A user opens the app, is handed a session, browses manga → page → method → Spec 3 run → Spec 4 run → language, all enumerated from disk, and opens one page that returns its bundle with a status. The benchmark-selected method is marked as the default.

**Independent Test**: Start the server against the fixture tree, `GET /api/session`, walk `/api/catalog/*` to a page, `GET /api/page` for it, and `GET /api/artifact` for its original image. Every enumerated value comes from disk; deleting a fixture from disk removes it from the listing rather than falling back to a default.

### Tests for User Story 1 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T018 [P] [US1] Contract test for `GET /api/session` in `tests/contract/test_web_session_api.py` (FR-022, FR-022a).
- [ ] T019 [P] [US1] Contract test for `GET /api/catalog/{mangas,pages,methods,runs,languages}` in `tests/contract/test_web_catalog_api.py`: nothing hard-coded (FR-015), empty-on-disk yields empty rather than a default, and pages come back in manifest order (FR-014).
- [ ] T020 [P] [US1] Unit test for the benchmark default method selection in `tests/unit/test_web_benchmark_default.py` covering **FR-017** and **SC-003**: IoU and F1 are primary, Precision/Recall/time secondary, no results is an error, and an unbroken tie is an error.
- [ ] T021 [P] [US1] Contract test for `GET /api/page` in `tests/contract/test_web_page_api.py`: the eight **FR-020** states are each reachable, and **FR-019**'s missing-value case returns null rather than a fabricated value.
- [ ] T022 [P] [US1] Contract test for `GET /api/artifact` in `tests/contract/test_web_artifact_api.py`: a run-relative reference resolves, a traversal attempt and a `data/no-need-to-read/` attempt are refused, and no route other than this one returns file bytes.
- [ ] T023 [P] [US1] Integration test for the full open-a-page journey in `tests/integration/test_web_open_page.py` (SC-001's first fully rendered page, exercised against fixtures).

### Implementation for User Story 1

- [ ] T024 [US1] Implement `src/manga_text_seg/web/routes/catalog.py` (FR-014–FR-017): enumerate mangas, pages, methods, Spec 3/Spec 4 runs and languages from disk via the existing `manifest.py` and `availability.py`, with the benchmark default marked on the methods response.
- [ ] T025 [US1] Wire the FR-017 benchmark default into the methods response in `src/manga_text_seg/web/routes/catalog.py`, reusing the existing `selection.py` rather than reimplementing ranking.
- [ ] T026 [US1] Implement `src/manga_text_seg/web/routes/page.py` (FR-018–FR-027, FR-030): assemble the page bundle — status, provenance read from upstream only, artifact references and the eight-state status — and return 200 even when the page itself is in a failure state.
- [ ] T027 [US1] Implement `src/manga_text_seg/web/routes/artifacts.py` (FR-013d, FR-023, FR-028): the single byte-serving route, resolving through `web/security.py` and never accepting a caller-supplied filesystem path.
- [ ] T028 [US1] Implement page-level failure isolation in `src/manga_text_seg/web/routes/page.py` and the run error report (FR-026): a page failure is recorded in the run's error report and the request that carries the bundle still succeeds.

**Checkpoint**: User Story 1 is fully functional and testable independently — a page opens, its original image serves, and every listing is disk-derived.

---

## Phase 4: User Story 2 - View the full visual chain of one page (Priority: P1)

**Goal**: The user sees the complete chain for one page — original → prediction mask → processed mask → inpainted → rendered — with each stage resolvable as an artifact and each missing stage reported precisely.

**Independent Test**: Open a fixture page, request every stage's artifact reference from the bundle, fetch each through `/api/artifact`, and confirm byte equality. Then repeat with each stage omitted in turn and confirm the specific FR-079 category and `detail.expected` for that stage.

### Tests for User Story 2 ⚠️

- [ ] T029 [P] [US2] Contract test for the five chain stage references in `tests/contract/test_web_page_chain.py` (FR-018, FR-024).
- [ ] T030 [P] [US2] Integration test for byte-identical stage serving in `tests/integration/test_web_artifact_bytes.py`: **FR-029** byte equality for every stage, and **SC-002**'s incomplete-artifact matrix — each omission yields its own category with `detail.expected` naming the missing location (FR-083).

### Implementation for User Story 2

- [ ] T031 [US2] Extend `src/manga_text_seg/web/routes/page.py` with the chain stage references for all five stages, each carrying its own availability state so a missing intermediate stage does not fail the bundle.
- [ ] T032 [US2] Extend `src/manga_text_seg/web/routes/page.py` with per-stage missing-artifact resolution (FR-079: `missing_prediction_mask`, `missing_metadata_sidecar`, `missing_inpainting_output`, `missing_ocr_output`, `missing_translation_output`), reusing `availability.py`.

**Checkpoint**: The full chain displays for a complete page, and every incomplete-page case names exactly the missing stage.

---

## Phase 5: User Story 3 - Inspect per-region OCR text and translations (Priority: P1)

**Goal**: The user reads, per region, the bounding box, the OCR transcript with its confidence, the translation, and where each value came from.

**Independent Test**: Open a fixture page, read its region list, and confirm each region carries a bbox, OCR text, confidence, translation and provenance; then open a page whose OCR or translation has a missing value and confirm the region reports it as missing rather than blank or fabricated.

### Tests for User Story 3 ⚠️

- [ ] T033 [P] [US3] Contract test for the region payload in `tests/contract/test_web_regions_api.py` (FR-024, FR-025): every region carries bbox, OCR text, OCR confidence, translation and provenance.
- [ ] T034 [P] [US3] Unit test for **FR-019**'s missing-value handling in `tests/unit/test_web_missing_values.py`: a missing OCR or translation value is reported as missing, never as an empty string or a substituted default.

### Implementation for User Story 3

- [ ] T035 [US3] Extend `src/manga_text_seg/web/routes/page.py` with the region list — bbox, OCR transcript, confidence, translation and provenance — reading the existing Spec 4 OCR/translation artifacts rather than re-running any stage.
- [ ] T036 [US3] Implement FR-019's missing-value representation across the region payload in `src/manga_text_seg/web/routes/page.py`.

**Checkpoint**: All three P1 stories work independently — select, view the chain, and inspect regions.

---

## Phase 6: User Story 4 - Edit OCR text or translation and re-render (Priority: P2)

**Goal**: The user corrects a region's OCR text or translation, re-renders, and sees the corrected page — without re-running segmentation or inpainting.

**Independent Test**: Edit one region's translation, re-render, and confirm the rendered image changed for that page only; confirm no segmentation or inpainting was executed; confirm the Spec 1–4 trees are checksum-identical afterwards.

### Tests for User Story 4 ⚠️

- [ ] T037 [P] [US4] Contract test for `POST /api/page/edit` in `tests/contract/test_web_edit_api.py` (FR-031, FR-034): the edit is recorded with its original and edited values.
- [ ] T038 [P] [US4] Contract test for `POST /api/page/render` in `tests/contract/test_web_render_api.py` (FR-032, FR-033, FR-036): re-rendering executes neither segmentation nor inpainting, and only the affected page is re-rendered.
- [ ] T039 [P] [US4] Integration test for edit persistence and upstream invariance in `tests/integration/test_web_edit_persistence.py`: the edit materializes into the run on open and is replaced on re-render (FR-036), and **SC-011**'s upstream checksums are unchanged.

### Implementation for User Story 4

- [ ] T040 [US4] Implement `src/manga_text_seg/web/routes/edit.py` — `POST /api/page/edit` recording a `ManualEditRecord` (FR-031, FR-034).
- [ ] T041 [US4] Implement `POST /api/page/render` in `src/manga_text_seg/web/routes/edit.py`, calling Spec 4's existing renderer through `translation/` (FR-032, FR-033) and re-rendering only the affected regions and page.
- [ ] T042 [US4] Persist edits and re-rendered outputs into the run in `src/manga_text_seg/web/session.py` and `src/manga_text_seg/web/routes/edit.py`: materialize on open, replace on re-render (FR-036).

**Checkpoint**: Editing and re-rendering work, and Specs 1–4 are provably untouched.

---

## Phase 7: User Story 5 - Generate page and chapter summaries (Priority: P2)

**Goal**: The user generates a summary for one page or for a chapter given as an explicit ordered page list, and repeated requests reuse the cache instead of re-calling the provider.

**Independent Test**: Generate a page summary with a mock provider, request it again and confirm the provider was not called twice; generate a chapter summary from a three-page ordered list and confirm order is preserved and any missing page is listed; confirm no secret value appears in any log, metadata or output.

### Tests for User Story 5 ⚠️

- [ ] T043 [P] [US5] Unit test for the summary provider adapter boundary in `tests/unit/test_web_summary_adapter.py` (FR-039–FR-041) using a mock provider — the default suite must not call a real provider.
- [ ] T044 [P] [US5] Unit test for secret handling in `tests/unit/test_web_secrets.py`: **FR-042/FR-085** — the secret comes only from its configured environment variable, is validated at startup, and a no-leak scan of logs, metadata and outputs finds no secret value.
- [ ] T045 [P] [US5] Contract test for `POST /api/summary/page` and `POST /api/summary/chapter` in `tests/contract/test_web_summary_api.py`, including **FR-043**'s cache reuse.
- [ ] T046 [P] [US5] Integration test for chapter summarization in `tests/integration/test_web_chapter_summary.py` covering **SC-010**: the explicit ordered page list is preserved, the chapter id is stored in the run (FR-046), and missing pages are listed.

### Implementation for User Story 5

- [ ] T047 [US5] Implement the `SummaryProviderAdapter` boundary and the env-var secret validation at startup in `src/manga_text_seg/web/routes/summary.py` and `src/manga_text_seg/web/app.py` (FR-042, FR-085).
- [ ] T048 [US5] Implement `POST /api/summary/page` in `src/manga_text_seg/web/routes/summary.py` (FR-039–FR-044), writing `summary.txt` and `metadata.json` into the page's run directory (FR-063).
- [ ] T049 [US5] Implement summary caching in `src/manga_text_seg/web/routes/summary.py` (FR-043): a repeated request for the same inputs returns the cached summary without calling the provider.
- [ ] T050 [US5] Implement `POST /api/summary/chapter` in `src/manga_text_seg/web/routes/summary.py` (FR-045, FR-046): a chapter is an explicit user-supplied ordered page list with a stable chapter id stored in the run, and its `chapter_summary/<chapter_id>/` directory holds `selection.json` and `summary.txt`.

**Checkpoint**: Page and chapter summaries generate, cache and persist, with no secret in any output.

---

## Phase 8: User Story 6 - Generate the quantitative comparison report (Priority: P2)

**Goal**: The user generates the quantitative report into a run — a metrics CSV and a comparison PNG — either into an existing run or into one auto-created for the purpose, and downloads both.

**Independent Test**: Generate the quantitative report twice into two different runs and once into an auto-created third; confirm all three coexist, and confirm generation reads only existing artifacts and executes no model.

### Tests for User Story 6 ⚠️

- [ ] T051 [P] [US6] Contract test for `POST /api/reports/quantitative` in `tests/contract/test_web_quant_report_api.py` (FR-047–FR-053).
- [ ] T052 [P] [US6] Integration test for report generation into a run in `tests/integration/test_web_quant_report.py` covering **SC-012** (generation from artifacts alone, no model executed), **SC-013** (three coexisting runs), and **FR-054** (the report is always written into a run — an existing target or an auto-created one).
- [ ] T053 [P] [US6] Contract test for report artifact serving in `tests/contract/test_web_report_serving.py` (FR-030a): the CSV and PNG serve byte-identically through `/api/artifact`.

### Implementation for User Story 6

- [ ] T054 [US6] Implement `src/manga_text_seg/web/routes/reports.py` — `POST /api/reports/quantitative` (FR-047–FR-053), reusing the existing `report.py` for the metrics rather than recomputing them.
- [ ] T055 [US6] Write `report/metrics_summary.csv` and `report/metrics_comparison.png` into the target run (FR-048, FR-050, FR-063).
- [ ] T056 [US6] Implement report-artifact serving through `src/manga_text_seg/web/routes/artifacts.py` (FR-030a), and report generation into an auto-created run when no target run exists (FR-054).

**Checkpoint**: The quantitative report generates from artifacts alone, into an existing or auto-created run, and downloads.

---

## Phase 9: User Story 7 - Generate the qualitative report (Priority: P2)

**Goal**: The user generates a self-contained HTML qualitative report whose page selection follows a configurable policy, defaulting to best + worst + typical, and whose failure sections enumerate every failure rather than a sample.

**Independent Test**: Generate the qualitative report once per policy (`first-N`, `best-N`, `worst-N`, `typical-N`, `manual-list`) and once with no policy for the `combined` default; confirm the selected set matches the policy, confirm the HTML opens with no network access, and confirm the failure section lists every failure.

### Tests for User Story 7 ⚠️

- [ ] T057 [P] [US7] Unit test for the five selection policies in `tests/unit/test_web_qual_policies.py` (FR-056), including that `combined` (best + worst + typical) is the default.
- [ ] T058 [P] [US7] Contract test for `POST /api/reports/qualitative` in `tests/contract/test_web_qual_report_api.py` (FR-055–FR-062).
- [ ] T059 [P] [US7] Integration test for report self-containment and failure enumeration in `tests/integration/test_web_qual_report.py` covering **FR-058** (failure sections enumerate ALL failures, never a sample) and **FR-061** (the HTML is self-contained).

### Implementation for User Story 7

- [ ] T060 [US7] Implement the five selection policies in `src/manga_text_seg/web/routes/reports.py` (FR-056), reusing the existing `qualitative.py` for the report body.
- [ ] T061 [US7] Implement `POST /api/reports/qualitative` in `src/manga_text_seg/web/routes/reports.py` (FR-055–FR-060), writing `report/qualitative_report.html` into the run (FR-063).
- [ ] T062 [US7] Make the HTML written by `src/manga_text_seg/web/routes/reports.py` self-contained (FR-061): no external stylesheet, script, font or image reference, so it renders offline.

**Checkpoint**: Both reports generate into runs and download; the qualitative report is offline-renderable.

---

## Phase 10: User Story 8 - Run a batch of pages (Priority: P3)

**Goal**: The user starts a batch over a page list, watches progress while it runs, and reads the job's outcome — including the pages that failed — after the session that started it has ended.

**Independent Test**: Start a batch over a fixture page list containing one deliberately failing page; confirm progress advances to completion, confirm the failing page is recorded as a failure without aborting the batch, and confirm the job status is still readable after the starting session is closed.

### Tests for User Story 8 ⚠️

- [ ] T063 [P] [US8] Contract test for `POST /api/jobs/batch` and `GET /api/jobs/{job_id}` in `tests/contract/test_web_jobs_api.py` (FR-073, FR-084).
- [ ] T064 [P] [US8] Integration test for batch failure isolation and job durability in `tests/integration/test_web_batch.py` covering **SC-009**: a failing page inside a batch does not abort the batch, and the job status is readable after the session closes.

### Implementation for User Story 8

- [ ] T065 [US8] Implement `src/manga_text_seg/web/jobs.py` — the `BatchJob` registry, background execution and completed/total progress (FR-073, FR-084).
- [ ] T066 [US8] Implement `src/manga_text_seg/web/routes/jobs.py` — `POST /api/jobs/batch` and `GET /api/jobs/{job_id}`, reusing the same orchestration functions the page and report routes call so batch and in-app behaviour cannot diverge (FR-078).
- [ ] T067 [US8] Make the job registry in `src/manga_text_seg/web/jobs.py` outlive the session that started it (FR-084) and record each failed page in the run's error report without aborting the batch (FR-026).

**Checkpoint**: A batch runs to completion, survives its session, and reports every failure.

---

## Phase 11: User Story 9 - Download results from the app (Priority: P3)

**Goal**: The user sees what a run contains and downloads any artifact from it, byte-identical to what is on disk.

**Independent Test**: Open a completed run, read its status and its artifact listing, download each listed artifact, and confirm every download is byte-identical to the file on disk.

### Tests for User Story 9 ⚠️

- [ ] T068 [P] [US9] Contract test for `GET /api/run/{run_id}/status` in `tests/contract/test_web_run_status_api.py` (FR-077): run configuration, per-page statuses and the error summary.
- [ ] T069 [P] [US9] Integration test for downloading a whole run in `tests/integration/test_web_run_download.py`: every artifact in the FR-063 tree is listed and every one downloads byte-identically through `/api/artifact`.

### Implementation for User Story 9

- [ ] T070 [US9] Implement `GET /api/run/{run_id}/status` in `src/manga_text_seg/web/session.py` and its router (FR-077), reusing the existing `runs.py` for run records.
- [ ] T071 [US9] Implement run artifact enumeration (FR-063, FR-064, FR-067–FR-072) so the run's full tree — `report/`, `pages/<manga>/<NNN>/`, `chapter_summary/<chapter_id>/` — is listed with run-relative references that `/api/artifact` resolves.

**Checkpoint**: A run's contents are visible and every artifact downloads byte-identically.

---

## Phase 12: User Story 10 - Perform equivalent operations from the CLI (Priority: P3)

**Goal**: Every operation the app offers is available from the CLI with identical behaviour, including the ambiguity errors.

**Independent Test**: For each in-app operation, run its CLI equivalent against the same fixture tree and confirm the produced artifacts and the reported errors are identical — including the `undeterminable_run` and `undeterminable_segmentation_method` ambiguity cases, which must list the same candidates.

### Tests for User Story 10 ⚠️

- [ ] T072 [P] [US10] Integration test for CLI parity in `tests/integration/test_web_cli_parity.py` (FR-073–FR-078): each CLI operation produces the same artifacts as its in-app equivalent, and the ambiguity errors list the same candidates as the API's `detail.candidates` (FR-082).

### Implementation for User Story 10

- [ ] T073 [US10] Add the CLI subcommands for every in-app operation to `src/manga_text_seg/cli.py`, dispatching to the same orchestration functions the API handlers call (FR-078).
- [ ] T074 [US10] Route the CLI through the shared error taxonomy in `src/manga_text_seg/web/errors.py` so CLI and API report identical categories, candidates and expected locations (FR-079, FR-082, FR-083).

**Checkpoint**: All ten user stories are independently functional and CLI parity is demonstrated.

---

## Phase 13: Polish & Cross-Cutting Concerns

**Purpose**: Completeness, invariance and governance checks that span every story.

- [ ] T075 [P] Audit the error taxonomy end to end (FR-079–FR-084): confirm every one of the 18 categories is actually raised somewhere in `src/manga_text_seg/web/` rather than only declared, and that no route returns an unenveloped error.
- [ ] T076 [P] Audit the **FR-087** nineteen-item mandatory test-coverage list against `tests/unit/`, `tests/contract/` and `tests/integration/` and add any missing test; confirm each item traces to a task above.
- [ ] T077 [P] Verify upstream invariance (SC-011, FR-090): run the Spec 1–4 trees' checksums before and after a full demo run and confirm they are identical, and confirm no task above edited a Spec 1–4 module in `src/manga_text_seg/`.
- [ ] T078 [P] Verify **SC-014**: swap the static root's bundle directory mid-session, without restarting the backend or changing code, and confirm the new UI is served by `src/manga_text_seg/web/app.py`.
- [ ] T079 [P] Update `README.md` with the Spec 005 demo: how to start it, the configured static root, and the `outputs/end-to-end/<run_id>/` layout.
- [ ] T080 Ponytail full-mode pass over `src/manga_text_seg/web/` (Principle III pre-commit gate): no abstraction with a single implementation, no config for a fixed value, no scaffolding for later; confirm the never-weaken list (validation, security, accessibility, data-loss handling) is intact.
- [ ] T081 Run the end-to-end demo from `src/manga_text_seg/cli.py` against the real fixture tree and time launch → first fully rendered page against **SC-001**'s two-minute budget.
- [ ] T082 Code review of the full Spec 005 diff in `src/manga_text_seg/web/`, `src/manga_text_seg/cli.py` and `tests/`; address CRITICAL and HIGH findings before finish-branch (constitution Review gate).

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately.
- **Foundational (Phase 2)**: Depends on Setup — **BLOCKS all user stories**. Every story below is an HTTP route, so `app.py`, `security.py`, `errors.py`, `session.py` and `schemas.py` must exist first.
- **User Stories (Phases 3–12)**: All depend on Foundational. US1 has no dependency on any other story; the rest may integrate with earlier stories but each is independently testable.
- **Polish (Phase 13)**: Depends on all desired stories being complete.

### User Story Dependencies

- **US1 (P1)**: Starts after Foundational — no dependency on other stories. Creates `routes/catalog.py`, `routes/page.py`, `routes/artifacts.py`.
- **US2 (P1)**: Starts after Foundational — **extends `routes/page.py`**, so run it after US1 to avoid a same-file conflict, though its contract tests are independent.
- **US3 (P1)**: Starts after Foundational — **also extends `routes/page.py`**. Same note as US2.
- **US4 (P2)**: Starts after Foundational — creates `routes/edit.py`; its re-render reuses Spec 4's renderer. Independent of US2/US3.
- **US5 (P2)**: Starts after Foundational — creates `routes/summary.py`; touches `app.py` for startup secret validation.
- **US6 (P2)**: Starts after Foundational — creates `routes/reports.py`; extends `routes/artifacts.py` for report serving.
- **US7 (P2)**: Starts after Foundational — **extends `routes/reports.py`** from US6.
- **US8 (P3)**: Starts after Foundational — creates `web/jobs.py` and `routes/jobs.py`; reuses the orchestration functions from US1, US5, US6 and US7 for its batch work.
- **US9 (P3)**: Starts after Foundational — extends `web/session.py`; its download test exercises artifacts produced by earlier stories.
- **US10 (P3)**: Starts after all in-app operations exist, because parity is defined against them.

### Within Each User Story

- Tests MUST be written and FAIL before the implementation they cover.
- Schemas and route modules before endpoint wiring.
- Core implementation before integration.
- Story complete before moving to the next priority.

### Parallel Opportunities

- T002–T006 (Setup) run in parallel.
- T007, T009, T011, T013, T016 (Foundational tests and schemas) run in parallel — different files.
- Every `[P]` test task across Phases 3–12 can run in parallel within its story.
- Once Foundational completes, US1, US4, US5, US6 and US8 touch disjoint files and can start in parallel; US2/US3 (both extend `routes/page.py`) and US7 (extends `routes/reports.py`) must follow their file's owner.
- T075–T079 (Polish audits) run in parallel.

---

## Parallel Example: User Story 1

```bash
# Launch all User Story 1 tests together (they are different files):
Task: "Contract test for GET /api/session in tests/contract/test_web_session_api.py"
Task: "Contract test for GET /api/catalog/* in tests/contract/test_web_catalog_api.py"
Task: "Unit test for the benchmark default method selection in tests/unit/test_web_benchmark_default.py"
Task: "Contract test for GET /api/page in tests/contract/test_web_page_api.py"
Task: "Contract test for GET /api/artifact in tests/contract/test_web_artifact_api.py"
Task: "Integration test for the open-a-page journey in tests/integration/test_web_open_page.py"
```

```bash
# Then the three route modules, each a different file:
Task: "Implement src/manga_text_seg/web/routes/catalog.py"
Task: "Implement src/manga_text_seg/web/routes/page.py"
Task: "Implement src/manga_text_seg/web/routes/artifacts.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL — blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: start the server, walk the catalog, open a page, fetch its original image
5. Demo the MVP

### Incremental Delivery

1. Setup + Foundational → the server starts and refuses the four SC-015 cases
2. US1 → a page opens and its artifacts serve → demo
3. US2 + US3 → the full chain and per-region detail → demo
4. US4 → editing and re-rendering → demo
5. US5 → page and chapter summaries → demo
6. US6 + US7 → both reports → demo
7. US8 + US9 + US10 → batch, downloads, CLI parity → demo
8. Polish → coverage, invariance and review gates

### Parallel Team Strategy

With multiple developers: complete Setup + Foundational together, then assign US1, US4, US5, US6 and US8 in parallel (disjoint files), with US2/US3 following US1 and US7 following US6.

---

## Notes

- [P] tasks = different files, no dependencies. Same-file extensions (US2→US1, US3→US1, US7→US6) carry no [P] and must be sequenced.
- [Story] label maps each task to its user story for traceability.
- The frontend bundle is a blocking precondition for a usable **demo** but not for the backend or its tests, which run against the Phase 1 fixture bundles (plan.md Dependencies). No task above writes frontend source.
- Every route returns JSON except `/api/artifact`, which returns raw bytes.
- Verify tests fail before implementing; commit after each task or logical group.
- Stop at any checkpoint to validate a story independently.
