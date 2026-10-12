# Implementation Plan: End-to-End Manga Translation Web Demo & Reporting

**Branch**: `005-web-e2e-reporting` | **Date**: 2026-10-10 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/005-web-e2e-reporting/spec.md`

## Summary

Spec 005 is the orchestration, demo-UI and reporting layer over Specs 1–4. It reads
their outputs read-only, presents one page's full artifact chain (original → prediction
mask → processed mask → inpainted → rendered), lets a user edit OCR/translation and
re-render, generates page/chapter summaries and quantitative/qualitative reports, and
persists everything into self-contained runs under `outputs/end-to-end/<run_id>/`.

**Technical approach.** A local client/server application. A Python backend
(`manga_text_seg.web`) owns all orchestration, artifact location, editing, re-rendering,
summary calls and report generation; it exposes a JSON + artifact-bytes HTTP API and
serves a static HTML/CSS/JS frontend bundle from one configurable root. The bundle is
authored outside this repository and is a blocking precondition (spec Assumptions); the
backend never generates UI. The CLI grows one subcommand, `serve`, that starts the
backend, and every other CLI operation reuses the same orchestration functions the API
handlers call, so in-app and CLI behaviour stay identical (FR-078).

## Technical Context

**Language/Version**: Python 3.10+ (existing `requires-python = ">=3.10"`)

**Primary Dependencies**: FastAPI (API layer) + Uvicorn (ASGI server). Everything else
is already in `pyproject.toml`: opencv-python, numpy, pandas, matplotlib, Pillow.
Existing internal modules are reused rather than reimplemented — `manifest`, `runs`,
`availability`, `selection`, `report`, `qualitative`, `boards`, `inpaint`,
`translation.*`.

**Storage**: Filesystem only. Upstream Spec 1–4 trees read-only; demo runs written under
`outputs/end-to-end/<run_id>/` (FR-063 tree). No database.

**Testing**: pytest (`testpaths = ["tests"]`, `addopts = "--cov=manga_text_seg ..."`).
Existing suites: `tests/unit`, `tests/integration`, `tests/contract`. New tests follow
the same three-way split. Run with `-o addopts=''` in this repo.

**Target Platform**: Researcher's local machine (Windows/Linux/macOS), CPU only
(FR-089). Loopback-bound, single user, no auth, no deployment (spec Assumptions).

**Project Type**: Single Python package (`src/manga_text_seg/`) with a local web-service
entry point. Not a frontend/backend monorepo — the frontend bundle is an external
artifact dropped into a static root, not source in this repository.

**Performance Goals**: Launch → first fully rendered page under 2 minutes (SC-001). No
throughput target; single user.

**Constraints**: Backend binds loopback only with a `Host`/`Origin` guard (FR-013c).
Every served path resolves inside an allow-listed artifact root, and
`data/no-need-to-read/` is never served (FR-013d, FR-008). Secrets come only from
environment variables, validated at startup, never in source/logs/metadata/outputs
(FR-042, FR-085). Upstream Spec 1–4 trees must remain checksum-identical (SC-011).

**Scale/Scope**: Manga109 benchmark pages already fully artifacted by Specs 2–4. 10 user
stories, 99 FRs, 15 SCs. One new module package, one new CLI subcommand, no change to
Spec 1–4 code paths.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Constitution v1.0.0, three principles.

| Principle | Gate | Status |
|---|---|---|
| **I. Task List Is the Contract** | No `tasks.md` exists for Spec 005 yet, so nothing is binding. Once `/speckit-tasks` output is approved, this plan must not be re-planned; deviations get surfaced for amendment. | PASS (not yet engaged) |
| **II. Spec-Scoped Test-First (NON-NEGOTIABLE)** | Every task below is written as a red-green-refactor pair: the test names the FR/SC it covers, is written first, and fails before the implementation lands. Tests cover only the current task's acceptance criteria — no speculative coverage. | PASS by construction |
| **III. Simplicity and Reuse (Ponytail Full Mode)** | FastAPI + Uvicorn are new dependencies. They are not strictly unavoidable — stdlib `http.server` could serve this API. This is therefore a recorded violation, justified in **Complexity Tracking** below. All other work reuses existing modules. | PASS with one recorded violation |

Additional quality gates:

- **Plan-stage ponytail gate**: satisfied — the module list below adds no abstraction
  with a single implementation, no config for a fixed value, and no scaffolding for
  later.
- **Security/never-weaken list**: `.gitignore`'s `.streamlit/secrets.toml` entry is
  **kept** (deleting a secret guard is exactly what Principle III forbids weakening),
  even though Streamlit is gone. FR-013c/FR-013d add the guards a threaded loopback
  server needs, which `st.production`-style single-process Streamlit did not require.
- **Pre-commit / test-scope / review gates**: unchanged from the constitution.

## Project Structure

### Documentation (this feature)

```text
specs/005-web-e2e-reporting/
├── plan.md              # This file
├── spec.md              # Feature specification
├── checklists/
│   └── requirements.md  # Spec quality checklist
└── tasks.md             # Phase 2 output (/speckit-tasks — NOT created by this plan)
```

`research.md`, `data-model.md`, `quickstart.md` and `contracts/` are not produced by this
plan: the upstream output contracts already exist in Specs 1–4, the data model is fixed
by spec.md's Key Entities, and the API contract lives in the section below rather than in
a separate `contracts/` directory.

### Source Code (repository root)

```text
src/manga_text_seg/
├── web/                      # NEW — the only new package
│   ├── __init__.py
│   ├── app.py                # FastAPI app factory; mounts API routers + static root
│   ├── security.py           # loopback bind, Host/Origin guard, artifact-root allow-list
│   ├── session.py            # BackendSession store (FR-022a), run ownership (FR-065)
│   ├── jobs.py               # BatchJob registry + progress (FR-084)
│   ├── errors.py             # FR-079 category enum → HTTP status mapping
│   ├── schemas.py            # request/response models for the endpoints below
│   └── routes/
│       ├── catalog.py        # /api/catalog/*  — selection enumeration (FR-014–017)
│       ├── page.py           # /api/page/*     — page bundle, regions, provenance
│       ├── edit.py           # /api/page/edit, /api/page/render (FR-031–038)
│       ├── summary.py        # /api/summary/*  (FR-039–046)
│       ├── reports.py        # /api/reports/*  (FR-047–062)
│       ├── artifacts.py      # /api/artifact   — the only byte-serving route (FR-013d)
│       └── jobs.py           # /api/jobs/*     — batch start/status (FR-084)
├── cli.py                    # MODIFIED — add `cmd_serve` + `serve` subparser + dispatch
├── manifest.py               # reused as-is
├── runs.py                   # reused as-is
├── availability.py           # reused as-is
├── selection.py              # reused as-is
├── report.py                 # reused as-is
├── qualitative.py            # reused as-is
├── boards.py                 # reused as-is
├── inpaint.py                # reused as-is
└── translation/              # reused as-is (ocr, translate, render, regions, intake, runs)

tests/
├── contract/                 # NEW: API surface contract tests (status codes, envelope)
├── integration/              # NEW: session lifecycle, artifact serving, batch isolation
└── unit/                     # NEW: security guards, error mapping, session store
```

Frontend bundle (not in this repository): a static HTML/CSS/JS directory dropped into
the configured static root, served verbatim.

**Structure Decision**: Option 1 (single project) with one added sub-package. The
repository already has one Python package with a flat module layout and a
three-way test split; Spec 005 adds `web/` beside it rather than restructuring. Option 2
(backend/ + frontend/) was rejected: the frontend is an external artifact with no source
here, so a `frontend/` tree would be an empty placeholder. The static bundle is served
from a configured path, so it needs no repository layout of its own.

## Application API Contract

Served by `manga-text-seg serve`. All responses are JSON except `/api/artifact`, which
returns raw bytes. The frontend bundle is served from the static root at `/`.

### Endpoints

| Method | Path | Purpose | Spec FRs |
|---|---|---|---|
| `GET` | `/api/session` | Issue or resume the session id; return current selection and owned run | FR-022, FR-022a, FR-065 |
| `GET` | `/api/catalog/mangas` | Manga/book list from the manifest | FR-014, FR-015 |
| `GET` | `/api/catalog/pages?manga=` | Pages (image_ids) for a manga, manifest order | FR-014, FR-015 |
| `GET` | `/api/catalog/methods` | Segmentation methods with on-disk masks; benchmark-selected default marked | FR-015, FR-017 |
| `GET` | `/api/catalog/runs?spec=3\|4` | Spec 3 / Spec 4 runs available for the selection | FR-014–016 |
| `GET` | `/api/catalog/languages` | Target languages recorded by compatible Spec 4 runs | FR-011, FR-015 |
| `GET` | `/api/page?image_id=&method=&run3=&run4=&lang=` | Full page bundle: status, provenance, region list, artifact refs | FR-018–027, FR-030 |
| `POST` | `/api/page/edit` | Record a manual edit (region_id, original, edited) | FR-031, FR-034 |
| `POST` | `/api/page/render` | Re-render affected regions/page using Spec 4's renderer | FR-032, FR-033, FR-036 |
| `POST` | `/api/summary/page` | Generate or return a cached page summary | FR-039–044 |
| `POST` | `/api/summary/chapter` | Generate a chapter summary from an ordered page list | FR-045, FR-046 |
| `POST` | `/api/reports/quantitative` | Generate the quantitative report into a run | FR-047–054 |
| `POST` | `/api/reports/qualitative` | Generate the qualitative report into a run | FR-055–062 |
| `GET` | `/api/artifact?path=` | Serve one stored artifact's bytes, byte-identical (FR-029) | FR-013d, FR-023, FR-028, FR-030a |
| `POST` | `/api/jobs/batch` | Start a batch over a page list; returns job id | FR-073, FR-084 |
| `GET` | `/api/jobs/{job_id}` | Job status and completed/total progress | FR-084 |
| `GET` | `/api/run/{run_id}/status` | Run configuration, page statuses, error summary | FR-077 |

`/api/artifact` is the **only** route that returns file bytes. It accepts a run-relative
artifact reference, never a caller-supplied filesystem path, resolves it inside the
allow-listed artifact roots, refuses traversal and refuses `data/no-need-to-read/`
(FR-013d, FR-008). Report artifacts (FR-048 CSV, FR-050 PNG, FR-061 HTML) and the
original page image (FR-023a) are served through this same route, which is what gives
FR-030a its serving path.

### Error envelope

Every non-2xx JSON response has the same shape:

```json
{ "error": { "category": "missing_prediction_mask", "message": "…", "detail": { } } }
```

`category` is one of the FR-079 set. HTTP status mapping:

| Status | Categories |
|---|---|
| 400 | `invalid_request` |
| 404 | `missing_prediction_mask`, `missing_metadata_sidecar`, `missing_inpainting_output`, `missing_ocr_output`, `missing_translation_output`, `undeterminable_run` |
| 409 | `duplicate_run_id`, `cross_run_mismatch`, `undeterminable_segmentation_method` |
| 422 | `page_without_text_regions` |
| 429 | `api_timeout_rate_limit` |
| 500 | `ocr_failure`, `translation_failure`, `rendering_failure`, `summary_failure`, `output_write_failure`, `internal` |
| 503 | `not_ready` |

A page-level failure never fails the request that carries the page bundle: `/api/page`
returns 200 with the page's status set to one of the eight FR-020 states, and the failure
is recorded in the run's error report (FR-026). The error envelope is for request-level
failures only. Ambiguity errors (`undeterminable_run`,
`undeterminable_segmentation_method`) put every candidate in `detail.candidates`
(FR-082); missing-artifact errors put the expected location in `detail.expected`
(FR-083).

## Frontend build and static-asset serving

- The bundle is a directory of static files. `serve` takes the static root as a config
  value (FR-086); the default is a path outside the Python package.
- The backend mounts that directory for `GET /` and its subpaths. No templating, no
  server-side rendering, no bundler step, no build hook (FR-013a, FR-013b).
- Replacing the UI is a directory swap; the backend is not restarted and no code changes
  (SC-014).
- If the static root is missing or empty, `serve` still starts, logs the condition, and
  `/` returns a `not_ready` envelope naming the expected root. It never substitutes a
  built-in UI.
- Session state lives server-side in `BackendSession` (FR-022a). The bundle holds only a
  session id, so a reload resumes the same session and the same run.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| New dependencies: FastAPI + Uvicorn (Principle III — avoid new dependencies unless the task cannot be completed without them) | The backend must serve a static bundle, ~17 JSON endpoints with typed request/response validation, and raw artifact bytes; run a long-lived batch job in the background while serving progress; and map ~19 error categories onto HTTP statuses consistently. Request validation and response modelling are the bulk of the surface. | Stdlib `http.server` + `json` was the first rung considered and is genuinely viable for a single-user loopback demo. It was rejected because it supplies no request parsing, no response serialization, no validation layer and no ASGI server: each of the ~17 endpoints would hand-roll query parsing and dict validation, and every FR-079 category would need manual status wiring — more code, not less, and the exact place where a validation gap becomes a path-traversal or a silently accepted bad request. FastAPI is ~2 lines to mount a static directory and gives typed validation at the trust boundary for free, which Principle III explicitly does not allow us to simplify away. Uvicorn is FastAPI's required ASGI server, not a separate choice. |

No other violations. No abstraction with one implementation, no config for a fixed
value, no scaffolding for later.

## Phase Plan

Phases are ordered so each one is independently testable against fixtures. Each item is
a red-green-refactor pair per Principle II.

**Phase 0 — Backend skeleton and guards**
1. `web/app.py` app factory + `serve` subcommand in `cli.py`; a health endpoint. Test:
   `serve` starts, binds loopback, `/` serves a fixture bundle.
2. `web/security.py`: Host/Origin guard, artifact-root allow-list, traversal refusal,
   `data/no-need-to-read/` refusal. Test: SC-015's four refusal cases.
3. `web/errors.py`: FR-079 category enum → status mapping, envelope serializer. Test:
   every category maps to its documented status.

**Phase 1 — Catalog and session**
4. `web/session.py`: session issue/resume, one run per session, reload resumes. Test:
   FR-022a reload case; two sessions get two runs.
5. `/api/catalog/*`: enumerate mangas, pages, methods, runs, languages from disk. Test:
   nothing hard-coded; empty-on-disk yields empty, not a default.
6. Benchmark default method selection with the FR-017 tie/no-result error. Test: SC-003
   fixtures.

**Phase 2 — Page bundle and artifact serving**
7. `/api/page`: assemble PageBundle from Spec 2–4 artifacts; eight FR-020 states;
   provenance from upstream only. Test: FR-020 state matrix; FR-019 missing-value case.
8. `/api/artifact`: byte-identical serving of images, JSON, CSV, PNG, HTML. Test:
   FR-029 byte equality; SC-002 incomplete-artifact matrix.
9. Page-level failure isolation. Test: FR-026 — failure recorded, session survives.

**Phase 3 — Editing and re-rendering**
10. `/api/page/edit` + `/api/page/render` via Spec 4's renderer. Test: FR-032 no
    segmentation/inpainting executed; FR-033 only affected page re-rendered.
11. Run persistence of edits: materialize on open, replace on re-render. Test: FR-036;
    SC-011 upstream checksums unchanged.

**Phase 4 — Summaries**
12. Summary adapter boundary, caching, env-var secret validation. Test: mock provider;
    FR-043 cache reuse; FR-085 no-leak scan.
13. Page and chapter summaries; chapter order preserved, missing pages listed. Test:
    SC-010.

**Phase 5 — Reports**
14. Quantitative report into a run (target run or auto-created). Test: SC-012
    artifact-only generation; SC-013 three coexisting runs.
15. Qualitative report with configurable selection policy. Test: FR-056 policy
    fixtures; FR-061 self-contained HTML.
16. Serve report artifacts through `/api/artifact`. Test: FR-030a.

**Phase 6 — Batch and CLI parity**
17. `web/jobs.py` + `/api/jobs/*`: background batch, progress, survives session end.
    Test: SC-009 failing page inside a batch; job status readable after session close.
18. CLI parity for every in-app operation. Test: FR-078 identical behaviour including
    ambiguity errors.

## Dependencies

Specs 1–4 are specification-only and are blocking preconditions (spec Dependencies). The
frontend bundle is likewise a blocking precondition for a usable demo, but not for the
backend or its tests, which run against fixture bundles.
