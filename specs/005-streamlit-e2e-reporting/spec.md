# Feature Specification: Streamlit End-to-End Manga Translation Demo & Reporting

**Feature Branch**: `005-streamlit-e2e-reporting`

**Created**: 2026-09-10

**Status**: Draft — ready for /speckit-plan. Clarification session 2026-09-10 resolved 4 questions: demo-run lifecycle (one auto-created run per app session / per CLI invocation, FR-065); self-contained runs (page artifacts materialized into the run on open, re-render replaces them, FR-063/FR-036); chapter = explicit user-supplied ordered page list with stable chapter/selection id stored in the run, book ≠ chapter (FR-046/FR-063/FR-075); reports always belong to a run — target an existing run for regeneration or auto-create one (FR-054/FR-068/FR-074). 90 FRs, 13 SCs, 10 user stories, 16 edge cases, 0 [NEEDS CLARIFICATION] markers.

**Input**: User description: "Xây dựng feature “Streamlit End-to-End Manga Translation Demo và Reporting” cho pipeline xử lý manga. Spec 1 cung cấp dataset, manifest, image_id, classical baseline và metrics. Spec 2 cung cấp prediction mask từ các phương pháp segmentation (deep learning chạy độc lập trên Google Colab). Spec 3 cung cấp kết quả xóa văn bản và inpainting. Spec 4 cung cấp OCR, translation và text rendering. Spec 5 là lớp orchestration, giao diện demo Streamlit và reporting cuối cùng. Không chạy lại segmentation model trong feature này. Không thay đổi các output của Spec 1–4. MVP chỉ làm việc với các ảnh benchmark đã có đầy đủ artifact từ Spec 2–4; upload ảnh manga bất kỳ là ngoài phạm vi MVP. Ứng dụng chính là một Streamlit app chạy local (`streamlit run app.py`) cho phép chọn manga/page/segmentation method/inpainting run/translation run/ngôn ngữ đích, xem toàn bộ kết quả từ ảnh gốc đến ảnh đã dịch, chỉnh sửa OCR/bản dịch rồi render lại, tạo summary, tải kết quả. Báo cáo định lượng đọc metrics từ Spec 1–2 so sánh bốn phương pháp; báo cáo định tính đọc output Spec 3–4 với chính sách chọn sample cấu hình được. Batch nhiều trang là chức năng mở rộng nhưng được thiết kế trong contract. Mỗi run có run_id, không ghi đè, upstream read-only. Không có secret trong source code, log hoặc output. Chỉ tạo specification, chưa viết code."

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Select and open a fully-artifacted manga page (Priority: P1)

As a user of the local demo application, I want to select a manga book, a page from the manifest, a segmentation method, a Spec 3 inpainting run, a Spec 4 translation run and a target language, and have the application load that page's complete end-to-end artifact chain, so that I can inspect a real translated page without running any model.

**Why this priority**: This is the MVP's core capability — connecting the outputs of Specs 1–4 into one viewable page. Everything else (editing, summaries, reports, batch) builds on a correctly loaded page bundle.

**Independent Test**: With fixture artifacts for one page, launch the app, make the six selections, and confirm the page loads with its provenance; repeat with an ambiguous configuration (two candidate runs, no explicit choice) and confirm an explicit error is shown instead of a random pick.

**Acceptance Scenarios**:

1. **Given** a manifest page with complete Spec 2–4 artifacts, **When** the user selects the manga, page, segmentation method, Spec 3 run, Spec 4 run and target language, **Then** the page loads and the provenance panel shows the image_id, segmentation method, inpainting algorithm, target language and run IDs recorded in the upstream metadata.
2. **Given** two Spec 3 runs exist and none is explicitly configured, **When** the user opens the page, **Then** the application shows a clear error naming the candidate runs and requires an explicit choice instead of selecting one at random.
3. **Given** the selected Spec 4 run was built on a different Spec 3 run than the one selected, **When** the user opens the page, **Then** the application reports the cross-run mismatch explicitly and does not present the mixed artifacts as if they were consistent.
4. **Given** a target language for which no Spec 4 run exists, **When** the user selects that language, **Then** the application shows an explicit error and does not re-run translation to satisfy the choice.

---

### User Story 2 - View the full visual chain of one page (Priority: P1)

As a user, I want to see, for the selected page, the original image, the prediction mask, the processed (post-dilation) mask, the inpainted image, the text-region overlay, and the finally rendered translated page, so that I can inspect every stage of the pipeline visually.

**Why this priority**: Seeing the chain from original to translated page is the demo's headline deliverable (MVP goal 4) and the fastest way to spot where quality is lost.

**Independent Test**: Load one fixture page and confirm all six visual artifacts render; load a page whose Spec 3 output is missing and confirm a distinct missing-artifact state instead of a broken view.

**Acceptance Scenarios**:

1. **Given** a complete page bundle, **When** the page is opened, **Then** the original image, Spec 2 prediction mask, Spec 3 processed (dilated) mask, Spec 3 inpainted image, region overlay (bounding boxes) and Spec 4 rendered page are all displayed.
2. **Given** a page whose prediction mask file is absent, **When** the page is opened, **Then** the application displays a distinct "missing prediction mask" state for that page while the rest of the session stays usable.
3. **Given** a page with no detected text regions, **When** the page is opened, **Then** the application shows the page with an explicit "no text regions" state and omits OCR/translation/rendering panels gracefully rather than erroring.
4. **Given** one region whose OCR failed, **When** the page is opened, **Then** that region is marked failed and every other region still displays its OCR text and translation.

---

### User Story 3 - Inspect per-region OCR text and translations (Priority: P1)

As a user, I want to see, for each text region of the selected page, its region identifier, position, recognized Japanese text and translated text together with their per-region statuses, so that I can verify what was recognized and translated region by region.

**Why this priority**: Region-level inspection is what makes the demo a research tool rather than a slideshow, and it is the substrate the editing story (US4) operates on.

**Independent Test**: Load a fixture page with three regions, one with a failed translation, and confirm the region list shows all three with correct statuses and the recorded texts, byte-identical to the upstream JSON artifacts.

**Acceptance Scenarios**:

1. **Given** a page with N regions in the Spec 4 metadata, **When** the page is opened, **Then** N region entries are listed with region_id, bbox, OCR text, translation and per-region statuses exactly as recorded upstream.
2. **Given** a region whose recorded translation status is "failed", **When** the user inspects it, **Then** the failure is shown for that region only and the other regions are unaffected.

---

### User Story 4 - Edit OCR text or translation and re-render (Priority: P2)

As a user, I want to edit the OCR text or the translation of individual regions and re-render the affected page, so that I can correct recognition or translation mistakes without re-running segmentation or inpainting.

**Why this priority**: Editing closes the loop from inspection to correction and is an explicit MVP goal (goal 5, acceptance criteria 6–7), but it depends on the page being loadable (US1–US3).

**Independent Test**: Edit the translation of one region on a fixture page, trigger re-render, and confirm: only that page's rendering is redone, no upstream file changes, the edit is logged with before/after text, and the output lands in the demo run's own directory.

**Acceptance Scenarios**:

1. **Given** an open page, **When** the user edits one region's translation and triggers re-render, **Then** only the affected page is re-rendered using the Spec 4 rendering configuration, and segmentation and inpainting are not re-run.
2. **Given** an edited region, **When** the re-render completes, **Then** the demo run stores the re-rendered image and updated OCR/translation JSON under its own page directory, and the Spec 1–4 outputs are byte-identical to before.
3. **Given** any manual edit, **When** the edit is saved, **Then** a manual-edit record exists containing region_id, the original text, the edited text, the manual-edit status and — where available — the editor identity and timestamp.

---

### User Story 5 - Generate page and chapter summaries (Priority: P2)

As a user, I want to generate a content summary for the open page from its dialogue texts, and optionally a chapter summary across an explicitly selected ordered page list (the "chapter", per FR-046), so that I can grasp a page's (or chapter's) content quickly.

**Why this priority**: Summaries are MVP goal 8 and add narrative value, but they depend on an external provider and are explicitly fail-soft: their failure must never take the translated image down with them.

**Independent Test**: With a mock summary provider, request a page summary and confirm the provider receives the page's dialogue texts, the result is cached on identical re-request, and a forced provider failure leaves the page's translated image intact with the failure recorded.

**Acceptance Scenarios**:

1. **Given** a page with OCR/translation results, **When** the user requests a summary, **Then** a summary is produced through the provider adapter in the configured language (default Vietnamese) and stored in the demo run.
2. **Given** the same page, texts, language, provider and model, **When** a summary is requested again, **Then** the cached summary is reused and no new provider call is made.
3. **Given** the summary provider fails, **When** the summary is requested, **Then** the page's translated image and all other outputs are retained, the failure is recorded in the run's error report, and no summary text is fabricated.
4. **Given** a chapter whose page summaries are incomplete, **When** a chapter summary is requested for its selected ordered page list, **Then** the missing pages are explicitly listed in selected order and no content is invented for them.

---

### User Story 6 - Generate the quantitative comparison report (Priority: P2)

As a researcher, I want a quantitative report that reads the persisted metrics of Spec 1 and Spec 2 and compares the four segmentation methods (classical baseline, Manga-Text-Segmentation, comic-text-detector, UNet++ EfficientNetV2), so that I can present method performance without recomputing anything.

**Why this priority**: The quantitative report is one of the feature's two headline deliverables and must work purely from existing artifacts (acceptance criterion 17).

**Independent Test**: Point the report generator at fixture metric files and confirm the tables reproduce the persisted values exactly, the four methods appear, an unavailable method is shown as unavailable with its reason, and no value is fabricated.

**Acceptance Scenarios**:

1. **Given** persisted per-method metrics from Spec 1 and Spec 2, **When** the report is generated, **Then** tables of IoU, Precision, Recall and F1 (mean and standard deviation), average processing time and success/failure page counts appear for every available method.
2. **Given** one deep-learning method with no benchmark result, **When** the report is generated, **Then** that method is listed as unavailable with its recorded reason and no metric value is interpolated or invented for it.
3. **Given** the report is generated, **Then** it records the benchmark provenance including Method A's leave-one-fold-out checkpoint strategy and fold/checkpoint attribution, the run IDs consumed, and the execution environment.

---

### User Story 7 - Generate the qualitative report (Priority: P2)

As a researcher, I want a qualitative report built from Spec 3 and Spec 4 outputs that presents best, worst and typical cases plus failure categories, with a configurable sample-selection policy, so that I can show where the pipeline succeeds and fails concretely.

**Why this priority**: The qualitative report is the second headline deliverable; its default sample policy was the one explicitly flagged decision in the requester's brief and was resolved as the combined best-N + worst-N + typical-N default.

**Independent Test**: With fixture artifacts ranked by known segmentation metrics, apply each selection policy and confirm the selected pages match the policy exactly; confirm failure sections enumerate all recorded failures and the report keeps the five quality dimensions separate.

**Acceptance Scenarios**:

1. **Given** a configured selection policy and N, **When** the report is generated, **Then** the showcased pages match the policy exactly (e.g. best-N are the N highest-ranked pages by the selected segmentation metric) and the report states which policy produced each section.
2. **Given** pages with recorded OCR, translation or rendering failures, **When** the report is generated, **Then** the corresponding failure sections enumerate all of them, not a sample.
3. **Given** the report, **When** read, **Then** segmentation quality, inpainting quality, OCR result, translation result and rendering result are each presented under their own dimension, and no downstream quality substitutes for a segmentation metric.

---

### User Story 8 - Run a batch of pages (Priority: P3)

As a user, I want to run the demo over a list of image_ids or the whole manifest with visible progress, so that I can produce outputs and a batch summary for many pages while individual page failures do not stop the run.

**Why this priority**: Batch is an explicitly designated extension (MVP is single-page), but its contract — batch summary fields, error isolation, progress — must be designed now so the MVP does not paint itself into a corner.

**Independent Test**: Run a batch fixture of three pages where one page fails at artifact loading, and confirm the other two complete, the batch summary counts reconcile, and the failure is in the error report.

**Acceptance Scenarios**:

1. **Given** a batch of pages where one page fails, **When** the batch runs, **Then** the remaining pages complete and the failed page is recorded in the error report.
2. **Given** a completed batch, **When** the batch summary is read, **Then** it contains total/successful/failed page counts, region counts, OCR/translation/rendering success and failure counts, per-stage timings and error categories.

---

### User Story 9 - Download results from the app (Priority: P3)

As a user, I want to download the rendered translated image and the metadata, OCR and translation results of the open page, so that I can take the outputs out of the demo for slides, reports or further processing.

**Why this priority**: Downloads make the demo's outputs portable (MVP goal, acceptance criteria 8), but they are a convenience layered on already-stored artifacts.

**Independent Test**: Open a fixture page, download each offered artifact, and confirm each download is byte-identical to the stored file in the demo run.

**Acceptance Scenarios**:

1. **Given** an open page with a rendered result, **When** the user downloads the image, **Then** the downloaded file is byte-identical to the stored rendered image in the demo run.
2. **Given** an open page, **When** the user downloads the metadata and OCR/translation results, **Then** the downloaded JSON files are byte-identical to the stored artifacts.

---

### User Story 10 - Perform equivalent operations from the CLI (Priority: P3)

As a user working in a terminal or scripting the demo, I want a CLI covering the app's core operations — running the demo for one or many image_ids, generating reports from existing outputs, creating page/chapter summaries, selecting method/runs/language, checking run status, retrying failed pages and launching the Streamlit app — so that the feature is usable without the browser.

**Why this priority**: The Streamlit app is the primary deliverable; the CLI mirrors it for automation and reproducibility and reuses the same orchestration layer, so it is naturally third.

**Independent Test**: Execute each CLI operation against fixture artifacts and confirm each behaves identically to its in-app counterpart, including error messages for ambiguous selections.

**Acceptance Scenarios**:

1. **Given** fixture artifacts, **When** the CLI runs the demo for one image_id, **Then** the produced demo run is identical in structure and metadata to what the app would produce for the same configuration.
2. **Given** an existing demo run, **When** the CLI is asked for run status, **Then** it reports the run's configuration, page statuses and error summary without modifying anything.
3. **Given** existing artifacts, **When** the CLI generates a report, **Then** no segmentation, OCR, translation or summary model is executed.

---

### Edge Cases

- What happens when the prediction mask is missing for a selected page? → distinct per-page state, error recorded, nothing substituted.
- What happens when the Spec 3 metadata sidecar is missing? → the Spec 3 bundle is invalid input; explicit error, per inherited Spec 3 contract.
- What happens when the inpainting output is missing? → distinct page state; the remaining viewable artifacts are still shown where possible.
- What happens when OCR or translation output is missing for a page? → distinct page state; region panels omitted with explanation.
- What happens when no segmentation method can be determined (no benchmark result, or an unbroken tie)? → explicit error; never a random choice.
- What happens when several Spec 3 or Spec 4 runs could serve the request? → explicit error naming all candidates; the user must choose.
- What happens when a page has no text regions? → explicit "no text regions" state; not treated as a hard error.
- What happens when one region's OCR or translation fails? → that region is marked failed; other regions and the page continue.
- What happens when rendering fails for a page? → page marked rendering-failed; upstream artifacts unaffected; error recorded.
- What happens when the summary provider times out or is rate-limited? → summary marked failed; translated image retained; retry/timeout policy applies as for any provider call.
- What happens when writing an output file fails (disk, permissions)? → recorded as output-write failure; the run reports which artifact could not be written.
- What happens when a run_id already exists? → the run is refused unless an explicit overwrite option was given.
- What happens when the selected Spec 4 run's segmentation method or inpainting source disagrees with the other selections? → explicit cross-run mismatch error; no silent mixing.
- What happens when a page in the manifest has no entry in the selected Spec 4 run? → distinct page state; listed as missing OCR/translation output.
- What happens when chapter summaries are incomplete? → chapter summary lists the missing pages in order; no invented content.
- What happens when an upstream artifact changes or disappears mid-session? → the next load re-validates every location; stale views are never silently served.

## Clarifications

### Session 2026-09-10

- Q: When is a demo run (run_id) created — automatically per app session, or only when the
  user explicitly starts one? → A: Automatically: the app creates one new run with an
  auto-generated run_id per app session; the CLI equivalently creates one run per demo
  invocation. Bound by amending FR-065.
- Q: For a page that was never edited, are the rendered image and OCR/translation JSON copied
  into the demo run on page open, or only written after an edit/re-render? → A: Copied on open —
  every successfully opened page's artifacts are materialized into the run, making each run a
  self-contained snapshot; a later re-render replaces them. Bound by amending FR-063 and FR-036.
- Q: What does a "chapter" correspond to for chapter summaries — all pages of one manga/book per
  manifest order, or a user-supplied page list? → A: An explicitly selected ordered page list
  supplied by the user. Manga109 data has no chapter-boundary metadata, so a book is never assumed
  to equal a chapter; the selected page list and a stable chapter/selection id are stored in the run
  output; whole-book summarization is out of scope and would be a separate future book-summary mode.
  Bound by amending FR-046, FR-063, FR-075, US5, SC-010, ChapterSummaryRecord and Out of Scope.
- Q: Where are reports generated from existing artifacts written — always into a demo run, or a
  standalone location outside the run tree? → A: Always into a run: a report command MAY target an
  existing run to regenerate into; otherwise it auto-creates a new run to hold the reports, mirroring
  the per-invocation run lifecycle of FR-065. Nothing is ever written outside the run layout. Bound
  by amending FR-054, FR-068 and FR-074.

## Requirements *(mandatory)*

### Reuse of the Spec 1–4 foundation (FR-001–012)

- **FR-001**: The system MUST consume Spec 1–4 outputs exclusively through each spec's documented output contracts and MUST treat all data and outputs of Specs 1–4 as read-only.
- **FR-002**: The system MUST use the Spec 1 manifest as the single page index and MUST NOT re-derive, re-create or extend it.
- **FR-003**: The system MUST locate Spec 2 prediction masks by page identifier plus segmentation method within a selected Spec 2 run identified by its experiment name/run ID (inheriting Spec 2's run-identification contract).
- **FR-004**: The system MUST locate Spec 3 outputs (raw prediction mask copy, processed/dilated mask, inpainted images, per-page metadata sidecar) by Spec 3 run ID, inpainting algorithm and image_id, and MUST treat a mask whose sidecar is absent as invalid input (inheriting Spec 3 FR-009).
- **FR-005**: The system MUST locate Spec 4 outputs (crops, ocr.json, translations.json, rendered.png, metadata.json) by Spec 4 run ID, segmentation method and image_id, per Spec 4's FR-048 tree.
- **FR-006**: The system MUST NOT run any segmentation model, re-run inpainting, or modify any prediction mask, ground-truth mask, inpainting output or Spec 4 output; text edits re-render using Spec 4's rendering component only.
- **FR-007**: The MVP MUST operate only on precomputed benchmark artifacts that already exist for the selected page; uploading new manga images is out of scope for the MVP.
- **FR-008**: The main demo MUST NOT use ground-truth masks, and MUST NOT read `data/no-need-to-read/` (inherited prohibitions).
- **FR-009**: The end-to-end pipeline stages MUST consume the immediately preceding spec's outputs in order: manifest → segmentation method → Spec 2 mask → Spec 3 inpainting bundle → Spec 4 OCR/translation/rendering bundle → display → edit → re-render → summary → demo-run persistence.
- **FR-010**: Before presenting a page as consistent, the system MUST validate cross-run consistency: the selected Spec 4 run's recorded segmentation method and inpainting source MUST match the selected segmentation method and selected Spec 3 run; any mismatch MUST produce an explicit error naming the disagreement.
- **FR-011**: The target-language selection MUST resolve to Spec 4 runs whose recorded configuration names that language; if no such run exists the system MUST show an explicit error and MUST NOT re-run translation to satisfy the choice.
- **FR-012**: The system MUST recover the Spec 2 run ID for provenance display from the Spec 3 sidecar's recorded upstream mask provenance (or the Spec 4 metadata where present) and MUST NOT infer or guess it.

### Streamlit application and artifact selection (FR-013–022)

- **FR-013**: The primary deliverable MUST be a local web application startable with the documented command (`streamlit run app.py`); Streamlit is a deliberately mandated product requirement from the requester, as are the named upstream method and algorithm contracts. The application runs locally only; deployment is out of scope.
- **FR-014**: The application MUST offer selection of: manga/book; page (image_id) from the manifest; segmentation method; Spec 3 inpainting run; Spec 4 translation run; and target language.
- **FR-015**: All selectable choices MUST be enumerated from artifacts that actually exist on disk (present runs, methods, pages, recorded languages); nothing may be hard-coded.
- **FR-016**: The system MUST NOT select any artifact randomly or arbitrarily; whenever multiple candidates exist and no deterministic rule resolves them, it MUST show a clear error naming the candidates and require the user to choose.
- **FR-017**: When no segmentation method is specified, the default MUST be the Spec 2 benchmark-selected method, chosen by IoU and F1-score as primary criteria with Precision, Recall and processing time as secondary criteria; if the benchmark has no results, or multiple methods are tied with no tie-breaking rule, the system MUST raise a clear error instead of picking. The user MAY always override the method.
- **FR-018**: The application MUST display, for every loaded page, a provenance panel containing: image_id; segmentation method; Spec 2 run ID; Spec 3 inpainting run ID; Spec 4 translation run ID; inpainting algorithm; target language; OCR engine/version; translation provider/model; rendering font/configuration; and summary provider/model/version where a summary exists.
- **FR-019**: Every provenance value MUST come from upstream metadata or sidecars; values that are not recorded upstream MUST be displayed as explicitly missing, never filled in.
- **FR-020**: The application MUST recognise and display distinctly the eight single-page demo states: valid input; missing prediction mask; missing inpainting output; missing OCR/translation output; page without text regions; region OCR failure; region translation failure; rendering failure.
- **FR-021**: Every selection change (method, page, run, language) MUST re-validate all artifact locations before anything is displayed.
- **FR-022**: The application state (current manga, page, method, runs, language) MUST be preserved across interactions within a session so the user does not lose their selections while inspecting or editing.

### Page display, regions and downloads (FR-023–030)

- **FR-023**: For a valid page the application MUST display: the original page image; the Spec 2 prediction mask; the Spec 3 processed (post-dilation) mask; the Spec 3 inpainted image for the selected algorithm; a region overlay with bounding boxes over the page; and the Spec 4 rendered page.
- **FR-024**: The application MUST present a per-region list containing region_id, bounding box, reading-order index, OCR text, translation and per-region statuses, exactly as recorded in the Spec 4 metadata.
- **FR-025**: Region isolation MUST hold in the UI: one failed region MUST NOT hide or corrupt the display of other regions.
- **FR-026**: A page-level failure MUST be displayed as such and recorded in the run's error report; it MUST NOT terminate the application session.
- **FR-027**: Everything displayed MUST be exactly what the upstream artifacts record — the system MUST NOT fabricate data or silently substitute artifacts.
- **FR-028**: The application MUST let the user download the rendered image and the metadata, OCR and translation JSON artifacts of the open page.
- **FR-029**: Downloaded content MUST be byte-identical to the stored demo-run artifacts.
- **FR-030**: A page without text regions MUST be displayed with an explicit "no text regions" state, omitting OCR/translation/rendering panels gracefully.

### OCR/translation editing and re-rendering (FR-031–038)

- **FR-031**: The application MUST allow editing the OCR text and the translation of individual regions.
- **FR-032**: Re-rendering after a text edit MUST use the Spec 4 rendering component and its recorded rendering configuration, and MUST NOT re-run segmentation or inpainting.
- **FR-033**: The system MUST re-render only the affected regions or page — unaffected pages in the same session MUST NOT be re-rendered.
- **FR-034**: Every manual edit MUST be recorded with: region_id; the original text; the edited text; the manual-edit status; and, where available, the editing user and timestamp.
- **FR-035**: Edited outputs MUST be stored in a new demo run or an explicitly identified revision; upstream Spec 1–4 outputs MUST NEVER be overwritten.
- **FR-036**: The re-rendered page image and the updated OCR/translation JSON MUST be written into the demo run's page directory for that image_id, replacing the artifacts materialized on open (Clarifications, 2026-09-10).
- **FR-037**: Page provenance after an edit MUST reflect the manual modification (edit count or revision marker) so downstream viewers can tell edited pages from pristine ones.
- **FR-038**: Manual edits MUST NOT alter any metric and MUST NOT be used to evaluate segmentation.

### Page and chapter summaries (FR-039–046)

- **FR-039**: Page summaries MUST be generated from the page's dialogue texts as recognised (OCR) and translated.
- **FR-040**: The summary language MUST be configurable with Vietnamese as the default.
- **FR-041**: The summary provider MUST be accessed through a provider adapter; the concrete provider, model and version MUST be recorded for every generated summary.
- **FR-042**: Summary API keys/secrets MUST be supplied exclusively through environment variables, validated at startup, and MUST NEVER appear in source code, logs, metadata or outputs.
- **FR-043**: The system MUST cache summaries keyed on the input content, target language, provider and model; an identical request MUST reuse the cached result instead of calling the provider again.
- **FR-044**: A summary failure MUST be isolated: the page's translated image and all other outputs MUST be retained, the failure MUST be recorded in the error report, and no summary text may be fabricated.
- **FR-045**: Summaries MUST NOT be used to evaluate segmentation, OCR or translation quality.
- **FR-046**: A chapter summary MUST be generated from an explicitly selected, user-supplied ordered page list (the "chapter") — the app and the CLI MUST accept such a list — because the Manga109 data carries no chapter-boundary metadata and a book MUST NEVER be assumed to equal a chapter (Clarifications, 2026-09-10). The selected page list and a stable chapter/selection id MUST be stored in the demo run. The chapter summary MUST preserve the selected page order; any page lacking a summary MUST be explicitly listed as missing, and no content may be invented for it. A chapter summary is not required unless the constituent page summaries exist.

### Quantitative reporting (FR-047–054)

- **FR-047**: The quantitative report MUST read the persisted metrics of Spec 1 and Spec 2 and MUST NOT recompute any metric from images.
- **FR-048**: The report MUST include, per available method: IoU, Precision, Recall and F1-score (mean and standard deviation), average processing time with standard deviation, and successful/failed page counts.
- **FR-049**: The report MUST present all four methods: classical baseline, Manga-Text-Segmentation, comic-text-detector and UNet++ EfficientNetV2.
- **FR-050**: The report MUST include a comparison chart of the four methods across the primary metrics.
- **FR-051**: A method without benchmark results MUST be shown as unavailable with its recorded reason; the report MUST NOT interpolate, estimate or fabricate any metric for an unavailable method.
- **FR-052**: The report MUST record the benchmark provenance: the checkpoint strategy of Method A (the leave-one-fold-out strategy inherited from Spec 2), the fold/checkpoint attribution, whether LOFO was used, the Spec 1/Spec 2 run identifiers consumed, and the execution environment.
- **FR-053**: The report MUST be reproducible from existing artifacts alone, with no model execution.
- **FR-054**: The report artifacts (metrics summary table and comparison chart) MUST be written into a demo run's report directory — reports never live outside the run layout (Clarifications, 2026-09-10): generation MAY target an explicitly selected existing run for regeneration, and otherwise writes into a newly created run (FR-068, FR-074).

### Qualitative reporting (FR-055–062)

- **FR-055**: The qualitative report MUST contain sections for: best cases; worst cases; typical cases; OCR failures; translation failures; rendering failures; and inpainting artifacts.
- **FR-056**: The sample-selection policy MUST be configurable among `first-N`, `best-N`, `worst-N`, `typical-N` and `manual-list`, with the count N configurable; the DEFAULT MUST be the combined policy that generates the best-N, worst-N and typical-N sections together (N pages per section), per the requester's resolution of 2026-09-10; any single policy (e.g. best-N only) MUST remain selectable by configuration; the report MUST always state which policy produced each section.
- **FR-057**: Best/worst/typical sections MUST be ranked by the selected method's segmentation metric (IoU/F1 from the persisted Spec 1/2 metrics); "typical" MUST mean pages whose metric lies closest to that method's mean.
- **FR-058**: The failure sections (OCR, translation, rendering) MUST enumerate ALL pages/regions with the corresponding recorded failure status — they are never sampled down.
- **FR-059**: The inpainting-artifacts section MUST be populated via the configured selection policy or an explicit manual list; since no quantitative inpainting metric exists in the pipeline, the report MUST NOT fabricate an inpainting score for any page.
- **FR-060**: The report MUST present segmentation quality, inpainting quality, OCR result, translation result and rendering result each under its own clearly separated dimension, and MUST NOT use OCR, translation or rendering quality as a substitute for segmentation metrics.
- **FR-061**: The qualitative report MUST be delivered as a self-contained HTML document that embeds the referenced page images and states each showcased page's identifiers and provenance.
- **FR-062**: Report generation MUST NOT modify any upstream artifact.

### Output structure and run management (FR-063–072)

- **FR-063**: Demo outputs MUST follow this structure:

```text
outputs/end-to-end/
  <run_id>/
    config.json
    report/
      metrics_summary.csv
      metrics_comparison.png
      qualitative_report.html
      batch_summary.json
      errors.json
    pages/
      <manga>/
        <NNN>/
          rendered.png
          summary.txt
          metadata.json
          ocr.json
          translations.json
    chapter_summary/
      <chapter_id>/
        selection.json
        summary.txt
```

  Files not applicable to a given run (e.g. chapter_summary when no chapter summary was requested) MAY be absent, but nothing outside this layout may be invented. Each requested chapter gets one `<chapter_id>/` directory whose selection.json records the user-supplied ordered page list and whose summary.txt holds the chapter summary (FR-046). The run is a self-contained snapshot (Clarifications, 2026-09-10): when a page is first opened successfully, the rendered image and the OCR/translation JSON shown for it MUST be materialized into that page's directory in the run, so every displayed page's artifacts exist in the run even without any edit.
- **FR-064**: Each page's metadata.json MUST contain: image_id; segmentation method; Spec 2 run ID; Spec 3 run ID; Spec 4 run ID; inpainting method; target language; OCR engine/version; translation provider/model/version; rendering configuration; summary provider/model/version (when a summary exists); page status; per-region statuses; per-stage timings; errors; and manual edits.
- **FR-065**: Every demo run MUST have a run_id; a colliding run_id MUST be refused unless an explicit overwrite option was given by the user. The run lifecycle is fixed (Clarifications, 2026-09-10): the app automatically creates one new run with an auto-generated run_id per app session, and the CLI creates one run per demo invocation.
- **FR-066**: Different runs MUST coexist side by side; the run's configuration MUST be saved with the run (config.json).
- **FR-067**: Demo outputs MUST be separated from the benchmark outputs of Specs 1–4 (the `outputs/end-to-end/` tree), and upstream trees MUST remain read-only.
- **FR-068**: The system MUST support generating reports from already-existing artifacts without executing any model, writing them into a run per FR-054 (a targeted existing run for regeneration, or an auto-created new run).
- **FR-069**: Locally executed stages (loading, selection, validation, re-rendering with fixed configuration) MUST be deterministic; translation and summary nondeterminism, where a provider exhibits it, MUST be recorded as a provider property.
- **FR-070**: The system MUST record per-stage timings for the pipeline stages it executes itself (loading, validation, re-rendering, summary request, report generation).
- **FR-071**: The system MUST produce an error report (errors.json) covering the error categories of FR-079.
- **FR-072**: Page metadata MUST validate against the FR-064 schema; an invalid metadata file MUST be reported as an error, never silently accepted.

### CLI (FR-073–078)

- **FR-073**: The CLI MUST support running the demo for a single image_id and for a list of image_ids (or the whole manifest).
- **FR-074**: The CLI MUST support generating the quantitative report, the qualitative report and the batch summary from existing outputs; the command MUST accept an optional target run to regenerate into and MUST otherwise auto-create a new run to hold the reports (FR-054, FR-065).
- **FR-075**: The CLI MUST support creating a page summary and a chapter summary; a chapter summary invocation MUST take the explicit ordered page list defining the chapter (FR-046).
- **FR-076**: The CLI MUST support explicit selection of: segmentation method; Spec 3 inpainting run; Spec 4 translation run; target language; and run ID.
- **FR-077**: The CLI MUST support checking run status and retrying failed pages of an existing run.
- **FR-078**: The CLI MUST provide a command that launches the Streamlit application, and every CLI operation MUST behave identically to its in-app counterpart, including ambiguity errors.

### Error handling (FR-079–084)

- **FR-079**: The system MUST detect and record the following error categories: missing prediction mask; missing metadata sidecar; missing inpainting output; missing OCR output; missing translation output; undeterminable segmentation method; undeterminable run; OCR failure; translation failure; rendering failure; summary failure; API timeout/rate limit; output write failure; page without text regions; duplicate run ID.
- **FR-080**: A failed region MUST NOT invalidate its page's other regions, and a failed page MUST NOT stop a batch.
- **FR-081**: The system MUST NOT fabricate data or silently substitute artifacts under any error condition.
- **FR-082**: Ambiguity errors (multiple candidate runs/methods) MUST name every candidate so the user can choose explicitly.
- **FR-083**: Missing-artifact errors MUST name the expected location or contract so the missing artifact can be produced by the right upstream spec.
- **FR-084**: Batch runs MUST expose visible progress (pages completed / total) while running.

### Secrets, configuration and testing (FR-085–090)

- **FR-085**: No API key, secret or credential MUST appear in source code, repository configuration, logs, metadata or outputs; provider secrets MUST come exclusively from environment variables validated at startup.
- **FR-086**: All configuration (method/run/language defaults, selection policy, N values, rendering and summary settings, output locations) MUST live outside source code in configuration files or explicit options.
- **FR-087**: The implementation MUST include tests covering: benchmark-based best-method selection; unavailable-method rejection; artifact loading from Specs 2–4; single-page demo from fixtures; pages missing one artifact; pages without text regions; one failing page inside a batch; best/worst/typical sample selection; report metric correctness; application state across method/page/run changes; OCR/translation editing; re-render after edit; summary adapter with a mock provider; summary caching; secret-leak prevention; run-id non-overwrite; chapter-summary page ordering; and upstream-artifact invariance.
- **FR-088**: The default test suite MUST run on small fixtures only — no full-dataset pass, no real provider calls, no large model downloads.
- **FR-089**: All locally executed stages MUST run on CPU; the feature adds no GPU requirement.
- **FR-090**: The system MUST provide a way to verify upstream invariance (that Spec 1–4 trees are unchanged by demo activity), e.g. checksums or recorded modification times.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user can go from launching the application to viewing a complete translated page (all six visual artifacts plus the region list) in under 2 minutes for any manifest page whose artifacts exist.
- **SC-002**: 100% of manifest pages with complete Spec 2–4 artifacts can be selected and displayed; every incompletely artifacted page produces one of the eight defined page states, never a crash or a blank screen.
- **SC-003**: Zero random artifact selections occur: in every ambiguity fixture (multiple candidate runs, tied benchmark methods, no benchmark results), the system shows an explicit error naming the candidates, verified by automated tests.
- **SC-004**: An OCR/translation edit plus re-render completes without any segmentation or inpainting execution, verified by execution monitoring and by upstream checksums remaining unchanged.
- **SC-005**: A secrets audit of the source tree, logs, metadata and outputs finds zero API keys or credentials.
- **SC-006**: Every value in the quantitative report matches the persisted upstream metric files exactly (no recomputation drift), verified by automated comparison.
- **SC-007**: Every method without benchmark results appears as unavailable with its reason, and the report contains zero fabricated metric values, verified by automated inspection.
- **SC-008**: For every selection policy — including the default combined best-N + worst-N + typical-N policy — the showcased pages match the policy exactly (e.g. best-N are exactly the N highest-ranked pages by the configured metric), verified by automated tests on ranked fixtures.
- **SC-009**: A batch containing a deliberately failing page completes all remaining pages, and the batch summary's counts reconcile exactly with the per-page statuses and the error report.
- **SC-010**: A chapter summary preserves the selected page order and explicitly lists every page lacking a summary; zero invented content, verified by automated test with a fixture chapter (an explicit ordered page list per FR-046) missing one page.
- **SC-011**: After any demo activity — browsing, editing, re-rendering, summarising, reporting — the Spec 1–4 trees are checksum-identical to their pre-session state.
- **SC-012**: Reports can be regenerated from existing artifacts in a session in which no segmentation, OCR, translation or summary provider is reachable, proving report generation is artifact-only.
- **SC-013**: At least three demo runs with different configurations coexist under `outputs/end-to-end/` without any overwrite.

## Key Entities

- **EndToEndRun**: One demo execution. Attributes: run_id, DemoConfig, page bundles, report artifacts, error report, per-stage timings; creation refused on run_id collision without explicit overwrite.
- **DemoConfig**: The run's fixed configuration. Attributes: segmentation method (explicit or benchmark default), Spec 3 run ID, Spec 4 run ID, target language, summary language, selection policy and N, rendering settings, output locations, overwrite flag.
- **PageBundle**: Everything displayable for one image_id. Attributes: original image, prediction mask, processed mask, inpainted image, region list, OCR texts, translations, rendered image, summary, page status, per-region statuses, provenance; assembled read-only from Spec 1–4 artifacts plus demo-run edits.
- **ProvenanceRecord**: The FR-018 field set for one page; every value sourced from upstream metadata or sidecars, or explicitly marked missing.
- **ArtifactLocator**: Resolves (spec, run ID, method, image_id) tuples to concrete artifact paths per the upstream contracts; reports missing/ambiguous instead of guessing.
- **ManualEditRecord**: One manual edit. Attributes: region_id, original text, edited text, editor identity (when available), timestamp (when available), manual-edit status.
- **PageSummaryRecord**: One page summary. Attributes: image_id, summary text, language, provider/model/version, cache key inputs, status.
- **ChapterSummaryRecord**: One chapter summary. Attributes: chapter/selection id, user-supplied ordered page list (FR-046), per-page summary presence, missing-page list, language, provider/model/version.
- **BatchSummary**: Aggregate of a batch run. Attributes: total/successful/failed page counts, region counts, OCR/translation/rendering success and failure counts, per-stage timings, error categories.
- **QuantitativeReport**: The FR-047–054 artifact set: metric tables, comparison chart, availability statuses, benchmark provenance (LOFO, folds, run IDs, environment).
- **QualitativeReportSpec**: Configuration of one qualitative report. Attributes: selection policy (`first-N`/`best-N`/`worst-N`/`typical-N`/`manual-list`, with `combined` — best-N + worst-N + typical-N together — as the default), N per section, manual page list, ranking metric, sections to include.
- **SummaryProviderAdapter**: The pluggable provider boundary for summaries. Attributes: provider/model/version identity, retry/timeout/rate-limit policy, cache, secret requirements (environment-variable key).
- **ErrorReport**: The run's structured error list over the FR-079 categories, each entry naming the affected page/region and the expected artifact location.

## Assumptions

- **Upstream maturity (blocking precondition)**: Specs 1–4 are specification-only as of 2026-09-10. Their implementations are hard prerequisites; this spec defines the orchestration contract against their documented output trees.
- **Summary provider pattern**: consistent with the requester's resolved Spec 4 decision, the summary provider is a cloud LLM API behind the adapter, with its key supplied via an environment variable, caching, and recorded provider/model/version. A different concrete provider can be plugged via the adapter without spec change.
- **Summary language default**: Vietnamese (`vi`), configurable.
- **"Processed mask"**: the post-dilation mask that Spec 3 exports alongside the raw prediction mask copy; the "prediction mask" displayed is Spec 2's raw output (also preserved by Spec 3's raw copy).
- **Method identity**: the four method names are fixed as in Spec 2: `classical_baseline`, `manga_text_segmentation`, `comic_text_detector`, `unetpp_efficientnetv2`.
- **Language selection semantics**: because a Spec 4 run fixes its target language (Spec 4 FR-049), the app's language selector filters/identifies compatible Spec 4 runs rather than triggering new translation.
- **Benchmark provenance**: Method A = Manga-Text-Segmentation, whose checkpoint strategy (LOFO, five fold checkpoints, seed-42 book→fold mapping) is inherited from Spec 2 FR-013a and reported, not recomputed.
- **Qualitative report format**: self-contained HTML per the requester's proposed tree (`qualitative_report.html`); the metrics summary is CSV and the comparison chart PNG, per the same tree.
- **Local-only, single-user**: the app runs on the researcher's machine with no authentication, no concurrent multi-user access, and no cloud deployment.
- **Downloads**: handled by the local UI for images and JSON artifacts; the CLI equivalents are file copies of the same stored artifacts.
- **English specification language**: kept for tooling consistency with Specs 1–4; user communication remains Vietnamese.

## Dependencies

- **Spec 1 — Data Foundation & Classical Baseline** (`specs/001-data-foundation-classical-baseline`): manifest, image_id convention, aligned page space, classical baseline metrics (IoU/Precision/Recall/F1, mean/std/time).
- **Spec 2 — Deep-Learning Segmentation Benchmark** (`specs/002-deep-learning-segmentation-benchmark`): prediction masks per method per run, per-image metrics and summaries, LOFO checkpoint strategy for Method A, availability/unavailable reasons.
- **Spec 3 — Text Removal & Image Inpainting** (`specs/003-text-removal-inpainting`): raw + processed masks, inpainted images, per-page metadata sidecar with upstream provenance.
- **Spec 4 — OCR, Translation & Text Rendering** (`specs/004-ocr-translation-rendering`): crops, ocr.json, translations.json, rendered.png, per-page metadata, run configuration fixing segmentation method + inpainting source + target language.

All four are specification-only as of 2026-09-10 and are blocking preconditions for implementation.

## Out of Scope

- Training or fine-tuning any segmentation, OCR, translation or summary model.
- Running the segmentation model inside Streamlit or anywhere in this feature.
- Creating a new benchmark or changing benchmark methodology.
- Modifying ground-truth masks, prediction masks or original inpainting outputs.
- Using OCR, translation or summary quality to score segmentation.
- Uploading arbitrary new manga images in the MVP (future extension; masks must come from the external segmentation pipeline, chiefly Google Colab).
- Deploying Streamlit to the cloud.
- Treating chapter summaries as mandatory when the constituent page outputs are incomplete.
- Whole-book summarization (a book is not a chapter; if ever needed it would be a separate book-summary mode) — Clarifications, 2026-09-10.
- Batch processing as an MVP deliverable (designed in contract; delivered as the designated extension).
