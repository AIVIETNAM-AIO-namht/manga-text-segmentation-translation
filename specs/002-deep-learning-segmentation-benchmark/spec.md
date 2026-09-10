# Feature Specification: Deep Learning Segmentation Benchmark

**Feature Branch**: `002-deep-learning-segmentation-benchmark`

**Created**: 2026-09-10

**Status**: Draft — complete. No clarification markers remain (an automated scan of this file finds
zero); six clarification questions were answered by the requester across two decision rounds on
2026-09-10. The first round's answers are bound as FR-002, FR-052…FR-060 (including FR-054a), FR-055,
FR-056, FR-018b, FR-050a and the amendments listed in the quality checklist; this round's answers bind
the evaluation scope (all 390 pairs, no subset — FR-001), method A's leave-one-fold-out checkpoint
strategy (FR-013a) and method B's pinned source (upstream `dmMaze`, forks as recorded patch
deviations). All three methods are verified against primary sources. 67 functional requirements,
14 success criteria, 24 edge cases. Ready for `/speckit-plan`.

> **Material finding (2026-09-10).** This benchmark's ground truth was proven to be the published
> training dataset of methods A and C: the upstream 45-book training list is set-identical to the local
> ground-truth tree, and the upstream dataset record's own description of its post-processing rule
> (multiple-of-8 padding, small components removed, holes filled, 450 images) matches local measurement
> clause for clause. Two of the three methods therefore trained on essentially 100% of the evaluation
> ground truth. This does not invalidate the feature, but it does bound what its numbers can claim, and
> it is why SC-011 makes disclosure a precondition of reporting any score. For method A the requester
> has since required the leave-one-fold-out recovery (FR-013a), so its reported score is
> de-contaminated by construction; method C remains fully contaminated and method B's extent is
> unknowable. See `## Assumptions` → *Training-data contamination*.

**Input**: User description:

```text
Xây dựng feature "Deep Learning Segmentation Benchmark" cho hệ thống phân đoạn văn bản trong ảnh manga.

Bối cảnh:
Spec 1 đã xây dựng data foundation, manifest, image/mask loader, alignment, metric interface và classical baseline. Hãy đọc các artifact và contract do Spec 1 tạo ra trước khi viết đặc tả. Không được tạo lại dataset loader hoặc thay đổi format dữ liệu nếu không có lý do rõ ràng.

Dataset sử dụng:
- Ảnh manga: data/raw/Manga109s_released_2026_05_21/images/
- Ground-truth mask: data/groundtruth/post-processed/
- Benchmark sử dụng manifest đã có từ Spec 1.
- Sử dụng cùng danh sách ảnh và ground-truth với classical baseline.
- Không tạo benchmark subset mới.
- Không sử dụng dữ liệu trong data/no-need-to-read/ trừ khi có yêu cầu rõ ràng.

Mục tiêu feature:
1. Tích hợp ba phương pháp deep learning pretrained vào pipeline chung.
2. Chạy inference trên cùng benchmark với classical baseline.
3. Chuẩn hóa output của mọi model thành binary text mask.
4. Tính metric và thời gian xử lý thống nhất.
5. So sánh ba model deep learning với classical baseline từ Spec 1.

Các phương pháp cần tích hợp:
A. Manga-Text-Segmentation: U-Net, backbone ResNet34, pretrained checkpoint/inference implementation phù hợp, output cuối là segmentation mask mức pixel.
B. comic-text-detector: pipeline pretrained gồm YOLOv5, DBNet, U-Net nếu repository/model cung cấp; dùng segmentation mask cuối làm output chính; KHÔNG dùng bounding box làm output benchmark chính; nếu pipeline sinh nhiều loại output phải xác định rõ output nào tạo pixel-level mask.
C. UNet++ EfficientNetV2: Manga-Text-Segmentation của ContemporaryCat, backbone EfficientNetV2, chuẩn hóa output thành binary text mask.

Yêu cầu model adapter: interface chung; nhận đường dẫn ảnh hoặc image tensor, cấu hình model, device, checkpoint path nếu cần; trả về binary prediction mask, kích thước ảnh đầu vào, thời gian inference, metadata model, trạng thái lỗi nếu thất bại. Không để logic riêng của từng model lan sang metric hoặc reporting. Chạy từng model riêng lẻ hoặc cả ba.

Pretrained weights và dependency: ưu tiên implementation/checkpoint chính thức hoặc được chỉ định trong đề tài; không huấn luyện lại; không tải checkpoint vào repository nếu quá lớn; cấu hình checkpoint path, repository path, cache directory bên ngoài source code; kiểm tra license, version compatibility, dependency trước khi chạy. Nếu model không chạy được: không giả lập kết quả; ghi lỗi rõ ràng; cung cấp hướng dẫn cài đặt/cấu hình; cho phép chạy các model còn lại; không dừng toàn bộ benchmark. Cần bước kiểm tra model availability trước khi benchmark.

Chuẩn hóa preprocessing và output: tôn trọng preprocessing của từng model; không thay đổi ground-truth gốc; dùng alignment contract của Spec 1; prediction mask cuối cùng cùng kích thước với ảnh/GT sau alignment, convention binary thống nhất, format output thống nhất; nếu model trả probability map cần cấu hình threshold; ghi lại threshold, preprocessing, resize, postprocessing; không âm thầm resize hoặc crop mà không ghi vào metadata.

Đánh giá: chạy toàn bộ model trên cùng manifest; metric IoU, Precision, Recall, F1-score; ghi nhận metric từng ảnh, trung bình, độ lệch chuẩn, thời gian inference trung bình/ảnh, thời gian preprocessing nếu có, thời gian postprocessing nếu có, tổng thời gian; so sánh classical baseline (Spec 1) + Manga-Text-Segmentation + comic-text-detector + UNet++ EfficientNetV2; KHÔNG dùng Pixel Accuracy làm metric chính; không dùng chất lượng OCR/dịch/inpainting để xếp hạng model.

Output mong muốn: model adapters; availability checker; file cấu hình từng model; prediction mask từng model; metadata inference từng ảnh; metrics từng ảnh; bảng tổng hợp so sánh; biểu đồ so sánh IoU/Precision/Recall/F1; biểu đồ hoặc bảng so sánh thời gian; danh sách ảnh thành công/thất bại; visualization (ảnh gốc, GT mask, prediction mask, overlay prediction-GT); báo cáo trường hợp tốt/thất bại; README hướng dẫn cài đặt checkpoint, dependency, chạy inference.

Reproducibility: ghi lại tên model, checkpoint, version code/repository, device, input size, threshold, preprocessing, postprocessing, seed, thời gian chạy, môi trường và package version; chạy lại được với cùng manifest; không ghi đè kết quả lần chạy trước nếu chưa được yêu cầu; dùng experiment name hoặc run ID; cache model và output nếu phù hợp.

CLI: kiểm tra model/dependency/checkpoint; chạy một model; chạy tất cả model DL; chạy benchmark kết hợp classical baseline; tạo metrics report; tạo visualization; xem trạng thái các model bị lỗi.

Testing: test adapter với input giả lập hoặc fixture nhỏ; test output mask đúng shape; test mask chỉ chứa giá trị binary hợp lệ; test threshold probability map; test alignment sau inference; test metric calculation bằng output chuẩn; test một model lỗi không làm mất kết quả model khác; test metadata ghi đúng preprocessing và checkpoint; không yêu cầu test tải model lớn trong unit test mặc định, có thể tách integration test.

Ngoài phạm vi: không huấn luyện/fine-tune; không xây classical baseline mới; không inpainting; không OCR; không dịch máy; không text rendering; không end-to-end demo; không thay đổi GT hoặc manifest của Spec 1 trừ khi phát hiện lỗi và có migration rõ ràng.

Acceptance criteria: 13 mục (ba adapter cùng interface; availability check; chạy từng model; chạy cả ba trên cùng manifest; binary mask đúng convention và kích thước sau alignment; metric thống nhất với classical baseline; bảng so sánh đầy đủ; mask và metadata tách biệt theo model và experiment; lỗi không làm hỏng kết quả hợp lệ khác; visualization và báo cáo; tái lập được; output contract tương thích Spec 3 cho inpainting; chỉ tạo specification, chưa viết code).

Trước khi hoàn tất specification: đọc và đối chiếu output contract của Spec 1; kiểm tra project hiện có model code/checkpoint/dependency nào chưa; xác định chính xác cách tích hợp từng model; phân biệt official / third-party / fallback; không tự giả định checkpoint hoặc API nếu chưa kiểm tra; nêu các vấn đề có thể khiến model không chạy được; đặt câu hỏi làm rõ nếu cần quyết định ảnh hưởng phạm vi hoặc tính tái lập.
```

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Benchmark one pretrained deep-learning method end-to-end (Priority: P1)

A team member takes one of the three pretrained text-segmentation methods, exports the benchmark page
list from the project, and runs that method over it in their own environment — a hosted notebook, with
the method's external repository cloned there and its checkpoint fetched there. The method produces one
binary text mask per page in the standardised format, together with per-page timing, geometry metadata
and a provenance record. They hand that back to the project, which validates it, admits it into a named
run, and computes the same four segmentation quality metrics the classical baseline already produces.
The results land in their own run directory without touching anything from a previous run.

**Why this priority**: This is the whole feature in miniature. If a single method cannot be driven
through the shared contracts and produce comparable numbers, nothing else — the three-way comparison,
the charts, the failure report — has anything to stand on. It is also the slice that proves the
interface really is compatible with the Spec 1 metric module rather than requiring changes to it, and
the slice that proves a result produced in someone else's environment can be scored by the project's
own single evaluation procedure.

**Independent Test**: Register a stand-in method that implements the adapter interface and returns a
synthetic mask for a two-page fixture. Run the benchmark for that one method. Confirm it produces, per
page: a binary mask of the aligned page size, the four metrics, an inference time, and a metadata
record — and that the metric and reporting modules were not modified to accept it. Then take the same
synthetic result, express it as a hand-off produced "elsewhere" with a provenance record, admit it
through the receipt path, and confirm it yields byte-identical metrics — proving the two paths meet at
one evaluation procedure.

**Acceptance Scenarios**:

1. **Given** the Spec 1 page list and an available method, **When** the runner runs that single
   method against the exported list, **Then** every page in the list receives a binary prediction mask
   in the standardised format, with per-page timing and geometry metadata, and the hand-off names the
   list identity it was produced against.
2. **Given** a returned result that passes validation, **When** the project admits it, **Then** it
   computes per-page IoU, Precision, Recall and F1 with its own evaluation procedure, writes them under
   that method's own directory inside the run directory, and stores no score that came from the runner.
3. **Given** a method whose availability check passes, **When** the result is admitted, **Then** the
   run directory is named by the experiment name or run ID and no directory from any earlier run is
   modified or deleted.
4. **Given** a method that emits a probability map rather than a binary mask, **When** the runner
   produces it, **Then** the configured threshold is applied to produce the binary mask and the
   threshold effectively applied is recorded in that method's metadata.
5. **Given** a method that internally resizes or pads the page before inference, **When** the
   prediction is produced, **Then** the mask is returned at the aligned page size and the internal
   geometry change plus its inverse mapping are recorded in metadata.
6. **Given** a stand-in method that implements the adapter interface, **When** it is benchmarked,
   **Then** metrics are produced through the unchanged Spec 1 metric pipeline, demonstrating the
   interface needs no method-specific branching in the metric or reporting modules.
7. **Given** the result has been admitted, **When** the researcher inspects the run record, **Then**
   it names the method, checkpoint identity, external repository and exact revision, code and weight
   licence, evidence that the checkpoint loaded, device, input size, threshold, preprocessing,
   postprocessing, seed, timings, and environment package versions — each attributed to the
   environment that produced them.
8. **Given** a returned result whose provenance record omits the checkpoint-loading evidence, **When**
   validation runs, **Then** the result is refused with the missing field named, nothing is written
   into the run, and the run remains reportable over the results already admitted.

---

### User Story 2 - Know before running whether each method can actually run (Priority: P2)

Before committing hours of inference time, whoever is about to run a method asks the system whether it
can run *in the environment they are in*. For that method they get a pass/fail verdict covering the
checkpoint, the third-party code at its pinned revision, the required dependencies, the licence, and
the compute device, plus concrete remediation steps for anything missing. Methods that fail are
excluded; methods that pass still run. A verdict is only meaningful for the environment that produced
it, so the verdict records that environment.

**Why this priority**: All three methods depend on external pretrained weights and third-party code
that is not present in this project, and at least one of them is known to be broken against current
library versions while another needs an interpreter older than the development machine has. Under
distributed execution these are three separate environment problems solved by three different people,
and a verdict recorded on one machine says nothing about another. A benchmark that fails halfway
through page 200 of 390 wastes far more time than a one-minute pre-flight check. This story is also
what makes the "never fabricate results" requirement enforceable rather than aspirational.

**Independent Test**: Run the availability check with no checkpoints installed and confirm the method
reports unavailable with a named missing artefact and an install instruction. Then place a valid
checkpoint for that method and confirm it flips to available, that the verdict records the environment
it was produced in, and that a run proceeds with that one method while the others are listed as not yet
returned.

**Acceptance Scenarios**:

1. **Given** no checkpoints have been downloaded, **When** the runner runs the availability check in
   their environment, **Then** the method reports unavailable, identifies the specific missing
   artefact, and states where to obtain it and how large it is.
2. **Given** a method whose required dependency is absent or at an incompatible version, **When**
   the availability check runs, **Then** the verdict names the dependency and the version conflict
   rather than deferring the failure to inference time.
3. **Given** one method available and two not yet returned, **When** the run is reported, **Then** the
   available method's results appear normally, the two others are recorded as "not run" with their
   reasons or their awaited status, and no synthetic or imputed metric values appear anywhere in the
   output.
4. **Given** a checkpoint that is present but does not match the expected architecture or is
   truncated, **When** the availability check runs, **Then** the method is reported unavailable with
   the mismatch described.
5. **Given** an availability check that has run, **When** the researcher asks for failed-method status
   later, **Then** the recorded verdicts, their reasons and the environments that produced them are
   retrievable without re-running the check.
6. **Given** a method whose upstream code needs a workaround to run at all, **When** the runner applies
   it in their own environment, **Then** the workaround is recorded in the provenance record as a
   deviation from the pinned revision, and no change is committed to the project (FR-055).
7. **Given** a method that needs an accelerator or a legacy interpreter this machine lacks, **When** the
   check reports the device or environment failure, **Then** the method is re-checked in a suitable
   environment before being declared unavailable, rather than abandoned on the strength of one
   machine's verdict.

---

### User Story 3 - Compare the three deep-learning methods against the classical baseline (Priority: P3)

Three team members each run one method, in three separate hosted environments, against the same exported
page list, and hand their results back at different times. The researcher admits each result as it
arrives, and once the classical baseline plus whatever deep-learning methods have returned are in,
receives one summary table and one set of charts showing IoU, Precision, Recall, F1 and processing time
per method, each accuracy metric with mean and standard deviation. The accuracy figures are directly
comparable because every method was scored by the project's own single evaluation procedure over the
same pages. The timing figures are not, because they were produced on three different machines, so each
one is labelled with the device that produced it.

**Why this priority**: This is the deliverable the thesis actually needs: the comparison. It ranks
below P1 and P2 only because it is the composition of them — it cannot exist before a single method can
be benchmarked, its availability known in the environment that runs it, and its result admitted.

**Independent Test**: Assemble a combined run on a small fixture from the classical baseline plus two
stand-in methods whose results were admitted through the receipt path from two separately constructed
hand-offs. Confirm the summary table contains one row per method with all four metrics plus mean and
standard deviation and timing, that every admitted result carries the same page-list identity, that each
timing figure is labelled with its producing device, and that every score in the table was computed by
the project's own evaluation procedure.

**Acceptance Scenarios**:

1. **Given** the classical baseline and at least one admitted deep-learning result, **When** the
   comparison is produced, **Then** one summary table reports IoU, Precision, Recall, F1, mean, standard
   deviation, mean per-page inference time, and total processing time for every method admitted.
2. **Given** results admitted from different environments at different times, **When** the run is
   assembled, **Then** every admitted result carries the same page-list identity, admitting a later
   result leaves earlier results byte-identical, and a run with one method still outstanding is
   reportable over those already admitted.
3. **Given** the combined run, **When** the page lists per method are compared, **Then** they are
   identical, and the alignment applied to predictions and to ground truth is the same for every method.
4. **Given** the summary results, **When** charts are generated, **Then** there is a chart comparing
   IoU, Precision, Recall and F1 across methods and a chart or table comparing processing time across
   methods, with each timing figure labelled by its producing device and no cross-device timing ranking
   presented.
5. **Given** a method that could not run or has not yet returned, **When** the comparison table is
   produced, **Then** that method appears as "not run" with its recorded reason or as awaited, and the
   table contains no placeholder numeric value for it.
6. **Given** the comparison report, **When** a method's published training data is known to overlap the
   evaluation dataset, **Then** the report discloses that overlap next to the method's scores.
7. **Given** any output of this feature, **When** it is inspected, **Then** Pixel Accuracy does not
   appear as a primary ranking metric and no OCR, translation or inpainting quality measure is used to
   rank methods.
8. **Given** the project's stored tree, **When** it is audited, **Then** it contains no third-party
   model source file and no model weight file of any size, and for each method holds only its
   configuration, its returned standardised results and its documentation.

---

### User Story 4 - Inspect where methods succeeded and where they failed (Priority: P4)

The researcher looks past the averages. They get a per-method list of successful and failed pages
with reasons, side-by-side visualisations showing the raw page, the ground-truth mask, the
prediction mask and the prediction-versus-ground-truth overlay, and a written report of
representative good and failure cases.

**Why this priority**: Mean IoU hides the failure modes that matter for a thesis discussion —
which page types break which method, and why. It is last because it is purely diagnostic: the
benchmark is already complete and comparable without it.

**Independent Test**: Run the visualisation and reporting step against an existing run directory
containing one deliberately failed page. Confirm the failed page appears in the failure list with
its reason, that a configurable number of cases produce four-panel visualisations, and that the
report separates good from failure cases.

**Acceptance Scenarios**:

1. **Given** a completed run, **When** the researcher requests the success/failure lists, **Then**
   each method has a list of pages that succeeded and a list of pages that failed, each failure
   carrying its recorded reason.
2. **Given** a completed run, **When** visualisations are generated, **Then** each selected case
   shows the raw page, the ground-truth mask, the prediction mask, and the prediction-versus-
   ground-truth overlay, and the number of cases is configurable.
3. **Given** a page on which inference raised an error in the runner's environment, **When** the result
   is admitted, **Then** the page is recorded as failed for that method only, the remaining pages of
   that method's hand-off are still scored, and no other method's results are affected.
4. **Given** a completed run, **When** the case report is generated, **Then** it identifies
   representative good cases and representative failure cases per method.
5. **Given** a method that never returned a result at all, **When** the failure report is generated,
   **Then** it distinguishes "pages that failed inside a method that ran" from "method did not run",
   and does not present the absent method as having zero failed pages.

---

### Edge Cases

- **Degenerate ground truth.** One page in the dataset has an entirely white mask, i.e. zero text
  pixels. IoU, Precision and F1 are undefined when both prediction and ground truth are empty, and
  Precision is undefined whenever the prediction is empty. The convention used MUST be declared and
  applied identically to every method, and such pages MUST NOT be silently dropped from the mean.
- **Thin-stroke ground truth.** Measured stroke width on this dataset ranges from about 3 to about
  14.5 pixels, and a single-pixel 3×3 erosion removes roughly half the text area on the thin pages.
  A prediction that is dilated by even one pixel, or binarised at a slightly different threshold,
  therefore moves IoU substantially. Thresholds and morphological post-processing are treated as
  method-level configuration that is fixed for the whole run and disclosed, never tuned per page.
- **Region-level versus stroke-level output.** A method that emits merged text regions or
  rasterised text-line polygons will systematically over-cover relative to stroke-level ground
  truth. If a method's consumed output is not naturally stroke-level, that fact MUST be recorded in
  metadata so the comparison can be interpreted.
- **Internal resolution mismatch.** Methods that letterbox or pad to a fixed network input size
  (one method in scope pads to a 1024×1024 multiple-of-64 canvas) must have the inverse geometry
  mapping applied and recorded; a mask left at network resolution is a defect, not a result.
- **Channel-order mismatch.** At least one method in scope has a suspected red/blue channel reversal
  between its training-time and inference-time preprocessing. Such a mismatch degrades accuracy
  without raising an error, so the channel order actually used MUST be recorded in metadata.
- **Hardcoded threshold overriding configuration.** One method in scope accepts a threshold
  parameter that its own code ignores in favour of a hardcoded constant. Configuring a value that
  the code does not honour is a silent lie in the metadata, so the effective threshold MUST be
  verified against the produced mask, not merely recorded from configuration.
- **Checkpoint present but unusable.** A file exists at the configured path but is truncated, is a
  different architecture, or lacks expected component keys. Treated as unavailable, not as an
  inference failure.
- **Checkpoint absent but inference succeeds anyway.** The most dangerous case in scope, and a real
  behaviour of one method's upstream code rather than a hypothetical: its checkpoint path is a
  hardcoded working-directory-relative filename, and a missing-file error is caught, printed as a
  warning and execution continues with a **randomly initialised decoder**. Inference then returns a
  mask of the correct shape and value range, so every output-shape and output-domain check passes while
  the numbers are meaningless. An availability check that only verifies "a mask came back" is useless
  here. The check MUST positively establish that weights were loaded from the configured checkpoint —
  by identity, not by inference success — and FR-018 forbids running a method whose weights cannot be
  proven to be the published ones.
- **Two different padding conventions in one pipeline.** The ground truth is padded to a multiple of 8,
  one method pads its own input to a multiple of 8, and another pads to a multiple of 32. The aligned
  page size is a multiple of 8 but **not** of 32. A geometry mapping that assumes all padding in the
  pipeline is the same transformation will misplace the mask by a few pixels on every page — and on a
  dataset whose mean stroke width is 3–14 px, a few pixels is a large fraction of the signal. Each
  method's padding multiple MUST be recorded separately and inverted separately (FR-024, FR-025).
- **Runtime incompatibility discovered late.** An upstream repository references a numeric-library
  alias that no longer exists in the installed version. The availability check is expected to catch
  this class of failure before inference; if it surfaces mid-run, the method fails as a whole with a
  recorded reason and other methods continue.
- **Interpreter-version ceiling.** One method in scope pins a framework stack whose maximum supported
  interpreter version is below the one installed on the development machine, and its checkpoints can
  only be deserialised by that legacy stack. Under the distributed execution model this is no longer a
  scope decision but an environment requirement: the method MUST be attempted in an environment that
  supplies the legacy interpreter, which a hosted notebook environment can. If it still cannot be made
  to run there, it MUST be reported as unavailable with the specific reason — it MUST NOT be dropped
  silently, and MUST NOT be run against a different framework version, which would silently produce
  different weights.
- **Upstream defect requiring a workaround.** A method's published code does not run as committed —
  a numeric-library alias that no longer exists, an import of a graphical component absent from a
  headless environment, a configuration value the code ignores. Under FR-055 the project cannot patch
  upstream in-tree, so the workaround MUST be applied in the runner's own environment and MUST be
  recorded in the provenance record as a deviation from the pinned revision, together with what was
  changed and why. A workaround that changes the model's output rather than merely letting it run — a
  different threshold, a different preprocessing constant, a re-exported checkpoint — MUST be treated
  as a different method, not as the pinned one.
- **Result produced against a stale page list.** The page list is exported, a runner works from it days
  later, and in the meantime the benchmark's page set changed. Page identifiers alone would let the
  stale result be admitted and silently evaluated over the wrong set. This is why the list carries an
  identity that every result must echo (FR-054) and why receipt validation checks it (FR-058): a
  mismatch is a refusal with a named reason, never a name-based realignment.
- **Runner sourced its images elsewhere.** A runner cannot reach the distributed images, so it downloads
  the same dataset from its public release instead. The page names match and the sizes probably match,
  but the bytes may not: a re-release, a re-compression, a colour-profile change or a resampling of the
  same page produces a different input and therefore a different mask, and the difference is invisible
  in any per-page check the project could run afterwards. This is why FR-054a requires the image
  distribution to carry a verifying identity per page and every returned result to echo it, and why a
  mismatch is refused rather than investigated after the fact. The classical baseline and all three
  methods must be run on identical bytes for the accuracy comparison to mean anything.
- **Provenance the project never witnessed.** Under FR-052 the project does not execute the method, so
  every claim about how a result was produced — which revision, which checkpoint, which device, whether
  weights really loaded — arrives as an assertion from the runner's environment. Plausible masks are not
  evidence. The provenance record is the only admissible evidence, it MUST be complete (FR-056), and an
  incomplete one MUST be refused rather than accepted with gaps (FR-018b, SC-014).
- **Ephemeral runner environment.** A hosted notebook session ends; its cloned repository, downloaded
  checkpoint and model cache all disappear. Nothing durable is lost provided the returned result and its
  provenance record were handed off before the session ended, because those — not the cache — are the
  record of what was used (FR-046). A result whose hand-off did not complete is simply absent: the
  method shows as not yet returned, the run stays reportable (FR-059), and no partial or cached output
  is substituted for it.
- **Two methods, two devices, one timing column.** Methods produced on different accelerators, or one
  on an accelerator and one on a CPU, yield timings that differ by an order of magnitude for reasons
  that have nothing to do with the methods. Presenting those numbers side by side in a single ranking
  would report hardware, not architecture. Every timing figure carries its producing device (FR-033,
  FR-037, FR-060) and the timing comparison is labelled per-device rather than ranked across devices.
  Accuracy is unaffected: it depends only on the shared page list and the shared evaluation procedure,
  and stays directly comparable.
- **Failure on one page.** Out-of-memory, decode error, or an exception inside third-party code on a
  single page. That page is recorded as failed for that method; the run continues; no other method's
  results are affected.
- **Run-identifier collision.** Re-running with an experiment name or run ID that already exists on
  disk. Prior results MUST NOT be overwritten unless the researcher explicitly requested it.
- **Orphan ground truth.** Sixty masks in the dataset have no corresponding source image and are not
  part of the benchmark. They MUST NOT be evaluated and MUST NOT be reported as failures.
- **No compute device.** A method that requires an accelerator on a machine that has none. Under FR-052
  the remedy is to run that method in an environment that does have one, not to abandon it; the
  availability check reports the device failure in the environment it ran in (FR-017), and the method is
  re-checked in a suitable environment before being declared unavailable.
- **Partial-run interruption.** The process is killed mid-benchmark. Already-written per-page results
  MUST remain valid and readable, and a resumed or repeated run MUST NOT corrupt them.
- **Absent Spec 1 foundation.** The page list, loaders, alignment or metric module this feature
  depends on are not present. This is a blocking precondition reported by the availability check; it
  MUST NOT be resolved by silently building a replacement loader, and it MUST NOT be resolved by
  exporting a locally derived page list for the runners to consume (FR-002, FR-054).
- **Ground truth reaching a runner.** The distributable page list tells a runner which pages to process;
  it does not carry ground-truth pixel data, because a runner needs no answers to produce predictions
  and a runner holding the answers could produce a mask against them. Any request to include ground
  truth in the export is a scope violation of the evaluation model (FR-053, FR-057), where scoring
  happens once, in the project, from returned masks.

---

## Clarifications

### Session 2026-09-10

- Q: Spec 001 evaluated all 390 valid pairs and left any deep-learning subsampling as Spec 2's
  decision, but the course outline (đề cương §5.1) suggests "một tập con khoảng 100–200 ảnh" — which
  evaluation scope does this feature use? → A: All 390 pairs, for all four methods. No subset is
  created (the input instruction "Không tạo benchmark subset mới" stands); the outline's underlying
  requirement — "bảo đảm các ảnh được sử dụng nhất quán cho việc so sánh giữa các phương pháp" — is
  satisfied by evaluating the identical full set rather than by subsampling.
- Q: Method A publishes five cross-validation-fold checkpoints and the specification did not say which
  one to run — which checkpoint strategy does the benchmark use for it? → A: Full leave-one-fold-out
  (LOFO). All five released checkpoints are used; every page is scored by the checkpoint of the fold
  whose training split provably did not contain that page's book, with the book→fold mapping derived
  from the upstream fold construction's published seed (42) and verified rather than assumed. Method A's
  reported score is therefore de-contaminated by construction. Bound as FR-013a.
- Q: The course outline cites the source of method B as the Ajatt-Tools/comic_text_detector fork
  (forked from dmMaze/comic-text-detector), while the specification recommends upstream plus
  fork-supplied patches — which repository is pinned as method B's primary source? → A: The upstream
  `dmMaze/comic-text-detector` at a pinned revision. Forks (kha-white, Ajatt-Tools) are patch sources
  only: any patch applied in the runner's environment is a recorded deviation from the pinned upstream
  revision under the *Upstream defect requiring a workaround* edge case. Citing upstream in the report
  is consistent with the outline, which itself identifies the Ajatt-Tools fork as derived from dmMaze.

---

## Requirements *(mandatory)*

### Functional Requirements

#### Reuse of the Spec 1 foundation

- **FR-001**: The system MUST consume the existing Spec 1 benchmark manifest as the single source of
  truth for the evaluated image list, and MUST NOT construct a new manifest, a new benchmark subset,
  or a different image list. The evaluated scope is **all 390 valid image/ground-truth pairs** in the
  manifest, for the classical baseline and every deep-learning method alike — the course outline's
  (~100–200-image) subset suggestion is deliberately not adopted, and no subsampling artefact is
  created (Clarifications, 2026-09-10). The manifest is Spec 1's object and stays inside the project;
  what a runner receives is its export, the distributable page list (FR-054), which MUST be derived
  from the manifest and MUST carry the manifest's identity so that the two can be shown to denote the
  same pages. The two terms are not interchangeable: "manifest" is the source of truth, "page list" is
  the artefact derived from it and handed out.
- **FR-002**: The system MUST reuse the Spec 1 image loader, ground-truth mask loader and alignment
  implementation, and MUST NOT re-implement them. This feature is specified against Spec 1's contracts
  as written and its scope MUST NOT include building any part of Spec 1. Because Spec 1 exists today as
  a specification only, its implementation is a **blocking precondition** of this feature: the
  availability check MUST report it as unmet while it is absent, and MUST NOT silently substitute a new
  loader, a new alignment rule or a locally derived image list to make progress. The confirmed
  sequencing is Spec 1 implementation → external model runs → benchmark integration.
- **FR-003**: The system MUST reuse the Spec 1 metric definitions for IoU, Precision, Recall and F1
  without modification, and MUST NOT define alternative metric formulas for deep-learning methods.
- **FR-004**: The system MUST reuse the Spec 1 binary mask convention — single channel, unsigned
  8-bit, background = 0, text = 255 — for every prediction mask it produces.
- **FR-005**: The system MUST NOT modify ground-truth masks or the manifest. If a defect is found in
  either, work MUST stop and the finding MUST be reported together with a proposed migration; silent
  correction is prohibited.
- **FR-006**: The system MUST NOT read any data from the directory reserved as out-of-scope
  (`data/no-need-to-read/`) unless a later requirement explicitly asks for it.
- **FR-007**: The system MUST evaluate the classical baseline and every deep-learning method on
  exactly the same image/ground-truth pairs so that all reported numbers are directly comparable.

#### Distributed execution and result hand-off

- **FR-052**: Each deep-learning method MUST be executable independently, in an environment chosen by
  whoever runs it, and MUST NOT be required to share one local environment with the other methods or
  with the classical baseline. "Running all three methods" (FR-012, acceptance criterion 4) is
  therefore satisfied by three independent executions whose standardised results are later combined;
  it MUST NOT be read as requiring one process, one machine or one dependency set to run all three.
- **FR-053**: Because execution is distributed, exactly six things are the shared invariants that make
  separately produced results combinable, and each MUST be fixed once and consumed identically by every
  method: (a) the benchmark page list; (b) the input images those pages denote; (c) the preprocessing and
  alignment convention; (d) the prediction-mask format; (e) the evaluation procedure; (f) the recorded
  environment metadata. A result from a method that deviates from any one of these MUST NOT be admitted
  to the comparison.
- **FR-054**: The evaluated page list MUST be distributable as a self-contained artefact that a runner
  can consume without access to the project's own data tree, and every runner MUST consume the same
  list instance. The list MUST carry an identity that is recorded with each returned result, so that
  the evaluation can verify all methods were run against the same pages rather than merely against
  pages with similar names.
- **FR-054a**: The input images MUST be distributable to a runner, and every method MUST be run on the
  same image bytes that the classical baseline was run on. The distribution MUST record, per page, an
  identity strong enough for the project to verify on receipt that the images a runner actually processed
  are those it was given, and each returned result MUST carry that record. A runner that obtained its
  images from any other source — a different release of the same dataset, a re-download, a converted or
  re-compressed copy — MUST be refused rather than admitted, because accuracy comparability across
  methods depends on identical inputs and not merely on identically named ones. Distributing the input
  images is permitted and required; distributing the ground truth is not (see the *Ground truth reaching
  a runner* edge case), because the images are the question and the masks are the answer.
- **FR-055**: Third-party model source code and pretrained weights MUST NOT be vendored, copied,
  committed or otherwise stored in the main project. Each method MUST run from its external repository
  in the runner's own environment. What the main project stores MUST be limited to the shared
  contracts, the evaluation code, configuration, standardised results and documentation.
- **FR-056**: Each returned result MUST be accompanied by a provenance record captured in the
  environment that produced it, naming at minimum: the external repository and the exact revision
  (commit) used, the checkpoint identity, the licence of the code and of the weights, the compute
  device, the interpreter and package versions, and the time of the run. A result whose provenance
  record is missing or incomplete MUST be refused rather than accepted with gaps.
- **FR-057**: Metrics MUST be computed by one shared evaluation procedure inside the main project,
  applied identically to the classical baseline and to every returned result. Metrics MUST NOT be
  computed in a runner's environment and MUST NOT be transcribed from a runner's own numbers: a runner
  returns prediction masks and provenance, not scores.
- **FR-058**: Returned results MUST be validated on receipt — page-list identity, input-image identity
  (FR-054a), mask size against the aligned page size, the permitted binary value set, completeness
  of the provenance record, and — for method A — the per-page fold attribution of FR-013a — before any
  metric is computed. A result failing validation MUST be reported
  as invalid with the specific reason, and MUST NOT be silently corrected, resized, re-thresholded or
  dropped.
- **FR-059**: Results MUST be admitted to a named run incrementally. A result arriving later than the
  others MUST be added without recomputing or invalidating results already admitted, and a run MUST
  remain reportable while one or more methods have not yet returned.
- **FR-060**: Because methods may execute on different hardware, every timing figure MUST be attributed
  to the device and environment that produced it, and MUST NOT be presented as a cross-method
  performance comparison without that attribution. Accuracy metrics depend only on the shared page list
  and the shared evaluation procedure, and remain directly comparable across methods regardless of where
  each was produced.

#### Model adapters

- **FR-008**: The system MUST wrap each deep-learning method in an adapter that exposes one common
  interface conforming to the Spec 1 segmentation-method contract (`segment(image) -> binary mask`),
  so that the metric and reporting modules accept a new method without modification. The adapter is the
  boundary at which a method's own repository is called; under FR-055 the adapter MUST reference an
  external repository by configured path and revision and MUST NOT contain a copy of that repository's
  model code.
- **FR-009**: Each adapter MUST accept an image reference (a path or an in-memory image), a method
  configuration, a target compute device, and an optional checkpoint path.
- **FR-010**: Each adapter MUST return a binary prediction mask, the input image size, the inference
  time, model metadata, and an error status when inference fails.
- **FR-011**: The system MUST keep all method-specific behaviour — preprocessing, thresholds, tensor
  layout, framework calls, geometry mapping — inside the adapter and its configuration, and MUST NOT
  let it appear in the metric, reporting, visualisation or command-line modules.
- **FR-012**: The system MUST allow running a single method, all deep-learning methods together, or
  the deep-learning methods combined with the classical baseline. Under FR-052 the "together" and
  "combined" cases are satisfied by combining independently produced results (FR-059), not by a single
  execution.
- **FR-013**: The system MUST support exactly three deep-learning methods in this feature: (A) a
  U-Net with a ResNet34 encoder producing a pixel-level segmentation mask; (B) the composite
  comic-text-detector pipeline; (C) a UNet++ with an EfficientNetV2 encoder. For method B the
  benchmark output MUST be the pixel-level segmentation mask; bounding boxes and text-line polygons
  MUST NOT be used as the benchmark output.
- **FR-013a**: Method A's benchmark run MUST use a **leave-one-fold-out (LOFO)** checkpoint strategy,
  decided 2026-09-10 (see Clarifications). All five released fold checkpoints MUST be used, and each
  page MUST be scored by exactly one checkpoint — that of the fold whose training split did not contain
  the page's book. The book→fold mapping MUST be derived from the upstream fold construction's published
  fixed seed (42) and MUST be verified against the upstream split definition before any page is scored;
  a mapping that cannot be reproduced and verified makes method A's result inadmissible — reported
  unavailable with the reason (FR-018) — not approximated. Per-page run metadata (FR-027) and the
  provenance record (FR-056) MUST record, for every page, which fold checkpoint scored it, and FR-058
  MUST refuse a method A result with any missing or inconsistent fold attribution. The availability
  check (FR-016) MUST verify the presence and integrity of all five checkpoints, not one. Each page is
  still inferred exactly once: the strategy multiplies the checkpoints loaded and downloaded, not the
  inference passes.
- **FR-014**: Where a method's pipeline emits more than one output type, the method configuration
  MUST name exactly which output is consumed to produce the pixel-level mask, and that choice MUST be
  recorded in the run metadata.
- **FR-015**: The system MUST NOT train or fine-tune any model in this feature; only pretrained
  inference is in scope.

#### Availability check

- **FR-016**: The system MUST provide a per-method availability check reporting a pass/fail verdict
  for: checkpoint presence and integrity, third-party code availability and pinned revision, required
  dependencies and their versions, licence identifier, and compute-device capability — each failure
  with a human-readable reason.
- **FR-017**: The availability check MUST run before the benchmark and MUST also be runnable
  standalone. Because a method's environment may not be the environment where the check is invoked
  (FR-052), the check MUST be executable in the environment that will actually run that method, and
  its verdict MUST be recorded together with the identity of the environment that produced it. A verdict
  obtained in a different environment MUST NOT be relied upon as evidence that the method can run in
  the one that will execute it.
- **FR-018**: The system MUST NOT fabricate, impute, simulate or copy results for an unavailable
  method. An unavailable method MUST appear in comparison output as "not run" together with its
  recorded reason.
- **FR-018a**: The system MUST positively verify, before any inference, that the weights actually used
  were loaded from the configured checkpoint. A method whose upstream implementation tolerates a missing
  checkpoint and continues with untrained weights MUST be detected and reported as unavailable by this
  check, and MUST NOT be run. Successful production of a correctly-shaped, correctly-valued mask is NOT
  sufficient evidence that published weights were loaded, and MUST NOT be accepted as such.
- **FR-018b**: Under FR-052 the project may never itself witness a method executing, so the FR-018a
  verification MUST be performed in the environment that runs the method, and its outcome MUST be
  returned as part of the provenance record (FR-056) in a form that distinguishes "loaded from the
  configured checkpoint, evidenced" from "not evidenced". A result whose provenance record does not
  carry positive evidence MUST be treated as unverified and refused by the FR-058 validation, rather
  than being accepted on the strength of plausible-looking masks.
- **FR-019**: The system MUST provide remediation guidance for every failed check — installation or
  configuration steps, the expected checkpoint source and size, and pinned dependency versions.
- **FR-020**: An unavailable or failing method MUST NOT prevent the available methods from running,
  and MUST NOT abort the benchmark while at least one method can proceed.
- **FR-021**: The availability check MUST verify dependency and version compatibility before
  inference wherever feasible, rather than allowing an incompatible-version failure to surface
  mid-benchmark.
- **FR-022**: The system MUST record availability verdicts so that failed-method status can be
  retrieved later without re-running the check.

#### Preprocessing, normalisation and alignment

- **FR-023**: The system MUST apply each method's own required preprocessing and MUST NOT force one
  shared preprocessing onto all methods.
- **FR-024**: The system MUST map every prediction into the Spec 1 aligned space — the raw native
  page size, which is uniform across this benchmark — using the Spec 1 alignment contract, and MUST
  NOT resize or otherwise alter the ground truth to accommodate a prediction.
- **FR-024a**: Where a method pads its own input to an internal multiple (8, 32 or otherwise), the
  system MUST record that multiple per method and MUST invert it separately per method. Padding applied
  by a method MUST NOT be assumed to be the same transformation as the multiple-of-8 padding already
  present in the ground truth, and MUST NOT be collapsed into a single shared rule. The aligned page
  size is a multiple of 8 but not of 32; any mapping that treats all padding as one convention is a
  defect.
- **FR-024b**: Where a method's training-time label space distinguishes text sub-classes or ignore
  classes that this benchmark's binary ground truth has already collapsed, the system MUST verify —
  not assume — that collapsing the method's prediction to the FR-004 binary convention is lossless with
  respect to this ground truth, and MUST record the collapse rule in the run metadata.
- **FR-025**: The system MUST record in per-page metadata every geometry transformation applied,
  including internal resize or letterbox dimensions, padding, cropping, interpolation method and
  scale factors. A resize or crop that is not recorded in metadata is prohibited.
- **FR-026**: Where a method emits a probability or confidence map, a threshold MUST be configured,
  applied and recorded. Thresholds MUST be fixed per method for the whole run and MUST NOT be tuned
  per page.
- **FR-027**: The system MUST record, per method: preprocessing steps, the input size actually fed to
  the model, normalisation constants, postprocessing steps (including any morphological operation,
  connected-component filter, or rasterisation of regions into a mask), the threshold, the internal
  padding multiple, and the output geometry mapping.
- **FR-028**: Every final prediction mask MUST be single-channel, binary under the FR-004 convention,
  exactly the aligned page size, and stored in one unified output format across all methods.
- **FR-029**: The system MUST NOT apply morphological dilation or erosion to predictions by default.
  Where such an operation is explicitly configured, it MUST be recorded and MUST be identical for
  every page of that method.
- **FR-030**: Where a method's configuration exposes a threshold that its own implementation ignores,
  the system MUST record the effective threshold actually applied rather than the configured-but-
  unused value.

#### Evaluation

- **FR-031**: The system MUST compute IoU, Precision, Recall and F1 per page in the aligned space,
  and MUST NOT use Pixel Accuracy as a primary metric.
- **FR-032**: The system MUST declare and record the metric convention for degenerate pages — where
  the ground truth contains no text pixels, or the prediction contains no text pixels — and MUST
  apply that convention identically to every method without silently dropping such pages.
- **FR-033**: The system MUST record per-page metrics and, per method, the mean, the standard
  deviation, the mean inference time per page, the preprocessing time where measurable, the
  postprocessing time where measurable, and the total processing time. Every timing figure MUST carry
  the device and environment attribution required by FR-060.
- **FR-034**: The system MUST produce a combined comparison covering the Spec 1 classical baseline
  and every deep-learning method that ran, on the same page list (FR-054) and the same alignment, with
  all metrics computed by the shared evaluation procedure (FR-057).
- **FR-035**: The system MUST NOT rank methods using OCR quality, translation quality or inpainting
  quality in this feature.

#### Outputs and artefacts

- **FR-036**: The system MUST persist per-method prediction masks, per-page inference metadata and
  per-page metrics, separated by method and by run.
- **FR-037**: The system MUST produce a summary comparison table covering every method that ran, with
  IoU, Precision, Recall, F1 (mean and standard deviation) and processing time. The processing-time
  column MUST state, per method, the device and environment that produced it, and the table MUST NOT
  present timing as a cross-method ranking where the producing devices differ (FR-060).
- **FR-038**: The system MUST produce charts comparing IoU, Precision, Recall and F1 across methods,
  and a chart or table comparing processing time across methods. Where the compared methods were
  produced on differing devices, the timing chart MUST label each bar with its producing device or MUST
  be presented per-device rather than as a single ranking.
- **FR-039**: The system MUST produce, per method, a list of successful pages and a list of failed
  pages with recorded failure reasons.
- **FR-040**: The system MUST produce qualitative visualisations for a configurable number of cases,
  each showing the raw page, the ground-truth mask, the prediction mask, and the prediction-versus-
  ground-truth overlay, reusing the Spec 1 visualisation convention.
- **FR-041**: The system MUST produce a report identifying representative good cases and
  representative failure cases per method.
- **FR-042**: The system MUST ship a README covering checkpoint acquisition, dependency installation,
  configuration, and how to run inference and the benchmark. Because a method may be run by someone with
  no access to this project's environment (FR-052), the README MUST also cover, per method: the external
  repository and revision to use, how to obtain the distributable page list, what a compliant returned
  result consists of, and what provenance must accompany it. A runner MUST be able to produce an
  admissible result from the README alone.
- **FR-043**: Prediction masks MUST be stored so that a downstream inpainting feature (Spec 3) can
  consume them using only the manifest page identifier and the method name: full aligned resolution,
  the FR-004 binary convention, a deterministic per-page path, and no dependence on run-specific
  metadata to interpret the mask.

#### Reproducibility

- **FR-044**: The system MUST record per run: method name, checkpoint identity (file name, size,
  checksum, source), third-party code version or revision, compute device, input size, threshold,
  preprocessing, postprocessing, seed, timings, and environment package versions. Because a run is
  assembled from independently produced results (FR-052), these attributes MUST be recorded **per
  method, from the environment that produced that method's result**, and MUST NOT be recorded once for
  the run as a whole on the assumption that all methods shared it. The run as a whole additionally
  records which results it combined and the page-list identity they were all produced against.
- **FR-045**: The system MUST identify each run by an experiment name or run ID, and MUST NOT
  overwrite the results of a previous run unless explicitly requested. A run MUST accept results
  arriving at different times without rewriting already-admitted results (FR-059).
- **FR-046**: Checkpoint paths, third-party repository paths and cache directories MUST be configured
  outside source code, and large checkpoint files MUST NOT be committed to the repository. Under FR-055
  this extends to third-party source code as well: no external model implementation may be committed,
  in any size, in any form, including a partial copy, a patch series against upstream, or a
  single-file extract. Where a runner's environment is ephemeral, its cache is ephemeral with it; the
  durable record of what was used is the provenance record (FR-056), not the cache.
- **FR-047**: The system MUST be re-runnable against the same page list, yielding the same evaluated
  image list and the same metric definitions; any remaining source of nondeterminism MUST be listed
  in the run record. Under FR-052 re-running one method in a different environment is permitted and is
  not a defect provided the page list, alignment convention, mask format, evaluation procedure and
  provenance recording are unchanged; the change of environment MUST then appear in the run record as
  a difference between the two results.
- **FR-048**: The system MAY cache loaded models and outputs where caching does not change results;
  any cache MUST be keyed so that a configuration change invalidates it.

#### Command-line interface

- **FR-049**: The system MUST expose command-line operations equivalent to: checking
  model/dependency/checkpoint availability; running one method; running all deep-learning methods;
  running the combined benchmark including the classical baseline; generating the metrics report;
  generating visualisations; and viewing the status of failed methods. Under FR-052 it MUST additionally
  expose: exporting the distributable page list and the input images it denotes (FR-054, FR-054a);
  validating and admitting a returned result
  into a named run (FR-058, FR-059); and showing which methods a run has admitted and which it still
  awaits.

#### Testing

- **FR-050**: The system MUST test adapters using a small fixture or simulated model output — without
  downloading a checkpoint — covering at minimum: output mask shape, output mask containing only
  valid binary values, probability-map thresholding, alignment after inference, metric calculation
  against a reference output, isolation of results when one method fails, and metadata correctly
  recording preprocessing and checkpoint identity.
- **FR-050a**: The system MUST additionally test the receipt path introduced by FR-052 using synthetic
  results, covering at minimum: a result produced against a stale or different page list is rejected
  rather than aligned by name; a result whose input-image identity record does not match the distributed
  images is rejected (FR-054a); a result whose provenance record lacks the checkpoint-loading evidence
  required by FR-018b is refused; a result of the wrong mask size or value set is refused with a named
  reason and is not silently resized; admitting a third result after two are already admitted leaves the
  first two byte-identical; and a metric computed from a synthetic result equals the same metric
  computed from the same mask fed through the adapter path.
- **FR-051**: Default unit tests MUST NOT require loading a real large pretrained model; integration
  tests that do so MAY be separated and made explicitly opt-in.

### Key Entities *(include if feature involves data)*

- **Method Adapter**: The single wrapper around one pretrained deep-learning method. Attributes:
  method identifier, the common segmentation interface it satisfies, its configuration, its
  availability verdict, and its error status. It is the only place method-specific behaviour lives.
- **Method Configuration**: Per-method settings held outside source code. Attributes: checkpoint
  path, third-party code path and pinned revision, compute device, model input size, threshold,
  preprocessing parameters, post-processing parameters, which pipeline output is consumed, licence
  identifier, and cache location. Under FR-052 the configuration travels with the method: the copy a
  runner uses is authoritative for that method's result, and the copy retained in the project records
  what was configured at the time the result was admitted.
- **Distributable Page List**: The self-contained export of the benchmark page list (FR-054).
  Attributes: list identity, the page identifiers in evaluation order, and for each page the raw image
  reference and the aligned target size that predictions must be returned at, so a runner can apply the
  alignment convention without access to the project's data tree. It carries no ground-truth reference
  and no ground-truth pixel data: a runner needs to know *which* pages to process and *what size* to
  return, not what the answers are. It is paired with the image distribution of FR-054a, which supplies
  the bytes the references denote and their verifying identity; neither alone is sufficient to run a
  method.
- **Returned Result**: One method's complete hand-off back to the project (FR-053, FR-058). Attributes:
  method identifier, the page-list identity it was produced against, the input-image identity record
  (FR-054a), one prediction mask per page in the standardised format, per-page geometry and timing
  metadata, the provenance record, and the validation verdict.
- **Provenance Record**: The environment evidence accompanying a returned result (FR-056, FR-018b).
  Attributes: external repository and exact revision, checkpoint identity including size and checksum,
  code licence and weight licence, evidence that the configured checkpoint was actually loaded, compute
  device, interpreter and package versions, seed, and the time the run was produced.
- **Availability Report**: The pre-flight verdict set for one run. Attributes: per method, a
  pass/fail verdict, the specific missing or incompatible artefact, remediation guidance, the identity
  of the environment the check ran in (FR-017), and the timestamp of the check.
- **Run Record (Experiment)**: One identified benchmark execution, assembled from one or more returned
  results (FR-044, FR-059). Attributes: experiment name or run ID, page-list identity, methods admitted,
  methods still awaited, per-method configuration snapshot, per-method provenance, per-method device,
  admission timestamps, start and end times, and the list of known nondeterminism sources. It holds no
  single environment or package-version field of its own, because under FR-052 there is no single one.
- **Prediction Result**: The per-page output of one method. Attributes: page identifier, method
  identifier, binary prediction mask at the aligned size, input image size, network input size,
  geometry transformations applied and their inverse, threshold effectively applied, preprocessing
  and post-processing description, inference time, preprocessing time, post-processing time, and
  success/failure status with reason.
- **Per-Page Metrics**: One row per page per method. Attributes: page identifier, method identifier,
  IoU, Precision, Recall, F1, and any degenerate-case convention flag applied to that page.
- **Method Summary**: One row per method per run. Attributes: mean and standard deviation for each of
  the four metrics, mean inference time per page, mean preprocessing and post-processing time where
  measurable, total processing time, count of successful pages, count of failed pages, and a training-
  data-overlap disclosure flag.
- **Failure Record**: One entry per failed page or unavailable method. Attributes: scope (method-level
  or page-level), identifier, recorded reason, and remediation guidance where applicable.
- **Comparison Summary**: The cross-method view for one run. Attributes: one row per method including
  the classical baseline, all four metrics with mean and standard deviation, timing, run status, and
  disclosure notes.
- **Visualisation Case**: One selected qualitative example. Attributes: page identifier, method
  identifier, raw page, ground-truth mask, prediction mask, prediction-versus-ground-truth overlay.

### Out of Scope

- Training or fine-tuning any model.
- Building a new classical baseline, or changing the Spec 1 baseline.
- Inpainting, OCR, machine translation, text rendering, and any end-to-end demonstration pipeline.
- Creating a new benchmark subset or a disjoint evaluation set.
- Reading data from the directory reserved as out-of-scope.
- Modifying Spec 1's ground truth or manifest, except through a reported defect with an explicit
  migration.
- Producing the inpainting feature itself; this feature only guarantees that its prediction masks are
  consumable by that feature.
- Vendoring, copying, committing or storing third-party model source code or pretrained weights in the
  main project in any form (FR-055) — including partial copies, patch series against upstream, and
  single-file extracts. Runners clone and run the external repositories in their own environments.
- Building or maintaining any part of Spec 1's implementation. Its delivery is a blocking precondition
  of this feature (FR-002), not part of its scope.
- Requiring all three methods to share one environment, one dependency set or one machine (FR-052).
- Presenting cross-method timing as a performance ranking where the producing devices differ (FR-060).

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For every method whose result is admitted to the run, 100% of the pages in the page list it
  was produced against receive a prediction mask of exactly the aligned page size containing only the two
  permitted binary values, with zero pages silently skipped, zero pages left at network resolution, and
  the page-list identity matching the manifest-derived export exactly. A method that never returned, or
  whose result was refused by validation, appears as "not run" or "invalid" with a reason rather than as
  a partial coverage figure.
- **SC-002**: A stand-in method implementing the adapter interface produces per-page metrics through
  the Spec 1 metric pipeline with zero modifications to the metric and reporting modules — verified by
  a diff of those modules showing no change.
- **SC-003**: One combined comparison table reports IoU, Precision, Recall and F1 (mean and standard
  deviation) plus mean per-page and total processing time for the classical baseline and every
  deep-learning method that ran, with every unavailable method shown as "not run" with a reason and
  no numeric placeholder anywhere in the table. Every timing cell states its producing device and
  environment; no timing figure appears without that attribution.
- **SC-004**: A run executed with one method's checkpoint deliberately removed completes all remaining
  methods with zero valid results lost, and records exactly one method-level failure with a named
  missing artefact.
- **SC-005**: Two runs with different run identifiers both remain on disk afterwards with distinct
  directories and byte-identical evaluated image lists; a repeated run identifier refuses to overwrite
  prior results unless overwrite was explicitly requested.
- **SC-006**: For 100% of evaluated pages and methods, the stored metadata names the threshold
  effectively applied, the network input size, every geometry transformation, preprocessing,
  postprocessing, checkpoint identity, device and seed — verified by an automated completeness check
  over the metadata files.
- **SC-007**: The availability check returns a verdict for a method in under one minute without loading
  any model weights onto a compute device, and the verdict records the environment it was produced in.
- **SC-008**: The default unit test suite completes without downloading a checkpoint or loading a real
  pretrained model, and covers all seven adapter behaviours listed in FR-050 and all six receipt-path
  behaviours listed in FR-050a.
- **SC-009**: No file under the out-of-scope data directory is opened by any availability check,
  benchmark run, report generation or visualisation generation — verified by an access audit.
- **SC-010**: A consumer that knows only a page identifier and a method name can locate and correctly
  interpret that page's prediction mask, with no access to run-specific metadata — satisfying the
  downstream inpainting hand-off.
- **SC-011**: For 100% of methods in the comparison report, the report states the method's training-data
  contamination status next to its scores — either that no overlap between the method's published
  training data and the evaluation dataset could be found, or that an overlap exists, with its extent
  quantified where the training data is identified (as it is for two of the three methods, where the
  evaluation ground truth is the published training dataset itself — for method A this disclosure
  additionally states that the reported score is the leave-one-fold-out score of FR-013a, computed only
  by checkpoints that provably never saw the page's book) and declared unknowable where the
  training split is unpublished (as it is for the third). No method's scores appear without this
  disclosure, and no accuracy ranking of methods appears without it attached.
- **SC-012**: A single run admits results for all three methods produced in three mutually independent
  environments, every admitted result carrying the same page-list identity **and** the same input-image
  identity record (FR-054, FR-054a), and every reported metric for every method computed by the project's
  own single evaluation procedure. Verified by inspecting the recorded page-list and input-image
  identities on all admitted results and confirming that no score value in the report originated in a
  runner's environment.
- **SC-013**: The project's stored tree contains zero third-party model source files and zero model
  weight files, of any size and in any form — verified by an audit of the tree against the configured
  external repository paths, in the same manner as SC-009's access audit. What is stored for each method
  is limited to its configuration, its returned standardised results and its documentation.
- **SC-014**: For 100% of admitted results, the provenance record names the external repository and its
  exact revision, the checkpoint identity including checksum, the code licence and the weight licence,
  positive evidence that the configured checkpoint was loaded, the compute device and the package
  versions — verified by an automated completeness check. A result submitted with any of these absent is
  refused, the refusal is recorded with the specific missing field, and the run remains reportable over
  the results already admitted.

---

## Assumptions

### Dependency on Spec 1 — verified state of the project

- **Spec 1 exists as a specification only.** Verified on 2026-09-10: `specs/001-data-foundation-classical-baseline/`
  contains `spec.md` and `checklists/requirements.md`, and nothing else — there is no `plan.md`, no
  `tasks.md`, no `research.md`, no `data-model.md` and no `contracts/` directory. The project has no
  `src/`, no `tests/`, no dependency manifest, no generated `outputs/`, no manifest file, no
  configuration file and no command-line entry point. The project is not a git repository, so the
  constitution's worktree stage is currently inoperable.
- **Consequence for this specification.** The contracts this feature binds to are inherited from Spec
  1's `spec.md` at the specification level only: the segmentation-method contract
  (`segment(image) -> binary mask`, Spec 1 FR-021/FR-022/FR-035), the binary convention (FR-012),
  the alignment rule (FR-016/FR-017/FR-018), the metric set (FR-027…FR-031), the log-and-continue
  failure rule (FR-031), the four-panel visualisation (FR-032/FR-033), the module separation rule
  (FR-034), configuration-outside-code (FR-036) and CPU operation (FR-037). Concrete function
  signatures, the manifest schema, configuration keys and output path layouts are **not** knowable
  today and MUST be inherited from Spec 1's implementation once it exists rather than invented here.
- **Ordering — decided by the requester on 2026-09-10.** This feature is finalised against Spec 1's
  contracts as written and does **not** need to be implemented immediately. The confirmed sequence is:
  Spec 1 is implemented first; then the external model runs; then benchmark integration. Spec 1's
  implementation is therefore a *blocking precondition* of this feature, not a dependency to be
  satisfied by building it here — FR-002 states the precondition and forbids substituting a locally
  derived page list or loader to make progress in the meantime. The practical consequence is that this
  specification is ready to hand to `/speckit-plan` now, while its first executable task cannot start
  until Spec 1 exists.
- Spec 1's own success criterion SC-008 already anticipates this feature: a stub implementing the
  common interface must produce metrics through the unchanged pipeline. FR-008 and SC-002 here are the
  same obligation seen from this side.

### Execution model — distributed, decided by the requester on 2026-09-10

- **The decision.** Each team member runs one model independently in Google Colab. The models do not
  need to run in the same local environment. The requester named the requirements that make separately
  produced results combinable: a shared manifest, a shared preprocessing/alignment convention, a shared
  prediction-mask format, a shared evaluation script, and recorded environment metadata. Those five are
  elevated to FR-053 verbatim in substance, and the surrounding mechanics to FR-052 and FR-054…FR-060.
  FR-053 nevertheless carries **six** invariants, not five, because this specification adds one the
  requester's list did not name: **the input images those pages denote** (FR-054a). The addition is not
  a disagreement with the decision but a consequence of it — the requester's list was written on the
  assumption that the images were already locally available to whoever ran a model, and moving execution
  into three separate Colab sessions removes that assumption. Naming the page list without supplying the
  images it denotes would leave each runner free to obtain them from a different source, which page
  names and page sizes alone cannot detect. The five named invariants are reproduced faithfully; the
  sixth closes the gap the move to distributed execution opened.
- **What this changes architecturally.** Inference moves out of the project. What the project owns is
  the export of the page list, the definition of the standardised result, the receipt-and-validation
  path, and the single evaluation procedure. What it does not own is any model runtime, any checkpoint,
  or any third-party code. The adapter (FR-008) remains the conceptual boundary at which a method's own
  repository is called, but under this model the adapter is exercised in the runner's environment and
  the project consumes its output rather than invoking it.
- **What stays comparable, and what does not.** Accuracy stays fully comparable: it depends only on the
  shared page list and the shared evaluation procedure (FR-057), both of which live in the project, so
  IoU/Precision/Recall/F1 are directly comparable across three differently-produced results and against
  the classical baseline. Timing does not: three Colab sessions are three unknown devices, possibly
  mixed accelerator and CPU, and a timing column across them would report hardware rather than
  architecture. FR-060 binds this — every timing figure carries its producing device, and no cross-device
  timing ranking is presented. This is a genuine loss against the requester's original "so sánh thời
  gian xử lý" goal, and it is reported as such rather than papered over: the comparison still exists,
  per method and per device, but it stops being a ranking.
- **Spec 1's CPU mandate is reinterpreted, not violated.** Spec 1 FR-037 requires the *benchmark* to run
  on CPU with no GPU required. That mandate now applies to what the project itself executes: the
  evaluation procedure, the metric computation, the reporting and the visualisation, all of which remain
  CPU-only and run on the development machine. It cannot apply to the per-method inference runs, which
  are the whole reason Colab was chosen — a hosted accelerator environment. The classical baseline stays
  CPU-only and in-project, so the two are compared on accuracy (comparable) and reported on timing
  (attributed, not ranked).
- **Method A's environment problem is dissolved by this decision.** Method A pins a framework stack
  whose interpreter ceiling is below the one installed here, which on the development machine was a
  scope decision requiring an isolated legacy environment. Colab can supply a legacy interpreter, so the
  question changes from "is it in scope to build one" to "does the hosted environment work". The
  requester directed: attempt Method A in its required Colab environment; if it cannot run, report it as
  unavailable with the reason. That is now FR-002's availability behaviour plus the *Interpreter-version
  ceiling* edge case, and it is no longer an open question.
- **Ephemeral environments are expected, not exceptional.** A Colab session ends and takes its clone,
  its checkpoint and its cache with it. Nothing durable is lost provided the hand-off completed first,
  because the durable record is the returned result plus its provenance record, not the cache (FR-046).
  An incomplete hand-off simply means the method has not returned: the run stays reportable over what
  has (FR-059), and no cached fragment is substituted for a missing result.

### Repository boundary — no vendoring, decided by the requester on 2026-09-10

- **The decision.** Use external repositories. Team members may clone and run the original repositories
  in Colab, but the main project must not vendor or copy third-party source code or weights. The main
  project stores only shared contracts, evaluation code, configurations, standardised outputs and
  documentation. Record repository, commit, checkpoint, licence and environment information for every
  method. This is FR-055 and FR-056, audited by SC-013 and SC-014.
- **The requester's list extends what FR-044 already required.** FR-044 already named checkpoint
  identity and code revision; the decision adds *commit* precision (an exact revision, not a branch or a
  tag that can move), *licence* for both code and weights, and *environment* as a first-class recorded
  field per method. FR-056 records all five.
- **The consequence for upstream defects.** Three concrete upstream problems are already verified in
  this feature's evidence base: a numeric-library alias removed in the installed version, an import of a
  graphical component that a headless environment lacks, and a threshold parameter the code ignores in
  favour of a hardcoded constant. Under a vendoring model these would be patched in-tree. Under this
  boundary they may not be: the workaround is applied in the runner's own environment and recorded in the
  provenance record as a deviation from the pinned revision, with what changed and why (*Upstream defect
  requiring a workaround* edge case). A workaround that alters the model's output rather than merely
  letting it run — a different threshold, different preprocessing constants, a re-exported checkpoint —
  is a different method and must not be reported as the pinned one. If no workaround lets a method run
  without changing its output, FR-018 applies and the method is reported unavailable.
- **Why this boundary is also a legal one, not only a tidiness one.** The three methods carry three
  different licence positions (see *Licence* below). Not copying third-party code means the project never
  redistributes GPL-3.0 code and never copies code that has no licence at all, which for method C is the
  only defensible position since there is no grant to copy from. The boundary therefore discharges the
  requester's own requirement to distinguish official from third-party from fallback: each method is
  identified by repository, revision and licence, and nothing about it is reproduced in the project.

### Environment — verified on 2026-09-10

- Python 3.10.1, package installer 21.2.4, no virtual environment active.
- Present: numpy 2.2.6, opencv-python 5.0.0.93, pandas 2.3.3, matplotlib 3.10.8, Pillow 12.1.1,
  scipy 1.15.3, scikit-learn 1.7.2, pytest 9.0.2.
- **Absent: every deep-learning runtime.** No torch, no torchvision, no segmentation-models-pytorch,
  no ultralytics, no mmsegmentation, no onnxruntime, no timm, no albumentations, no
  opencv-contrib-python, no scikit-image. All three methods therefore require dependencies this machine
  does not have, and an availability check run **here** is expected to report all three unavailable until
  they are installed. This is the normal, designed behaviour of FR-016, not a defect.
- **That verdict describes this machine only, and FR-017 forbids relying on it elsewhere.** Under the
  execution model decided on 2026-09-10 the three inference runs happen in hosted notebook environments
  chosen by whoever runs them, not here. A "unavailable on a fresh checkout" result obtained in the
  project's own environment is therefore *not* evidence about the runners' environments, and a verdict
  obtained in a runner's environment is *not* evidence about any other runner's. Each availability check
  records the environment it was performed in, and only that environment's verdict is admissible for it.
  What this machine must supply is the evaluation procedure and its dependencies, which are already
  present above.
- **No checkpoints and no vendored third-party model code exist anywhere under `D:\UTE\DIPR`.** Verified
  by filesystem search. Nothing in this project can be reused as a model source — and under FR-055
  nothing is to be added to it as one either. Checkpoints are obtained by each runner in its own
  environment; the project stores their identity, size, checksum and licence, not their bytes.

### Dataset — measured directly, not assumed

- 450 ground-truth masks across 45 manga, 10 per manga; 390 masks pair with a source image by exact
  filename stem; 60 masks are orphans across 6 manga that have no images in the release. The
  benchmark evaluates the 390 pairs.
- **Page geometry is perfectly uniform across all 390 pairs.** Source images are all 1654×1170
  (width×height); ground-truth masks are all 1656×1176; exactly **one** distinct size combination
  exists across the whole benchmark. Spec 1's alignment contract is therefore a single fixed rule for
  this feature, not a per-page decision, and the aligned evaluation space is 1654×1170.
- **The +2/+6 delta is not an arbitrary canvas pad — it is pad-to-multiple-of-8.** Verified by
  measurement across all 450 masks: every mask dimension is an exact multiple of 8, and
  `ceil(1654/8)×8 = 1656` and `ceil(1170/8)×8 = 1176`. This reproduces, exactly, the padding rule that
  the upstream mask dataset documents for its "post-processed" variant. Two consequences follow:
  (a) Spec 1's top-left crop to 1654×1170 is confirmed **lossless**, because the padding carries no
  content by construction rather than by inspection alone; and (b) a method that pads its own input to a
  multiple of 8 is performing the *same* transformation the ground truth already received, so
  FR-024's geometry mapping MUST account for that shared convention rather than treating the two pads
  as independent.
- **Ground truth is stroke- and glyph-level, not region-level.** Measured over a random sample: text
  occupies 0.86%–11.9% of the page area; pages contain 69–724 connected text components; the ratio of
  text area to component bounding-box area is 0.42–0.65; mean stroke width ranges from 3.06 px to
  14.50 px; and the IoU between a mask and its own single-pixel 3×3 erosion falls as low as 0.446 on
  thin-stroke pages. Two consequences are binding on this specification: (a) a published claim that
  this dataset's annotations are region-level rectangles is **false** and must not be used to explain
  away low IoU; (b) one-pixel boundary disagreement is heavily penalised, which is why FR-026, FR-029
  and FR-030 forbid per-page threshold tuning and undisclosed morphological post-processing.

### Method B (comic-text-detector) — verified in detail

- Source repository `dmMaze/comic-text-detector`, licence **GPL-3.0**, last upstream commit
  2023-08-13, no published releases.
- Checkpoint `comictextdetector.pt`, 79,948,869 bytes, hosted on a **third-party** release
  (`manga-image-translator`, tag `beta-0.2.1`) rather than by the model author. Reachability verified
  by HTTP HEAD (status 200); the file was downloaded and verified to contain all three expected
  component keys (`blk_det`, `text_seg`, `text_det`). Because it is large, it is subject to FR-046 and
  must live in a configured cache directory outside the repository.
- **Architecture verified by reading the code, not inferred**: a single YOLOv5s backbone with two
  heads — a U-Net-style transposed-convolution segmentation head with a sigmoid output producing the
  pixel mask, and a DB head producing ICDAR-style text-line polygons. The forward pass returns the
  backbone block features, the segmentation mask and the text lines together. **This feature consumes
  only the segmentation head's refined mask** (FR-013, FR-014); the polygon output is discarded.
- The refined mask is available at original page resolution as a binary {0,255} unsigned 8-bit array —
  verified against a committed example output for a Manga109s page at 1654×1170, which is exactly this
  benchmark's aligned size.
- **Preprocessing verified**: letterbox to a 1024×1024 canvas with stride-64 bottom-right zero
  padding, scaled by 1/255 only, with **no** ImageNet mean/std normalisation. This is why FR-025 and
  FR-027 require the geometry mapping and normalisation constants to be recorded rather than assumed
  shared.
- **No command-line interface exists upstream** (zero argument-parsing constructs in the repository) and
  the reference inference script blocks on an interactive image window. A headless entry point must
  therefore be supplied by the adapter.
- Known breakage against the current environment, each of which is a concrete availability-check
  target under FR-016 and FR-021:
  1. The upstream I/O helpers reference two numeric-library aliases that were removed in numpy 2.x, so
     they raise an attribute error against the installed numpy 2.2.6.
  2. Two geometry dependencies are missing from the upstream requirements file; the corresponding
     upstream issue is still open.
  3. A stray interactive-window import breaks headless execution.
  4. A configurable mask threshold is accepted but never used; the code hardcodes an integer cutoff
     equivalent to approximately 0.235 on the sigmoid output. FR-030 exists precisely for this case.
  5. A suspected red/blue channel-order reversal between training-time and inference-time
     preprocessing. Its effect on accuracy is unmeasured; FR-025 requires the channel order actually
     used to be recorded.
- **Pinned source — decided by the requester on 2026-09-10 (Clarifications): the upstream repository
  `dmMaze/comic-text-detector` at a pinned revision is method B's primary source**, recorded in the
  provenance record (FR-056). The actively maintained fork (`kha-white`, 10 commits ahead, last touched
  2026-07-19, same GPL-3.0 licence) and the `Ajatt-Tools` fork named in the course outline are **patch
  sources only**: where one supplies the fix for one of the five breakages above, the patch is applied
  in the runner's environment and recorded as a deviation from the pinned upstream revision under the
  *Upstream defect requiring a workaround* edge case. The course outline cites the `Ajatt-Tools` fork;
  citing upstream in the report is consistent with it, because the outline itself identifies that fork
  as derived from `dmMaze/comic-text-detector`. Both fork and upstream are **third-party**; neither is
  an official release from the model author.

### Training-data contamination — verified, benchmark-wide, and partly recoverable

This is the single most consequential finding in this specification, and unlike the other assumptions
here it was **proven from primary sources** rather than inferred from a README.

- **The evaluation ground truth is the published training dataset of methods A and C, identified
  exactly.** The upstream training code publishes an explicit list of 45 book folders. That list was
  fetched and compared set-wise against the 45 book folders under this project's ground-truth tree:
  **the two sets are identical, with zero difference in either direction.** Both contain 45 books × 10
  masks = 450 masks. The upstream dataset record states it "contains 450 images with the text
  segmentation of images from Manga109 dataset", that its *post-processed* variant is "after
  automatically removing small connected components and filling small holes", and that these masks
  "are also slightly bigger in width/height in order to be multiples of 8". Every clause of that
  description matches an independent measurement made on this project's data: 450 masks, all
  dimensions multiples of 8, the +2/+6 delta reproduced exactly by `ceil(1654/8)×8` and
  `ceil(1170/8)×8`, and a local directory named `post-processed`. This benchmark's ground truth **is**
  that dataset, not merely a sample drawn from the same source.
- **Methods A and C therefore saw approximately 100% of the evaluation ground truth during training**,
  not a fraction of it. Method C's own documentation cites the same dataset record as its training
  source. Method B's upstream README states its training corpus drew about one third from Manga109-s,
  with the split unpublished.
- **The contamination is structural, not accidental.** The upstream training code splits those 45 books
  with 5-fold cross-validation (shuffle enabled, fixed seed 42), trains one model per fold, and
  publishes all five checkpoints. Consequently **every book was held out from exactly one fold and
  trained on by the other four.** For any given page, at most one of the five released checkpoints
  never saw that page's book — and which one, per book, is computable from the published seed.
- **A partial recovery is possible for method A, and on 2026-09-10 the requester chose to require it.**
  The user forbids creating a new benchmark subset, so the evaluation remains the full manifest;
  leave-one-fold-out adds no subset and removes no page — it evaluates the same 390 pages, scoring each
  by the one checkpoint of the five whose training split provably never saw that page's book. The
  requester's decision (Clarifications, 2026-09-10) makes this method A's **primary** evaluation, bound
  as FR-013a: all five checkpoints are used, the book→fold mapping is derived from the published seed
  and verified before use, and method A's reported score is de-contaminated by construction. **It does
  not extend to method C**, which publishes a single checkpoint trained on all 45 books — C's scores
  remain fully contaminated and can carry only disclosure. This changes the shape of the comparison:
  method A is now the only deep-learning method whose reported accuracy is defensible as a
  generalisation estimate.
- **What is not recoverable.** Method B's split is unpublished, so its contamination extent remains
  genuinely unknowable — it can be neither measured nor excluded. Its scores can only carry a
  disclosure.
- **Mandatory disclosure in every case.** SC-011 requires the comparison report to state the
  contamination status next to every method's scores, with the extent quantified where it is known
  (methods A and C: the dataset is identified and the overlap is total) and declared unknowable where it
  is not (method B). No method's scores appear without it. No attempt to de-contaminate, re-sample or
  exclude pages from the primary benchmark is in scope.
- **Interpretation constraint.** Because two of the three deep-learning methods trained on essentially
  the whole benchmark while the Spec 1 classical baseline is not contaminated at all, a raw accuracy
  comparison between them measures memorisation as much as capability. After the 2026-09-10 LOFO
  decision the asymmetry also runs within the deep-learning group itself: method A's reported score is
  de-contaminated (FR-013a) while method C's is inflated by memorisation, so a result favouring C over
  A measures the contamination as much as the capability. Any statement of the form "method X
  outperforms method Y" in this feature's report MUST be accompanied by the contamination disclosure
  stating which of the two scores is de-contaminated, and MUST NOT be used to rank methods without it.

### Method A (U-Net / ResNet34) — verified in detail

- Source repository `juvian/Manga-Text-Segmentation`, licence **MIT** — the only permissively licensed
  method in this benchmark. This is the *original* implementation, not a fork: an identically-named
  repository that was an earlier candidate does not exist (HTTP 404).
- **Architecture confirmed from source**, matching the feature request: `unet_learner` with a
  torchvision `resnet34` encoder producing a dynamic U-Net. The released artefacts for this exact
  configuration are five checkpoints, one per cross-validation fold, each approximately 344 MB, hosted
  on the repository's own GitHub release (tag `v1.0`) and verified reachable. Being large, they fall
  under FR-046 and must live in a configured cache directory outside the repository.
- **There is no inference entry point of any kind.** The repository ships notebooks and library modules
  only — zero argument-parsing constructs, no script that loads a checkpoint and writes a mask. Its own
  prediction notebook saves raw floating-point tensors per page rather than masks, and only for a
  hand-picked model. A headless inference path must therefore be constructed by this feature's adapter,
  from the training-time data pipeline, under FR-009 and FR-013.
- **Preprocessing and output convention verified from source**: training normalises with ImageNet
  mean/std; the training-time pipeline pads every item to a multiple of **8** — the same multiple as
  this benchmark's ground truth, which is direct further evidence of the shared provenance established
  above. The loss is a fixed combination of focal (γ=1) and Dice. Inference applies a **fixed 0.5
  sigmoid threshold that is not exposed as a parameter**, so FR-026's configurability must be satisfied
  by the adapter and FR-030 applies.
- **The label space is multi-class, not binary**: 0 = background, 1 = "easy" text (black, mostly inside
  speech bubbles), 2 = "hard" text (pink — covers, sound effects, outside bubbles), and 3–5 = ignore.
  The upstream evaluation counts classes 1 and 2 as text and excludes class 3 from both hits and misses.
  This benchmark's ground truth is the *post-processed binary* variant, in which the easy/hard colour
  split has already been collapsed, so no ignore-class pixels are present and the collapse to Spec 1's
  binary convention (FR-004) is lossless. That this is so MUST be verified by the adapter rather than
  assumed, because it is what makes the comparison to the other two methods valid at all.
- **Method A cannot run in the project's own environment, and the 2026-09-10 decision resolved what to
  do about it.** Its stack pins `fastai 1.0.60` with `torch 1.4.0` and `torchvision 0.5.0`, which cap at
  **Python 3.7**; the interpreter installed here is Python 3.10.1 with no virtual environment active. The
  checkpoints are fastai-v1 pickles whose loader can only deserialise them with that exact legacy stack.
  The repository additionally ships no dependency manifest for the undeclared packages it imports (image
  processing, scientific, array, plotting and a Manga109 parser library), and hardcodes absolute Linux
  data paths that do not exist on this machine. On the development machine this was a genuine scope
  decision — build a dedicated isolated legacy environment, or reduce the feature to a two-method
  comparison — and was raised as clarification question 2.
- **Resolution: attempt method A in its required hosted notebook environment; if it cannot run, report it
  as unavailable with the reason.** A hosted environment can supply a legacy interpreter, so the question
  changes from "is building an isolated environment in scope" to "does the hosted environment work", and
  the answer is determined by attempting it rather than by deciding scope in advance. This is why the
  *Interpreter-version ceiling* edge case now reads as attempt-then-report rather than as a scope fork,
  and why FR-017 requires the availability verdict to name the environment it was obtained in.
- The remaining obstacles are unchanged by that resolution and are what the attempt has to clear: the
  missing dependency manifest, the undeclared parser library, and the hardcoded absolute data paths. Each
  is a configuration or environment problem solvable in the runner's own environment, and none may be
  fixed by copying or patching the repository into this project (FR-055). Any change needed to make it run
  is applied in the runner's environment and recorded as a deviation from the pinned revision under the
  *Upstream defect requiring a workaround* edge case.
- Under FR-018 and FR-020, if the attempt fails, method A is recorded as **unavailable with a named
  reason and remediation guidance**, the benchmark continues with the remaining methods, and no result is
  fabricated. A method reported unavailable from a hosted notebook environment is an *observed* fact about
  that environment, not an inference drawn from this machine's package inventory.
- **Checkpoint strategy — leave-one-fold-out, decided by the requester on 2026-09-10.** The five
  released fold checkpoints (~344 MB each, ~1.7 GB total) are all required, and every page is scored by
  the checkpoint of the fold that held its book out (FR-013a). The book→fold mapping is recomputed from
  the upstream KFold construction (shuffle enabled, seed 42) and verified against the upstream split
  definition before any page is scored; each page's fold attribution is recorded in the per-page
  metadata (FR-027) and the provenance record (FR-056), and checked at receipt (FR-058). Each page is
  inferred exactly once — the strategy multiplies model loads and checkpoint downloads, not inference
  passes — and method A's availability check MUST confirm all five checkpoints before any page is
  scored.

### Method C (UNet++ / EfficientNetV2) — verified in detail

- Source repository `ContemporaryCat/Manga-Text-Segmentation`. **This carries no licence at all** —
  neither the repository nor its model card states one. It is therefore *not* a fork of method A's
  repository (an earlier assumption in this specification, now corrected) but an **independent
  reimplementation**, and under default copyright the author reserves all rights: there is no grant to
  copy, modify or redistribute its inference code or weights. This is the most restrictive licence
  position of the three methods despite looking the most modern, and it was the core of clarification
  question 3.
- **Architecture confirmed from source**, matching the feature request: a UNet++ decoder over a
  `timm`-provided EfficientNetV2-M encoder, with scSE decoder attention, one input channel set of 3 and
  one output class, no built-in activation, and BatchNorm converted to GroupNorm in the decoder. The
  encoder is downloaded from the model hub at construction time (approximately 215 MB).
- Checkpoint `model.pth`, **216,911,417 bytes**, hosted on a Hugging Face model repository and verified
  reachable. Large, so FR-046 applies.
- **The output is a floating-point probability map**, not a binary mask, and the reference code applies
  a default 0.5 threshold. FR-026 therefore binds directly: the threshold must be configurable and the
  effective value recorded.
- **Preprocessing verified from source**: ImageNet mean/std normalisation, and zero-padding to a
  multiple of **32**. Note this is *not* the same multiple as the ground truth's pad-to-8: 1656 and
  1176 are multiples of 8 but **not** of 32. FR-024's geometry mapping must therefore invert a different
  padding from the one the ground truth received, and FR-025 must record that the two are not the same
  transformation. Optional horizontal/vertical flip test-time augmentation and mixed-precision
  inference are present in the reference code; both change results and/or timing, so both must be
  configuration that is recorded per FR-027 and FR-044, never silent.
- **No command-line interface, no file I/O, no batch mode.** The only entry point is an interactive web
  demo that accepts an upload and displays a result. Like method A, a headless inference path must be
  constructed by the adapter.
- **Method C contains a silent-failure trap that directly engages the user's "no fabricated results"
  prohibition.** Its checkpoint path is a hardcoded, working-directory-relative filename, and the code
  that loads it catches a missing-file error and merely prints a warning before *continuing*. The
  consequence is that a missing, misnamed or wrongly-located checkpoint yields a **randomly initialised
  decoder that still produces plausible-looking masks and raises no error**. Every downstream
  requirement would appear satisfied while the numbers are meaningless. This is why FR-016 and FR-018
  are written as they are: the availability check MUST positively verify that weights were actually
  loaded from the configured checkpoint — not merely that inference returned a mask of the right shape
  and value range — and a method that cannot prove its weights are the published ones MUST be reported
  as unavailable rather than run. It is recorded as an edge case below.
- The repository ships **no dependency manifest whatsoever**, and its documentation cites the same
  upstream mask dataset record identified in the contamination section as its training source.

### Compute device and runtime budget — resolved by inheritance, then re-scoped by decision

- Spec 1 FR-037 requires the pipeline to run on CPU with no accelerator, and the user's instruction to
  reuse Spec 1's foundation rather than rebuild it means this feature inherits that constraint. **What
  the decision of 2026-09-10 settled is which part of this feature it applies to.** The mandate binds
  what the project itself executes — the shared evaluation procedure, the metric computation, the
  reporting and the visualisation, plus the classical baseline it compares against. All of that remains
  CPU-only and runs on the development machine, so the benchmark is reproducible by anyone without an
  accelerator. It cannot bind the per-method inference runs, which happen in hosted notebook
  environments precisely because those environments supply an accelerator this project does not.
- **There is therefore no single device for this feature, and the specification does not pretend there
  is.** Each method's inference device is whatever its runner's environment provided, discovered rather
  than demanded, and recorded per method by FR-044 — which is why FR-044 requires that record to come
  *from the environment that produced the result* and forbids recording one environment for the whole
  run. The availability check reports the device it actually found rather than demanding a specific one
  (FR-016), and its verdict is scoped to its own environment (FR-017).
- **The consequence for the timing comparison is a real reduction in what can be claimed, and it is
  stated rather than hidden.** Three Colab sessions are three unknown devices, and one method in scope
  additionally uses mixed-precision inference that is meaningless without an accelerator. FR-027 and
  FR-044 require the device, and any precision or test-time-augmentation setting that affects it, to be
  recorded per run. FR-060 then binds the reporting: every timing figure carries its producing device,
  and no cross-method timing ranking is presented where the devices differ. The timing comparison
  demanded by FR-033 and SC-003 is therefore a comparison **on the declared device**, never an absolute
  claim about a method's speed, and never a ranking across methods that ran on different hardware.
  Accuracy is unaffected: it depends only on the shared page list and the single evaluation procedure
  (FR-057), so it stays directly comparable across all four approaches however they were produced.
- No wall-clock budget is imposed by this specification; planning measures it. An accelerator MAY be
  used anywhere and MUST then be recorded identically. Neither point is a clarification question: the
  project-side default is fixed by an inherited requirement, and the runner-side device is fixed by
  observation.

### Licence — three different positions, not one

Verified per method, and materially different from one another:

| Method | Source | Licence | What it permits |
| --- | --- | --- | --- |
| A | Original implementation of the published baseline | **MIT** | Copy, modify and redistribute with attribution. The least restrictive position in this benchmark. |
| B | Original implementation plus an actively maintained third-party fork | **GPL-3.0** | Same, but copyleft: derivative work that is distributed must carry the same licence. |
| C | Independent reimplementation | **No licence stated anywhere** | Default copyright applies. There is **no grant** to copy, modify or redistribute its code or weights. |

- **Resolution — decided by the requester on 2026-09-10.** Use external repositories: team members may
  clone and run the original repositories in their own hosted environments, but the main project must not
  vendor or copy third-party source code or weights. The project stores only shared contracts, evaluation
  code, configuration, standardised outputs and documentation, and records repository, commit, checkpoint,
  licence and environment for every method. This is FR-055 and FR-056, audited by SC-013 and SC-014. The
  question of *how* each method is integrated — vendored-and-patched versus configured-external-repository
  versus reimplemented-from-the-paper — is therefore settled on the second option for all three, and the
  user's requirement to distinguish official from third-party from fallback is discharged by the recorded
  repository identity, revision and licence rather than by any copy held in the project.
- **The decision inverts method C's position from most-constrained to comfortably compliant.** Under
  default copyright there is no grant to copy, modify or redistribute — so the *only* lawful way to use
  method C is exactly what was chosen: run it where the author's own repository and hosted checkpoint
  live, and take back nothing but masks and a record of what was used. The project never redistributes
  method C's code or weights, so the absence of a licence stops being a blocker and becomes a documented
  constraint on what may be stored. The same reasoning makes method B's GPL-3.0 copyleft moot for
  distribution purposes: the project carries no derivative work, only a description of one, so no
  copyleft obligation is triggered by the project's own artefacts.
- **What the licence positions now constrain is not integration but repair.** Because nothing may be
  copied in-tree, the three verified upstream defects — a removed numeric-library alias, an import of a
  graphical component a headless environment lacks, and a threshold parameter ignored in favour of a
  hardcoded constant — cannot be patched inside the project. Each is worked around in the runner's own
  environment and recorded as a deviation from the pinned revision. Where a licence permits modification
  (method A, MIT) a runner may modify its local clone; where it does not (method C) the runner may still
  *use* the code as the author published it, and any change it must make to get it running is recorded as
  a deviation with what changed and why. Method B sits between, and its copyleft applies to the runner's
  own clone, not to this project.
- Regardless of the above, the licence identifier is required in the availability report (FR-016), in the
  provenance record (FR-056) and in the run record (FR-044), for the code **and** for the weights
  separately, and the checkpoint licence/usage terms are required to be checked before running per the
  user's instructions. If a method's code cannot lawfully be used even from its own repository, the
  availability check reports it unavailable with that reason rather than the adapter quietly
  reimplementing it.

### Scope boundaries and conventions

- This feature produces a specification and, later, an implementation that **runs** pretrained models.
  It produces no trained artefacts of its own.
- The out-of-scope data directory is never read (FR-006, SC-009).
- Following Spec 1's convention, this specification is written in English for tooling consistency while
  the user communicates in Vietnamese.
- The technology stack assumed by Spec 1 (Python with OpenCV, NumPy, pandas and matplotlib) remains
  non-binding here and is fixed at planning time; the deep-learning runtime and any third-party model
  repositories are additional and MUST be justified against the constitution's simplicity-and-reuse
  principle, since this feature cannot be delivered without them.
