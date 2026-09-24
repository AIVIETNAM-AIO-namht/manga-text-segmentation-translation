# Bảng phân công công việc — Spec 003 & Spec 004

**Nguồn**: [specs/003-text-removal-inpainting/tasks.md](specs/003-text-removal-inpainting/tasks.md) (60 task) và
[specs/004-ocr-translation-rendering/tasks.md](specs/004-ocr-translation-rendering/tasks.md) (82 task).
Tổng **142 task**, chia cho **3 người**.

---

## 0. Quy ước chung — đọc trước khi bắt đầu

### 0.1 Nhịp làm việc (không thương lượng)

Constitution Principle II là *Spec-Scoped Test-First (NON-NEGOTIABLE)*. Mỗi task test phải:

1. Viết test trước.
2. Chạy — **phải FAIL**.
3. Viết code tối thiểu cho tới khi PASS.
4. Refactor.
5. Commit (một commit cho mỗi task hoặc mỗi nhóm logic).

Lệnh chạy suite — **bắt buộc có `-o addopts=''`** (`pyproject.toml` đặt `--cov=` nhưng `pytest-cov`
không có trong môi trường):

```bash
pytest -o addopts='' -q
```

### 0.2 Ranh giới an toàn — MUST NOT, áp dụng cho cả 3 người

| # | Cấm | Nguồn |
|---|-----|-------|
| 1 | **Không ground truth nào được chạm tới output.** Prediction mask là mask duy nhất được mở. Không chế độ GT-assisted nào là mặc định. | 003 FR-033, 004 FR-008, US1 scenario 4 |
| 2 | **Không đọc `data/no-need-to-read/`** dưới bất kỳ hình thức nào. | 003 FR-007, 004 FR-007 |
| 3 | **Không vendor thứ ba**: không source model, không weights, ở mọi kích cỡ, mọi định dạng. `manga-ocr` là extra tuỳ chọn sau lazy import. | 003 FR-055/SC-013, 004 FR-005/FR-028 |
| 4 | **Không secret nào vào file**: chỉ đọc từ biến môi trường, kiểm tra lúc khởi động. `provider`/`model`/`version` được ghi; **giá trị** key thì không, ở đâu cũng không. | 004 FR-034/FR-035/FR-057 |
| 5 | **Nguồn là read-only**: `data/raw/…`, `data/groundtruth/…`, manifest Spec 1, mask Spec 2, output Spec 3 — không sửa, không move, không rename, không xoá. Cần đưa vào cây output thì **copy, không move**. | 003 FR-006, 004 FR-006 |
| 6 | **`page_list_identity` / `input_image_identity` phải trích nguyên văn** từ `benchmark/`, không bao giờ tính lại. Không tự dựng manifest mới bằng cách ghép tên trong `images/` và `post-processed/`. | 003 FR-054a, 004 FR-001, ONBOARDING §8 |
| 7 | **Không tự tính metric.** Metric do project tính, runner không nộp. Không PSNR/SSIM/ground-truth tổng hợp/xếp hạng một điểm trong artefact chính. | 003 FR-030/FR-031, 002 FR-057 |
| 8 | **`src/manga_text_seg/run.py` (số ít) là module run-record của Spec 2** — không đụng vào. Module của Spec 003 là `runs.py` (số nhiều); của Spec 004 là `translation/runs.py`. | cả hai tasks.md |
| 9 | **Không tự sửa/thay thế/bịa** input, vùng, OCR text, bản dịch hay bản render hỏng — ghi nhận, không che. | 004 FR-055 |
| 10 | **`.reissue-page.html` giữ nguyên untracked** — không commit. | lựa chọn của chủ dự án |

### 0.3 Sở hữu file — tránh đụng độ

| File | Ai giữ | Ghi chú |
|------|--------|---------|
| `src/manga_text_seg/runs.py` (003) | **A** cho T007–T011, T021, T027, T029; **B** cho T036–T038, T051 | B **không** mở file này cho tới khi A xong US2 |
| `src/manga_text_seg/inpaint.py` (003) | A (T026, T028, T029, T030) → B (T035) | tuần tự theo story, không song song |
| `src/manga_text_seg/translation/runs.py` (004) | **chỉ C** | T009–T012 và T066–T070 cùng một file — serialize |
| `src/manga_text_seg/config.py` | A (003-T006) và C (004-T008) | **đụng độ giữa 2 spec** — ai xong trước land trước, người sau rebase |
| `src/manga_text_seg/cli.py` | A/B (003) và C (004) | **đụng độ giữa 2 spec** — cùng cách xử lý |
| `tests/conftest.py` | A (003-T003) và C (004-T007) | **đụng độ giữa 2 spec** — cùng cách xử lý |
| `assets/fonts/`, `pyproject.toml` | chỉ C | |

Ba file đụng độ (`config.py`, `cli.py`, `conftest.py`) đều là edit kiểu **thêm mới** (hàm mới,
subcommand mới, fixture mới) nên conflict nhỏ và dễ rebase. Không cần branch riêng.

---

## 1. Tổng quan

| Người | Spec | Phạm vi | Số task | Vai trò |
|-------|------|---------|---------|---------|
| **A** | 003 | Phase 1–4: T001–T031 | **31** | Đường găng của Spec 003 — không ai làm US1/US2 nếu A chưa xong |
| **B** | 003 | Phase 5–9: T032–T060 | **29** | Nửa sau của Spec 003: batch, boards, qualitative, ablation, polish |
| **C** | 004 | Toàn bộ: T001–T082 | **82** | Spec 004 trọn vẹn |

**Chuỗi dài nhất là của C (82 task).** Xem §5 để biết cách san tải khi A hoặc B xong sớm.

**Mốc kiểm tra (checkpoint)** — mỗi người dừng lại và tự kiểm trước khi đi tiếp:

- A: hết T011 (nền xong) → hết T021 (cổng intake đứng một mình) → hết T031 (US1+US2 chạy độc lập).
- B: hết T039 → T044 → T047 → T051 (sáu user story đều chạy độc lập).
- C: hết T012 (nền xong) → T019 (MVP) → T029 → T038 → T048 → T057 → T064 → T070 → T074 → T082.

---

## 2. BẢNG CÔNG VIỆC — NGƯỜI A

> Spec 003, Phase 1–4. A là người **duy nhất** được sửa `runs.py` cho tới hết T031.

### Phase 1 — Setup

| Mã | Việc | Tác vụ con | File | Phụ thuộc | Cỡ |
|----|------|-----------|------|-----------|-----|
| T001 | Tạo file cấu hình inpainting | 1. Bốn định danh method + input root theo research R1 (`classical_baseline` → `outputs/segmentation/default/adaptive/`; ba method DL → `deliverables/<run_id>/<method>/`)<br>2. Khối mask-processing dùng chung: dilation bật, `ellipse`, `[3,3]`, 1 vòng lặp<br>3. `inpaint.radius=3`, `inpaint.algorithms=["telea","ns"]`, `inpaint.output_format="png"`<br>4. `selection.n=5`, `selection.manual_artifact_flags=[]`, output root `outputs/inpainting/`<br>**Xong khi**: không path/tham số nào còn nằm trong code (FR-038) | `configs/inpainting.json` | — | M |
| T002 | [P] Xác nhận gitignore | 1. Kiểm `.gitignore` dòng 23 (`outputs/`) có phủ `outputs/inpainting/`<br>2. Kiểm `configs/inpainting.json` được track<br>3. Ghi kết quả vào commit message — **không sửa `.gitignore`** | `.gitignore` | — | S |
| T003 | [P] Fixture dùng chung | 1. Thêm factory `inpaint_fixture(method, image_id)` vào `conftest.py`<br>2. Ghi page image + prediction mask ở `ALIGNED_SHAPE` `(1170, 1654)` (`tests/conftest.py:35`) + sidecar JSON<br>3. Đường dẫn: `<root>/masks/<manga>/<stem>.png`, `<root>/metadata/<manga>/<stem>.json`<br>4. Tham số hoá để perturb được size / pixel value / rỗng / thiếu sidecar | `tests/conftest.py` | T001 | M |

### Phase 2 — Foundational (chặn mọi user story)

| Mã | Việc | Tác vụ con | File | Phụ thuộc | Cỡ |
|----|------|-----------|------|-----------|-----|
| T004 | [P] Test config loader — **viết trước, phải FAIL** | 1. `load_inpaint_config()` trả về 4 định danh + root đã resolve<br>2. Trả về mask-processing, inpainting config và `selection.n`<br>3. Thiếu key bắt buộc → `ConfigError`<br>4. Không path nào hard-code | `tests/unit/test_config_inpaint.py` | T001 | M |
| T005 | [P] Test runs — **viết trước, phải FAIL** | 1. Hai template addressing resolve chỉ từ `(run_id, method, algorithm, image_id)`, không tra index<br>2. Trùng run ID → từ chối<br>3. Yêu cầu overwrite tường minh → chấp nhận<br>4. Hai run ID cùng tồn tại, không đụng nhau | `tests/unit/test_runs.py` | T001 | M |
| T006 | Cài `load_inpaint_config()` | 1. Theo convention `load_config()`/`load_dl_config()` trong cùng file (dataclass, validate kiểu `_require_mapping`, `_resolve` theo repo root)<br>2. Expose `MaskProcessingConfig`, `InpaintingConfig` (data-model.md) và `selection.N` | `src/manga_text_seg/config.py` | T004 | M |
| T007 | Run-ID + addressing | 1. `outputs/inpainting/<run_id>/<method>/<algorithm>/<manga>/<page_id>.png` + sibling `.json`<br>2. Biến thể ablation: `outputs/inpainting/ablation/<run_id>/…`<br>3. `<method>` chỉ nhận 4 định danh, `<algorithm>` chỉ `telea`/`ns` — còn lại từ chối | `src/manga_text_seg/runs.py` | T005 | M |
| T008 | Scaffolding + luật không ghi đè | 1. Tạo cây run<br>2. Từ chối tái dùng run ID trừ khi có yêu cầu overwrite tường minh<br>3. Không bao giờ chạm thư mục của run khác | `src/manga_text_seg/runs.py` | T007 | M |
| T009 | Writer `SampleMetadata` | 1. Ghi đủ mọi field của `contracts/sample-metadata.schema.json`<br>2. Tôn trọng `additionalProperties: false`<br>3. Hai conditional `allOf`: có `error_message` khi `status: failed`; có `kernel_shape`/`kernel_size`/`iterations` khi dilation bật | `src/manga_text_seg/runs.py` | T007 | M |
| T010 | Accumulator `ErrorReport` + `errors.json` | 1. Đủ **9 category**: missing source image, missing prediction mask, wrong image ID / mapping mơ hồ, wrong size, empty mask, non-binary mask, corrupt image, inpainting failure, output write failure<br>2. Mỗi entry mang sample identifier, category, reason | `src/manga_text_seg/runs.py` | T007 | M |
| T011 | Writer `run.json` (`InpaintRun`) | 1. run ID, các method đã xử lý, mask-processing config, inpainting config<br>2. `page_list_identity` / `input_image_identity` **trích nguyên văn** từ `benchmark/` — không tính lại<br>3. Ghi lại method nguồn đã resolve cho `classical_baseline` và lý do (research R1) | `src/manga_text_seg/runs.py` | T007 | M |

**Checkpoint A1**: config + cây run tồn tại → user story bắt đầu được.

### Phase 3 — US1: Nhận và kiểm tra prediction mask của Spec 2 (P1, MVP)

**Independent Test**: đưa vào 5 mask — một map đúng, một sai page identity, một sai size, một
non-binary, một thiếu source image — và khẳng định đúng những cái phải pass mới pass, mọi lỗi được
ghi dưới category của nó, không input nào bị resize/re-threshold/rename, các mẫu hợp lệ vẫn chạy tiếp.

| Mã | Việc | Tác vụ con | File | Phụ thuộc | Cỡ |
|----|------|-----------|------|-----------|-----|
| T012 | [P] [US1] Test intake — **viết trước, phải FAIL** | 1. Map theo tên chính xác tới **đúng một** page manifest, `image_id = <manga>/<NNN>`<br>2. Zero-match = image-ID mismatch; multi-match = ambiguous mapping; **không** giải quyết bằng similarity<br>3. Mỗi check trong 7 check của `contracts/intake-validation.md` sinh đúng category đã ghi<br>4. Thiếu sidecar = `missing metadata`, không join từ chỗ khác | `tests/unit/test_intake.py` | T003 | L |
| T013 | [P] [US1] Test maskproc — **viết trước, phải FAIL** | 1. Chuẩn hoá về nền 0 / chữ 255; giá trị 128 bị từ chối là non-binary<br>2. Pass-through nguyên trạng ở 1654×1170<br>3. Crop góc trên-trái lossless từ 1656×1176; size khác → validation error<br>4. Ghi `pre_size`/`post_size`; dilation theo kernel shape/size/iterations từ config và ghi lại config hiệu dụng | `tests/unit/test_maskproc.py` | T003 | L |
| T014 | [US1] Test tích hợp từ chối | 1. Chạy 5 input của Independent Test qua đường intake<br>2. Khẳng định đúng tập pass, đúng category + reason của từng lỗi<br>3. Không input bị từ chối nào bị sửa trên đĩa<br>4. Các mẫu hợp lệ vẫn pass | `tests/integration/test_intake_rejections.py` | T012, T013 | M |
| T015 | [US1] Định vị input | 1. **Một** đường code phục vụ cả 4 định danh<br>2. Đọc `<root>/masks/<manga>/<stem>.png` và `<root>/metadata/<manga>/<stem>.json`, chỉ root lấy từ config<br>3. `notebooks/deliverables/` **không** là đường intake; `benchmark/availability.json` **không** được tra — cả hai không được import | `src/manga_text_seg/intake.py` | T011 | M |
| T016 | [US1] Mapping | 1. Resolve mask về đúng một page manifest bằng tên thư mục manga + stem zero-padded<br>2. `image_id = <manga>/<NNN>`<br>3. Zero-match và multi-match **đều là từ chối** (check 3) — không similarity, không đoán, không đánh số lại | `src/manga_text_seg/intake.py` | T015 | M |
| T017 | [US1] Các check khi nhận | 1. Source image tồn tại + decode được<br>2. Mask tồn tại + decode được<br>3. Identity nhất quán; size; tập giá trị nhị phân sau chuẩn hoá; rỗng; có sidecar<br>4. Mỗi cái raise rejection có category, **không fatal**<br>5. Mẫu bị từ chối sinh một entry error-report và **không sinh artefact nào** (FR-013) | `src/manga_text_seg/intake.py` | T016 | L |
| T018 | [US1] Chuẩn hoá mask | 1. Đọc giữ thông tin kênh<br>2. Chuẩn hoá về convention nhị phân<br>3. Từ chối mọi giá trị ngoài `{0, 255}` sau chuẩn hoá — **không ép, không re-threshold** | `src/manga_text_seg/maskproc.py` | T013 | M |
| T019 | [US1] Căn chỉnh mask | 1. Pass-through ở size aligned<br>2. Crop góc trên-trái lossless từ canvas GT padded 1656×1176<br>3. Size khác → validation error<br>4. Ghi `rule`, `pre_size`, `post_size` trên mọi mẫu được nhận | `src/manga_text_seg/maskproc.py` | T018 | M |
| T020 | [US1] Check rỗng + dilation | 1. Mask toàn 0 → category `empty mask`<br>2. Dilation lấy kernel shape/size/iterations từ config và ghi lại giá trị hiệu dụng (FR-018) | `src/manga_text_seg/maskproc.py` | T018 | M |
| T021 | [US1] Nối rejection vào `ErrorReport` | 1. Mẫu lỗi được ghi nhận<br>2. Caller tiếp tục với các mẫu hợp lệ còn lại | `src/manga_text_seg/runs.py` | T017, T010 | S |

**Checkpoint A2**: cổng intake chạy độc lập — mọi từ chối đều có tên, không gì bị sửa âm thầm.

### Phase 4 — US2: Inpaint bằng TELEA và NS (P1)

**Independent Test**: chạy pipeline lõi trên fixture nhỏ và khẳng định mỗi mẫu có đủ hai output ở
đúng size aligned, mask đã xử lý khác mask thô **chỉ** bởi dilation đã cấu hình, mọi tham số xuất
hiện trong metadata, và input + config giống nhau cho ra output byte-identical.

| Mã | Việc | Tác vụ con | File | Phụ thuộc | Cỡ |
|----|------|-----------|------|-----------|-----|
| T022 | [P] [US2] Contract test cây output | 1. Đủ **6 artefact FR-025** mỗi mẫu: ảnh gốc, bản copy mask thô, mask sau dilation, kết quả TELEA, kết quả NS, metadata JSON<br>2. Ở đúng địa chỉ `contracts/inpainted-output.md` quy định<br>3. Trang inpainted đúng 1654×1170 ở định dạng đã cấu hình<br>4. `run.json`, `errors.json`, `performance.json` nằm ở run root | `tests/contract/test_inpainted_output.py` | T011 | M |
| T023 | [P] [US2] Contract test schema metadata | 1. Mọi sidecar validate được với `contracts/sample-metadata.schema.json`<br>2. `additionalProperties: false` được tôn trọng, đủ 15 property bắt buộc<br>3. Hai conditional (status failed, dilation bật) thoả | `tests/contract/test_sample_metadata_schema.py` | T009 | M |
| T024 | [P] [US2] Unit test inpaint | 1. Cả hai thuật toán chạy trên fixture, cả hai output đúng size aligned<br>2. `--algorithm telea` và `--algorithm ns` mỗi cái chỉ sinh output của mình<br>3. Radius + output format lấy từ config<br>4. Timing ghi riêng theo từng thuật toán<br>5. `cv2.inpaint` nhận mask **sau** dilation; bản copy thô byte-identical với nguồn | `tests/unit/test_inpaint.py` | T003 | L |
| T025 | [US2] Test tích hợp pipeline | 1. Pipeline đầy đủ theo mẫu trên fixture<br>2. Cả hai output đúng size aligned<br>3. `001.mask.png` khác `001.raw.png` **đúng** bởi dilation đã cấu hình và không gì khác<br>4. Metadata đầy đủ<br>5. Hai run liên tiếp byte-identical, trừ `processing_time_seconds` và run ID | `tests/integration/test_inpaint_pipeline.py` | T022, T023, T024 | L |
| T026 | [P] [US2] Wrapper inpaint | 1. `INPAINT_TELEA` và `INPAINT_NS` trên mask sau dilation<br>2. Radius + output format lấy từ `InpaintingConfig`<br>3. Trong benchmark chính, **cả hai** thuật toán luôn chạy trên cùng tập input | `src/manga_text_seg/inpaint.py` | T024 | M |
| T027 | [P] [US2] Writer artefact | 1. Bản copy mask thô byte-identical (`<page_id>.raw.png`, **copy không move**, FR-006)<br>2. Mask sau dilation (`<page_id>.mask.png` — "processed mask" của Spec 5)<br>3. Bản copy ảnh gốc<br>4. Trang inpainted ở định dạng lossless đã cấu hình (FR-028) | `src/manga_text_seg/runs.py` | T022 | M |
| T028 | [US2] `process_sample()` | 1. Một mẫu hợp lệ đi qua: normalize → align → dilate → cả hai thuật toán<br>2. Sinh đủ 6 artefact + sidecar `SampleMetadata`<br>3. `processing_time_seconds` ghi theo từng thuật toán (FR-022) và là field phi tất định **duy nhất** | `src/manga_text_seg/inpaint.py` | T026, T027 | L |
| T029 | [US2] Làm cho tất định | 1. Không timestamp wall-clock trong bất kỳ metadata per-sample nào — run ID là field định thời duy nhất<br>2. Mọi vòng lặp qua page / method identity / algorithm theo thứ tự sắp xếp toàn phần, **không** dùng `set` hay thứ tự `dict` ngẫu nhiên | `src/manga_text_seg/inpaint.py`, `src/manga_text_seg/runs.py` | T028 | M |
| T030 | [US2] Ghi timing per-sample | 1. Bắt thời gian mỗi mẫu<br>2. Ghi **riêng theo từng thuật toán** để aggregate method × algorithm của FR-022 có đầu vào | `src/manga_text_seg/inpaint.py` | T028 | S |
| T031 | [US2] Subcommand `inpaint` | 1. Theo convention subparser hiện có (`_common_parser`, handler `cmd_*`, `src/manga_text_seg/cli.py:333`)<br>2. Cờ: `--config`, `--run`, `--method`, `--image-id`, `--algorithm {telea,ns}`, `--overwrite`<br>3. Phủ 3 kịch bản: xử lý một prediction mask; xử lý một method; chạy TELEA-only / NS-only | `src/manga_text_seg/cli.py` | T028 | M |

**Checkpoint A3 (mốc bàn giao)**: US1 + US2 chạy độc lập — một mask hợp lệ thành hai trang inpainted
với metadata đầy đủ. **B được mở `runs.py` và `inpaint.py` từ đây.**

---

## 3. BẢNG CÔNG VIỆC — NGƯỜI B

> Spec 003, Phase 5–9. B chỉ bắt đầu sau **Checkpoint A3**. Trước đó xem §5 để nhận việc san tải.

### Phase 5 — US3: Batch, cô lập lỗi, tách run (P2)

**Independent Test**: chạy batch có cố ý một lỗi thuộc mỗi category; khẳng định run hoàn tất trên mọi
mẫu còn lại, error report liệt kê đủ mọi lỗi được gài theo category + sample identifier + reason,
run thứ hai với run ID khác cùng tồn tại, và tái dùng run ID bị từ chối trừ khi có overwrite.

| Mã | Việc | Tác vụ con | File | Phụ thuộc | Cỡ |
|----|------|-----------|------|-----------|-----|
| T032 | [P] [US3] Test cô lập batch | 1. Gài đủ **9 category** — bảng perturb trong quickstart.md §3<br>2. Batch exit 0<br>3. Mọi mẫu khác hoàn tất<br>4. `errors.json` ghi mọi lỗi được gài với category + sample identifier + reason<br>5. **Không** lỗi nào không được ghi | `tests/integration/test_batch_isolation.py` | T031 | L |
| T033 | [P] [US3] Test tách run | 1. Hai run ID cùng còn trên đĩa, byte-identical với trạng thái hoàn tất của chúng<br>2. Trùng run ID bị từ chối khi không có `--overwrite`, chạy được khi có<br>3. Run bị từ chối **không ghi gì** | `tests/integration/test_run_separation.py` | T031 | M |
| T034 | [P] [US3] Test performance summary | 1. `performance.json` nhóm theo method × algorithm<br>2. Mỗi nhóm báo: thời gian xử lý trung bình mỗi ảnh, số mẫu, số lỗi, metadata runtime/device | `tests/unit/test_performance_summary.py` | T031 | S |
| T035 | [US3] Điều phối batch | 1. Duyệt page list của manifest theo thứ tự `(manga, stem)` cho một method, hoặc cả 4 định danh<br>2. Cô lập lỗi từng mẫu để batch **luôn** chạy tới hết | `src/manga_text_seg/inpaint.py` | T032, T033 | M |
| T036 | [US3] Đường overwrite | 1. Mặc định từ chối run ID trùng<br>2. `--overwrite` là **cách duy nhất** vượt qua<br>3. Bị từ chối thì run trước còn nguyên | `src/manga_text_seg/runs.py`, `src/manga_text_seg/cli.py` | T033 | M |
| T037 | [US3] Performance summary | 1. Aggregate timing per-sample đã ghi<br>2. Nhóm theo segmentation method × inpainting algorithm<br>3. Kèm mean time/image, sample count, failure count, runtime/device metadata | `src/manga_text_seg/runs.py` | T034 | M |
| T038 | [US3] Ghi run counts | 1. Ghi vào `run.json`: số page đã thử / thành công / thất bại<br>2. Tách theo method × algorithm (quickstart.md §2) | `src/manga_text_seg/runs.py` | T035 | S |
| T039 | [US3] Mở rộng subcommand | 1. Thêm dạng all-methods ("processing all prediction masks")<br>2. Đảm bảo run có lỗi vẫn exit 0 | `src/manga_text_seg/cli.py` | T035 | S |

**Checkpoint B1**: US1+US2+US3 chạy độc lập — một method đầy đủ hoàn tất qua lỗi thật, output tồn tại.

### Phase 6 — US4: Board so sánh (P3)

**Independent Test**: sinh board cho mẫu được chọn bởi từng luật và khẳng định mỗi board có đủ 5
panel được dán nhãn, đúng image ID, method và nhãn cấu hình.

| Mã | Việc | Tác vụ con | File | Phụ thuộc | Cỡ |
|----|------|-----------|------|-----------|-----|
| T040 | [P] [US4] Test luật chọn mẫu | 1. Mỗi luật trong **5 luật** chọn tối đa N=5 mẫu **theo từng segmentation method**<br>2. Luật có ít hơn N ứng viên → ghi shortfall, **không** độn thêm<br>3. `manual_artifact_flags` chọn tối đa N từ danh sách đã cấu hình, ghi shortfall chứ không bịa flag<br>4. Metric của `classical_baseline` lấy từ `metrics.csv` của Spec 1 cho method mà nó resolve tới (research R1) | `tests/unit/test_selection.py` | T031 | M |
| T041 | [P] [US4] Test board | 1. Board có **đúng 5 panel**: ảnh gốc, mask dự đoán thô, mask sau dilation, kết quả TELEA, kết quả NS<br>2. Mỗi panel dán nhãn segmentation method, inpainting algorithm, mask-processing config và `image_id`<br>3. **Không** panel nào là mask ground truth<br>4. **Không** panel nào là overlay prediction-vs-GT | `tests/unit/test_boards.py` | T031 | M |
| T042 | [US4] Cài 5 luật chọn | 1. IoU/F1 cao nhất; IoU/F1 thấp nhất; nhiều false positive nhất; nhiều false negative nhất; mẫu được gắn cờ thủ công<br>2. N mỗi luật mỗi method, ghi shortfall<br>3. Kết quả ghi ra `selection.json` | `src/manga_text_seg/selection.py` | T040 | L |
| T043 | [US4] Render board | 1. Ghi tại `boards/<method>/<rule>/<manga>_<stem>.png`<br>2. **Chỉ** đọc artefact của chính feature này<br>3. **Không** tái dùng màu GT-overlay của `src/manga_text_seg/visualize.py` — không ground truth nào xuất hiện trong board | `src/manga_text_seg/boards.py` | T041 | M |
| T044 | [US4] Subcommand `boards` | 1. Cờ `--config`, `--run`, `--method`<br>2. Theo convention subparser hiện có | `src/manga_text_seg/cli.py` | T042, T043 | S |

**Checkpoint B2**: US4 giao board review được, không phụ thuộc US5/US6.

### Phase 7 — US5: Báo cáo định tính (P3)

**Independent Test**: sinh scaffold trên một tập mẫu và khẳng định mỗi entry mang segmentation method,
inpainting algorithm, mask-processing config, image ID, một field rating **rỗng** cho mỗi tiêu chí
trong 7 tiêu chí, và một field comment rỗng — không rating nào được tự điền.

| Mã | Việc | Tác vụ con | File | Phụ thuộc | Cỡ |
|----|------|-----------|------|-----------|-----|
| T045 | [P] [US5] Test qualitative | 1. Mỗi entry dán nhãn segmentation method, inpainting algorithm, mask-processing config, `image_id`<br>2. Đủ **7 tiêu chí**: mức độ xoá sạch chữ; lượng chữ bị bỏ sót; lượng nền bị xoá quá; độ tự nhiên vùng phục hồi; artefact/nhiễu; hiệu ứng halo/viền; hư hại linework/texture/viền panel — mỗi cái trên một thang ordinal đã ghi<br>3. Mọi field rating và observation **rỗng**<br>4. **Không** PSNR, **không** SSIM, **không** ground-truth nền sạch tổng hợp, **không** xếp hạng một điểm | `tests/unit/test_qualitative.py` | T031 | M |
| T046 | [US5] Sinh scaffold | 1. Sinh `qualitative.md` và `qualitative.json` với nội dung giống nhau và **độ rỗng giống nhau**<br>2. Hệ thống **không được** tự cho điểm bất kỳ tiêu chí nào | `src/manga_text_seg/qualitative.py` | T045 | M |
| T047 | [US5] Subcommand `qualitative` | 1. Cờ `--config`, `--run`<br>2. Từ chối ghi đè scaffold mà reviewer đã điền (FR-027, quickstart.md §6) | `src/manga_text_seg/cli.py` | T046 | S |

**Checkpoint B3**: US5 giao deliverable đánh giá, vẫn kiểm thử độc lập được.

### Phase 8 — US6: Ablation trên dilation và radius (P4)

**Independent Test**: chạy ablation với vài cấu hình dilation/radius và khẳng định output mỗi cấu hình
nằm **ngoài** cây benchmark chính, mỗi cấu hình ghi lại config của nó, và output benchmark chính
không bị chạm.

| Mã | Việc | Tác vụ con | File | Phụ thuộc | Cỡ |
|----|------|-----------|------|-----------|-----|
| T048 | [P] [US6] Test ablation | 1. Output mỗi cấu hình nằm dưới `outputs/inpainting/ablation/<run_id>/`, **không bao giờ** vào cây chính<br>2. `run.json` của ablation ghi `varied`, `baseline_config` được so sánh, và `ablation: true`<br>3. Thư mục run của benchmark chính không bị chạm | `tests/unit/test_ablation.py` | T031 | M |
| T049 | [US6] Sweep ablation | 1. Quét qua danh sách cấu hình dilation/radius (FR-024)<br>2. Ghi vào namespace tách biệt qua addressing ablation của T007<br>3. Ghi lại từng cấu hình | `src/manga_text_seg/ablation.py` | T048 | M |
| T050 | [US6] Subcommand `ablate` | 1. Cờ `--config`, `--run`, `--vary` (vd `dilation.kernel_size`), `--values` (vd `3,5,7`) | `src/manga_text_seg/cli.py` | T049 | S |
| T051 | [US6] Loại ablation khỏi so sánh chính | 1. Báo cáo chính **không bao giờ** đọc `outputs/inpainting/ablation/`<br>2. Áp ở cả `runs.py` và `selection.py` | `src/manga_text_seg/runs.py`, `src/manga_text_seg/selection.py` | T049 | S |

**Checkpoint B4**: cả 6 user story của Spec 003 đều chạy độc lập.

### Phase 9 — Polish & cross-cutting

| Mã | Việc | Tác vụ con | File | Phụ thuộc | Cỡ |
|----|------|-----------|------|-----------|-----|
| T052 | [P] Test bất biến cây nguồn | 1. Checksum hoặc mtime bất biến trên `data/raw/…`, `data/groundtruth/…`, manifest Spec 1, cây mask Spec 2 sau một run đầy đủ<br>2. Audit truy cập chứng minh **không** file nào dưới `data/no-need-to-read/` bị mở (FR-007) | `tests/integration/test_source_tree_invariance.py` | T051 | M |
| T053 | [P] Test đồng nhất cấu hình | 1. Qua cả 4 định danh trong một run benchmark chính<br>2. Mask-processing config và inpaint radius ghi trong mọi mẫu **giống hệt nhau** (SC-003) | `tests/integration/test_config_uniformity.py` | T051 | S |
| T054 | [P] Test không GT trong artefact | 1. Grep một thư mục run đã hoàn tất và boards của nó tìm mọi đường dẫn ground truth — kỳ vọng **0 hit**<br>2. Khẳng định không board nào chứa overlay prediction-vs-GT | `tests/integration/test_no_ground_truth_in_artifacts.py` | T044 | S |
| T055 | [P] Test không vendor model | 1. Quét cây được track<br>2. Khẳng định **0** file source model segmentation và **0** file weight ở mọi kích cỡ, mọi định dạng | `tests/unit/test_no_vendored_models.py` | — | S |
| T056 | [P] Test tiêu thụ hạ nguồn | 1. Consumer chỉ biết `(run_id, method, image_id)` định vị và hiểu được ảnh inpainted + metadata của page đó<br>2. Không tra index, không đọc manifest | `tests/integration/test_downstream_consumption.py` | T051 | S |
| T057 | [P] Test không tính lại metric | 1. Không metric segmentation nào bị tính lại từ output inpainting<br>2. Không PSNR, SSIM, ground truth tổng hợp hay xếp hạng một điểm trong bất kỳ artefact chính nào | `tests/integration/test_no_metric_recomputation.py` | T051 | S |
| T058 | Chạy quickstart §0–§8 | 1. Chạy mọi kịch bản trong `quickstart.md` §0–§8 ở quy mô fixture<br>2. Sửa mọi lệch giữa lệnh đã ghi và CLI đã cài (FR-036) | `specs/003-text-removal-inpainting/quickstart.md` | T051 | M |
| T059 | [P] Cập nhật README | 1. Thêm các subcommand mới<br>2. Ghi contract `configs/inpainting.json`<br>3. Ghi lý do gitignore `outputs/inpainting/` (research R6) | `README.md` | T051 | S |
| T060 | [P] Suite đầy đủ CPU-only | 1. Chạy `pytest -o addopts='' -q`<br>2. Không checkpoint, không model runtime nào hiện diện (FR-037, FR-039, SC-012) | — | T051–T059 | S |

---

## 4. BẢNG CÔNG VIỆC — NGƯỜI C

> Spec 004, toàn bộ T001–T082. C là người **duy nhất** sửa `translation/runs.py`.

### Phase 1 — Setup

| Mã | Việc | Tác vụ con | File | Phụ thuộc | Cỡ |
|----|------|-----------|------|-----------|-----|
| T001 | File cấu hình dịch | 1. Đường dẫn manifest Spec 1; segmentation method đã chọn; nguồn inpainting (`run_id` + `algorithm`); `target_language`<br>2. Khối region-extraction: min/max area, merge distance, crop padding, reading order<br>3. Khối OCR: engine, preprocessing tuỳ chọn<br>4. Khối provider: name, model, version, base URL, `secret_env_var`, timeout, retries, backoff, `failure_policy`, cache (kể cả `bypass`)<br>5. Khối rendering: font path, fallback font path, size, min size, colour, outline, background, wrap, line spacing, alignment, `text_direction`, `expansion_allowance`<br>6. Output roots<br>**Xong khi**: không path/tham số nào nằm trong code (FR-061) | `configs/translation.json` | — | L |
| T002 | [P] Font đi kèm | 1. Copy `DejaVuSans.ttf` từ bundle mà matplotlib đã có sẵn<br>2. Kèm `DejaVuSans.LICENSE.txt` chứa văn bản giấy phép<br>3. Đường render mặc định resolve font **theo đường dẫn file**, không tra font hệ thống — kết quả không phụ thuộc máy | `assets/fonts/` | — | S |
| T003 | [P] Skeleton package | 1. Tạo `__init__.py` cho subpackage mới<br>2. Docstring nêu tên 7 module và dải FR mỗi module phụ trách<br>3. **Chưa** import gì — module land ở phase riêng | `src/manga_text_seg/translation/__init__.py` | — | S |
| T004 | [P] Xác nhận gitignore | 1. Kiểm `outputs/translation/` và `cache/translation/` được `.gitignore` phủ (đã có `outputs/` và `cache/`)<br>2. Kiểm `configs/translation.json` và `assets/fonts/` được track<br>3. Ghi kết quả — **không sửa `.gitignore`** | `.gitignore` | — | S |
| T005 | [P] Khai báo dependency | 1. Thêm `Pillow>=10.0.0` vào `[project.dependencies]`<br>2. Thêm group `[project.optional-dependencies]` mới: `ocr = ["manga-ocr"]`<br>3. `manga-ocr` giữ **tuỳ chọn** để bản cài mặc định và suite mặc định không cần OCR engine (FR-060) | `pyproject.toml` | — | S |

### Phase 2 — Foundational (chặn mọi user story)

| Mã | Việc | Tác vụ con | File | Phụ thuộc | Cỡ |
|----|------|-----------|------|-----------|-----|
| T006 | [P] Test config loader — **viết trước, phải FAIL** | 1. `load_translation_config()` trả về khối extraction, OCR, provider, cache, rendering với path đã resolve<br>2. Thiếu key bắt buộc → `ConfigError`<br>3. Không path nào hard-code | `tests/unit/test_config_translation.py` | T001 | M |
| T007 | [P] Fixture dùng chung | 1. Thêm `translation_bundle(method, image_id)` vào `conftest.py`<br>2. Dưới tmp root: ghi ảnh gốc ở `ALIGNED_SHAPE` `(1170, 1654)` (`tests/conftest.py:35`), prediction mask + sidecar JSON theo layout hand-off của Spec 2, trang inpainted + sidecar Spec 3, và entry manifest nối chúng<br>3. Tham số hoá để perturb sự tồn tại / size / pixel value / độ rỗng của **từng** artefact<br>**Đây là thứ cho phép cả 13 vùng test của FR-060 chạy không cần dataset** | `tests/conftest.py` | T001 | L |
| T008 | Cài `load_translation_config()` | 1. Theo convention `load_config()`/`load_dl_config()` (dataclass, validate kiểu `_require_mapping`, `_resolve` theo repo root)<br>2. Expose `TranslationConfig` và các sub-config (`RegionExtractionConfig`, `RenderingConfig` — data-model.md) | `src/manga_text_seg/config.py` | T006 | M |
| T009 | Run-ID + addressing | 1. Đường dẫn `outputs/translation/<run_id>/<segmentation_method>/<manga>/<NNN>/{crops/,mask.png,ocr.json,translations.json,rendered.png,metadata.json}`<br>2. `run.json` + `errors.json` ở run root<br>3. `<segmentation_method>` chỉ nhận 4 định danh FR-009 — còn lại từ chối<br>4. Consumer chỉ biết (run ID, method, `image_id`) resolve được, **không** cần index file (SC-010) | `src/manga_text_seg/translation/runs.py` | T008 | M |
| T010 | Scaffolding + không ghi đè | 1. Tạo cây run<br>2. Từ chối tái dùng run ID trừ khi có overwrite tường minh<br>3. Không chạm thư mục run khác; bị từ chối thì cây run trước còn nguyên | `src/manga_text_seg/translation/runs.py` | T009 | M |
| T011 | `ErrorReport` + `errors.json` | 1. Đủ **11 category** trong `counts` và `categories` **kể cả khi = 0**: `missing_original_image`, `missing_prediction_mask`, `missing_inpainted_image`, `empty_mask`, `no_text_region_detected`, `ocr_failure`, `translation_failure`, `font_missing`, `text_overflow`, `api_timeout_or_rate_limit`, `output_write_failure`<br>2. Mỗi entry mang `image_id`, `region_id` (`null` ở mức page), `stage` (`intake`/`regions`/`ocr`/`translate`/`render`/`write`) và reason cụ thể | `src/manga_text_seg/translation/runs.py` | T009 | M |
| T012 | Writer `run.json` (`TranslationRun`) | 1. run ID; method cố định; nguồn inpainting (run ID + algorithm); target language; OCR engine + version; provider/model/version; font + rendering config<br>2. Input image IDs; **toàn bộ config đã resolve** lưu cùng run để tái áp dụng được (FR-059)<br>3. Độ tất định provider ghi như **thuộc tính của provider**, không phải bảo đảm (FR-058)<br>4. `page_statuses`<br>5. `page_list_identity`/`input_image_identity` **trích nguyên văn** từ `benchmark/`, không tính lại | `src/manga_text_seg/translation/runs.py` | T009 | L |

**Checkpoint C1**: config, fixture, cây run và error report tồn tại → user story bắt đầu được.

### Phase 3 — US1: Nhận và kiểm tra input từ Spec 1–3 (P1, MVP)

**Independent Test**: dựng các bundle gồm một hợp lệ hoàn toàn, một thiếu ảnh gốc, một thiếu
prediction mask, một thiếu sidecar, một thiếu ảnh inpainted, và một có size ảnh inpainted lệch —
khẳng định đúng những cái phải pass mới pass, mọi lỗi ghi dưới category của nó, không gì bị thay thế
hay sửa âm thầm.

| Mã | Việc | Tác vụ con | File | Phụ thuộc | Cỡ |
|----|------|-----------|------|-----------|-----|
| T013 | [P] [US1] Contract test addressing | 1. Cho (run ID, method, `image_id`), bundle resolve **không** cần index file và **không** dựng lại manifest<br>2. Đường dẫn thư mục chính là `image_id` dùng nguyên văn, không phải biến đổi của nó | `tests/contract/translation/test_intake_addressing.py` | T007 | M |
| T014 | [P] [US1] Test tích hợp từ chối | 1. Sáu bundle của Independent Test<br>2. Đúng tập pass<br>3. Mỗi từ chối rơi đúng category với reason cụ thể<br>4. Các page còn lại chạy tiếp<br>5. Không gì bị thay thế hay sửa | `tests/integration/test_intake_rejections.py` | T007 | M |
| T015 | [US1] Lắp bundle | 1. Định vị ảnh gốc<br>2. Định vị prediction mask của method đã chọn **và sidecar JSON của nó**<br>3. Định vị `<NNN>.png` và `<NNN>.json` của run inpainting đã cấu hình<br>4. Nối **chỉ** bằng `image_id` (`<manga>/<NNN>`) — không tự ghép path, không manifest mới, không subset benchmark | `src/manga_text_seg/translation/intake.py` | T012 | M |
| T016 | [US1] Validate bundle | 1. Chuẩn hoá mask về convention nhị phân Spec 1 (1 kênh, uint8, nền 0, chữ 255)<br>2. Kiểm size ảnh inpainted **bằng** size trang aligned 1654×1170<br>3. Lệch size → ghi là intake failure; ảnh **không bao giờ bị resize âm thầm**<br>4. **Tái dùng** `normalize.py` và `imaging.py`, không viết lại (research R1) | `src/manga_text_seg/translation/intake.py` | T015 | M |
| T017 | [US1] Nối từ chối vào accumulator | 1. `missing_original_image`, `missing_prediction_mask`, `missing_inpainted_image`, `empty_mask` và reason lệch size<br>2. Mỗi cái kèm page identifier + reason cụ thể<br>3. Mỗi cái để batch chạy tiếp | `src/manga_text_seg/translation/intake.py` | T011, T016 | M |
| T018 | [US1] Bảo đảm upstream read-only | 1. Mọi đường dẫn upstream chỉ mở để đọc<br>2. `mask.png` trong cây run là **bản copy** của prediction mask, byte-identical với mask đã admit<br>3. Không gì dưới `data/raw/`, `data/groundtruth/`, manifest Spec 1, mask Spec 2 hay output Spec 3 bị sửa/move/rename/xoá | `src/manga_text_seg/translation/intake.py` | T015 | M |
| T019 | [US1] Resolve method + nguồn, có từ chối | 1. Resolve method mặc định bằng cách đọc **bản ghi** lựa chọn của Spec 2 — **không** import `availability.py`<br>2. Không có lựa chọn nào và config cũng không có → từ chối với lỗi rõ<br>3. Nguồn inpainting mơ hồ hoặc vắng → từ chối — **không bao giờ chọn bừa**<br>4. **Ground truth không bao giờ được tra**: prediction mask là mask duy nhất feature này mở, và không chế độ GT-assisted nào là mặc định (FR-008, US1 scenario 4) | `src/manga_text_seg/translation/intake.py` | T015 | M |

**Checkpoint C2 (MVP)**: cổng intake đứng một mình — `test_intake_rejections.py` pass, mọi từ chối có tên.

### Phase 4 — US2: Trích vùng chữ ổn định từ prediction mask (P1)

**Independent Test**: chạy extraction trên mask tổng hợp (một khối, nhiều khối tách rời, hai component
đủ gần để merge, một component quá lớn, một mask rỗng) và khẳng định vùng sinh ra, hành vi merge, thứ
tự và ID khớp luật đã cấu hình, ID giống nhau qua các lần chạy, và edge case được ghi lại.

| Mã | Việc | Tác vụ con | File | Phụ thuộc | Cỡ |
|----|------|-----------|------|-----------|-----|
| T020 | [P] [US2] Unit test regions | 1. Một khối; nhiều khối tách rời; một cặp merge; một component quá lớn — trên fixture `aligned_mask`<br>2. Mỗi vùng mang **cả** `bbox` và `polygon`<br>3. Crop có padding chạm viền trang → clip vào biên trang, ghi `clipped: true`, **không** cắt ký tự nào bên trong vùng | `tests/unit/translation/test_regions.py` | T007 | M |
| T021 | [P] [US2] Unit test region ID | 1. `region_id` là vị trí theo thứ tự đọc, 1-based, dưới config đang hoạt động<br>2. **Giống hệt** qua hai lần chạy trên cùng mask<br>3. `merge_distance: 0` để component tách rời; giá trị khác 0 merge chúng và `merged_from` ghi lại số lượng đã gộp<br>4. Orientation được ghi theo từng vùng | `tests/unit/translation/test_region_ids.py` | T007 | M |
| T022 | [US2] Trích component + lọc diện tích | 1. `connectedComponentsWithStats` trên mask nhị phân với `min_area`/`max_area` từ `configs/translation.json`<br>2. Component ngoài biên bị loại và **việc loại bị ghi lại**<br>3. Component quá lớn **không bao giờ bị tự động tách** (FR-016, edge case) | `src/manga_text_seg/translation/regions.py` | T020 | M |
| T023 | [US2] Merge cấu hình được | 1. Một lượt union-find gộp các component gần nhau hơn `merge_distance` thành một vùng (cùng bubble / cùng dòng chữ)<br>2. Ghi lại số component đã gộp vào<br>3. Giá trị `0` tắt merge<br>4. Merge distance đến từ config, **không** là literal | `src/manga_text_seg/translation/regions.py` | T021 | M |
| T024 | [US2] Sinh bbox + polygon | 1. `findContours` + `approxPolyDP` trên mỗi vùng đã merge<br>2. Lưu **cả** bounding box axis-aligned và polygon của vùng chữ trong không gian trang aligned | `src/manga_text_seg/translation/regions.py` | T022 | M |
| T025 | [US2] Padding crop + clip viền | 1. Pad theo lượng đã cấu hình để không ký tự nào của vùng bị cắt ở biên crop<br>2. Crop sau pad vượt biên trang → clip vào trang và ghi `clipped: true` | `src/manga_text_seg/translation/regions.py` | T022 | M |
| T026 | [US2] Thứ tự đọc + `region_id` ổn định | 1. Thứ tự toàn phần: tâm ngang giảm dần → tâm dọc tăng dần → diện tích bbox giảm dần → top-left tăng dần<br>2. `region_id` suy ra từ vị trí thứ tự đọc, nên cùng mask + cùng config luôn cho cùng ID<br>3. **Chỉ** dùng sort key tường minh; không dựa vào thứ tự lặp của dict hay set (research R8) | `src/manga_text_seg/translation/regions.py` | T024 | M |
| T027 | [US2] Phát hiện orientation | 1. Ngang hoặc dọc<br>2. Xác định từ hình học của mask và ghi lại theo từng vùng | `src/manga_text_seg/translation/regions.py` | T024 | S |
| T028 | [US2] Kết cục zero-region | 1. Page không ra vùng nào — mask rỗng, hoặc mọi component bị lọc — được ghi với 0 vùng và status tương ứng<br>2. **Không** phải run failure (FR-021) | `src/manga_text_seg/translation/regions.py` | T022 | S |
| T029 | [US2] Nối 2 kết cục page-level | 1. `empty_mask` khi mask không có pixel chữ<br>2. `no_text_region_detected` khi có pixel chữ nhưng không vùng nào qua được bộ lọc FR-016<br>3. Cả hai là **kết cục**, không phải lỗi; không cái nào làm run fail | `src/manga_text_seg/translation/regions.py` | T011, T028 | S |

**Checkpoint C3**: extraction tất định và kiểm thử độc lập — `region_id` ổn định qua các run.

### Phase 5 — US3: OCR chữ Nhật theo vùng (P1)

**Independent Test**: chạy adapter OCR trên bộ crop fixture với recognizer stub trả output đã biết và
một lỗi được gài; khẳng định record mỗi vùng mang crop path, `region_id`, text, status, timing; lỗi
được cô lập trong vùng của nó; page hoàn tất.

| Mã | Việc | Tác vụ con | File | Phụ thuộc | Cỡ |
|----|------|-----------|------|-----------|-----|
| T030 | [P] [US3] Unit test adapter OCR | 1. Adapter OCR với recognizer fixture trả output đã biết + một lỗi được gài<br>2. Record mỗi vùng mang `region_id`, `crop_path`, `text`, `confidence`, `processing_time_seconds`, `status`<br>3. Lỗi cô lập trong vùng của nó; page hoàn tất<br>4. `confidence` là `null` khi engine không báo — **không bao giờ bịa `0.0`**<br>5. Chạy được khi **không** cài OCR engine (FR-060) | `tests/unit/translation/test_ocr_adapter.py` | T007 | M |
| T031 | [US3] Trích crop | 1. Crop mỗi vùng lấy từ **ảnh gốc trước inpaint**, không bao giờ từ trang đã inpaint<br>2. Ảnh crop lưu tại `crops/<region_id>.png` (FR-046) | `src/manga_text_seg/translation/ocr.py` | T030 | S |
| T032 | [US3] Adapter OCR | 1. `manga-ocr` là engine ưu tiên, đặt sau một adapter để có thể thay bằng công cụ tương đương<br>2. Import **lazy và nằm trong adapter**, để module import được và suite chạy được khi extra tuỳ chọn vắng mặt<br>3. Ghi danh tính + version của engine cho mọi run (FR-056) | `src/manga_text_seg/translation/ocr.py` | T031 | M |
| T033 | [US3] Record OCR per-vùng | 1. `region_id`, `crop_path`, `text`, `confidence` (hoặc `null`), `processing_time_seconds`, `status`, và `error` khi lỗi<br>2. Ghi ra `ocr.json`, một entry mỗi vùng | `src/manga_text_seg/translation/ocr.py` | T032 | S |
| T034 | [US3] Cô lập lỗi mức vùng | 1. Lỗi hoặc kết quả rỗng của một vùng được ghi lại<br>2. **Mọi vùng khác trên page vẫn được xử lý** | `src/manga_text_seg/translation/ocr.py` | T033 | S |
| T035 | [US3] Tiêu thụ bản sửa tay | 1. `ocr.json` bị sửa tay được stage sau tôn trọng<br>2. Vùng đó được gắn cờ `manually_edited`, để bản sửa sống tới bước dịch | `src/manga_text_seg/translation/ocr.py` | T033 | M |
| T036 | [US3] Vô hiệu hoá bản sửa cũ | 1. Re-extraction làm đổi tập vùng → vô hiệu hoá các record đã sửa không còn khớp<br>2. **Báo cáo** việc vô hiệu hoá đó — không bao giờ revert âm thầm | `src/manga_text_seg/translation/ocr.py` | T035 | M |
| T037 | [US3] Preprocessing crop tuỳ chọn | 1. Bước preprocessing cấu hình được<br>2. Khi bật, được ghi vào run để input OCR tái lập được | `src/manga_text_seg/translation/ocr.py` | T032 | S |
| T038 | [US3] Đường thiếu engine | 1. Engine vắng → adapter ghi `ocr_failure` nêu tên engine thiếu<br>2. **Không bao giờ** thay text rỗng cho một lỗi, không bao giờ bịa kết quả (FR-055) | `src/manga_text_seg/translation/ocr.py` | T032 | S |

**Checkpoint C4**: OCR chạy trên fixture khi không cài engine; lỗi một vùng để các vùng khác nguyên vẹn.

### Phase 6 — US4: Dịch sang ngôn ngữ đích (P2)

**Independent Test**: chạy dịch trên mock provider (thành công, timeout, rate-limit, lỗi cứng) và
khẳng định record mapping đầy đủ, retry + back-off theo đúng policy đã cấu hình, failure policy được
tôn trọng, và **không** secret nào xuất hiện trong bất kỳ record hay log nào.

| Mã | Việc | Tác vụ con | File | Phụ thuộc | Cỡ |
|----|------|-----------|------|-----------|-----|
| T039 | [P] [US4] Unit test adapter dịch | 1. Adapter dịch trên mock provider: thành công, timeout, rate-limit, lỗi cứng<br>2. Mapping **đầy đủ kể cả vùng thất bại**<br>3. Retry + back-off theo policy đã cấu hình<br>4. `api_timeout_or_rate_limit` được ghi kèm lần retry tiếp theo<br>5. Mỗi nhánh `failure_policy` của lỗi được áp và ghi lại | `tests/unit/translation/test_translate_adapter.py` | T007 | L |
| T040 | [P] [US4] Unit test cache dịch | 1. Run thứ hai y hệt trên cùng text phục vụ mọi vùng từ cache với **0 provider call**<br>2. `cache_status` là `hit`/`miss` theo từng vùng<br>3. `bypass: true` ép gọi mới **mà không xoá gì** trong `cache/translation/` (research R10) | `tests/unit/translation/test_translation_cache.py` | T007 | M |
| T041 | [P] [US4] Unit test không rò secret | 1. Đặt một giá trị key sentinel trong môi trường<br>2. Khẳng định `provider`/`model`/`version` xuất hiện trong `run.json`, mọi entry `translations.json` và `metadata.json`<br>3. Khẳng định **giá trị** key không xuất hiện ở đâu — không trong record, log, exception message hay chuỗi `reason`. **Tên** key thì được phép xuất hiện<br>4. Thiếu biến → từ chối run **lúc khởi động**, trước khi thử bất kỳ vùng nào | `tests/unit/translation/test_no_secret_leakage.py` | T007 | M |
| T042 | [US4] Adapter provider | 1. Adapter pluggable, có một bản fixture và một bản thật<br>2. Dùng `urllib.request` của stdlib cho HTTP — **không** thêm dependency HTTP mới<br>3. Adapter tự khai báo độ tất định của nó (FR-058) | `src/manga_text_seg/translation/translate.py` | T039 | M |
| T043 | [US4] Request per-vùng | 1. **Một request provider cho mỗi vùng chữ**, không gọi gộp, để timeout một vùng không ảnh hưởng vùng khác<br>2. Mapping vùng → tiếng Nhật → bản dịch ghi ra `translations.json`, **kể cả vùng dịch thất bại** | `src/manga_text_seg/translation/translate.py` | T042 | M |
| T044 | [US4] Policy retry/timeout/rate-limit | 1. Timeout, số retry, back-off lấy từ `configs/translation.json`<br>2. Timeout hoặc rate-limit được retry theo policy đó trước khi vùng bị tuyên bố thất bại<br>3. Mỗi lần xảy ra ghi thành `api_timeout_or_rate_limit` kèm lần retry tiếp theo | `src/manga_text_seg/translation/translate.py` | T042 | M |
| T045 | [US4] Failure policy | 1. Thất bại cuối cùng → nhánh đã cấu hình áp dụng<br>2. `keep_ocr_text` (mặc định) mang text tiếng Nhật đi tiếp với vùng gắn cờ translation-failed<br>3. `exclude_from_rendering` loại vùng đó ra<br>4. `failure_policy` được ghi trong cả hai trường hợp để lựa chọn hiện ra trong metadata | `src/manga_text_seg/translation/translate.py` | T043 | M |
| T046 | [US4] Cache theo nội dung | 1. Entry tại `cache/translation/<cache_key>.json`<br>2. `cache_key` = SHA-256 trên `(text, target_language, provider, model)`<br>3. `cache_status` báo theo từng vùng<br>4. Công tắc `bypass` ép gọi mới mà không xoá gì | `src/manga_text_seg/translation/translate.py` | T042 | M |
| T047 | [US4] Xử lý secret | 1. API key đến **duy nhất** từ biến môi trường do `configs/translation.json` nêu tên<br>2. Không gì hard-code<br>3. Secret bắt buộc nhưng thiếu bị phát hiện **lúc khởi động** với lỗi rõ, không phải fail từng vùng<br>4. Không key/token/secret nào bị ghi vào source, config, log, metadata hay bất kỳ output nào | `src/manga_text_seg/translation/translate.py` | T041 | M |
| T048 | [US4] Ghi provenance provider | 1. `provider`, `model`, `version` ghi vào `run.json`, mọi entry `translations.json` và `metadata.json`<br>2. Là chuỗi thuần, **không** có secret nào bên cạnh (SC-004, SC-005) | `src/manga_text_seg/translation/translate.py` | T047 | S |

**Checkpoint C5**: dịch hoàn tất trên mock provider không cần mạng; audit SC-005 tìm ra **0** secret.

### Phase 7 — US5: Render bản dịch lên trang đã inpaint (P2)

**Independent Test**: render các trang fixture với text ngắn (vừa), text dài (cần shrink/wrap), text
vượt cả font nhỏ nhất (kỳ vọng expand-rồi-warning), font thiếu, và một vùng ở sát viền trang — khẳng
định vị trí đặt, hành vi vừa khung, chiến lược overflow, record cảnh báo và cô lập lỗi đều theo spec.

| Mã | Việc | Tác vụ con | File | Phụ thuộc | Cỡ |
|----|------|-----------|------|-----------|-----|
| T049 | [P] [US5] Unit test fit | 1. Text ngắn vừa ở size đã cấu hình<br>2. Text dài wrap và shrink<br>3. Text vượt cả `min_font_size` đi hết thang FR-041 **theo đúng thứ tự** và kết thúc bằng cảnh báo `text_overflow` được ghi<br>4. **Cấm clip âm thầm** — khẳng định **không** pixel glyph nào rơi ra ngoài vùng hiển thị của vùng đó<br>5. Fallback font: font path không đọc được → rơi về `fallback_font_path`; không có fallback → vùng đó `font_missing` trong khi phần còn lại của page chạy tiếp | `tests/unit/translation/test_render_fit.py` | T007 | L |
| T050 | [P] [US5] Unit test vị trí render | 1. Text đặt bên trong bbox/polygon đã lưu của vùng, **không bao giờ** ra ngoài trừ khi `expansion_allowance` cho phép nấc thứ ba của FR-041<br>2. Vùng có chữ gốc dọc → ghi orientation phát hiện được và render theo hướng config chọn, `direction_used` ghi theo từng vùng | `tests/unit/translation/test_render_position.py` | T007 | M |
| T051 | [US5] Render | 1. **Chỉ Pillow**, nằm trong module này<br>2. Font family, size, colour, outline/stroke, background, word wrap, line spacing, alignment đều đọc từ `configs/translation.json`<br>3. Vẽ vào vùng hiển thị đã lưu của vùng<br>4. Pillow khai ở T005 và **không dùng ở chỗ nào khác** | `src/manga_text_seg/translation/render.py` | T049, T050 | L |
| T052 | [US5] Thang fit-rồi-overflow | 1. Áp chiến lược **theo đúng thứ tự**: giảm font size xuống tới `min_font_size` → tăng số dòng → nới nhẹ vùng hiển thị trong `expansion_allowance` | `src/manga_text_seg/translation/render.py` | T051 | M |
| T053 | [US5] Cảnh báo overflow | 1. Vùng vẫn không vừa sau nấc cuối → ghi `text_overflow` kèm reason<br>2. `render_status` là `overflow_warning`<br>3. Kèm một `render_warning`<br>4. Text **không bao giờ bị clip âm thầm** (SC-006) | `src/manga_text_seg/translation/render.py` | T052 | S |
| T054 | [US5] Resolve font | 1. Font resolve **theo đường dẫn file** từ `assets/fonts/DejaVuSans.ttf`<br>2. `fallback_font_path` là lần thử thứ hai<br>3. **Không** tra font hệ thống<br>4. Path không đọc được và không có fallback → ghi `font_missing`, phần còn lại của page chạy tiếp | `src/manga_text_seg/translation/render.py` | T051 | S |
| T055 | [US5] Render theo orientation | 1. Tiêu thụ orientation đã phát hiện của vùng<br>2. Render theo hướng config chọn<br>3. Ghi `direction_used` theo từng vùng | `src/manga_text_seg/translation/render.py` | T051 | S |
| T056 | [US5] Writer trang đã render | 1. `rendered.png` là PNG lossless 1654×1170<br>2. Cùng size với trang inpainted mà nó được vẽ lên | `src/manga_text_seg/translation/render.py` | T051 | S |
| T057 | [US5] Bảo đảm không resize, read-only | 1. Không gì trong đường render resize ảnh<br>2. Ảnh gốc, prediction mask, trang inpainted và mọi artefact Spec 1–3 không bị thao tác ghi chạm tới (FR-006, FR-042) | `src/manga_text_seg/translation/render.py` | T056 | S |

**Checkpoint C6**: một trang fixture render end-to-end và thang overflow quan sát được trong `render_status`.

### Phase 8 — US6: Batch với cô lập lỗi mức vùng và mức ảnh (P2)

**Independent Test**: chạy batch có gài một lỗi cho mỗi category trong 11 category ở mức vùng và mức
page; khẳng định batch hoàn tất, error report liệt kê đủ mọi lỗi theo category kèm identifier, run thứ
hai với run ID khác cùng tồn tại, và tái dùng run ID bị từ chối trừ khi có overwrite.

| Mã | Việc | Tác vụ con | File | Phụ thuộc | Cỡ |
|----|------|-----------|------|-----------|-----|
| T058 | [P] [US6] Test tích hợp cô lập lỗi batch | 1. Gài đủ **11 category**: thiếu ảnh gốc; thiếu mask; thiếu ảnh inpainted; mask toàn 0; mọi component dưới `min_area`; crop không đọc được; provider luôn fail; font không đọc được và không fallback; text dài hơn vùng hiển thị ở size nhỏ nhất; provider timeout rồi rate-limit; thư mục output chỉ-đọc<br>2. Batch hoàn tất<br>3. Lỗi mức vùng để các vùng khác của page hoàn tất; lỗi mức page để batch chạy tiếp<br>4. Đủ **11 key** trong `counts` **kể cả khi = 0**, để category rỗng phân biệt được với category chưa xử lý | `tests/integration/test_batch_error_isolation.py` | T012 | L |
| T059 | [P] [US6] Test tách run ID | 1. Run thứ hai dưới run ID khác cùng tồn tại, run đầu không bị chạm<br>2. Chạy lại vào run ID đã có → **từ chối** trừ khi có `--overwrite` tường minh<br>3. Bị từ chối → cây run trước **byte-identical**<br>4. Thư mục của page thất bại chứa đúng những gì run đã sinh tới lúc lỗi, không gì bị sửa/thay thế/bịa | `tests/integration/test_run_id_separation.py` | T012 | M |
| T060 | [US6] Điều phối stage | 1. Chạy intake → regions → OCR → translate → render cho một page<br>2. Mỗi stage tiêu thụ output đã lưu của stage trước làm input<br>3. Expose mỗi stage như một entry point để gọi riêng được (FR-050) | `src/manga_text_seg/translation/pipeline.py` | T058, T059 | L |
| T061 | [US6] Cô lập mức vùng | 1. Lỗi một vùng ở bất kỳ stage nào được ghi dưới category của nó<br>2. Các vùng còn lại của page vẫn hoàn tất | `src/manga_text_seg/translation/pipeline.py` | T060 | S |
| T062 | [US6] Cô lập mức page | 1. Lỗi một page được ghi dưới category của nó và batch chạy tiếp<br>2. `page_statuses` trong `run.json` mang cùng sự thật đó ở dạng tóm tắt | `src/manga_text_seg/translation/pipeline.py` | T060 | S |
| T063 | [US6] Batch quy mô manifest | 1. Duyệt page của manifest Spec 1 hiện có — nguồn sự thật **duy nhất** cho page list và ánh xạ ảnh–mask<br>2. **Không** dựng manifest mới, page list mới hay subset benchmark nào (FR-001) | `src/manga_text_seg/translation/pipeline.py` | T060 | M |
| T064 | [US6] Nối từ chối ghi đè vào batch | 1. Run ID đã tồn tại → từ chối trừ khi có overwrite tường minh<br>2. Từ chối được báo rõ ràng, **không** âm thầm ghi chỗ khác | `src/manga_text_seg/translation/pipeline.py` | T059 | S |

**Checkpoint C7**: batch với 11 lỗi được gài vẫn hoàn tất và kê khai đủ cả 11.

### Phase 9 — US7: Output trung gian kiểm tra được + metadata pipeline (P3)

**Independent Test**: xử lý một page end-to-end và khẳng định bộ artefact đầy đủ tồn tại đúng vị trí,
metadata JSON chứa mọi field bắt buộc, và mỗi stage chạy lại một mình được dựa trên output đã lưu của
stage trước.

| Mã | Việc | Tác vụ con | File | Phụ thuộc | Cỡ |
|----|------|-----------|------|-----------|-----|
| T065 | [P] [US7] Contract test schema metadata | 1. `metadata.json` validate được với `contracts/pipeline-metadata.schema.json`<br>2. `additionalProperties: false` được tôn trọng, conditional `allOf` thoả (`error_message` có khi `status: failed`)<br>3. Đủ field bắt buộc: `image_id`, `segmentation_method`, `inpainting_method`, `target_language`, OCR engine, translation provider, rendering config<br>4. Mảng `regions` có một entry mỗi vùng đã trích, **kể cả vùng thất bại** (FR-031) | `tests/contract/translation/test_pipeline_metadata_schema.py` | T012 | M |
| T066 | [US7] Writer `metadata.json` | 1. `image_id`, `run_id`, `segmentation_method`, `inpainting_method`, `target_language`, `ocr_engine`, `translation_provider`, `rendering`, `regions`, `stage_timings`, `status`<br>2. Mỗi entry vùng mang `region_id`, `bbox`, `polygon`, `orientation`, `ocr_text`, `translation`, `ocr_status`, `translation_status`, `render_status`, `cache_status`, `failure_policy`, `direction_used`, `manually_edited` và các warning | `src/manga_text_seg/translation/runs.py` | T065 | L |
| T067 | [US7] Writer artefact trung gian | 1. `crops/<region_id>.png`, `mask.png`, `ocr.json`, `translations.json`, `rendered.png` — mỗi cái là artefact riêng<br>2. Để bất kỳ stage nào chạy lại một mình dựa trên output đã lưu của stage trước, cho kết quả **giống** như khi chạy trong pipeline đầy đủ (FR-050, US7 scenario 3) | `src/manga_text_seg/translation/runs.py` | T066 | M |
| T068 | [US7] Trích provenance nguyên văn | 1. `inpainting_method` trích **nguyên văn** từ sidecar của Spec 3 — algorithm, radius và cấu hình dilation<br>2. **Không** suy diễn lại, **không** dùng giá trị mặc định (FR-004) | `src/manga_text_seg/translation/runs.py` | T066 | S |
| T069 | [US7] Bảo đảm addressing FR-052 | 1. Consumer chỉ biết (run ID, segmentation method, `image_id`) resolve được thư mục page<br>2. Hiểu được nó **chỉ từ** `metadata.json` — không index file, không tra manifest, không kiến thức riêng của run (SC-010) | `src/manga_text_seg/translation/runs.py` | T066 | S |
| T070 | [US7] Ghi timing + status per-stage | 1. Thời gian mỗi stage mỗi page vào `stage_timings`<br>2. `status` của page là một trong `ok`, `no_regions`, `failed`<br>3. `stage_timings` là field phi tất định **duy nhất** trong file này (research R8) | `src/manga_text_seg/translation/runs.py` | T066 | S |

**Checkpoint C8**: thư mục của một page đã hoàn tất tự mô tả được, metadata validate được với schema.

### Phase 10 — US8: Chạy lại có chọn lọc (P4)

**Independent Test**: gài một lỗi vùng và một lỗi page, chạy lại **đúng** những phạm vi đó dựa trên
output trung gian đã lưu, khẳng định chỉ phạm vi được nhắm bị xử lý lại, kết quả khớp một run đầy đủ,
và không output không liên quan nào bị sửa.

| Mã | Việc | Tác vụ con | File | Phụ thuộc | Cỡ |
|----|------|-----------|------|-----------|-----|
| T071 | [P] [US8] Test tích hợp chạy lại chọn lọc | 1. Gài một lỗi vùng và một lỗi page<br>2. Chạy lại đúng phạm vi đó dựa trên output trung gian đã lưu<br>3. Chỉ phạm vi được nhắm bị xử lý lại, record của nó được cập nhật<br>4. Output đã lưu của các stage trước được **tiêu thụ nguyên trạng**, không tính lại<br>5. Kết quả khớp run đầy đủ | `tests/integration/test_selective_rerun.py` | T070 | M |
| T072 | [US8] Chạy lại một stage | 1. Stage gọi riêng tiêu thụ output đã lưu của stage trước **mà không chạy lại nó**<br>2. Cho kết quả giống như khi chạy trong pipeline đầy đủ (FR-050) | `src/manga_text_seg/translation/pipeline.py` | T071 | M |
| T073 | [US8] Chạy lại một vùng | 1. Chạy lại một vùng chỉ xử lý lại vùng được nêu tên và cập nhật record của nó<br>2. Các vùng khác của page để nguyên (FR-053) | `src/manga_text_seg/translation/pipeline.py` | T072 | S |
| T074 | [US8] Bảo đảm bán kính ảnh hưởng | 1. Một lần chạy lại để output của các vùng, page và run **không liên quan** byte-unchanged (US8 scenario 3) | `src/manga_text_seg/translation/pipeline.py` | T073 | S |

**Checkpoint C9**: một vùng thất bại sửa được mà không chạm gì khác trên đĩa.

### Phase 11 — Polish & cross-cutting

| Mã | Việc | Tác vụ con | File | Phụ thuộc | Cỡ |
|----|------|-----------|------|-----------|-----|
| T075 | [P] Test tất định | 1. Hai run trên cùng input + config cho artefact extraction, OCR và rendering **byte-identical**<br>2. Đúng **hai** ngoại lệ được ghi: field thời gian (`stage_timings`, `processing_time_seconds`) và `translations.json` — độ tất định của nó là thuộc tính provider khai báo, được ghi trong `run.json` chứ không bảo đảm<br>3. Không timestamp wall-clock trong bất kỳ field metadata per-page nào; run ID là giá trị định thời duy nhất (research R8) | `tests/integration/test_determinism.py` | T070 | M |
| T076 | [P] Test audit secret | 1. Audit tự động trên output + log của một run đã hoàn tất tìm thấy **0** lần xuất hiện API key hay secret material<br>2. Trong khi field `provider`/`model`/`version` vẫn có mặt<br>3. Chuỗi `reason` nêu **tên** biến môi trường, không bao giờ nêu giá trị (SC-005) | `tests/integration/test_secret_audit.py` | T070 | M |
| T077 | [P] Test không vendor + không GT | 1. Không file source model nào và không file weight nào ở mọi kích cỡ, mọi định dạng tồn tại trong cây được lưu<br>2. Không runtime model segmentation nào bị import ở đâu trong `src/`<br>3. Không artefact nào dưới `outputs/translation/` bắt nguồn từ mask ground truth hay copy nó<br>4. Không file nào dưới `data/no-need-to-read/` bị mở (SC-011, SC-013) | `tests/integration/test_no_vendoring_and_no_gt.py` | T070 | M |
| T078 | Subcommand `translate` | 1. Cờ: `--config`, `--run`, `--image-id`, `--stage`, `--region`, `--segmentation-method`, `--inpainting-run`, `--target-lang`, `--overwrite`<br>2. Theo convention `cmd_*` / subparser hiện có tại `src/manga_text_seg/cli.py:341-386`<br>3. **Không** làm xáo trộn các subcommand đang có | `src/manga_text_seg/cli.py` | T070 | M |
| T079 | Nối thao tác FR-053 vào pipeline | 1. Xử lý một ảnh; xử lý cả manifest; chỉ trích vùng; chỉ OCR; chỉ dịch; chỉ render; pipeline đầy đủ; chạy lại một vùng thất bại; export output trung gian<br>2. Mỗi cái map tới một entry point của `pipeline.py`<br>3. Tên stage khớp với giá trị `stage` trong error report | `src/manga_text_seg/cli.py` | T078 | M |
| T080 | Export output trung gian | 1. Copy output trung gian đã lưu của page được yêu cầu tới một đích do caller đặt tên, **ngoài** cây run<br>2. **Copy, không move** — cây run còn nguyên (FR-006) | `src/manga_text_seg/translation/pipeline.py` | T079 | S |
| T081 | Kiểm 13 vùng test FR-060 | 1. Xác nhận mỗi vùng trong **13 vùng** đều có test và cả suite pass với **không GPU, không checkpoint, không OCR engine, không provider key, không mạng** — `pytest -o addopts='' -q`<br>2. Ghi lại file test nào phủ vùng nào: trích vùng từ mask nhị phân; bounding box và padding; thứ tự vùng; adapter OCR với fixture; adapter dịch với mock provider; cache kết quả dịch; không rò API key; wrap chữ; fallback font size; overflow chữ; vị trí render; một vùng lỗi trong khi vùng khác hoàn tất; ánh xạ metadata `image_id`/`region_id`/OCR/bản dịch | — | T075–T080 | M |
| T082 | Chạy quickstart + lint | 1. Chạy `quickstart.md` kịch bản 1–3 và 5–10 end-to-end trên fixture bundle (nửa live provider của kịch bản 4 vẫn tuỳ chọn, **không** chạy trong CI)<br>2. Xác nhận `ruff check` và `black --check` sạch trên các file mới | `specs/004-ocr-translation-rendering/quickstart.md` | T081 | M |

---

## 5. Điểm bàn giao & phối hợp

### 5.1 Thứ tự thực thi

```
A: T001 ─ T011 ─ T021 ─ T031 ──┐
                               ├─→ B: T032 ─ T039 ─ T044 ─ T047 ─ T051 ─ T060
C: T001 ─ T012 ─ T019 ─ T029 ─ T038 ─ T048 ─ T057 ─ T064 ─ T070 ─ T074 ─ T082
```

- **A chặn B** ở Checkpoint A3 (hết T031): B cần `runs.py` và `inpaint.py` đứng yên.
- **A chặn C** ở T006 (`config.py`) và C chặn A ở T008 cùng file — ai tới trước land trước.
- **C độc lập hoàn toàn** với A và B về mặt logic; chỉ đụng `config.py`, `cli.py`, `conftest.py`.

### 5.2 San tải — chuỗi của C dài nhất (82 task)

Khi A xong T031 hoặc B xong T051, **chuyển nguyên khối** cho người rảnh, không cắt lẻ:

| Khối | Task | Vì sao bàn giao được | Điều kiện |
|------|------|----------------------|-----------|
| 004 US5 — rendering | T049–T057 (9 task) | `render.py` là module lá: chỉ cần regions + translations từ fixture. **Không** phụ thuộc OCR thật hay provider thật | C đã xong T012 (Foundational) + T002 (font) |
| 004 Polish | T075–T082 (8 task) | Test audit + CLI + quickstart. Cần C xong hết US1–US8 | C đã xong T074 |

Bàn giao khối nào thì **C dừng hẳn** ở khối đó — không hai người cùng mở `render.py` hay cùng sửa
`cli.py` trong một khoảng thời gian.

### 5.3 Commit

Theo `common/git-workflow.md`: `<type>: <description>`, type ∈ feat / fix / refactor / docs / test /
chore / perf / ci. Commit sau mỗi task hoặc mỗi nhóm logic. Kết thúc commit message bằng:

```
Co-Authored-By: Claude Code <noreply@anthropic.com>
```

Nhánh hiện tại: `feature/002-dl-benchmark-export`. Thống nhất nhánh trước khi bắt đầu — cả 3 người
nên làm trên cùng một nhánh tính năng hoặc mỗi người một nhánh rồi merge theo checkpoint.

---

## 6. Checklist trước khi báo "xong" một task

- [ ] Test được viết **trước** và đã **FAIL** (với task test-first).
- [ ] `pytest -o addopts='' -q` pass.
- [ ] Không đọc `data/no-need-to-read/`.
- [ ] Không ground truth nào lọt vào output hay board.
- [ ] Không secret, không key, không token trong source / config / log / metadata / output.
- [ ] Không file model hay weight nào được thêm vào cây.
- [ ] Không nguồn read-only nào bị sửa, move, rename hay xoá.
- [ ] Không metric nào bị tự tính lại.
- [ ] `run.py` (số ít, Spec 2) không bị chạm.
- [ ] `ruff check` và `black --check` sạch.
- [ ] Commit đúng format, có dòng `Co-Authored-By`.
