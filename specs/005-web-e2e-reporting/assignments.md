# Spec 005 — Phân công 3 người

> **File này KHÔNG sửa `tasks.md`.** Hiến pháp Nguyên tắc I biến `tasks.md` thành hợp đồng: task ID, thứ tự, tiêu chí nghiệm thu ở đó là bất biến. File này chỉ là lớp phối hợp — gán chủ sở hữu theo task ID và chốt thứ tự chạy. Không thêm task mới, không đổi phạm vi task nào.

- **Nguồn duy nhất**: [tasks.md](tasks.md) — 82 task, T001–T082, 13 phase.
- **Ngày chốt**: 2026-10-12
- **Phương án**: 19 / 20 / 26

---

## 0. Sprint 0 — Làm chung (T001–T017, 17 task)

**Cả ba người cùng làm, trên MỘT nhánh chung.** Không tách nhánh trước khi phase này xong.

Lý do: Phase 2 tạo ra `web/schemas.py`, `web/errors.py`, `web/security.py`, `web/session.py`, `web/app.py` — đây là **hợp đồng interface giữa ba người**. Tách nhánh trước khi chốt `schemas.py` thì cả ba nhánh sẽ merge conflict ở đúng những file đó.

Phân công nội bộ Sprint 0:

| Task | Nội dung | Chủ trì | Ghi chú |
|---|---|---|---|
| **T001** | `configs/web.json` | **A** | Chặn tất cả — không path/host/port/root nào được nằm trong code. A viết, B+C review. |
| T002–T004 | `pyproject.toml` + `uv.lock`, `web/__init__.py`, kiểm `.gitignore` | B | Cơ học, 3 task `[P]`. |
| **T005** | Fixture-bundle factory trong `tests/conftest.py` | **C** | **Đòn bẩy lớn nhất của cả spec.** Mọi test của cả ba người chạy trên nó. Cờ omission theo từng stage (SC-002) phải đủ — thiếu thì US2/US3/US5 đều không test được. C viết, A+B review kỹ. |
| T006 | Ba-way test split rỗng | B | |
| T007–T008 | Error taxonomy: test + `web/errors.py` | B | 18 category → HTTP status. |
| T009–T010 | Security: test + `web/security.py` | C | 4 ca từ chối SC-015. |
| T011–T012 | Session store: test + `web/session.py` | A | FR-022a, FR-065. |
| T013–T014 | App skeleton: test + `web/app.py` | A | |
| T015 | `cmd_serve` + subparser `serve` trong `cli.py` | C | |
| T016 | `web/schemas.py` | **B** | **Chốt cuối Sprint 0.** B viết, cả ba review — sau đây là hợp đồng. |
| T017 | Run creation/materialization | A | `outputs/end-to-end/<run_id>/`. |

**Điều kiện thoát Sprint 0** (checkpoint của Phase 1 + Phase 2 trong tasks.md):

1. `pytest -o addopts='' -q` xanh.
2. `uv lock --check` pass.
3. `manga-text-seg serve` khởi động được trên fixture bundle.
4. Cả 4 ca từ chối SC-015 trả đúng envelope.
5. `schemas.py` được cả ba người review và **đóng băng**.

Xong 5 điều trên → merge vào `main` → tách 3 nhánh.

---

## 1. Ba làn việc

| | Người | Task | Số | File sở hữu |
|---|---|---|---|---|
| **A** | Đường đọc dữ liệu | US1, US2, US3 | **19** | `routes/catalog.py`, `routes/page.py`, `routes/artifacts.py` (tạo) |
| **B** | Sinh nội dung | US5, US6, US7 | **20** | `routes/summary.py`, `routes/reports.py`, `routes/artifacts.py` (mở rộng) |
| **C** | Vận hành, CLI, kiểm định | US4, US8, US9, US10, Polish | **26** | `routes/edit.py`, `web/jobs.py`, `routes/jobs.py`, `web/session.py`, `cli.py` |

### Vì sao chia thế này

Chia theo **file sở hữu**, không theo số task — conflict mới là thứ giết tiến độ, không phải khối lượng.

- **A là critical path.** US1 là MVP; demo chỉ sống khi A xong US1. US2 và US3 đều mở rộng `routes/page.py` — cùng một file, nên **buộc phải nằm chung một làn**. Không thể chia ba.
- **B không phải chờ A.** US5/US6/US7 chỉ đọc artifact của Spec 1–4 và tái dùng `report.py` / `qualitative.py` có sẵn — không cần page bundle. B bắt đầu được ngay sau Sprint 0.
- **C nặng nhất (26) nhưng phần lớn là task ít rủi ro.** US8/US9/Polish là cơ học. US10 (CLI parity) bắt buộc chờ A và B xong — C phải để nó cuối.

### Phương án thay thế (nếu muốn đều hơn)

Chuyển **US4 (6 task)** từ C sang B → **19 / 26 / 20**. Đổi lại: B phải chờ A xong US3 mới có region payload để sửa. Không khuyến nghị — nó biến B thành làn bị chặn, trong khi hiện tại cả A và B đều chạy độc lập.

---

## 2. Chi tiết từng người

### Người A — Đường đọc dữ liệu (19 task)

| Phase | Task | Nội dung |
|---|---|---|
| US1 | T018–T023 | 6 test `[P]`: session, catalog, benchmark default, page, artifact, open-page journey |
| US1 | T024 | `routes/catalog.py` — liệt kê từ disk qua `manifest.py` + `availability.py` |
| US1 | T025 | Nối FR-017 benchmark default, tái dùng `selection.py` |
| US1 | T026 | `routes/page.py` — page bundle |
| US1 | T027 | `routes/artifacts.py` — **route duy nhất trả bytes** |
| US1 | T028 | Cô lập lỗi cấp page + run error report (FR-026) |
| US2 | T029–T030 | 2 test `[P]`: chain references, byte equality |
| US2 | T031–T032 | Mở rộng `routes/page.py`: 5 stage + per-stage missing artifact |
| US3 | T033–T034 | 2 test `[P]`: region payload, missing values |
| US3 | T035–T036 | Mở rộng `routes/page.py`: region list + FR-019 |

**Thứ tự bắt buộc**: US1 → US2 → US3 (cùng file `routes/page.py`). Test của US2/US3 viết được sớm, nhưng implementation phải nối tiếp.

**Bàn giao cho B**: `routes/artifacts.py` sau khi US1 xong (T027) — xem mục 3.

### Người B — Sinh nội dung (20 task)

| Phase | Task | Nội dung |
|---|---|---|
| US5 | T043–T046 | 4 test `[P]`: summary adapter, secrets, summary API, chapter summary |
| US5 | T047 | `routes/summary.py` + startup secret validation trong `web/app.py` |
| US5 | T048–T050 | `POST /api/summary/page`, cache FR-043, `POST /api/summary/chapter` |
| US6 | T051–T053 | 3 test `[P]`: quant report API, generation, report serving |
| US6 | T054–T055 | `routes/reports.py` — quantitative + CSV/PNG |
| US6 | T056 | Report serving qua `routes/artifacts.py` + auto-created run (FR-054) |
| US7 | T057–T059 | 3 test `[P]`: 5 policy, qual report API, self-containment |
| US7 | T060–T062 | 5 policy + `POST /api/reports/qualitative` + HTML tự chứa |

**Thứ tự bắt buộc**: US6 → US7 (cùng file `routes/reports.py`).

**B là làn duy nhất đụng `web/app.py` sau Sprint 0** (T047) — chỉ một lần, ngay đầu US5. Báo trước cho A và C.

### Người C — Vận hành, CLI, kiểm định (26 task)

| Phase | Task | Nội dung |
|---|---|---|
| US4 | T037–T039 | 3 test `[P]`: edit API, render API, edit persistence |
| US4 | T040–T042 | `routes/edit.py` + persist vào `web/session.py` |
| US8 | T063–T064 | 2 test `[P]`: jobs API, batch failure isolation |
| US8 | T065–T067 | `web/jobs.py` + `routes/jobs.py` + durability FR-084 |
| US9 | T068–T069 | 2 test `[P]`: run status API, run download |
| US9 | T070–T071 | Run status + artifact enumeration |
| US10 | T072 | Test CLI parity |
| US10 | T073–T074 | Subcommand trong `cli.py` + route qua error taxonomy |
| Polish | T075–T082 | 8 audit: taxonomy, FR-087 coverage, SC-011 invariance, SC-014 static-root swap, README, ponytail gate, SC-001 timing, code review |

**T070 và T042 cùng đụng `web/session.py`** — cùng một người, không conflict.

**T072–T074 phải để cuối cùng** — CLI parity chỉ định nghĩa được khi A và B đã có đủ in-app operation.

**T077, T081, T082 chờ toàn bộ diff** — ba task này là cổng chặn cuối, không phải việc làm song song.

---

## 3. Bản đồ file sở hữu và điểm giao

| File | Ai tạo | Ai mở rộng | Ghi chú |
|---|---|---|---|
| `web/errors.py` | Sprint 0 | — | Đóng băng sau T008 |
| `web/security.py` | Sprint 0 | — | Đóng băng sau T010 |
| `web/schemas.py` | Sprint 0 (B) | — | **Đóng băng sau T016** |
| `web/app.py` | Sprint 0 (A) | **B** (T047) | C chỉ đọc (T078) |
| `web/session.py` | Sprint 0 (A) | **C** (T042, T070) | |
| `routes/catalog.py` | A (T024) | A (T025) | |
| `routes/page.py` | A (T026) | A (T028, T031, T032, T035, T036) | 6 task, một người |
| `routes/artifacts.py` | **A** (T027) | **B** (T056) | ⚠️ **Điểm giao duy nhất giữa A và B** |
| `routes/summary.py` | B (T047) | B (T048–T050) | |
| `routes/reports.py` | B (T054) | B (T055, T060–T062) | |
| `routes/edit.py` | C (T040) | C (T041, T042) | |
| `web/jobs.py` | C (T065) | C (T067) | |
| `routes/jobs.py` | C (T066) | — | |
| `cli.py` | Sprint 0 (C, T015) | **C** (T073, T074) | |

### ⚠️ Điểm giao A → B: `routes/artifacts.py`

- **A tạo** file ở T027 (US1) — route byte-serving duy nhất.
- **B mở rộng** ở T056 (US6) — thêm report serving (FR-030a) + auto-created run.

**Xử lý**: A báo B ngay khi T027 merge. B **không** tạo lại file, chỉ thêm nhánh report. Vì A's US1 đi trước theo thứ tự ưu tiên (P1 trước P2), điểm giao này tự giải quyết nếu tôn trọng thứ tự phase.

**Rủi ro nếu đảo**: B làm US6 trước khi A xong US1 → B phải tự viết route byte-serving → trùng lặp và lệch hành vi. Đừng đảo.

---

## 4. Cổng phụ thuộc

```
Sprint 0 (T001–T017, làm chung)
        │
        ├──► A: US1 ──► US2 ──► US3          (routes/page.py, tuần tự)
        │         └─► bàn giao routes/artifacts.py ──┐
        │                                            │
        ├──► B: US5 ──► US6 ──────────────────────► T056
        │                 └──► US7                   │
        │                                            │
        └──► C: US4 ──► US8 ──► US9 ────────────────┘
                              └──► US10 (chờ A + B) ──► Polish (T075–T082)
```

**Ba task không được bắt đầu sớm**: T072/T073/T074 (chờ A+B), T077/T081/T082 (chờ toàn bộ diff).

---

## 5. Bắt đầu từ đâu

### Làm được ngay, một mình, không chờ ai

**T001 — `configs/web.json`.** Task có đòn bẩy cao nhất trong Phase 1: FR-086/FR-013c/FR-013d cấm mọi path, host, port, root nằm trong code. Sai ở đây thì cả ba người viết code quanh một config sai. Làm trước, cả ba review.

**T005 — fixture-bundle factory.** Đòn bẩy cao thứ hai. Mọi test của cả ba người chạy trên nó; cờ omission theo stage (SC-002) là thứ US2 (T030), US3, US5 test dựa vào. Thiếu cờ → ba người tắc cùng lúc ở cùng một chỗ.

### Song song được ngay từ ngày 1, không cần code chạy

**Frontend bundle.** Xem mục 6.

### Không làm được cho tới khi Sprint 0 xong

Toàn bộ 10 user story. Phase 2 chặn cứng — mọi story đều là HTTP route, không route nào tồn tại trước `app.py`, `security.py`, `errors.py`, `session.py`, `schemas.py`.

---

## 6. ⚠️ Rủi ro chưa được gán: frontend bundle

**Không task nào trong `tasks.md` viết frontend source.** [tasks.md:426](tasks.md#L426) ghi rõ: *"No task above writes frontend source."* Bundle là artifact ngoài, thả vào static root cấu hình ở T001, phục vụ nguyên trạng.

Nhưng nó **chặn demo**, không chặn backend:

| Task | Cần frontend? |
|---|---|
| T018–T074 (backend + test) | **Không** — chạy trên fixture bundle của T005 |
| T078 (SC-014 static-root swap) | **Có** — cần bundle để swap |
| T081 (SC-001 two-minute timing) | **Có** — "first fully rendered page" |

**Hai việc cần chốt:**

1. **Ai làm frontend.** Nó là làn thứ ba hoàn toàn tách biệt — chỉ cần bảng 17 endpoint trong `plan.md`, không cần code chạy. Đây là ứng viên tự nhiên cho người rảnh nhất trong Sprint 0, hoặc làm song song từ ngày 1. Nó **không** tiêu tốn task nào của A/B/C.
2. **`stitch_manga_translation_pipeline_workbench/` đang untracked.** Quyết định: promote vào static root, hay xóa. `plan.md` chốt bundle **không** nằm trong repo này — phục vụ nguyên trạng từ static root cấu hình. Cần một quyết định dứt khoát trước T078.

---

## 7. Nhánh và commit

**Nhánh hiện có `feature/personA|B|C` đã mang công việc Spec 2–4.** Không tái sử dụng — tạo nhánh mới:

```
feature/005-foundation     ← Sprint 0, cả ba người, merge vào main trước
feature/005-personA        ← tách từ feature/005-foundation sau khi merge
feature/005-personB
feature/005-personC
```

**Điều kiện tiên quyết trước khi bắt đầu**: working tree đang bẩn — 22 file tracked đã sửa, 195 file untracked (183 là `.png` debug). Commit hoặc stash trước khi tạo nhánh, nếu không nhánh mới sẽ thừa hưởng toàn bộ thay đổi chưa commit.

**Commit**: theo `git-workflow.md` — `<type>: <description>`, type ∈ feat/fix/refactor/docs/test/chore/perf/ci. Commit sau mỗi task hoặc nhóm task logic.

**Hiến pháp Nguyên tắc II bắt buộc**: test viết trước, chạy cho **FAIL**, rồi mới implement. Không có ngoại lệ.

**Lệnh test**: `pytest -o addopts='' -q` — bắt buộc có `-o addopts=''` vì `pyproject.toml` đặt `--cov` nhưng `pytest-cov` chưa cài.
