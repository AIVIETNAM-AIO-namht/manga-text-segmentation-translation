# Outline báo cáo cuối kỳ — chia cho 4 người

**Môn**: Xử lý ảnh số (Digital Image Processing) — HCMUTE, Khoa Công nghệ Thông tin
**GVHD**: PGS.TS. Hoàng Văn Dũng · **Nhóm**: 9
**Đề tài**: Đánh giá so sánh các phương pháp truyền thống và học sâu trong phân đoạn văn bản Manga và dịch Manga từ đầu đến cuối (end-to-end)

**Nguồn của file này**: [`ref/FinaltermProject_Template.pdf`](ref/FinaltermProject_Template.pdf) (khung báo cáo bắt buộc) và [`ref/De_cuong_do_an_DIP.docx`](ref/De_cuong_do_an_DIP.docx) (đề cương đã nộp). Mọi số liệu trong đây trích từ artefact thật trong repo — **không được bịa thêm số nào**.

---

## 0. Cách dùng file này

- Mỗi mục có nhãn **`[TV4]` `[TV2]` `[TV3]` `[TV1]`**. Ai có nhãn nào thì viết mục đó.
- Mục ghi **`[CHUNG]`** là cả bốn người cùng viết, TV1 chốt bản cuối.
- Sau mỗi mục có dòng **Nguồn:** trỏ tới file trong repo để lấy số liệu/hình. **Trích nguyên văn, không tính lại.**
- Mục 7 là danh sách **bắt buộc nói thật** — đọc trước khi viết bất cứ phần nào. Trong `REPORT-OUTLINE.docx`, mục này là **Phụ lục A**, còn mục 8 là **Phụ lục B**; hai file cùng nội dung, chỉ khác số mục.

### Bốn người và phần việc đã làm

| Nhãn | Người | Việc đã làm trong repo | Spec |
|---|---|---|---|
| **TV4** | Nguyễn Quang Huy (`HuyyCter`) | Toàn bộ làn "nhận mask & inpainting" của Spec 003: `intake.py`, `maskproc.py`, `inpaint.py`, `runs.py`, `config.py`, `cli.py`, `configs/inpainting.json`. Chạy notebook **Manga-Text-Segmentation** (U-Net ResNet34) và chọn REISSUE Path 1 → phương pháp này **không được chấm điểm**. | 003-A |
| **TV2** | Hà Duy Anh (`PhemieIris`) | Làn batch / selection / boards / qualitative / ablation của Spec 003: `boards.py`, `qualitative.py`, `ablation.py`, `selection.py`. Chạy notebook **comic-text-detector** (YOLOv5 + DBNet + U-Net). | 003-B |
| **TV3** | `Penguin-cell34` *(điền tên thật)* | Toàn bộ Spec 004 (82 task): trích vùng chữ, OCR, dịch, render, batch, chạy lại có chọn lọc. Chạy notebook **UNet++ EfficientNetV2**. | 004-C |
| **TV1** | `CapyCoder` *(bạn)* | Spec 001 (baseline cổ điển 6 phương pháp), Spec 002 (benchmark, admission, metric, report, receipt), toàn bộ tài liệu (`ONBOARDING.md`, `REISSUE.md`, `TEAM-ASSIGNMENT.md`), và nền web Spec 005. | 001, 002, 005 |

### Mức độ đóng góp gợi ý (điền vào bảng bìa)

| Mã số SV | Họ và tên | % |
|---|---|---|
| *(điền)* | Nguyễn Quang Huy | 25 |
| *(điền)* | Hà Duy Anh | 25 |
| *(điền)* | *(tên thật của Penguin-cell34)* | 25 |
| *(điền)* | *(tên bạn)* | 25 |

> Chia đều 25% là mặc định hợp lý: bốn người, bốn làn việc tách bạch (TV4 = inpainting, TV2 = đánh giá định tính, TV3 = OCR/dịch/render, TV1 = baseline + benchmark + web). **Sửa lại nếu thực tế lệch** — đừng để bảng bìa nói sai.

---

## 1. Quy định hình thức (bắt buộc, từ template)

- Giấy A4, **font size 13**, **lề trái 3 cm**, **lề phải – trên – dưới 2 cm**, **dãn dòng 1.3 lines**.
- **Mỗi Phần bắt đầu ở trang mới.**
- Thứ tự trước nội dung: Trang bìa (bản tiếng Anh **và** bản tiếng Việt) → Mục lục → Danh mục các cụm từ/từ viết tắt → Danh sách hình vẽ → Danh sách bảng biểu.
- Nộp kèm **PPT slides** trình bày.
- **Tài liệu tham khảo phải chỉ rõ vị trí tham khảo trong báo cáo** (ví dụ: `[3, Phần 2.3]`).

---

## 2. Outline chi tiết

### Trang bìa — `[TV1]`

Hai trang, làm đúng khuôn template: trang tiếng Anh (`HCMC UNIVERSITY OF TECHNOLOGY AND ENGINEERING / FACULTY OF INFORMATION TECHNOLOGY`) và trang tiếng Việt (`TRƯỜNG ĐH CÔNG NGHỆ KỸ THUẬT TP. HỒ CHÍ MINH / KHOA CÔNG NGHỆ THÔNG TIN`).

- Course name: **Digital Image Processing** (bản VI: **Xử lý ảnh số**).
- Title: dùng đúng tên đề tài trong đề cương.
- Lecturer: **Assoc. Prof. Hoang Van Dung** / **PGS.TS. Hoàng Văn Dũng**.
- Bảng thành viên: **Mã số SV / Họ và tên / Mức độ đóng góp (%)** — bốn dòng, lấy từ bảng ở mục 0.
- `TP. Hồ Chí Minh, tháng 10 năm 2026`.

**Nguồn:** [`.scratch/template.txt`](.scratch/template.txt) trang 1 & 4.

---

### Mục lục · Danh mục viết tắt · Danh sách hình vẽ · Danh sách bảng biểu — `[TV1]`

Sinh tự động từ heading sau khi cả bốn người viết xong. Viết **sau cùng**, không viết trước.

**Danh mục viết tắt** — dùng danh sách này, ai thấy thiếu thì bổ sung:

| Viết tắt | Nghĩa đầy đủ |
|---|---|
| DL | Deep Learning (học sâu) |
| GT | Ground Truth (mask chuẩn) |
| IoU | Intersection over Union |
| MSER | Maximally Stable Extremal Regions |
| OCR | Optical Character Recognition |
| TELEA | Thuật toán inpainting của Telea (OpenCV `INPAINT_TELEA`) |
| NS | Navier–Stokes (OpenCV `INPAINT_NS`) |
| YOLO | You Only Look Once |
| DBNet | Differentiable Binarization Network |
| LLM | Large Language Model |
| E2E | End-to-End |
| CLI | Command-Line Interface |
| LOFO | Leave-One-Fold-Out |

---

## PHẦN 1. MỞ ĐẦU (mô tả về project) — `[TV1]`, có `[TV4][TV2][TV3]` góp

Đây là phần chuyển thể từ đề cương đã nộp, **viết lại theo văn phong báo cáo** (không copy nguyên văn), và **cập nhật theo thực tế đã làm**.

### 1.1. Phát biểu bài toán — `[TV1]`
- Manga là nội dung số có nhu cầu dịch tự động cao; xử lý ảnh manga khó hơn ảnh tài liệu thường: chữ nằm trong bong bóng thoại hình dạng không cố định, font cách điệu, hiệu ứng âm thanh, nền phức tạp, bố cục trang đa dạng.
- Trong pipeline dịch manga, **phân đoạn vùng văn bản mức pixel** và **xóa – phục hồi bằng inpainting** là hai bước xử lý ảnh quyết định.
- Đây là bài toán vận dụng đúng kiến thức môn học: biến đổi cường độ, phân ngưỡng, phát hiện vùng đặc trưng, phát hiện biên, hình thái học, phục hồi ảnh.

**Nguồn:** đề cương mục 1.

### 1.2. Mục đích, yêu cầu cần thực hiện — `[TV1]`
Năm mục tiêu, mỗi mục ghi rõ **đã đạt hay chưa**:

1. Baseline cổ điển cho phân đoạn mức pixel (phân ngưỡng, MSER, biên, hình thái học) — **đạt**.
2. Chạy và đánh giá ba phương pháp DL pretrained trên cùng bộ đánh giá — **đạt một phần**: comic-text-detector và UNet++ EfficientNetV2 được chấm; **Manga-Text-Segmentation không chấm được** (xem mục 7.1).
3. So sánh định lượng bốn phương pháp bằng IoU / Precision / Recall / F1 và thời gian trung bình mỗi ảnh — **đạt** (bảng ở mục 4.3).
4. Xóa và phục hồi vùng văn bản bằng inpainting OpenCV, đánh giá định tính — **đạt** (mục 4.5).
5. Pipeline demo end-to-end phân đoạn → xóa → OCR → dịch → chèn lại — **đạt** (mục 4.6, 4.7).

**Nguồn:** đề cương mục 2.

### 1.3. Phạm vi và đối tượng — `[TV1]`
- **Đối tượng**: ảnh trang truyện manga, tập trung vùng chứa văn bản (lời thoại, chú thích, hiệu ứng âm thanh).
- **Phạm vi chính**: hai bước xử lý ảnh — phân đoạn mức pixel và xóa/phục hồi bằng inpainting.
- **Ngoài phạm vi benchmark**: OCR, dịch máy, chèn chữ, tóm tắt nội dung — triển khai ở mức demo, dùng công cụ có sẵn, **không huấn luyện mới**, **không dùng để xếp hạng** bốn phương pháp phân đoạn.
- Ghi rõ: **không huấn luyện lại** bất kỳ model nào; **không vendor** source/weights của bên thứ ba vào repo.

**Nguồn:** đề cương mục 3; [`README.md`](README.md); [`ONBOARDING.md`](ONBOARDING.md).

---

## PHẦN 2. CƠ SỞ LÝ THUYẾT — chia theo phương pháp, mỗi người viết phần của mình

### 2.1. Công cụ, môi trường, ngôn ngữ — `[TV1]`
- **Ngôn ngữ**: Python 3.12.13 (pipeline chính); Python 3.10.1 (môi trường kiểm tra khả dụng trong `benchmark/availability.json`).
- **Thư viện**: OpenCV (phân ngưỡng, MSER, morphology, inpainting), NumPy, Pillow, FastAPI + Uvicorn (web), pytest.
- **Quản lý gói**: `uv` + `pyproject.toml` + `uv.lock`.
- **Môi trường chạy model DL**: Google Colab, GPU **Tesla T4**.
- **Cấu hình phần cứng CPU**: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel.
- Kiến trúc repo: `src/manga_text_seg/` (thư viện), `configs/` (tham số), `outputs/` (kết quả), `deliverables/` (bàn giao), `specs/` (đặc tả), `tests/`.

**Nguồn:** [`README.md`](README.md); `deliverables/spec002/run.json`; `outputs/inpainting/spec003-main/performance.json`.

### 2.2. Kỹ thuật xử lý ảnh cổ điển dùng trong baseline — `[TV1]`
Viết lý thuyết cho **từng** kỹ thuật, mỗi kỹ thuật nửa trang:

- **Chuyển ảnh xám** và chuẩn hóa cường độ.
- **Phân ngưỡng Otsu** — ngưỡng toàn cục tự động từ histogram.
- **Phân ngưỡng thích nghi (Adaptive Threshold)** — ngưỡng cục bộ theo vùng lân cận.
- **MSER** — phát hiện vùng cực trị ổn định, phù hợp chữ trên nền biến thiên.
- **Phát hiện biên** (Canny / Sobel) — biên chữ.
- **Hình thái học**: dilation, erosion, connected components — nối nét chữ, loại nhiễu, hoàn thiện vùng.

**Nguồn:** `src/manga_text_seg/` (module baseline Spec 001); `outputs/segmentation/default/metrics.csv`; giáo trình Gonzalez & Woods.

### 2.3. Manga-Text-Segmentation — U-Net + ResNet34 — `[TV4]`
- Kiến trúc **U-Net** (encoder–decoder, skip connection), backbone **ResNet34**.
- Ứng dụng cho phân đoạn mức pixel trên manga; huấn luyện trên mask Manga109.
- Nêu rõ **vì sao đề tài dùng bản pretrained**: không huấn luyện lại, chỉ suy luận.
- **Kết thúc mục này bằng một câu chuyển ý**: phương pháp đã được đưa vào quy trình nhưng không chấm được điểm — chi tiết ở mục 4.4.1 và 7.1.

**Nguồn:** [`notebooks/deliverables/manga-text-segmentation/README.md`](notebooks/deliverables/manga-text-segmentation/README.md); `benchmark/availability.json`.

### 2.4. comic-text-detector — YOLOv5 + DBNet + U-Net — `[TV2]`
- Ba tầng: **YOLOv5** phát hiện khối text → **DBNet** phát hiện dòng text → **U-Net** sinh mask phân đoạn.
- Huấn luyện trên ~13.000 ảnh (Manga109-s, Digital Comic Museum, dữ liệu tổng hợp).
- Ghi rõ quyết định của đề tài: **chỉ dùng mask phân đoạn cuối cùng**, **không dùng bounding box** làm đầu ra chính cho benchmark.

**Nguồn:** [`notebooks/deliverables/comic-text-detector/`](notebooks/deliverables/comic-text-detector/); đề cương mục 4.

### 2.5. UNet++ EfficientNetV2 (ContemporaryCat) — `[TV3]`
- **UNet++** (nested skip pathways, dense decoder) + backbone **EfficientNetV2**.
- Huấn luyện lại trên cùng bộ mask Manga109 của Del Gobbo & Matuk Herrera; weight công khai trên HuggingFace.
- Ghi rõ: đây là phương pháp **có cảnh báo nhiễm bẩn dữ liệu** — nói trước ở đây, phân tích ở mục 4.4.3 và 7.2.

**Nguồn:** [`notebooks/deliverables/unetpp-efficientnetv2/`](notebooks/deliverables/unetpp-efficientnetv2/); đề cương mục 4.

### 2.6. Inpainting: TELEA và Navier–Stokes — `[TV4]`
- Nguyên lý hai thuật toán inpainting cổ điển trong OpenCV.
- Vì sao dùng inpainting cổ điển thay vì model sinh ảnh: **không có ảnh sạch làm ground-truth** để huấn luyện hay đánh giá.
- Vì sao đánh giá **định tính** chứ không dùng PSNR/SSIM: bộ dữ liệu không cung cấp ảnh manga sạch không chứa văn bản.

**Nguồn:** `src/manga_text_seg/inpaint.py`; `configs/inpainting.json`.

### 2.7. OCR, dịch máy, render chữ — `[TV3]`
- **manga-ocr**: OCR tiếng Nhật cho manga (transformer, xử lý được furigana).
- **Dịch máy**: dùng LLM qua API (OpenRouter / Anthropic / Vilao / OpenAI) — nêu khái niệm, không nêu giá trị khoá API.
- **Render chữ**: canh chỉnh font size, word-wrap, chèn lên ảnh đã inpaint; tham khảo module render của `manga-image-translator`.
- Ghi rõ: cả ba bước là **demo minh họa ứng dụng**, không thuộc phạm vi đánh giá kỹ thuật chính.

**Nguồn:** `src/manga_text_seg/translation/`; `outputs/translation/*/run.json`; đề cương mục 5.5.

---

## PHẦN 3. PHÂN TÍCH, THIẾT KẾ GIẢI PHÁP

### 3.1. Sơ đồ khối tổng thể — `[TV1]`
Vẽ sơ đồ khối toàn pipeline, đúng chuỗi trong README:

```
Dataset → Text segmentation → Text removal & inpainting → Japanese OCR
        → Translation → Text rendering → Summary & web demo
```

Trên sơ đồ phải phân biệt bằng màu/khối:
- **Phạm vi benchmark chính**: phân đoạn + inpainting.
- **Phần demo minh họa**: OCR → dịch → render → tóm tắt.
- **Bốn nhánh phân đoạn** hội tụ về một quy trình inpainting chung.

**Nguồn:** [`README.md`](README.md) (khối pipeline); `specs/002-*/spec.md`.

### 3.2. Thiết kế baseline cổ điển — `[TV1]`
- Sơ đồ khối riêng cho nhánh cổ điển: tiền xử lý → phân ngưỡng/MSER/biên → hình thái học → connected components → binary mask.
- **Sáu biến thể** được cài: `otsu`, `adaptive`, `mser`, `edges`, `morphology`, `components`.
- Giải thích vì sao baseline có sáu biến thể: để tìm cấu hình tốt nhất trong họ cổ điển trước khi so với DL.

**Nguồn:** `src/manga_text_seg/` (Spec 001); `outputs/segmentation/default/metrics.csv`.

### 3.3. Hợp đồng mask và quy trình kết nạp kết quả — `[TV1]`
Đây là phần thiết kế **quan trọng nhất về mặt kỹ thuật**, phải viết kỹ:

- **Định dạng mask bắt buộc**: ảnh một kênh `uint8`, kích thước `(1170, 1654)`, giá trị chỉ thuộc `{0, 255}`.
- **Quy trình bàn giao**: mỗi thành viên chạy một model trên Colab, nộp prediction mask; **metric do project tính, runner không nộp metric**.
- **Sáu kiểm tra kết nạp** trước khi một kết quả được nhận vào benchmark.
- **`page_list_identity`** phải trích nguyên văn từ `benchmark/`, **không bao giờ tính lại**. Giá trị đang dùng:
  `b52aa60ded6c5c2cca115dce4335295d6795b0601b9cfa8324c675fb4cab88ef`
- **Quy tắc không chạm ground truth**: prediction mask là mask duy nhất được mở trong toàn bộ pipeline.
- **De-contamination / LOFO (leave-one-fold-out)**: mỗi trang chỉ được chấm bởi checkpoint mà fold của nó đã giữ cuốn sách đó ra ngoài.
- **Công bố mức nhiễm bẩn** của từng phương pháp (ba mức: không trùng / không xác định được / nhiễm hoàn toàn).

**Nguồn:** [`ONBOARDING.md`](ONBOARDING.md) §2, §7, §8; `benchmark/availability.json`; `deliverables/spec002/run.json`; `deliverables/spec002/comparison.csv` (cột `disclosure`).

### 3.4. Thiết kế quy trình xóa và phục hồi — `[TV4]`
- Sơ đồ: nhận prediction mask → **dilation ellipse 3×3, 1 vòng lặp** → áp mask lên ảnh gốc → inpainting **TELEA** và **NS**, **radius = 3** → ảnh PNG.
- Vì sao **cùng một quy trình** áp cho cả ba phương pháp: bảo đảm tính nhất quán khi so sánh.
- Vì sao có bước dilation: mask dự đoán thường sát nét chữ, để lại viền chữ sau khi xóa.
- Cấu hình nguồn mask cho từng phương pháp, và **lý do chọn `adaptive` làm đại diện cho baseline cổ điển** (IoU cao nhất trong sáu biến thể).

**Nguồn:** [`configs/inpainting.json`](configs/inpainting.json); `outputs/inpainting/spec003-main/run.json`.

### 3.5. Thiết kế lựa chọn trang và board so sánh — `[TV2]`
- **Bốn quy tắc chọn trang**, mỗi quy tắc lấy **5 trang**:
  `highest_iou_f1`, `lowest_iou_f1`, `most_false_positives`, `most_false_negatives`.
- Vì sao chọn cả ca tốt nhất **và** xấu nhất: ca tốt cho thấy trần năng lực, ca xấu cho thấy kiểu thất bại đặc trưng.
- **Board** = ảnh ghép gồm mask dự đoán + ảnh sau xóa + ảnh sau inpainting, để so sánh trực quan.
- Có một quy tắc thứ năm `manual_artifact_flags` — **hiện chưa có trang nào được gắn cờ** (xem mục 7.4).

**Nguồn:** `src/manga_text_seg/boards.py`, `selection.py`, `qualitative.py`; `outputs/inpainting/spec003-main/selection.json`.

### 3.6. Thiết kế trích vùng, OCR, dịch, render — `[TV3]`
- **Trích vùng chữ ổn định từ mask**: diện tích vùng tối thiểu 50 px, tối đa 12000 px, khoảng cách gộp 8 px, đệm crop 0, thứ tự đọc **manga**, epsilon đa giác 1.0.
- **OCR**: manga-ocr, tất định (`deterministic = true`), không tiền xử lý, không upscale.
- **Dịch**: qua provider cấu hình được, `provider`/`model` được ghi vào run record nhưng **giá trị khoá API thì không, ở đâu cũng không**.
- **Render**: font DejaVu Sans, cỡ 32, cỡ nhỏ nhất 12, bật word-wrap, giãn dòng 1.0, canh giữa/canh giữa dọc, hướng ngang.
- **Cô lập lỗi hai mức**: lỗi ở mức vùng không làm chết cả ảnh; lỗi ở mức ảnh không làm chết cả lô.
- **Chạy lại có chọn lọc**: chỉ chạy lại vùng/ảnh hỏng, không chạy lại cả run.

**Nguồn:** `src/manga_text_seg/translation/`; `outputs/translation/spec004-live/run.json`.

### 3.7. Thiết kế web demo — `[TV1]`
- Kiến trúc: FastAPI + Uvicorn, phục vụ một bundle giao diện tĩnh.
- **Ba route đang chạy thật** trong bản demo:
  `GET /api/demo/catalog`, `GET /api/demo/page?image_id=<manga>/<page>`, `GET /api/artifact?ref=<root>/<tail>`.
- Nguyên tắc thiết kế: **chỉ route `/api/artifact` được trả bytes**; mọi trường trong response là đường dẫn tương đối, **không bao giờ là đường dẫn tuyệt đối**.
- Giá trị thiếu thì **có mặt và bằng `null`**, không được vắng key — để giao diện hiển thị được "chưa ghi nhận".
- Nói thẳng: bản demo hiện chạy **3 route**, không phải 17 endpoint trong `specs/005-web-e2e-reporting/plan.md`. Phần còn lại là hướng phát triển (mục 5.2).

**Nguồn:** `src/manga_text_seg/web/demo.py`, `web/schemas.py`, `web/security.py`; `web-ui/index.html`; `stitch_manga_translation_pipeline_workbench/`.

---

## PHẦN 4. THỰC NGHIỆM, ĐÁNH GIÁ, PHÂN TÍCH KẾT QUẢ

### 4.1. Dữ liệu và giao thức đánh giá — `[TV1]`
- **Nguồn dữ liệu**: ảnh và mask phân đoạn từ bộ công khai Manga-Text-Segmentation, dựa trên **Manga109**.
- **Quy mô**: **390 ảnh thuộc 45 manga** — dùng nhất quán cho cả bốn phương pháp.
- Ánh xạ `<manga>/<page_id>`, ví dụ `AisazuNihaIrarenai/000`.
- **Không dùng Pixel Accuracy** làm độ đo chính — giải thích: pixel nền chiếm phần lớn diện tích nên accuracy cao ngay cả khi bỏ sót gần hết vùng chữ.
- **Bốn độ đo chính**: IoU, Precision, Recall, F1-score, tính trên **binary mask** giữa dự đoán và ground truth.
- **Độ đo thời gian**: thời gian trung bình xử lý một ảnh ở bước phân đoạn.
- **Vì sao inpainting không dùng PSNR/SSIM**: không có ảnh sạch làm ground truth.

**Nguồn:** `benchmark/`; `deliverables/spec002/run.json`; đề cương mục 5.1, 5.3.

### 4.2. Cấu hình phần cứng và tham số — `[CHUNG]`
Mỗi người ghi cấu hình của **phần mình chạy**:

- **CPU (baseline cổ điển, inpainting)**: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel; Python 3.12.13.
- **GPU (ba model DL)**: Tesla T4 trên Google Colab.
- **Ngưỡng quyết định** đã dùng: comic-text-detector `0.235`, UNet++ EfficientNetV2 `0.5`.
- **Tiền/hậu xử lý**: liệt kê theo `deliverables/spec002/run.json`.
- **Tham số inpainting**: dilation ellipse 3×3 ×1, radius 3, hai thuật toán TELEA và NS.
- **Ghi cả `sha256` của checkpoint** đã dùng.

**Nguồn:** `deliverables/spec002/run.json`; `outputs/inpainting/spec003-main/performance.json` (khối `runtime`).

### 4.3. Kết quả benchmark định lượng — `[TV1]` (bảng), `[TV4][TV2][TV3]` (đọc số của mình)

**Bảng chính của báo cáo.** Chép đúng số từ `deliverables/spec002/comparison.csv`:

| Phương pháp | IoU | Precision | Recall | F1 | Thời gian/ảnh (s) | Thiết bị |
|---|---|---|---|---|---|---|
| Otsu | 0.0081 | 0.0088 | 0.1848 | 0.0154 | 0.0022 | CPU |
| Adaptive Threshold | 0.0711 | 0.0753 | 0.8744 | 0.1261 | 0.0090 | CPU |
| MSER | 0.0435 | 0.0452 | 0.9549 | 0.0761 | 1.2676 | CPU |
| Edges | 0.0350 | 0.0356 | 0.9857 | 0.0630 | 0.0248 | CPU |
| Morphology | 0.0502 | 0.0553 | 0.4358 | 0.0888 | 0.0054 | CPU |
| Connected Components | 0.0563 | 0.0599 | 0.6402 | 0.1001 | 0.4173 | CPU |
| comic-text-detector | 0.0372 | 0.1142 | 0.0603 | 0.0646 | 0.3814 | Tesla T4 |
| UNet++ EfficientNetV2 | 0.1375 | 0.1582 | 0.6944 | 0.2240 | 0.2790 | Tesla T4 |
| Manga-Text-Segmentation | *không chấm* | — | — | — | — | — |

**Phải ghi kèm ngay dưới bảng:**
- Độ lệch chuẩn của từng chỉ số (có trong `comparison.csv`) — IoU của UNet++ có `std` 0.1219, tức phân tán rất lớn, không được chỉ báo trị trung bình.
- Cột **công bố nhiễm bẩn** cho từng phương pháp.
- Dòng Manga-Text-Segmentation ghi rõ **lý do không chấm** (mục 7.1).

**Hình**: `deliverables/spec002/charts/accuracy-by-method.png`.

**Nguồn:** `deliverables/spec002/comparison.csv`, `comparison.json`, `run.json`.

### 4.4. Phân tích từng phương pháp — mỗi người một mục

#### 4.4.1. Manga-Text-Segmentation — `[TV4]`
- Trình bày **quy trình đã thực hiện**: chạy notebook trên Colab, xuất prediction mask theo đúng hợp đồng.
- Trình bày **lý do không có kết quả điểm**: thiếu 5 checkpoint fold (mỗi file ~344 MB) từ `github.com/juvian/Manga-Text-Segmentation`, và không xác nhận được revision đã ghim `de8f148c78978d70ad0e0ae3242566da6d70f3a5`.
- Trình bày **quyết định**: chọn REISSUE Path 1 → phương pháp được ghi nhận trạng thái `unavailable`, **không được chấm điểm**.
- **Kết luận mục này**: đây là một kết quả âm hợp lệ và được báo cáo trung thực, không phải thất bại bị che giấu. Không suy diễn, không ngoại suy, không điền số thay.

**Nguồn:** `benchmark/availability.json`; [`REISSUE.md`](REISSUE.md); `notebooks/deliverables/manga-text-segmentation/`.

#### 4.4.2. comic-text-detector — `[TV2]`
- IoU 0.0372 / F1 0.0646 — **thấp hơn cả baseline cổ điển tốt nhất** (adaptive: IoU 0.0711 / F1 0.1261).
- Phân tích **vì sao**: Recall rất thấp (0.0603) trong khi Precision 0.1142 — hệ thống bỏ sót phần lớn vùng chữ ở mức pixel.
- Giả thuyết: đây là hệ thống **phát hiện khối/dòng văn bản**, mask cuối cùng không được tối ưu cho khớp pixel với ground truth mức nét chữ.
- Nêu rõ **công bố nhiễm bẩn**: *"contamination extent unknown"* — không xác định được mức trùng lặp giữa tập huấn luyện và tập đánh giá.
- Thời gian 0.3814 s/ảnh trên Tesla T4 — chậm hơn mọi phương pháp cổ điển.

**Nguồn:** `deliverables/spec002/comparison.csv`; `deliverables/spec002/comic-text-detector/`.

#### 4.4.3. UNet++ EfficientNetV2 — `[TV3]`
- IoU 0.1375 / F1 0.2240 — **cao nhất trong các phương pháp chấm được**, gấp ~1.9 lần baseline tốt nhất theo IoU.
- Recall 0.6944 — tìm được phần lớn vùng chữ, nhưng Precision chỉ 0.1582 — sinh nhiều dương tính giả.
- **Phân tích ca thất bại điển hình**: `EvaLady/006` có IoU = 0 và **310.000 pixel dương tính giả** với 0 âm tính giả — model tô kín gần như cả trang.
- **Bắt buộc nêu công bố nhiễm bẩn**: *"fully contaminated — checkpoint được huấn luyện trên toàn bộ 45 cuốn sách của tập ground truth, nên mọi điểm số nó báo cáo đều là điểm trên tập huấn luyện."*
- **Kết luận trung thực**: kết quả tốt nhất này **không thể coi là bằng chứng tổng quát hóa**. Đây là hạn chế lớn nhất của phần benchmark và phải nói rõ, không được lờ đi.
- Thời gian 0.2790 s/ảnh — nhanh nhất trong ba model DL.

**Nguồn:** `deliverables/spec002/comparison.csv`; `deliverables/spec002/unetpp-efficientnetv2/`; `outputs/inpainting/spec003-main/selection.json`.

#### 4.4.4. Baseline cổ điển — `[TV1]`
- **Adaptive Threshold là biến thể tốt nhất** (IoU 0.0711 / F1 0.1261) và **được chọn làm đại diện** cho baseline cổ điển trong các bước sau.
- Đặc trưng chung của họ cổ điển: **Recall rất cao nhưng Precision rất thấp** — edges đạt Recall 0.9857 với Precision 0.0356; mser Recall 0.9549 với Precision 0.0452. Nghĩa là bắt gần hết chữ nhưng tô lan ra nền.
- **Otsu tệ nhất** (IoU 0.0081) — ngưỡng toàn cục không phù hợp ảnh manga có nền gradient và screentone.
- Đánh đổi tốc độ: cổ điển nhanh hơn DL 1–2 bậc độ lớn (0.0022–1.2676 s/ảnh trên CPU so với 0.28–0.38 s/ảnh trên GPU).
- Nêu bất đối xứng phần cứng khi so thời gian: baseline chạy CPU, model DL chạy GPU.

**Nguồn:** `deliverables/spec002/comparison.csv`; `outputs/segmentation/default/metrics.csv`.

### 4.5. Kết quả xóa và phục hồi — `[TV4]` (số liệu) + `[TV2]` (đánh giá định tính)

#### 4.5.1. Thời gian inpainting — `[TV4]`
390 mẫu mỗi tổ hợp, **0 lỗi**, CPU Intel64, Python 3.12.13:

| Phương pháp | TELEA (s/ảnh) | NS (s/ảnh) |
|---|---|---|
| classical_baseline | 1.1375 | 1.0251 |
| comic_text_detector | 0.0746 | 0.0589 |
| unetpp_efficientnetv2 | 0.5324 | **7.8853** |

**Phân tích bắt buộc**: NS trên mask của UNet++ chậm **~14.8 lần** so với TELEA trên cùng mask (7.8853 vs 0.5324). Giải thích: mask của UNet++ dày hơn nhiều, Navier–Stokes phải giải phương trình vi phân trên vùng hỏng lớn hơn. Đây là quan hệ **thuật toán × mật độ mask**, không phải thuật toán chậm hơn.

**Nguồn:** `outputs/inpainting/spec003-main/performance.json`.

#### 4.5.2. Đánh giá định tính — `[TV2]`
Trình bày **board so sánh** và nhận xét theo **đúng năm tiêu chí** của đề cương:

1. Mức độ xóa hoàn toàn văn bản.
2. Mức độ bỏ sót văn bản.
3. Mức độ xóa nhầm vùng không phải văn bản.
4. Độ tự nhiên của vùng ảnh sau inpainting.
5. Artifact, viền hoặc nhiễu phát sinh sau phục hồi.

**Vật liệu có sẵn**: `outputs/inpainting/spec003-main/boards/<method>/<rule>/` — 3 phương pháp × 4 quy tắc × 5 trang = **60 ảnh board**.
Ba phương pháp có board: `classical_baseline`, `comic_text_detector`, `unetpp_efficientnetv2`.

**Lưu ý về tính trung thực của hình**: board **không được chứa ground truth**. Chỉ gồm mask dự đoán, ảnh sau xóa, ảnh sau inpainting.

**Nguồn:** `outputs/inpainting/spec003-main/boards/`; `src/manga_text_seg/boards.py`, `qualitative.py`.

#### 4.5.3. Năm trang được chọn cho pipeline demo — `[TV2]`
Quy tắc `highest_iou_f1`, phương pháp `unetpp_efficientnetv2`:

| Trang | IoU | F1 | FP | FN |
|---|---|---|---|---|
| Akuhamu/000 | 0.6318 | 0.7744 | 16.729 | 91.855 |
| AisazuNihaIrarenai/001 | 0.5866 | 0.7395 | 30.731 | 52.290 |
| BEMADER_P/000 | 0.5732 | 0.7287 | 72.846 | 36.930 |
| BakuretsuKungFuGirl/000 | 0.5415 | 0.7026 | 50.697 | 108.972 |
| Hamlet/000 | 0.5270 | 0.6903 | 72.172 | 58.612 |

**Nguồn:** `outputs/inpainting/spec003-main/selection.json`.

### 4.6. Kết quả OCR, dịch, render — `[TV3]`
- **Năm run đã thực hiện**, ghi rõ phương pháp phân đoạn, ngôn ngữ đích, provider/model, số trang:

| Run | Phương pháp phân đoạn | Đích | Provider / Model | Số trang |
|---|---|---|---|---|
| `spec004-vilao` | unetpp_efficientnetv2 | vi | vilao / deepseek-v4.1-flash | **28** |
| `spec004-live` | unetpp_efficientnetv2 | vi | openrouter / anthropic/claude-sonnet-5.5 | 1 |
| `spec004-real` | classical_baseline | vi | anthropic / claude-sonnet-5-5 | 1 |
| `smoke-dl` | comic_text_detector | vi | openai / gpt-4o-mini | 1 |
| `smoke-translate` | classical_baseline | vi | openai / gpt-4o-mini | 1 |

- **Nguồn inpainting của run chính**: `spec003-main`, thuật toán `telea`.
- **Ghi rõ tên biến môi trường chứa khoá** (`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `MANGA_TRANSLATE_API_KEY`) — **tuyệt đối không ghi giá trị khoá**.
- Trình bày kết quả: ảnh sau render, văn bản OCR, bản dịch, và ghi nhận **các vùng/ảnh lỗi** (có `errors.json` mỗi run) — ghi nhận, không che.
- Trình bày **chạy lại có chọn lọc** và **cô lập lỗi mức vùng** như minh chứng cho thiết kế ở mục 3.6.
- Nêu giới hạn: chỉ 1 trang cho phần lớn run; run 28 trang (`spec004-vilao`) là minh chứng rộng nhất.

**Nguồn:** `outputs/translation/*/run.json`, `errors.json`; `outputs/translation/spec004-vilao/`.

### 4.7. Giao diện phần mềm và hướng dẫn thực thi — `[TV1]`
- **Ảnh chụp giao diện** web demo (`web-ui/index.html` trên nền bundle Stitch).
- **Hướng dẫn thực thi** từng bước, kèm lệnh thật:

```bash
# Cài đặt
uv sync

# Phân đoạn cổ điển
manga-text-seg run --config configs/segmentation.json --method adaptive

# Kiểm tra khả dụng của model DL
manga-text-seg status --config configs/dl.json

# Kết nạp kết quả bàn giao vào benchmark
manga-text-seg admit --config configs/dl.json --run <run_id> --method <name> <hand-off>/

# Inpainting
manga-text-seg inpaint --config configs/inpainting.json --run spec003-main

# Board và báo cáo định tính
manga-text-seg boards --config configs/inpainting.json --run spec003-main
manga-text-seg qualitative --config configs/inpainting.json --run spec003-main

# Dịch end-to-end
manga-text-seg translate --config configs/translation.json --run spec004-vilao

# Chạy web demo
manga-text-seg serve
```

- **Lệnh chạy test** (ghi chú bắt buộc có `-o addopts=''`):

```bash
pytest -o addopts='' -q
```

- Ghi rõ: demo chạy trên máy người trình bày, mở browser tại loopback.

**Nguồn:** [`README.md`](README.md); `src/manga_text_seg/cli.py`; `web-ui/index.html`.

---

## PHẦN 5. KẾT LUẬN

### 5.1. Đánh giá những kết quả đã thực hiện được — `[CHUNG]`, TV1 chốt

Mỗi người viết **2–3 gạch đầu dòng** cho phần mình, TV1 gộp lại và viết đoạn mở/đoạn kết:

- `[TV1]` Baseline cổ điển sáu biến thể hoàn chỉnh; quy trình kết nạp kết quả có kiểm tra và có công bố nhiễm bẩn; bảng benchmark 390 ảnh × 45 manga; web demo chạy được trên kết quả thật.
- `[TV4]` Quy trình inpainting hoàn chỉnh, áp thống nhất cho mọi phương pháp, 390 mẫu × 3 phương pháp × 2 thuật toán, 0 lỗi; quyết định REISSUE Path 1 cho Manga-Text-Segmentation được ghi nhận minh bạch thay vì che giấu.
- `[TV2]` Hệ thống board so sánh và đánh giá định tính theo năm tiêu chí; 60 ảnh board; quy tắc lựa chọn trang tái lập được.
- `[TV3]` Pipeline OCR → dịch → render chạy end-to-end, cô lập lỗi hai mức, chạy lại có chọn lọc; run 28 trang hoàn chỉnh.

**Kết luận kỹ thuật trung thực cần có**: trên bộ dữ liệu này, baseline cổ điển tốt nhất (adaptive) **vượt comic-text-detector** về IoU/F1, và **thua UNet++ EfficientNetV2**; nhưng kết quả của UNet++ **bị nhiễm bẩn hoàn toàn** nên không thể dùng để khẳng định ưu thế của học sâu. Đây là kết luận đúng và phải được viết ra.

### 5.2. Định hướng phát triển — `[CHUNG]`
- `[TV4]` Bổ sung 5 checkpoint fold để chấm được Manga-Text-Segmentation theo đúng quy tắc LOFO.
- `[TV2]` Gắn cờ artifact thủ công cho quy tắc `manual_artifact_flags` (hiện đang trống).
- `[TV3]` Mở rộng đánh giá OCR/dịch/render lên nhiều trang hơn; hiện phần lớn run chỉ có 1 trang.
- `[TV1]` Hoàn thiện 17 endpoint còn lại của Spec 005 (session, jobs, summary, reports, edit); mở rộng demo xử lý theo lô một chương truyện; tổng hợp tóm tắt xuyên suốt nhiều trang thành tóm tắt chương.

---

## Tài liệu tham khảo — `[TV1]`

Tám tài liệu từ đề cương, **giữ nguyên thứ tự**, mỗi tài liệu ghi rõ vị trí được tham chiếu trong báo cáo:

1. Sachdeva, R., Zisserman, A. (2024). *The Manga Whisperer: Automatically Generating Transcriptions for Comics.* CVPR 2024. — Phần 2.7.
2. Hinami, R., Ishiwatari, S., Yasuda, K., Matsui, Y. (2021). *Towards Fully Automated Manga Translation.* AAAI 2021. — Phần 2.7, 4.6.
3. Del Gobbo, J., Matuk Herrera, R. (2020). *Unconstrained Text Detection in Manga: A New Dataset and Baseline.* ECCVW 2020. — Phần 2.3, 4.1.
4. dmMaze / kha-white. *comic-text-detector: Manga & Comic text detection.* GitHub (fork Ajatt-Tools/comic_text_detector). — Phần 2.4.
5. ContemporaryCat (2025). *Manga-Text-Segmentation (UNet++ EfficientNetV2 pretrained).* GitHub. — Phần 2.5.
6. kha-white. *manga-ocr: Japanese OCR for manga.* GitHub. — Phần 2.7, 4.6.
7. zyddnys. *manga-image-translator.* GitHub — nguồn module render text dùng trong demo. — Phần 3.6.
8. Gonzalez, R. C., Woods, R. E. *Digital Image Processing.* — Phần 2.2.

---

## Danh sách hình vẽ (dự kiến)

| # | Tên hình | Người cấp | Nguồn |
|---|---|---|---|
| 1 | Sơ đồ khối pipeline tổng thể | TV1 | `README.md` |
| 2 | Sơ đồ nhánh baseline cổ điển | TV1 | `src/manga_text_seg/` |
| 3 | Kiến trúc U-Net + ResNet34 | TV4 | `notebooks/deliverables/manga-text-segmentation/` |
| 4 | Kiến trúc comic-text-detector | TV2 | `notebooks/deliverables/comic-text-detector/` |
| 5 | Kiến trúc UNet++ + EfficientNetV2 | TV3 | `notebooks/deliverables/unetpp-efficientnetv2/` |
| 6 | Biểu đồ so sánh IoU/F1 theo phương pháp | TV1 | `deliverables/spec002/charts/accuracy-by-method.png` |
| 7 | Mask dự đoán của các phương pháp trên cùng một trang | TV2 | `outputs/inpainting/spec003-main/boards/*/highest_iou_f1/` |
| 8 | Board so sánh xóa và inpainting | TV2 | `outputs/inpainting/spec003-main/boards/` |
| 9 | Ca thành công điển hình | TV2 | board `highest_iou_f1` |
| 10 | Ca thất bại điển hình (`EvaLady/006`) | TV3 | board `lowest_iou_f1` |
| 11 | Kết quả OCR → dịch → render | TV3 | `outputs/translation/spec004-vilao/` |
| 12 | Giao diện web demo | TV1 | `web-ui/index.html` |

## Danh sách bảng biểu (dự kiến)

| # | Tên bảng | Người cấp | Nguồn |
|---|---|---|---|
| 1 | Tiêu chí đánh giá và mục đích đo lường | TV1 | đề cương, Bảng 2 |
| 2 | Tham số và cấu hình từng phương pháp | CHUNG | `deliverables/spec002/run.json` |
| 3 | Kết quả benchmark định lượng (IoU/P/R/F1/thời gian) | TV1 | `deliverables/spec002/comparison.csv` |
| 4 | Thời gian inpainting theo phương pháp × thuật toán | TV4 | `outputs/inpainting/spec003-main/performance.json` |
| 5 | Năm trang được chọn cho pipeline demo | TV2 | `outputs/inpainting/spec003-main/selection.json` |
| 6 | Cấu hình OCR / dịch / render | TV3 | `outputs/translation/*/run.json` |
| 7 | So sánh trực quan inpainting theo năm tiêu chí | TV2 | đề cương, Bảng 3 |
| 8 | Mức độ đóng góp của các thành viên | TV1 | trang bìa |

---

## 7. Việc PHẢI nói thật — không được che, không được bịa

Đọc mục này trước khi viết bất cứ phần nào. Bốn điểm dưới đây là **điểm mạnh** của báo cáo nếu trình bày trung thực, và là **gian lận học thuật** nếu bị làm mờ.

### 7.1. Manga-Text-Segmentation không có điểm — `[TV4]`

Phương pháp **không chấm được**, lý do là thiếu checkpoint. **Không được** suy diễn từ kết quả Colab thăm dò, **không được** lấy số từ nguồn khác, **không được** bỏ dòng này khỏi bảng. Ghi trạng thái `unavailable` và lý do.

### 7.2. UNet++ EfficientNetV2 nhiễm bẩn hoàn toàn — `[TV3]`

Đây là phương pháp có điểm cao nhất **và** là phương pháp duy nhất có công bố nhiễm bẩn ở mức nặng nhất: checkpoint được huấn luyện trên **toàn bộ 45 cuốn sách** của tập ground truth. Mọi điểm số của nó **là điểm trên tập huấn luyện**. Phải viết câu này ngay cạnh bảng kết quả, không được để ở phụ lục.

### 7.3. Không có ground truth trong bất kỳ hình nào — `[TV2][TV3]`

Mask dự đoán là mask duy nhất được mở. Board, hình so sánh, ảnh minh họa **không được chứa ground truth**. Không có chế độ "GT-assisted".

### 7.4. Quy tắc `manual_artifact_flags` trả về rỗng — `[TV2]`

`selection.json` ghi `manual_artifact_flags` với `shortfall: 5` — tức **không có trang nào được gắn cờ thủ công**. Nếu phần đánh giá định tính có nhắc tới quy tắc này thì phải nói rõ nó chưa được thực hiện, **không được viết như thể đã có kết quả**.

### 7.5. Không ghi giá trị khoá API — `[TV3]`

Run record ghi **tên biến môi trường** (`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `MANGA_TRANSLATE_API_KEY`). **Giá trị khoá không được xuất hiện ở bất kỳ đâu** trong báo cáo, hình, hay phụ lục.

### 7.6. Không nêu tên miền `data/no-need-to-read/` như một phần dữ liệu — `[TV1]`

Thư mục này nằm ngoài phạm vi xử lý. Chỉ nhắc một câu ở Phần 4.1 nếu cần, và nói rõ nó **không được đọc**.

---

## 8. Nhịp làm việc đề xuất

| Mốc | Việc | Ai |
|---|---|---|
| Ngày 1–2 | Phần 2 (mỗi người viết phương pháp của mình) | cả bốn, song song |
| Ngày 2–3 | Phần 3.4 `[TV4]`, 3.5 `[TV2]`, 3.6 `[TV3]` | TV4, TV2, TV3 |
| Ngày 3–4 | Phần 4.4 mỗi người một mục; 4.5.2 board | TV4, TV2, TV3 |
| Ngày 4–5 | Phần 4.6 `[TV3]`; Phần 1 + 4.1 + 4.3 + 4.7 `[TV1]` | TV3, TV1 |
| Ngày 5 | Phần 5 — mỗi người gửi gạch đầu dòng, TV1 gộp | cả bốn |
| Ngày 6 | Mục lục, danh mục, danh sách hình/bảng, định dạng, rà mục 7 | TV1 |
| Ngày 6 | PPT slides | cả bốn, TV1 gộp |

**Chốt trước khi nộp**: rà lại toàn báo cáo xem có câu nào vô tình khẳng định ưu thế của học sâu dựa trên điểm của UNet++ mà chưa kèm cảnh báo nhiễm bẩn. Đó là lỗi dễ mắc nhất và là lỗi nặng nhất.
