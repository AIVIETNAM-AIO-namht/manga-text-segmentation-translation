# Team Onboarding

Tài liệu này hướng dẫn thành viên tham gia project đọc specification, chạy model segmentation trên Google Colab và bàn giao kết quả theo cùng một format.

## 1. Đọc trước khi bắt đầu

Thành viên phải đọc theo thứ tự:

1. `README.md`
2. `specs/001-data-foundation-classical-baseline/spec.md`
3. `specs/002-deep-learning-segmentation-benchmark/spec.md`
4. Tài liệu này

Không bắt đầu clone hoặc chạy model trước khi hiểu:

- dataset và ground-truth được ánh xạ như thế nào;
- `image_id` là gì;
- mask được alignment ra sao;
- prediction mask cần format nào;
- metrics nào bắt buộc;
- metadata nào phải bàn giao;
- cách ghi nhận model không chạy được.

## 2. Mục tiêu công việc

Mỗi thành viên phụ trách chạy một phương pháp segmentation pretrained trên Google Colab và bàn giao prediction mask để project chính tổng hợp benchmark.

Phân công dự kiến:

- Thành viên 1: Manga-Text-Segmentation — U-Net với ResNet34.
- Thành viên 2: comic-text-detector.
- Thành viên 3: UNet++ với EfficientNetV2.

Project chính không yêu cầu ba model chạy trong cùng một môi trường Python. Mỗi model có thể sử dụng environment riêng trên Colab. Điều bắt buộc thống nhất là input, image ID, output mask, metrics và metadata.

## 3. Dataset

Dataset không được commit lên GitHub.

Đường dẫn local trong project chính:

```text
Ảnh gốc:
data/raw/Manga109s_released_2026_05_21/images/

Ground-truth:
data/groundtruth/post-processed/
```

Ground-truth hiện có khoảng 450 mask thuộc 45 manga. Cặp ảnh và mask được ánh xạ theo:

```text
<manga>/<page_id>
```

Ví dụ:

```text
Ảnh:  data/raw/Manga109s_released_2026_05_21/images/AisazuNihaIrarenai/000.jpg
Mask: data/groundtruth/post-processed/AisazuNihaIrarenai/000.png
image_id: AisazuNihaIrarenai/000
```

Phải dùng đúng manifest/danh sách page của project. Không tự ý:

- tạo dataset subset khác;
- đổi tên image ID;
- đổi mapping ảnh-mask;
- sửa ground-truth;
- đọc dữ liệu trong `data/no-need-to-read/`.

## 4. Quy trình chạy Colab

### Bước 1 — Chuẩn bị

- Đọc README, Spec 001, Spec 002.
- Xác nhận model được phân công.
- Xác định repository chính thức hoặc repository được project phê duyệt.
- Ghi lại commit/version của repository.
- Xác định checkpoint và license.
- Tạo notebook riêng cho model.

### Bước 2 — Chạy thử

Trước khi chạy toàn bộ dataset:

1. Chạy thử trên một vài page.
2. Kiểm tra prediction mask có đúng page không.
3. Kiểm tra mask có đúng kích thước aligned page không.
4. Kiểm tra mask có đúng binary convention không.
5. Kiểm tra metadata được ghi đầy đủ.
6. Kiểm tra inference time.

Nếu output sai, dừng và sửa trước khi chạy toàn bộ dataset.

### Bước 3 — Chạy benchmark

- Chạy trên toàn bộ page hợp lệ trong manifest.
- Dùng cùng danh sách page giữa các model.
- Không bỏ qua page mà không ghi lý do.
- Nếu một page lỗi, tiếp tục các page khác và ghi lỗi.
- Không tạo output giả cho page/model không chạy được.

## 5. Output mask bắt buộc

Prediction mask phải:

- là PNG;
- single-channel;
- unsigned 8-bit;
- `background = 0`;
- `text = 255`;
- đúng kích thước aligned page `1654 × 1170` theo Spec 001;
- giữ đúng `image_id`;
- không bị resize/crop ngầm định mà không ghi metadata.

Output cần giữ được mapping rõ ràng:

```text
<method>/<manga>/<page_id>.png
```

Mỗi mask phải có metadata sidecar tương ứng:

```text
<method>/<manga>/<page_id>.json
```

## 6. Metadata cần bàn giao

Mỗi page cần ghi tối thiểu:

```json
{
  "image_id": "AisazuNihaIrarenai/000",
  "method": "method_name",
  "repository": "repository_url",
  "commit": "commit_or_version",
  "checkpoint": "checkpoint_name_or_path",
  "input_size": [1654, 1170],
  "output_size": [1654, 1170],
  "preprocessing": [],
  "postprocessing": [],
  "threshold": null,
  "alignment": "spec-001-aligned-space",
  "device": "cuda",
  "inference_time_seconds": 0.0,
  "status": "success",
  "error": null
}
```

Nếu model có preprocessing hoặc padding riêng, phải ghi rõ:

- resize hoặc input size nội bộ;
- padding;
- crop;
- threshold;
- postprocessing;
- cách đưa output về aligned page space.

## 7. Metrics cần bàn giao

Metrics phải được tính trên cùng ground-truth và cùng convention với Spec 001:

- IoU;
- Precision;
- Recall;
- F1-score;
- inference time trung bình trên mỗi page.

Cần bàn giao:

- metrics theo từng page;
- metrics trung bình;
- độ lệch chuẩn nếu có;
- số page thành công;
- số page thất bại;
- danh sách page lỗi.

Không dùng Pixel Accuracy làm metric chính.

## 8. Visualization cần bàn giao

Chọn một số page tiêu biểu, gồm:

- ảnh gốc;
- ground-truth mask;
- prediction mask;
- overlay prediction và ground-truth.

Nên có cả:

- trường hợp tốt;
- trường hợp trung bình;
- trường hợp thất bại.

Không chỉ chọn các page đầu tiên trong manifest.

## 9. Những file cần bàn giao

Mỗi thành viên cần gửi:

- Colab notebook chạy được từ đầu;
- link repository model;
- commit/version đã sử dụng;
- checkpoint source;
- `requirements.txt` hoặc danh sách package/version;
- prediction masks;
- metadata sidecars;
- metrics theo page;
- metrics summary;
- visualization;
- error report;
- ghi chú các vấn đề đã gặp.

Đề xuất cấu trúc bàn giao:

```text
deliverables/
  <method_name>/
    notebook.ipynb
    requirements.txt
    README.md
    metrics_per_page.csv
    metrics_summary.json
    errors.json
    metadata/
    masks/
    visualizations/
```

## 10. License và repository bên ngoài

- Được phép clone repository model về Colab để chạy.
- Không copy toàn bộ source code bên thứ ba vào project chính.
- Không commit checkpoint lớn vào GitHub.
- Ghi rõ repository URL, commit/version và license.
- Không phân phối lại code, weights hoặc dataset nếu license không cho phép.
- Nếu repository không có license rõ ràng, báo lại trước khi đưa code vào project chính.

## 11. Khi model không chạy được

Nếu model lỗi vì dependency, checkpoint, Python version, CUDA hoặc license:

- không tạo output giả;
- ghi rõ lỗi đầy đủ;
- ghi environment đã thử;
- ghi checkpoint/repository đã dùng;
- đề xuất hướng khắc phục;
- đánh dấu model/page là `unavailable` hoặc `failed`.

Một model không chạy được không được làm mất kết quả của các model khác.

## 12. Checklist trước khi bàn giao

- [ ] Đã đọc README, Spec 001 và Spec 002.
- [ ] Dùng đúng dataset và manifest.
- [ ] Không thay đổi ground-truth.
- [ ] Giữ nguyên `image_id`.
- [ ] Prediction mask là PNG single-channel binary.
- [ ] Mask có giá trị `{0, 255}`.
- [ ] Mask đúng kích thước aligned page.
- [ ] Có metadata sidecar cho từng page.
- [ ] Có metrics per-page và summary.
- [ ] Có inference time.
- [ ] Có visualization tốt/thất bại.
- [ ] Có notebook chạy lại được.
- [ ] Có requirements/environment information.
- [ ] Có repository, commit và checkpoint provenance.
- [ ] Có error report.
- [ ] Không có API key, secret hoặc checkpoint lớn trong repository.

## 13. Nguyên tắc quan trọng

> Không tự đoán, không đổi format, không tạo output giả.

Nếu có điểm chưa rõ trong Spec 001 hoặc Spec 002, hãy hỏi trước khi chạy toàn bộ benchmark.
