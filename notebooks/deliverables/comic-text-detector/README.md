# Deliverable: comic-text-detector (Method B)

## Model & Provenance
- **Kiến trúc:** Pipeline tích hợp YOLOv5s backbone, U-Net segmentation head và DBNet text-line detection.
- **Repository:** https://github.com/Ajatt-Tools/comic_text_detector
- **Commit:** e958d8b4a7461528e8df8aa8156e3928b57b2fc6
- **Code License:** GPL-3.0
- **Checkpoint:** `comictextdetector.pt`
  - Source: https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/comictextdetector.pt
  - Dung lượng: 79948869 bytes (~76.25 MB)
  - SHA256: 1f90fa60aeeb1eb82e2ac1167a66bf139a8a61b8780acd351ead55268540cccb
  - Weight License: GPL-3.0
  - Evidence: File size và SHA256 checksum được kiểm tra runtime hợp lệ trước khi suy luận.

## Giới hạn Contamination (Spec 002)
Tập dữ liệu huấn luyện công bố của `comic-text-detector` có trích xuất từ Manga109-s nhưng không công bố phân chia train/val/test cụ thể. Theo quy định Spec 002, runner KHÔNG tự ý loại bỏ trang hoặc can thiệp dữ liệu (chạy đủ 390 trang). Điểm số và mức độ contamination sẽ do project chính công bố đối chứng theo tiêu chí SC-011.

## Đánh giá & Metrics
- **Quy định FR-057:** Runner **KHÔNG TÍNH VÀ KHÔNG NỘP METRICS** (`metrics_per_page.csv`, `metrics_summary.json`). Toàn bộ metrics (IoU, Precision, Recall, F1) sẽ do quy trình đánh giá dùng chung của project chính tính toán tập trung từ các prediction mask bàn giao.
- **Tổng số trang xử lý thành công:** 140/390 trang (0 trang lỗi).
- **Thiết bị suy luận:** Tesla T4 (loại: cuda).
- **Thời gian suy luận per-page:** Được ghi nhận chi tiết tại từng sidecar JSON (gồm `inference_time_seconds`, `preprocessing_time_seconds`, `postprocessing_time_seconds`).

## Preprocessing & Output Convention
- **Tiền xử lý:** Letterbox về kích thước 1024x1024 với stride-64 bottom-right zero-padding, scale 1/255.
- **Binarization:** Sigmoid threshold 0.235 (ngưỡng nội bộ của mô hình).
- **Lựa chọn đầu ra:** Chỉ trích xuất pixel segmentation mask (`ret[0]`), loại bỏ text line boxes.
- **Aligned Space:** Resize nearest-neighbor về kích thước chuẩn 1654 x 1170 (Width x Height).
- **Mask Format:** PNG single-channel, uint8, background = 0, text = 255.
- **Metadata Sidecar:** Mỗi mask có một file JSON sidecar đi kèm ghi đầy đủ 2 trường sống còn: `page_list_identity` và `input_image_identity` (SHA256 của ảnh đầu vào).
- **Visualization:** Chỉ bàn giao ảnh gốc, prediction mask và overlay prediction lên ảnh gốc (không nộp ground-truth hay overlay GT).

## Cấu trúc Deliverables
```text
deliverables/comic-text-detector/
├── README.md
├── requirements.txt
├── errors.json                 # Rỗng [] — không có trang lỗi
├── masks/                      # 140 masks PNG 1654x1170 {0, 255}
│   └── <manga>/<page_id>.png
├── metadata/                   # 140 metadata JSON sidecar
│   └── <manga>/<page_id>.json
└── visualizations/             # Ảnh minh họa không dùng GT
    ├── representative_cases.png
    └── <manga>/<page_id>_overlay.png
```
