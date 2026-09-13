# Deliverable: comic-text-detector (Method B)

## Model
- Kiến trúc: Pipeline kết hợp YOLOv5s backbone, U-Net segmentation head và DBNet text-line detection.
- Repository: https://github.com/Ajatt-Tools/comic_text_detector (fork từ upstream dmMaze/comic-text-detector).
- License: GPL-3.0.
- Checkpoint dùng: `comictextdetector.pt` (~76.25MB / 79,948,869 bytes), tải từ bản release chính thức của `manga-image-translator` (tag `beta-0.2.1`).
  Không lưu checkpoint vào repository — cần tải trực tiếp từ link lưu trữ theo hướng dẫn môi trường.

## Giới hạn quan trọng — ĐỌC TRƯỚC KHI DÙNG SỐ LIỆU
Theo Spec 002 (mục Training-data contamination), một phần tập dữ liệu Manga109-s đã được nhóm tác giả sử dụng trong quá trình huấn luyện pipeline gốc của `comic-text-detector` nhưng danh sách phân chia (split) cụ thể không được công bố công khai.

Hệ quả: Mức độ trùng lặp (contamination) giữa tập huấn luyện của mô hình và bộ 390 trang benchmark của đề tài là không thể xác định chính xác (unknowable).

→ **Khi công bố kết quả so sánh với các phương pháp khác, bắt buộc phải kèm theo ghi chú công bố (disclosure)** rằng điểm số có thể bị ảnh hưởng bởi độ trùng lặp không xác định của dữ liệu huấn luyện (SC-011).

## Kết quả (390/390 page, không có page lỗi)
| Metric | Mean | Std |
|---|---|---|
| IoU | 0.142 | 0.1255 |
| Precision | 0.3372 | 0.2388 |
| Recall | 0.1896 | 0.1567 |
| F1 | 0.2297 | 0.1752 |
| Inference time (s/page) | 0.3703 | 0.035 |

Device: CUDA (NVIDIA Tesla T4 - Google Colab GPU runtime — chi tiết device xem thêm trong metadata JSON từng page).
Chi tiết per-page: xem `metrics_per_page.csv`.

## Manifest / dataset
- Dùng đúng 390 cặp ảnh–mask hợp lệ (giao tên file giữa `data/raw/Manga109s_released_2026_05_21/images` và `data/groundtruth/post-processed`).
- Phát hiện và loại trừ đúng 60 mask mồ côi (orphan masks) thuộc 6 manga không có ảnh gốc theo đúng Spec 001.
- Dữ liệu được giải nén trực tiếp trên môi trường đĩa ảo Colab từ tệp sao lưu Google Drive (`/content/drive/MyDrive/DIP`).

## Preprocessing / output convention
- Tiền xử lý: Letterbox về kích thước 1024x1024 với stride-64 bottom-right zero-padding, chuẩn hóa giá trị pixel bằng tỉ lệ 1/255 (không áp dụng ImageNet mean/std).
- Threshold: Ngưỡng binarization nhị phân áp dụng trên xác suất đầu ra tương đương ~0.235 (ngưỡng nội bộ của mô hình).
- Lựa chọn đầu ra: Chỉ lấy pixel-level segmentation mask (`ret[0]`), loại bỏ polygon dòng chữ và bounding box theo đúng Spec 002 (FR-013, FR-014).
- Aligned output: Đưa mask dự đoán về đúng kích thước Aligned Space chuẩn 1654x1170 (W x H) theo Spec 001.
- Mask output: Định dạng PNG, single-channel, uint8, background=0, text=255.

## Cấu trúc thư mục deliverables
deliverables/comic-text-detector/
├── README.md
├── requirements.txt
├── metrics_per_page.csv
├── metrics_summary.json
├── errors.json                       # Rỗng [] — không có trang lỗi
├── masks/                            # Thư mục chứa 390 prediction masks PNG (0, 255)
│   └── <manga>/<page_id>.png
├── metadata/                         # Thư mục chứa 390 sidecar metadata JSON
│   └── <manga>/<page_id>.json
└── visualizations/                   # Trực quan hóa so sánh đa panel
    ├── representative_cases.png      # 3 ca đại diện: Tốt nhất, Trung bình, Kém nhất
    └── <manga>/<page_id>_*.png

## Cách chạy lại
1. Mở notebook `comic_text_detector_colab_HaDuyAnh.ipynb` trên Google Colab với runtime T4 GPU.
2. Chạy Cell 1 để kết nối Google Drive (`drive.mount('/content/drive')`).
3. Chạy Cell 2 để clone repository `Ajatt-Tools/comic_text_detector`, cài đặt `requirements.txt` và tải weights `comictextdetector.pt`.
4. Chạy Cell 3 để giải nén dataset từ Drive vào ổ đĩa ảo `/content/data/`.
5. Chạy Cell 4 & 5 để duyệt 390 trang hợp lệ, thực hiện suy luận, trích xuất masks và metadata.
6. Chạy các Cell tính toán để xuất `metrics_per_page.csv`, `metrics_summary.json` và bộ ảnh trực quan hóa.

## Việc còn tồn đọng cho người kế thừa
- [ ] So sánh đối chứng kết quả định lượng trên cùng tập trang với Method A (Manga-Text-Segmentation) và Method C (UNet++ EfficientNetV2).
- [ ] Tích hợp mask nhị phân đầu ra vào pipeline inpainting (Spec 3) và OCR/Translation (Spec 4).
