# Deliverable: Manga-Text-Segmentation (Method A)

## Model
- Kiến trúc: U-Net, backbone ResNet34 (fastai `unet_learner`).
- Repository: https://github.com/juvian/Manga-Text-Segmentation
- License: MIT.
- Checkpoint dùng: `fold.0.-.final.refined.model.2.pkl` (~328MB), tải từ GitHub Release `v1.0` của repo trên.
  Không copy checkpoint vào thư mục này — cần tự tải từ link gốc theo `setup_env.sh`.

## Giới hạn quan trọng — ĐỌC TRƯỚC KHI DÙNG SỐ LIỆU
Repo gốc publish 5 checkpoint (5-fold cross-validation). Đúng chuẩn benchmark (Spec 002, FR-013a)
yêu cầu chiến lược leave-one-fold-out: mỗi trang phải được chấm bởi checkpoint của fold KHÔNG
chứa manga đó lúc train.

**Deliverable này chỉ dùng 1 checkpoint (fold.0) cho toàn bộ 390 trang**, do giới hạn thời gian.
Hệ quả: các trang thuộc manga nằm trong tập train của fold 0 bị model "nhìn thấy trước" khi
train, khiến IoU đo được có thể cao hơn khả năng generalize thật của model.

→ **Không dùng IoU = 0.753 dưới đây để so sánh tuyệt đối với method B/C** cho tới khi có bản
chạy đầy đủ LOFO 5-checkpoint. Đây là hạn chế đã biết, không phải lỗi phát sinh ngoài ý muốn.

## Kết quả (390/390 page, không có page lỗi)
| Metric | Mean | Std |
|---|---|---|
| IoU | 0.7531 | 0.1443 |
| Precision | 0.8635 | 0.1190 |
| Recall | 0.8547 | 0.1346 |
| F1 | 0.8499 | 0.1138 |
| Inference time (s/page) | 0.600 | 0.046 |

Device: CUDA (Colab GPU runtime — loại GPU cụ thể xem thêm trong metadata JSON từng page).
Chi tiết per-page: xem `metrics_per_page.csv`.

## Manifest / dataset
- Dùng 390 cặp ảnh–mask hợp lệ (giao tên file giữa `raw/images` và `groundtruth/post-processed`).
- Manifest KHÔNG do Spec 1 cung cấp sẵn (Spec 1 tại thời điểm này chưa có implementation) —
  tự tính bằng cách giao 2 tập tên file, verify count = 390/60 khớp với số liệu trong
  `specs/002-deep-learning-segmentation-benchmark/spec.md`.
- File manifest: `/content/drive/MyDrive/Colab Notebooks/manifest.csv`.

## Preprocessing / output convention
- Threshold: sigmoid cố định 0.5 (không expose thành tham số ở bản gốc repo).
- Chuẩn hóa: ImageNet mean/std (mặc định fastai training pipeline).
- Aligned output: mọi mask (pred + GT) được pad/crop về đúng 1654×1170 trước khi tính metric.
- Mask output: PNG, single-channel, uint8, background=0, text=255.

## Cấu trúc thư mục
outputs/<manga>/<page_id>.png # prediction mask
outputs/<manga>/<page_id>.json # metadata (mask + metadata cùng thư mục, không tách masks/ và metadata/ riêng — quyết định giữ đơn giản, dễ đối chiếu 1-1)
visualizations/ # 6 case: 2 tốt nhất, 2 trung bình, 2 tệ nhất theo IoU
metrics_per_page.csv
metrics_summary.json
errors.json # rỗng — không có page lỗi
requirements.txt
notebook.ipynb


## Cách chạy lại
1. `bash setup_env.sh` (cài Python 3.8 + venv + clone repo + tải checkpoint + giải nén data).
2. Sinh `manifest.csv` bằng script giao tên file raw/gt (xem notebook).
3. `env38/bin/python run_benchmark.py`.

## Việc còn tồn đọng cho người kế thừa
- [ ] Chạy lại với đầy đủ 5 checkpoint theo LOFO (FR-013a) để có IoU không bị contamination.
- [ ] Tách `masks/` và `metadata/` thành 2 thư mục riêng nếu cần khớp 100% mẫu ONBOARDING mục 9.
