# Deliverable: Manga-Text-Segmentation (Method A)

## Model
- Kiến trúc: U-Net, backbone ResNet34 (fastai `unet_learner`).
- Repository: https://github.com/juvian/Manga-Text-Segmentation
- Code license: MIT (xác nhận qua badge + file `LICENSE` trên trang repo).
- Checkpoint dùng: `fold.0.-.final.refined.model.2.pkl`, tải từ GitHub Release `v1.0` của repo trên.
  Không copy checkpoint vào thư mục này — sha256 và nguồn tải đầy đủ nằm trong mỗi file
  `metadata/<manga>/<page>.json`, tự tải lại theo `checkpoint.source` nếu cần verify.
- Weight license: MIT (**suy luận**, không phải xác nhận tường minh từ tác giả — repo chỉ công bố
  1 license MIT cho toàn bộ nội dung, không có license riêng cho binary trong Release).

## Runner không tính metrics (FR-057)

Theo ONBOARDING §7, runner chỉ trả **mask + metadata + error report**, không tính hay nộp
IoU/Precision/Recall/F1 — quy trình đánh giá dùng chung do project chính chạy, để đảm bảo 3 phương
pháp so sánh được với nhau bằng đúng 1 convention. Vì vậy thư mục này **không có**
`metrics_per_page.csv` hay `metrics_summary.json`.

## Kết quả chạy (390/390 page thành công, 0 lỗi)

Không có con số IoU/F1 để báo — theo đúng quy định trên. Thông tin duy nhất về "chất lượng chạy" ở
mức runner là: 390/390 page suy luận thành công, không page nào lỗi. Chi tiết thời gian xử lý từng
page (`inference_time_seconds`, `preprocessing_time_seconds`, `postprocessing_time_seconds`) nằm
trong từng file `metadata/<manga>/<page>.json`.

## Ground-truth — không có trong deliverable này

Runner không đọc, không giữ, không nộp ground-truth mask hay bất kỳ overlay nào so với ground-truth
(ONBOARDING §3, §8) — *"the images are the question and the masks are the answer"*. Việc so khớp
với ground-truth để tính điểm là việc của project chính, thực hiện đúng 1 lần từ mask trả về ở đây.

## Cấu trúc thư mục

```text
masks/<manga>/<page_id>.png          # prediction mask, PNG single-channel uint8, {0,255}
metadata/<manga>/<page_id>.json      # provenance record — xem schema đầy đủ bên dưới
visualizations/<manga>_<page_id>.png # 6 case: 2 tỉ lệ text thấp nhất / trung bình / cao nhất
errors.json                          # rỗng — không có page lỗi trong 390 page
requirements.txt
manga_text_segmentation_colab_Nguyen-Quang-Huy.ipynb
```

Chọn case visualization theo tỉ lệ pixel-được-dự-đoán-là-text thay vì theo IoU (vì không có
ground-truth để tính IoU ở đây) — chỉ nhằm mục đích chọn đa dạng case để xem, không mang ý nghĩa
đúng/sai của prediction.

## Metadata schema — mỗi page ghi đầy đủ

```json
{
  "image_id": "manga/page_id",
  "page_list_identity": "sha256 của benchmark/page-list.json chính thức (b52aa60d…)",
  "input_image_identity": "sha256 của đúng file ảnh .jpg đã đọc cho page này",
  "method": "manga-text-segmentation",
  "repository": "...", "commit": "full 40-char SHA",
  "code_license": "MIT",
  "checkpoint": {"name": "...", "source": "...", "size_bytes": 0, "sha256": "...",
                 "weight_license": "...", "loaded_evidence": "..."},
  "fold_attribution": "fold_0 — KHÔNG ĐẠT check 6 (FR-013a): đây là 1 fold duy nhất, không phải LOFO 5-checkpoint",
  "checkpoint_strategy": "single-fold-fallback (fold.0 only)",
  "input_size": [w, h], "model_raw_output_size": [w, h], "output_size": [1654, 1170],
  "preprocessing": [...], "postprocessing": [...], "threshold": 0.5,
  "alignment": "spec-001-aligned-space",
  "device": {"type": "cuda", "name": "..."},
  "interpreter": "Python 3.8.x", "packages": {"torch": "...", "fastai": "..."},
  "seed": 42, "run_timestamp": "ISO8601",
  "inference_time_seconds": 0.0, "preprocessing_time_seconds": 0.0, "postprocessing_time_seconds": 0.0,
  "status": "success", "error": null
}
```

## Cách chạy lại

1. `bash setup_env.sh` — cài Python 3.8 + venv + clone repo + tải checkpoint + giải nén raw images.
   (Không cần giải nén groundtruth — runner không đọc GT, xem cảnh báo ở trên.)
2. `env38/bin/python run_benchmark.py` — chạy full 390 page, ghi mask + metadata + visualization.

## Việc còn tồn đọng cho người kế thừa

- [ ] Xoá 3 cell đọc ground-truth khỏi notebook đã nộp (cell `test_predict_iou.py`, cell chạy nó, và
      cell hiển thị overlay GT) — xem ghi chú bên dưới.
- [ ] Triển khai LOFO 5-checkpoint (FR-013a) để kết quả method A hợp lệ theo chuẩn hiện tại.
- [ ] Xác nhận tường minh `weight_license` với tác giả repo nếu cần độ chắc chắn cao hơn suy luận hiện tại.

## Ghi chú: 3 cell ground-truth trong notebook đã nộp

Notebook nộp kèm còn 3 cell vi phạm ONBOARDING §8 / FR-057 và **không** được phép có mặt trong
bản bàn giao:

| cell | nội dung | vấn đề |
|---|---|---|
| 3 | `%%writefile test_predict_iou.py` | đọc `groundtruth/post-processed`, tự tính IoU/Precision/Recall/F1, vẽ overlay FP/FN/TP so với GT |
| 4 | chạy `test_predict_iou.py` | sinh ra chính các số liệu đó |
| 5 | hiển thị overlay GT | ảnh prediction-vs-ground-truth |

Mask + metadata + `provenance.json` đã bàn giao **không** bị ảnh hưởng — 3 cell này chỉ là bước
kiểm tra thăm dò chạy rời, không ghi vào `masks/` hay `metadata/`. Nhưng chúng vẫn phải bị xoá khỏi
notebook trước khi tính là hợp lệ, vì `returned-result.md` ghi rõ phần bàn giao phải
*"Deliberately absent: any ground-truth mask, any prediction-vs-ground-truth overlay, any metric"*.

`run_benchmark.py` (cell 6) thì **đúng** — đọc `benchmark/page-list.json` chính thức, không đọc GT.
