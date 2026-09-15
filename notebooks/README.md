# Notebooks

Thư mục này chứa các notebook phục vụ kiểm tra dataset, chạy thực nghiệm và bàn giao kết quả từ Google Colab.

## Quy ước đặt tên

Notebook chạy model nên đặt tên theo mẫu:

```text
<method>_colab_<member>.ipynb
```

Ví dụ:

```text
manga_text_segmentation_colab_member01.ipynb
comic_text_detector_colab_member02.ipynb
unetpp_efficientnetv2_colab_member03.ipynb
```

## Trước khi chạy

Đọc các tài liệu sau:

1. `README.md`
2. `ONBOARDING.md`
3. `specs/001-data-foundation-classical-baseline/spec.md`
4. `specs/002-deep-learning-segmentation-benchmark/spec.md`

## Notebook cần có

Mỗi notebook nên bao gồm:

- thông tin model và repository;
- commit/version;
- checkpoint source + checksum;
- license của code và của weights;
- environment và package versions;
- cách lấy page list do project phát (không tự dựng manifest);
- cách mount dataset;
- cách chạy full 390 page và kiểm tra output trên đường đi;
- cách chạy benchmark;
- cách xuất prediction masks;
- cách xuất metadata sidecar;
- cách ghi nhận lỗi;
- cách đóng gói deliverables.

**Không tính metrics trong notebook.** Metrics do project chính tính bằng quy trình dùng chung
(FR-057). Xem `ONBOARDING.md` §7.

**Không đọc ground-truth khi chạy benchmark**, và **không** đưa ground-truth vào visualization hay
bất kỳ deliverable nào. Runner nhận ảnh, không nhận đáp án. Xem `ONBOARDING.md` §3 và §8.

## Không commit vào repository

- dataset;
- checkpoint/weights;
- API key hoặc secret;
- ảnh trung gian, ảnh debug, cache model;
- file environment chứa secret.

**Prediction masks thì PHẢI commit** — đó là deliverable chính, không phải "output ảnh lớn". Mask
PNG nhị phân nén rất tốt. Xem `ONBOARDING.md` §9.

Chi tiết output contract và checklist bàn giao nằm trong `ONBOARDING.md`.
