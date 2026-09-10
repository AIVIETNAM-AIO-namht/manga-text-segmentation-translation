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
- checkpoint source;
- environment và package versions;
- cách lấy hoặc mount dataset;
- cách chạy thử một vài page;
- cách chạy benchmark;
- cách xuất prediction masks;
- cách xuất metadata sidecar;
- cách tính hoặc xuất metrics;
- cách ghi nhận lỗi;
- cách đóng gói deliverables.

## Không commit vào repository

- dataset;
- checkpoint/weights;
- API key hoặc secret;
- toàn bộ output ảnh lớn;
- cache model;
- file environment chứa secret.

Chi tiết output contract và checklist bàn giao nằm trong `ONBOARDING.md`.
