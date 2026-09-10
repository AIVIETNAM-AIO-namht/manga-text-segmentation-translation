# Manga Text Segmentation and Translation

Đồ án môn Xử lý ảnh số về phân đoạn văn bản trong ảnh Manga, xóa và phục hồi vùng văn bản, OCR, dịch và chèn bản dịch trở lại ảnh.

## Mục tiêu

Project so sánh một baseline xử lý ảnh cổ điển với ba phương pháp deep learning pretrained:

- Classical baseline: Otsu, Adaptive Threshold, MSER, edge detection và morphology.
- Manga-Text-Segmentation: U-Net với backbone ResNet34.
- comic-text-detector: pipeline YOLOv5, DBNet và U-Net.
- UNet++ với backbone EfficientNetV2.

Pipeline tổng thể:

```text
Dataset
  -> Text segmentation
  -> Text removal and inpainting
  -> Japanese OCR
  -> Translation
  -> Text rendering
  -> Summary and Streamlit demo
```

## Trạng thái hiện tại

Năm feature specification đã được tạo:

1. `specs/001-data-foundation-classical-baseline`
2. `specs/002-deep-learning-segmentation-benchmark`
3. `specs/003-text-removal-inpainting`
4. `specs/004-ocr-translation-rendering`
5. `specs/005-streamlit-e2e-reporting`

Các specification mô tả yêu cầu và contract giữa các feature. Source code đang được triển khai sau giai đoạn plan và task breakdown.

## Dữ liệu

Dataset không được commit lên GitHub.

Các đường dẫn local hiện tại:

```text
Ảnh gốc:
data/raw/Manga109s_released_2026_05_21/images/

Ground-truth mask:
data/groundtruth/post-processed/
```

Ground-truth hiện có khoảng 450 mask thuộc 45 manga, được tổ chức theo tên manga và số trang. Ảnh gốc và mask được ánh xạ theo dạng:

```text
<manga>/<page_id>
```

Ví dụ:

```text
data/raw/Manga109s_released_2026_05_21/images/AisazuNihaIrarenai/000.jpg
data/groundtruth/post-processed/AisazuNihaIrarenai/000.png
```

Thư mục `data/no-need-to-read/` nằm ngoài phạm vi xử lý mặc định.

## Làm việc với team

Ba model deep learning được chạy độc lập trên Google Colab. Thành viên mới cần đọc
[`ONBOARDING.md`](ONBOARDING.md) trước khi clone model hoặc chạy notebook.

README chỉ ghi overview và output contract; quy trình Colab chi tiết, checklist bàn giao,
mask convention, metadata và cách xử lý model lỗi nằm trong `ONBOARDING.md`.

## Output contract cơ bản

Các output phải giữ nguyên `image_id` và không ghi đè dữ liệu gốc.

```text
outputs/
  segmentation/
    <run_id>/
      <method>/
        <manga>/
          <page_id>.png
          <page_id>.json
```

Metadata tối thiểu nên bao gồm:

- `image_id`;
- method/model;
- repository và commit;
- checkpoint;
- input size;
- preprocessing;
- threshold;
- alignment;
- device;
- inference time;
- status và error nếu có.

## Streamlit demo

Demo end-to-end chạy local trên project chính và sử dụng các artifact đã được tạo từ Spec 2–4. MVP không chạy deep-learning segmentation model trực tiếp trong Streamlit.

Khi đã có source code:

```bash
streamlit run app.py
```

Demo cho phép:

- chọn manga/page từ manifest;
- chọn segmentation method;
- chọn inpainting run;
- xem ảnh gốc, mask và ảnh inpainting;
- xem OCR và bản dịch theo từng region;
- chỉnh sửa OCR hoặc bản dịch;
- render lại ảnh;
- xem page summary;
- tải ảnh kết quả và metadata.

Upload ảnh manga mới là chức năng mở rộng vì ảnh mới cần chạy segmentation inference trước khi có prediction mask.

## Quy trình phát triển

```text
1. Hoàn thiện specifications
2. Chạy clarify/checklist/analyze
3. Chạy plan cho từng feature
4. Chạy tasks để chia việc
5. Team chạy model trên Colab
6. Implement data foundation và evaluation
7. Nhận prediction masks từ Colab
8. Implement inpainting, OCR, translation và rendering
9. Chạy Streamlit demo
10. Tạo báo cáo định lượng và định tính
```

## Bảo mật và dữ liệu lớn

Không commit các file sau lên repository:

- dataset và file `.zip`;
- pretrained checkpoint;
- API key hoặc secret;
- file `.env`;
- virtual environment;
- output ảnh lớn và cache.

API key translation/summary phải được cung cấp qua environment variable hoặc secret configuration bên ngoài repository.

## Dataset acknowledgement

Khi công bố kết quả, cần ghi rõ việc sử dụng Manga109-s, tuân thủ điều kiện sử dụng dataset và trích dẫn tài liệu liên quan. Không phân phối lại dataset hoặc công khai số lượng whole pages vượt quá giới hạn được quy định bởi dataset.
