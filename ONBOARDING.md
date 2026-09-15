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
- **vì sao runner không tính metrics** (§7);
- metadata nào phải bàn giao;
- cách ghi nhận model không chạy được.

## 2. Mục tiêu công việc

Mỗi thành viên phụ trách chạy một phương pháp segmentation pretrained trên Google Colab và bàn giao prediction mask để project chính tổng hợp benchmark.

Phân công dự kiến:

- Thành viên 1: Manga-Text-Segmentation — U-Net với ResNet34.
- Thành viên 2: comic-text-detector.
- Thành viên 3: UNet++ với EfficientNetV2.

Project chính không yêu cầu ba model chạy trong cùng một môi trường Python. Mỗi model có thể sử dụng environment riêng trên Colab.

Vì chạy phân tán, FR-053 chốt **đúng sáu thứ** phải cố định và được tiêu thụ giống hệt nhau ở mọi
method — lệch một thứ là kết quả **không được nhận** vào bảng so sánh:

1. **page list** của benchmark (do project phát — §3);
2. **ảnh input** mà page list trỏ tới;
3. **convention tiền xử lý và alignment** (Spec 001: crop-topleft về 1654×1170, không resize);
4. **format prediction mask** (§5);
5. **quy trình đánh giá** — do project chính chạy, **không phải runner** (§7);
6. **environment metadata** ghi lại được (§6).

Lưu ý điểm 5: runner **không** tạo ra "metrics" như một thứ để thống nhất — runner không tính metrics.

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

FR-054/FR-054a bắt buộc project phát cho runner một **page list tự chứa** (kèm identity) và **ảnh
input** (kèm record identity để project verify khi nhận). Runner **không được tự dựng manifest** bằng
cách giao tên file giữa `images/` và `post-processed/` — làm vậy vi phạm FR-001/002/054/054a và kết
quả bị từ chối, kể cả khi ra đúng 390 page.

> ⚠️ Artefact phân phối này **chưa tồn tại** — nó là output của `/speckit-plan` cho spec 002.
>
> **Bạn vẫn nên chạy full 390 page ngay** để de-risk môi trường và model. Chạy thử vài page **không**
> phát hiện được lỗi chỉ xuất hiện ở page khác (page hỏng, OOM ở page lớn, hết quota giữa chừng).
> Suy luận rẻ (Method A: ~0.6 s/page → vài phút GPU); cái đắt là setup — dependency, checkpoint,
> version — và đó chính là thứ chỉ chạy full mới lộ ra.
>
> Nhưng kết quả đó là **exploratory, KHÔNG admissible**: thiếu identity thì FR-058 từ chối khi nhận.
> Vì vậy: dán nhãn rõ trong README, **không** ghi headline metric, **không** nộp `metrics_*.csv`.
> Khi project phát export, chạy lại (rẻ) và điền 2 field identity — **không cần chờ mới bắt đầu**.
>
> Việc nên làm **sớm**, vì không phụ thuộc export: repo + commit SHA đầy đủ, license code **và**
> license weights, checkpoint sha256, và — riêng Method A — mapping book→fold của FR-013a (§4 Bước 3).
> Mapping đó không reproduce được thì cả 390 page vô hiệu, bất kể identity.

### Ground-truth — runner KHÔNG được nhận

Đường dẫn `data/groundtruth/` ở trên là để **bạn hiểu dataset được ánh xạ thế nào**, không phải để
bạn đọc khi chạy benchmark.

FR-054a và spec 002 (edge case *Ground truth reaching a runner*) nói rõ: page list phân phối cho
runner **không mang pixel ground-truth**, vì *"the images are the question and the masks are the
answer"*. Runner giữ đáp án thì có thể sinh mask dựa vào đáp án đó, và việc chấm điểm chỉ xảy ra một
lần, trong project, từ mask trả về (FR-053, FR-057).

Cụ thể:

- **Không** copy `post-processed/` lên Drive hay vào notebook.
- **Không** đọc GT trong lúc chạy benchmark.
- Chỉ đọc GT (nếu muốn) khi chạy thử vài page để biết model có ra gì — và khi đó dùng đúng convention
  ở §7, không dùng quy tắc tự chế.

### Một điều bạn nên biết trước — contamination

Spec 002 đã **chứng minh từ nguồn gốc** rằng ground-truth của benchmark này **chính là dataset train
đã publish của method A và method C** (45 book trùng khớp hoàn toàn, 450 mask). Nghĩa là hai model đó
đã "nhìn thấy" gần như 100% đáp án khi train; method B thì không xác định được.

Đây **không phải lỗi của bạn** và bạn **không phải sửa gì**:

- **Không** loại page, **không** lấy subset, **không** cố "de-contaminate" — spec nói rõ việc đó ngoài
  phạm vi. Chạy đủ page được phát.
- Project chính ghi nhận và công bố mức contamination cạnh số liệu của từng method (SC-011). Đó là
  việc của report, không phải của runner.
- Riêng method A đã có cơ chế khử riêng: LOFO 5 checkpoint (§4 Bước 3).

## 4. Quy trình chạy Colab

### Bước 1 — Chuẩn bị

- Đọc README, Spec 001, Spec 002.
- Xác nhận model được phân công.
- Xác định repository chính thức hoặc repository được project phê duyệt.
- Ghi lại commit/version của repository.
- Xác định checkpoint và license.
- Tạo notebook riêng cho model.

### Bước 2 — Smoke test rồi chạy full ngay

Suy luận rẻ; setup mới đắt. Nên **đừng** dành thời gian cho một bản chạy thử nhỏ rồi mới chạy full —
hãy chạy full, nhưng kiểm tra output trên đường đi.

Trên 1–2 page đầu, xác nhận nhanh:

1. Prediction mask có đúng page không.
2. Mask có đúng kích thước aligned page không (1654×1170).
3. Mask có đúng binary convention không ({0, 255}).
4. Metadata sidecar có được ghi không.

Rồi chạy tiếp **toàn bộ 390 page**. Đây mới là bài kiểm tra thật: page hỏng, OOM ở page lớn, hết
quota, dependency lệch — không cái nào lộ ra ở vài page.

### Bước 3 — Chạy benchmark

- Chạy trên toàn bộ page hợp lệ trong manifest.
- Dùng cùng danh sách page giữa các model.
- Không bỏ qua page mà không ghi lý do.
- Nếu một page lỗi, tiếp tục các page khác và ghi lỗi.
- Không tạo output giả cho page/model không chạy được.

**Riêng Method A — Manga-Text-Segmentation:** repo gốc publish **5 checkpoint** (5-fold
cross-validation). Spec 002 FR-013a bắt buộc dùng **leave-one-fold-out**: mỗi page phải được chấm bởi
checkpoint của fold **không chứa manga đó** lúc train, và mapping book→fold phải suy ra từ seed 42 đã
công bố của upstream rồi **verify lại** trước khi chấm bất kỳ page nào.

Dùng 1 checkpoint cho cả 390 page **không phải là kết quả yếu hơn — mà là không hợp lệ**. FR-013a nói
rõ: mapping không tái lập và verify được thì kết quả method A *inadmissible — báo `unavailable` kèm lý
do, không được xấp xỉ*. Mỗi page vẫn chỉ suy luận **một lần**; chiến lược này nhân số checkpoint phải
tải, không nhân số lần chạy.

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

Metadata sidecar là **provenance record** (FR-056). Thiếu bất kỳ trường nào dưới đây thì kết quả bị
**từ chối, không được nhận kèm chỗ trống** — không phải cảnh báo, mà là refuse.

Mỗi page cần ghi tối thiểu:

```json
{
  "image_id": "AisazuNihaIrarenai/000",
  "page_list_identity": "identity_cua_page_list_project_phat_ghi_nguyen_van",
  "input_image_identity": "sha256_cua_dung_file_anh_ban_da_doc",
  "method": "method_name",
  "repository": "repository_url",
  "commit": "commit_sha_day_du_40_ky_tu",
  "code_license": "MIT",
  "checkpoint": {
    "name": "fold.0.-.final.refined.model.2.pkl",
    "source": "github_release_url",
    "size_bytes": 343597383,
    "sha256": "hex_64_ky_tu",
    "weight_license": "MIT",
    "loaded_evidence": "log line / hash of loaded state_dict proving the configured checkpoint was the one used"
  },
  "fold_attribution": "fold_0",
  "input_size": [1654, 1170],
  "output_size": [1654, 1170],
  "preprocessing": [],
  "postprocessing": [],
  "threshold": null,
  "alignment": "spec-001-aligned-space",
  "device": {"type": "cuda", "name": "Tesla T4"},
  "interpreter": "Python 3.8.10",
  "packages": {"torch": "1.13.1", "fastai": "2.7.12"},
  "seed": 42,
  "run_timestamp": "2026-09-15T14:03:11+07:00",
  "inference_time_seconds": 0.0,
  "preprocessing_time_seconds": 0.0,
  "postprocessing_time_seconds": 0.0,
  "status": "success",
  "error": null
}
```

Bốn điểm dễ làm sai:

- **`page_list_identity` và `input_image_identity`** — hai field này là điều kiện sống còn. Project
  đối chiếu chúng khi nhận (FR-058) để chứng minh bạn chạy **đúng page list** và **đúng byte ảnh** mà
  classical baseline đã chạy. Lấy identity ở đâu, và `input_image_identity` tính thế nào, do page list
  project phát quy định — **ghi nguyên văn**, không tự chế, không tự băm lại theo cách của bạn. Sai
  hoặc thiếu một trong hai thì kết quả bị từ chối, **kể cả khi mask đúng**.
- **`sha256` của checkpoint** — không phải chỉ tên file. `size_bytes` một mình không đủ (FR-044).
- **`loaded_evidence`** — bằng chứng *dương* rằng checkpoint đã cấu hình thật sự được load, không
  phải suy đoán từ việc code chạy không lỗi (FR-018b).
- **`code_license` và `weight_license` là hai trường riêng** — repo MIT nhưng weights có thể khác
  (FR-056, ONBOARDING §10).
- **`packages`** — version thật của môi trường đã chạy, không phải version trong `requirements.txt`
  (FR-044: ghi theo từng method, từ chính môi trường đã sinh ra kết quả đó).

**Method A (Manga-Text-Segmentation) thêm `fold_attribution`** — mỗi page phải ghi rõ checkpoint của
fold nào đã chấm nó. FR-058 từ chối kết quả method A thiếu hoặc không nhất quán trường này.

Nếu model có preprocessing hoặc padding riêng, phải ghi rõ:

- resize hoặc input size nội bộ;
- padding;
- crop;
- threshold;
- postprocessing;
- cách đưa output về aligned page space.

## 7. Metrics — runner KHÔNG tính

> ⚠️ **Đọc kỹ mục này. Đây là chỗ dễ sai nhất.**

FR-057: metrics **phải** được tính bằng một quy trình đánh giá dùng chung **bên trong project chính**.
Metrics **KHÔNG ĐƯỢC** tính trong môi trường của runner và **KHÔNG ĐƯỢC** chép lại từ số của runner.

> *"a runner returns prediction masks and provenance, not scores."*

Nghĩa là runner **không nộp** `metrics_per_page.csv` hay `metrics_summary.json`, và **không** tự tính
IoU/Precision/Recall/F1 — dù chạy thử có đọc GT để debug (§3).

**Runner nộp:**

- prediction masks (đủ 390 page, đúng format §5);
- metadata sidecar từng page (§6);
- error report;
- visualization.

**Project chính tính:** IoU, Precision, Recall, F1, mọi con số tổng hợp. Runner **không cần** ground-truth
để hoàn thành phần việc của mình.
Điều này nghe ngược với bản ONBOARDING trước (bản cũ yêu cầu nộp metrics). Lý do: nếu mỗi runner tự
tính bằng ground-truth và convention riêng của mình, ba bộ số không so sánh được với nhau — đó chính
là điều benchmark này tồn tại để tránh.

**Được phép (không bắt buộc):** tự tính IoU trên vài page trong lúc chạy thử để biết model có ra gì
không. Số đó chỉ để bạn debug — **không đưa vào deliverable, không trích dẫn như kết quả benchmark.**

Không dùng Pixel Accuracy làm metric chính.

Thời gian suy luận: ghi **per-page trong metadata sidecar** (`inference_time_seconds`,
`preprocessing_time_seconds`, `postprocessing_time_seconds`), kèm `device` có tên GPU cụ thể
(FR-060 — `"cuda"` không đủ). Project sẽ tổng hợp; bạn không tổng hợp.

### Ground-truth convention (chỉ cần nếu bạn tự tính để debug)

GT phải chuyển sang binary bằng `normalize.normalize_mask` của Spec 001 (KMeans k=2, cluster **ít
pixel hơn** = text). **Không** dùng `~np.all(gt_rgb == 255)` ("cái gì không trắng thì là text") — quy
tắc đó cho ra mask khác biệt trên 308/390 page và làm số liệu không so sánh được với Spec 001.

## 8. Visualization cần bàn giao

Chọn một số page tiêu biểu, gồm:

- ảnh gốc;
- prediction mask;
- overlay prediction lên ảnh gốc.

**Không** nộp ground-truth mask và **không** nộp overlay prediction-vs-ground-truth — runner không giữ
ground-truth (§3), nên hai thứ đó không thể là deliverable của runner. Phần so sánh với ground-truth do
**project chính** ghép khi chấm điểm (§7), từ mask bạn trả về.

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
- checkpoint source + checksum;
- license của code và của weights;
- `requirements.txt` hoặc danh sách package/version;
- **prediction masks (đủ page, bắt buộc — xem cảnh báo dưới)**;
- metadata sidecars;
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
    errors.json
    metadata/
    masks/
    visualizations/
```

> ⚠️ **Prediction masks là deliverable bắt buộc, không được để trên Google Drive.**
>
> `notebooks/README.md` cấm commit *"toàn bộ output ảnh lớn"* — lệnh cấm đó nói về ảnh trung gian,
> ảnh debug, cache; **không áp dụng cho prediction masks**. Mask là thứ duy nhất runner phải trả về
> (FR-057), là input của Spec 3 (FR-043), và nếu thiếu thì project không thể chấm điểm — số liệu của
> bạn thành vô nghĩa.
>
> Mask PNG single-channel nhị phân nén rất tốt (390 page ≈ vài MB), commit thoải mái. Nếu vẫn quá
> nặng, nén thành một file `.zip` và commit file nén đó — nhưng **phải nằm trong repo**, không phải
> link Drive.
>
> **Không nộp** `metrics_per_page.csv` / `metrics_summary.json` — xem §7.

Layout mask + metadata: đặt cạnh nhau trong cùng thư mục cũng được, tách `masks/` và `metadata/` riêng
cũng được — cả hai đều hợp lệ. Chỉ cần giữ đúng mapping `<method>/<manga>/<page_id>.png`.

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
- [ ] Dùng đúng page list do project phát (không tự dựng manifest).
- [ ] Không thay đổi ground-truth.
- [ ] Giữ nguyên `image_id`.
- [ ] Prediction mask là PNG single-channel binary.
- [ ] Mask có giá trị `{0, 255}`.
- [ ] Mask đúng kích thước aligned page.
- [ ] Mask đã nằm **trong repo**, không phải link Drive.
- [ ] Có metadata sidecar cho từng page.
- [ ] Mỗi sidecar có `page_list_identity` và `input_image_identity` ghi nguyên văn từ page list (§6).
- [ ] Metadata có checksum checkpoint, license code + weights, package versions, seed, timestamp.
- [ ] Có ghi inference time per-page kèm tên GPU cụ thể.
- [ ] **Không** nộp metrics tự tính (FR-057).
- [ ] Visualization **không** chứa ground-truth mask hay overlay với ground-truth (§8).
- [ ] Có visualization tốt/trung bình/thất bại.
- [ ] Có notebook chạy lại được.
- [ ] Có requirements/environment information.
- [ ] Có repository, commit và checkpoint provenance.
- [ ] Có error report.
- [ ] Không có API key, secret hoặc checkpoint lớn trong repository.
- [ ] *(Method A)* Mỗi page ghi rõ `fold_attribution`; đã dùng đủ 5 checkpoint LOFO.

## 13. Nguyên tắc quan trọng

> Không tự đoán, không đổi format, không tạo output giả.

Nếu có điểm chưa rõ trong Spec 001 hoặc Spec 002, hãy hỏi trước khi chạy toàn bộ benchmark.
