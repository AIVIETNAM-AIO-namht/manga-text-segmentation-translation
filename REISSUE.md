# Re-issue: Method A chưa qua được FR-058

Tài liệu này là **trạng thái hiện tại** của ba hand-off trong `notebooks/deliverables/`, đối chiếu
với đúng cửa mà project dùng khi nhận bài (`receipt.validate_handoff`, FR-058).

**Cập nhật 2026-09-27 — ba bạn đã gửi lại, kiểm tra lại toàn bộ:**

| Method | Kết quả self-check | Trạng thái |
|---|---|---|
| B — `comic-text-detector` | **0 vấn đề** | ✅ sẵn sàng admit |
| C — `unetpp-efficientnetv2` | **0 vấn đề** | ✅ sẵn sàng admit |
| A — `manga-text-segmentation` | **320 vấn đề** (toàn bộ là `fold_attribution`) | ❌ còn chặn |

Mọi lỗi ở mục 0 và mục 2–3 của bản re-issue trước **đã được sửa hết**. Method A giờ chỉ còn **một**
việc: FR-013a. Xem mục 1.

Đây không phải đánh giá chất lượng model. Mask của cả ba **đúng convention**: single-channel `uint8`,
shape `(1170, 1654)`, giá trị `{0, 255}` — check 3 và check 4 pass toàn bộ 390 page ở cả ba hand-off.

---

## 0. Ba lỗi chung — đã sửa xong ở cả ba

Ghi lại để đối chiếu; **không cần làm gì thêm**.

| Lỗi (bản trước) | Trạng thái hiện tại |
|---|---|
| 0.1 `page_list_identity` ghi sai | ✅ cả ba ghi đúng `b52aa60d…`, đủ 390/390 page |
| 0.2 Thiếu `provenance.json` ở gốc hand-off | ✅ cả ba có, đủ **11/11** key bắt buộc, `method` khớp registry key |
| 0.3 Thư mục `UchiNoNyan_sDiary` (10 page mất) | ✅ cả ba đã đổi thành `UchiNoNyan'sDiary`, 39 thư mục, 390 file mỗi bên |

Refusal cũ, giữ lại để nhận diện nếu có hồi quy:

```text
[page-list-identity] ARMS/000: page list identity '39a6a4f8…' is not the identity of the
committed export (b52aa60d…); a result produced against another list is refused, not aligned by name
[provenance] no readable provenance.json in the hand-off
[page-list-identity] UchiNoNyan'sDiary/000: page list identity None is not the identity…
```

---

## 1. Method A — `manga-text-segmentation` — **VIỆC DUY NHẤT CÒN LẠI**

**Phải chạy lại inference.** Mọi thứ khác đã đúng; chỉ FR-013a chưa.

### Đã đúng (giữ nguyên)

- `repository` = `https://github.com/juvian/Manga-Text-Segmentation` — khớp `configs/dl.json`
- `commit` = `de8f148c78978d70ad0e0ae3242566da6d70f3a5` — khớp
- `code_license` = `MIT`, `threshold` = `0.5` — khớp
- `checkpoint.sha256` = `fa1d074002955c66f7df32ff60f6602d4aa21f1e1c1ca158dcb852b7e9eeb623` — khớp fold.0 đã pin
- `input_size` / `output_size` = `[1654, 1170]` — đúng
- `provenance.json` ghi `"method": "manga-text-segmentation"`, `checkpoint.identity` =
  `fold.0.-.final.refined.model.2.pkl`, `interpreter` = `Python 3.8.20` — đúng
- Mask: 390 file, `L`, `(1170, 1654)`, `{0, 255}` — đúng
- Check 1, 2, 3, 4, 5 **pass 390/390**. Tắt `check_folds` thì self-check báo **0 vấn đề**.

### Phải sửa: FR-013a — `fold_attribution` sai 320/390 page

Sidecar đang ghi `"fold_attribution": "fold_0"` cho **cả 390 page**, kèm dòng tự khai:

```json
"checkpoint_strategy": "single-fold-fallback (fold.0 only) — LOFO 5-checkpoint chưa triển khai, quyết định của runner"
```

FR-013a không cho phép điều đó. Upstream phát hành 5 checkpoint, mỗi checkpoint giữ lại đúng một
nhóm sách. Một page chỉ là **held-out prediction** khi nó được chấm bằng checkpoint của fold đã giữ
sách đó lại. Chỉ **70 page** thuộc 7 sách của fold 0 là đúng; **320 page còn lại** đang khai sai fold
— và sai theo hướng nguy hiểm: checkpoint fold 0 đã **train trên** sách của chúng, nên đó không phải
dự đoán held-out.

Refusal thực tế (lấy từ `receipt.validate_handoff`, đúng thông điệp code sinh ra):

```text
[fold-attribution] ARMS/000: scored by fold 0, but this page's book was held out of fold 2 —
the fold 0 checkpoint trained on it, so the page is not a held-out prediction (FR-013a)
```

**Việc phải làm:** lấy đủ **5 checkpoint** từ release v1.0 của upstream, chạy cả 5, và ghi
`fold_attribution` **theo từng sách** như bảng dưới. Bảng này project đã dựng lại và kiểm chứng được
từ seed công bố — nó khớp **chính xác** với `derive_book_folds()` trong
`src/manga_text_seg/adapters/manga_text_segmentation.py`, nên nó là thứ hai bên phải khớp, không phải
thứ bạn tự chọn:

| Fold | Số sách | Sách | Số page |
|---|---|---|---|
| `fold_0` | 7 | AosugiruHaru, BakuretsuKungFuGirl, HanzaiKousyouninMinegishiEitarou, HaruichibanNoFukukoro, TouyouKidan, YasasiiAkuma, YumeNoKayoiji | 70 |
| `fold_1` | 9 | Akuhamu, Arisa, Count3DeKimeteAgeru, Donburakokko, EienNoWith, EverydayOsakanaChan, Hamlet, ToutaMairimasu, UnbalanceTokyo | 90 |
| `fold_2` | 8 | ARMS, AppareKappore, DualJustice, HarukaRefrain, UchiNoNyan'sDiary, UchuKigekiM774, UltraEleven, YoumaKourin | 80 |
| `fold_3` | 8 | AisazuNihaIrarenai, AkkeraKanjinchou, BurariTessenTorimonocho, GakuenNoise, GinNoChimera, WarewareHaOniDearu, YamatoNoHane, YumeiroCooking | 80 |
| `fold_4` | 7 | BEMADER_P, DollGun, EvaLady, GarakutayaManta, HealingPlanet, TsubasaNoKioku, YukiNoFuruMachi | 70 |

Cả 39 sách đều có fold xác định được, nên không page nào phải bỏ. 70 + 90 + 80 + 80 + 70 = 390.

> Upstream có 45 sách; 6 sách không nằm trong benchmark này (Belmondo, BokuHaSitatakaKun,
> ByebyeC-BOY, GOOD_KISS_Ver2, YouchienBoueigumi, TotteokiNoABC) nên bị lọc ra. Đó là lý do bảng có
> 39 dòng chứ không phải 45.

**Nếu không lấy được đủ 5 checkpoint:** báo lại project. FR-013a nói rõ một mapping không reproduce
được thì Method A **inadmissible** — báo là unavailable kèm lý do, **không** được lấy fold 0 chấm
thay. Đây là kết cục hợp lệ; ghi `fold_0` cho tất cả thì không.

### Ghi chú nhỏ — không chặn admit, nhưng nên sửa cùng lượt

- `metadata/<manga>/<stem>.json` → `checkpoint.name` vẫn là `"model.pkl"` (tên file local). Nên ghi
  đúng tên asset đã pin: `fold.<n>.-.final.refined.model.2.pkl` — FR-044 nói tên file *là* identity.
  (README của A dòng 7 đã ghi đúng tên này, nên README và sidecar đang nói khác nhau.)
- `checkpoint_load_evidence` của A còn yếu: `sha256(first_layer_weights)=…; total_params=41221068`.
  Mức C đang làm (mục 3) là mức project muốn.
- `checkpoint` trong `provenance.json` là checkpoint nào? Với LOFO, một hand-off dùng 5 checkpoint.
  Ghi checkpoint của fold được dùng nhiều nhất, và mô tả đủ 5 trong `checkpoint_load_evidence.detail`
  (kèm sha256 của từng cái bạn đã hash được). Project chỉ pin sẵn sha256 của fold 0; bốn cái còn lại
  ghi giá trị **quan sát được**, không đoán.
- **3 cell ground-truth trong notebook đã nộp** (xem mục 6). Mask + metadata + `provenance.json` đã
  bàn giao **không** bị ảnh hưởng — 3 cell này chạy rời, không ghi vào `masks/` hay `metadata/`. Nhưng
  chúng vẫn phải bị xoá khỏi notebook trước khi tính là hợp lệ, vì `returned-result.md` ghi rõ phần
  bàn giao phải *"Deliberately absent: any ground-truth mask, any prediction-vs-ground-truth overlay,
  any metric"*.
- File `outputs-20260915T160219Z-1-001.zip` đã được chuyển khỏi thư mục A (nay nằm trong `tmp/`,
  gitignored). Không cần làm gì thêm — mask đã nằm rời trong `masks/` và đã được track.

---

## 2. Method B — `comic-text-detector` — ✅ đã sửa xong

**Không cần chạy lại model. Không cần sửa gì.** Self-check: **0 vấn đề**.

Đã sửa so với bản trước:

| | Bản trước | Hiện tại |
|---|---|---|
| URL | `https://github.com/Ajatt-Tools/comic_text_detector` (fork) | `https://github.com/dmMaze/comic-text-detector` ✅ khớp pin |
| Revision | `e958d8b4…` | `440b978563c71b758e31aaa315d100faba1efa2f` ✅ khớp pin |
| `packages.torchvision` | `"default"` | `0.26.0+cu128` ✅ version thật |
| `interpreter` | `"Python 3.10"` | `Python 3.13.15` ✅ |

Đã đúng từ trước, giữ nguyên: `checkpoint.sha256` = `1f90fa60…` khớp pin, `size_bytes` = `79948869`
khớp pin, `code_license` = `GPL-3.0`, `weight_license` = `GPL-3.0`, `threshold` = `0.235`,
`input_size`/`output_size` = `[1654, 1170]`, 390 mask + 390 sidecar không thừa không thiếu.

### Ghi chú nhỏ — không chặn admit

- `postprocessing` còn khai `'resize_nearest_to_aligned_space'`, và README dòng 29 ghi
  *"Aligned Space: Resize nearest-neighbor về kích thước chuẩn 1654 x 1170"*. Trên tập dữ liệu này
  **toàn bộ ảnh raw đã đúng 1654×1170**, nên nhánh resize **không bao giờ chạy** — đây là mô tả thừa,
  không phải lỗi kết quả. Sửa cho khớp thực tế khi có dịp chạy lại.
- Cell 4 của notebook **suy ra page list từ thư mục ground-truth** (liệt kê tên file mask, giao với
  ảnh gốc) thay vì đọc `benchmark/page-list.json`. Kết quả ra **đúng 390 trang** và
  `page_list_identity` trong sidecar là giá trị chính thức — nhưng cách suy ra thì không đúng hợp
  đồng: page list phải được **đọc nguyên văn** từ manifest, không được dựng lại. Đây là **lỗi phương
  pháp, không phải lỗi kết quả** — mask bàn giao hợp lệ, **không cần chạy lại**. Một lần chạy mới nên
  đọc thẳng manifest.
- Cell 5 patch sidecar sau khi chạy (`page_list_identity`, `repository`, `commit`,
  `packages.torchvision`, `interpreter`) — giá trị ghi vào là hằng số đã pin, khớp giá trị quan sát
  được, nên không có gì bị sửa lặng lẽ; nhưng lần chạy mới nên đúng ngay từ cell 2.

---

## 3. Method C — `unetpp-efficientnetv2` — ✅ đã sửa xong

**Không cần chạy lại model. Không cần sửa gì.** Self-check: **0 vấn đề**.

Đã sửa so với bản trước:

| | Bản trước | Hiện tại |
|---|---|---|
| Page list | tự dựng từ ground-truth, **925 mask** (thừa 545) | ✅ đúng **390 mask + 390 sidecar** |
| `page_list_identity` | `DERIVED_FROM_LOCAL_GROUNDTRUTH` | ✅ `b52aa60d…` |
| `errors.json` `page_list_source` | `"ground-truth name-matched discovery"` | ✅ `"project-issued"`, `page_count: 390`, `error_count: 0` |
| `input_size`/`output_size` | `[1170, 1654]` (đảo) | ✅ `[1654, 1170]` |
| `method` | `UNetPP_EfficientNetV2` | ✅ `unetpp-efficientnetv2` |
| `code_license` | `NOT_STATED_BY_UPSTREAM_REPOSITORY` | ✅ `none` |
| `weight_license` | `NOT_VERIFIED__VERIFY_HUGGING_FACE_MODEL_CARD` | ✅ `none` |
| `repository` | có đuôi `.git` | ✅ `https://github.com/ContemporaryCat/Manga-Text-Segmentation` |
| 10 page `UchiNoNyan'sDiary` | thiếu | ✅ đủ |

Còn một điểm vặt, không chặn admit: `provenance.json` của C ghi `"interpreter": "3.13.15"`, thiếu tiền
tố `"Python "` mà A (`"Python 3.8.20"`) và B (`"Python 3.13.15"`) đều có. Cửa FR-058 chỉ kiểm tra key
có mặt, không kiểm tra định dạng, nên nó qua — nhưng ba hand-off nên ghi giống nhau.

### Điểm làm tốt — nên giữ làm mẫu

`checkpoint.loaded_evidence` của C là bằng chứng dương đúng nghĩa FR-018a: **1396/1396 tensor key
khớp bitwise**, kèm `loaded_param_fingerprint_sha256` và `fixed_probe_output_sha256`. Đây chính xác
là thứ project muốn, và nó quan trọng nhất ở Method C — upstream của C **nuốt lỗi thiếu checkpoint,
in warning rồi chạy tiếp với decoder khởi tạo ngẫu nhiên**, vẫn ra mask trông hợp lệ. A nên nâng
`checkpoint_load_evidence` lên mức này.

---

## 4. Tóm tắt việc phải làm

| # | Việc | A | B | C |
|---|---|---|---|---|
| 1 | Sửa `page_list_identity` thành `b52aa60d…` | ✅ xong | ✅ xong | ✅ xong |
| 2 | Thêm `provenance.json` ở gốc hand-off | ✅ xong | ✅ xong | ✅ xong |
| 3 | Đổi `UchiNoNyan_sDiary` → `UchiNoNyan'sDiary` | ✅ xong | ✅ xong | ✅ xong |
| 4 | Sửa `fold_attribution` theo sách (FR-013a) | ❌ **CÒN** | — | — |
| 5 | Dùng repo/commit đã pin | ✅ xong | ✅ xong | ✅ xong |
| 6 | Cắt về đúng 390 page | ✅ xong | ✅ xong | ✅ xong |
| 7 | Sửa `input_size`/`output_size` thành `[1654, 1170]` | ✅ xong | ✅ xong | ✅ xong |
| 8 | Sửa `method` → tên registry | ✅ xong | ✅ xong | ✅ xong |
| 9 | Giải quyết `code_license` / `weight_license` | ✅ xong | ✅ xong | ✅ xong |
| 10 | Ghi version thật vào `packages` | ✅ xong | ✅ xong | ✅ xong |
| 11 | `checkpoint.name` = tên asset đã pin | ⚠️ nên sửa | ✅ | ✅ |

Mask **không** cần đổi ở bất kỳ đâu — convention đã đúng cả ba.

---

## 5. Tự soát trước khi gửi

Chạy đoạn này ở gốc repo trước khi bàn giao. Nó kiểm tra đúng những gì cửa FR-058 kiểm tra.

> Đánh số ở đây theo **code** (`receipt.validate_handoff`), không theo bảng trong
> `returned-result.md`. Code chạy 7 check: bảng contract liệt kê 6 và gộp `deviation` vào rule 7 dạng
> văn xuôi, nên check `fold_attribution` là **#7** trong code nhưng là **#6** trong bảng. Cùng một
> check, chỉ khác cách đếm — đừng để lệch số làm bạn tưởng thiếu bước.

```python
import json, pathlib, sys
import numpy as np, cv2
sys.path.insert(0, "src")
from manga_text_seg.adapters.manga_text_segmentation import derive_book_folds

pl  = json.loads(pathlib.Path("benchmark/page-list.json").read_text(encoding="utf-8"))
idr = json.loads(pathlib.Path("benchmark/image-identity.json").read_text(encoding="utf-8"))
W, H = pl["aligned_size"]
REQ = ("method", "repository", "checkpoint", "code_license", "weight_license",
       "checkpoint_load_evidence", "device", "interpreter", "packages", "seed", "run_timestamp")

# Bang fold o muc 1 — lay truc tiep tu adapter, khong chep tay.
FOLDS = derive_book_folds()

def selfcheck(handoff, check_folds=False):
    h, bad = pathlib.Path(handoff), []
    prov = h / "provenance.json"
    if not prov.is_file():
        bad.append("check5: thieu provenance.json o goc hand-off")
    else:
        miss = [k for k in REQ if k not in json.loads(prov.read_text(encoding="utf-8-sig"))]
        if miss:
            bad.append(f"check5: provenance thieu {miss}")
    for page in pl["pages"]:
        iid, manga, stem = page["image_id"], page["manga"], page["stem"]
        sc = h / "metadata" / manga / f"{stem}.json"
        if not sc.is_file():
            bad.append(f"check1: thieu sidecar {iid}"); continue
        s = json.loads(sc.read_text(encoding="utf-8-sig"))
        if s.get("page_list_identity") != pl["page_list_identity"]:
            bad.append(f"check1: {iid} page_list_identity sai")
        if s.get("input_image_identity") != idr[iid]["sha256"]:
            bad.append(f"check2: {iid} input_image_identity sai")
        if check_folds:
            want_fold = FOLDS.get(manga)
            if want_fold is None:
                bad.append(f"check7: {iid} sach {manga!r} khong co trong bang fold")
            elif s.get("fold_attribution") != f"fold_{want_fold}":
                bad.append(f"check7: {iid} fold_attribution {s.get('fold_attribution')!r} "
                           f"nhung phai la 'fold_{want_fold}'")
        m = cv2.imread(str(h / "masks" / manga / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        if m is None:
            bad.append(f"check3: {iid} thieu mask"); continue
        if m.ndim != 2 or m.dtype != np.uint8:
            bad.append(f"check3: {iid} khong phai single-channel uint8")
        elif m.shape != (H, W):
            bad.append(f"check3: {iid} mask {m.shape} != {(H, W)}")
        elif not set(np.unique(m).tolist()) <= {0, 255}:
            bad.append(f"check4: {iid} gia tri ngoai {{0,255}}")
    got  = {f"{d.name}/{f.stem}" for d in (h / "masks").iterdir() if d.is_dir() for f in d.glob("*.png")}
    want = {p["image_id"] for p in pl["pages"]}
    if got - want: bad.append(f"thua {len(got - want)} mask ngoai list, vd {sorted(got - want)[:2]}")
    if want - got: bad.append(f"thieu {len(want - got)} mask, vd {sorted(want - got)[:2]}")
    return bad

# Method A: selfcheck("notebooks/deliverables/manga-text-segmentation", check_folds=True)
# check_folds chi dung cho Method A — FR-013a chi ap cho method A. Chay no cho B/C thi
# sidecar thieu key fold_attribution la dung, khong phai loi.
for x in selfcheck("notebooks/deliverables/<method>"):
    print(x)
```

Không có dòng nào in ra = hand-off sẽ qua cửa.

**Kết quả chạy trên ba hand-off hiện tại (2026-09-27):**

| Method | `check_folds` | Số vấn đề |
|---|---|---|
| `manga-text-segmentation` | `False` | **0** |
| `manga-text-segmentation` | `True` | **320** — toàn bộ là `check7`, đúng bằng 390 − 70 page thuộc 7 sách của fold 0 |
| `comic-text-detector` | `False` | **0** |
| `unetpp-efficientnetv2` | `False` | **0** |

---

## 6. Không được thay đổi

Những thứ này đúng rồi và việc "sửa" chúng sẽ làm hỏng kết quả:

- **Mask convention** — single-channel `uint8`, `{0, 255}`, text = 255, shape `(1170, 1654)`. Cả ba
  đang đúng.
- **Alignment** — `crop-topleft` 1654×1170, **không** resize. Không crop lại, không scale lại mask.
- **`image_id`** — không đổi, không đánh số lại.
- **Không đọc ground-truth khi chạy benchmark.** Đường dẫn GT trong ONBOARDING §3 là để hiểu dataset
  được ánh xạ thế nào, không phải để đọc lúc chạy. *"The images are the question and the masks are the
  answer."*
- **Không nộp metrics.** Không `metrics_per_page.csv`, không `metrics_summary.json`, không tự tính
  IoU/Precision/Recall/F1. Metrics được tính trong project, một lần, từ mask trả về (FR-057). Kết quả
  tự tính sẽ bị bỏ qua, không đọc.
- **Không nhúng ground-truth vào visualization.** Không GT overlay. (A còn 3 cell GT trong notebook —
  xem mục 1.)
- **Không commit checkpoint hay weight vào repo.** SC-013: zero weight file, mọi kích thước, mọi
  dạng.

---

## 7. Sau khi gửi lại

Project sẽ chạy `manga-text-seg admit --run <run_id> --method <method> <hand-off>`. Nếu qua hết các
check, mask được copy vào `deliverables/<run_id>/<method>/` và **chỉ khi đó** mới chấm điểm — trên
máy project, bằng quy trình dùng chung.

Hiện tại **chưa method nào được admit**, và cả bốn method trong `benchmark/availability.json` đang là
`unavailable` vì máy project không có checkpoint nào.

**Nhưng đó không phải điều kiện để được chấm.** `availability.json` là pre-flight cho `run`/`sweep`
— nó trả lời "máy này có chạy được model không". Còn `admit` chỉ đọc page list, identity record,
manifest và mask bạn trả về; nó không import `availability` và không mở file checkpoint nào. Metrics
được tính từ ground-truth cộng với mask bạn gửi (FR-057), và inference đã xảy ra trên máy bạn rồi.

Hệ quả cụ thể:

- **B và C chấm được ngay** — hand-off đã hợp lệ, không phải tải weight nào cả.
- **A** vẫn cần đủ 5 checkpoint, nhưng chỉ vì phải **chạy lại inference**. Riêng check
  `fold_attribution` thì không mở checkpoint — nó suy bảng fold từ seed công bố rồi so với sidecar.
