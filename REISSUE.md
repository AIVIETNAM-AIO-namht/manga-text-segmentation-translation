# Re-issue: ba hand-off chưa qua được FR-058

Tài liệu này là **yêu cầu chạy lại** cho ba hand-off đang nằm trong
`notebooks/deliverables/`. Cả ba đã được đưa qua đúng cửa mà project sẽ dùng khi nhận bài
(`receipt.validate_handoff`, FR-058) và **cả ba bị từ chối ngay ở check 1**. Không có page nào
được chấm điểm, không có gì được ghi vào `deliverables/`.

Đây không phải đánh giá chất lượng model. Mask đã kiểm tra và **đúng convention**: cả ba đều là
single-channel `uint8`, shape `(1170, 1654)`, giá trị `{0, 255}` — check 3 và check 4 **pass toàn bộ**.
Vấn đề nằm ở phần *chứng minh nguồn gốc*, không nằm ở phần mask.

> Chạy lại **không** cần train lại, **không** cần sửa model. Phần lớn là sửa metadata và bổ sung một
> file còn thiếu; riêng Method A và Method C cần chạy lại inference.

---

## 0. Hai lỗi chung cho cả ba hand-off

### 0.1. `page_list_identity` đang ghi sai — đây là lý do bị từ chối

Mỗi sidecar phải chép **nguyên văn** identity của page list đã commit. Ba bạn đang ghi ba giá trị
khác nhau, và **không giá trị nào đúng**:

| Method | Đang ghi | Phải ghi |
|---|---|---|
| A | `39a6a4f8c365ccda47804233309f072ea01d5d5a571aa7f1c7c2235f4fdf3e6a` | ↓ |
| B | `spec-002-canonical-manifest-v1` | ↓ |
| C | `DERIVED_FROM_LOCAL_GROUNDTRUTH` | ↓ |
| **Tất cả** | | **`b52aa60ded6c5c2cca115dce4335295d6795b0601b9cfa8324c675fb4cab88ef`** |

Giá trị đúng nằm trong `benchmark/page-list.json`, field `page_list_identity`. **Chép nguyên văn từ
file đó.** Không tự tính, không tự đặt tên, không suy ra từ danh sách bạn tự dựng.

Refusal thực tế mà project nhận được:

```text
[page-list-identity] ARMS/000: page list identity '39a6a4f8…' is not the identity of the
committed export (b52aa60d…); a result produced against another list is refused, not aligned by name
```

### 0.2. Thiếu `provenance.json` ở gốc hand-off

Đây là file **riêng**, không phải sidecar từng page. `receipt.validate_handoff` đọc
`<hand-off>/provenance.json`; sidecar trong `metadata/` là chuyện khác. Cả ba hand-off đều không có
file này. Sau khi sửa 0.1, đây là lỗi tiếp theo mà cả ba gặp:

```text
[provenance] no readable provenance.json in the hand-off
```

Cấu trúc hand-off đúng (theo `specs/002-*/contracts/returned-result.md`):

```text
<hand-off>/
├── masks/<manga>/<stem>.png      # đã có, đúng convention
├── metadata/<manga>/<stem>.json  # đã có
├── provenance.json               # ← THIẾU, phải bổ sung (một file cho cả hand-off)
└── errors.json                   # đã có
```

`provenance.json` phải là **object phẳng** với đúng các key sau (`contracts/provenance.schema.json`,
`additionalProperties: false` — key lạ cũng bị coi là sai):

```json
{
  "method": "<tên method, xem mục riêng từng bạn>",
  "repository": {
    "url": "<repo URL đã pin>",
    "revision": "<commit SHA đủ 40 ký tự hex>"
  },
  "checkpoint": {
    "identity": "<tên file checkpoint>",
    "size_bytes": 123456789,
    "sha256": "<64 ký tự hex>"
  },
  "code_license": "<MIT | GPL-3.0 | none>",
  "weight_license": "<license của weights, ghi riêng>",
  "checkpoint_load_evidence": {
    "evidenced": true,
    "method": "<cách bạn chứng minh>",
    "detail": "<bằng chứng cụ thể>"
  },
  "device": { "type": "cuda", "name": "Tesla T4" },
  "interpreter": "<version đầy đủ>",
  "packages": { "<tên>": "<version thật>" },
  "seed": 42,
  "run_timestamp": "<ISO-8601 có offset>"
}
```

Ba điểm dễ sai ở file này:

- `checkpoint` ở đây dùng key **`identity`**, không phải `name`. (Sidecar per-page dùng `name` —
  hai schema khác nhau, đừng copy nguyên.)
- `revision` phải là **40 ký tự hex**. `"main"`, tag, hay short SHA đều bị từ chối.
- `packages` phải là **version thật** của môi trường đã chạy. `"default"` không phải version.

### 0.3. Sai chính tả tên thư mục — 10 page bị mất

Page list dùng `UchiNoNyan'sDiary` (**dấu nháy đơn**). Hand-off của A và C đặt thư mục là
`UchiNoNyan_sDiary` (**dấu gạch dưới**). Hệ quả: sidecar của 10 page đó không được tìm thấy, và
refusal đọc ra là `identity None`:

```text
[page-list-identity] UchiNoNyan'sDiary/000: page list identity None is not the identity…
```

Sửa: đổi tên thư mục trong cả `masks/` và `metadata/` thành `UchiNoNyan'sDiary`.

---

## 1. Method A — `manga-text-segmentation`

**Phải chạy lại inference.** Ngoài 0.1–0.3, Method A còn vi phạm FR-013a.

### Đã đúng (giữ nguyên)

- `repository` = `https://github.com/juvian/Manga-Text-Segmentation` — khớp `configs/dl.json`
- `commit` = `de8f148c78978d70ad0e0ae3242566da6d70f3a5` — khớp
- `code_license` = `MIT`, `threshold` = `0.5` — khớp
- `checkpoint.sha256` = `fa1d074002955c66f7df32ff60f6602d4aa21f1e1c1ca158dcb852b7e9eeb623` — khớp fold.0 đã pin
- `input_size` / `output_size` = `[1654, 1170]` — đúng
- Mask: 390 file, `L`, `(1170, 1654)`, `{0, 255}` — đúng

### Phải sửa: FR-013a — `fold_attribution` sai toàn bộ

Sidecar đang ghi `"fold_attribution": "fold_0"` cho **cả 390 page**, kèm dòng tự khai:

```json
"checkpoint_strategy": "single-fold-fallback (fold.0 only) — LOFO 5-checkpoint chưa triển khai, quyết định của runner"
```

FR-013a không cho phép điều đó. Upstream phát hành 5 checkpoint, mỗi checkpoint giữ lại đúng một
nhóm sách. Một page chỉ là **held-out prediction** khi nó được chấm bằng checkpoint của fold đã giữ
sách đó lại. Checkpoint fold 0 đã **train trên** 32 trong 39 sách của benchmark này, nên nó không
phải dự đoán held-out cho các page đó.

Refusal thực tế:

```text
[fold-attribution] ARMS/000: scored by fold 0, but this page's book was held out of fold 2 —
the fold 0 checkpoint trained on it, so the page is not a held-out prediction (FR-013a)
```

**Việc phải làm:** lấy đủ **5 checkpoint** từ release v1.0 của upstream, chạy cả 5, và ghi
`fold_attribution` **theo từng sách** như bảng dưới. Bảng này project đã dựng lại và kiểm chứng được
từ seed công bố (`KFold(n_splits=5, shuffle=True, random_state=42)`), nên nó là thứ hai bên phải
khớp — không phải thứ bạn tự chọn:

| Fold | Số sách | Sách |
|---|---|---|
| `fold_0` | 7 | AosugiruHaru, BakuretsuKungFuGirl, HanzaiKousyouninMinegishiEitarou, HaruichibanNoFukukoro, TouyouKidan, YasasiiAkuma, YumeNoKayoiji |
| `fold_1` | 9 | Akuhamu, Arisa, Count3DeKimeteAgeru, Donburakokko, EienNoWith, EverydayOsakanaChan, Hamlet, ToutaMairimasu, UnbalanceTokyo |
| `fold_2` | 8 | ARMS, AppareKappore, DualJustice, HarukaRefrain, UchiNoNyan'sDiary, UchuKigekiM774, UltraEleven, YoumaKourin |
| `fold_3` | 8 | AisazuNihaIrarenai, AkkeraKanjinchou, BurariTessenTorimonocho, GakuenNoise, GinNoChimera, WarewareHaOniDearu, YamatoNoHane, YumeiroCooking |
| `fold_4` | 7 | BEMADER_P, DollGun, EvaLady, GarakutayaManta, HealingPlanet, TsubasaNoKioku, YukiNoFuruMachi |

Cả 39 sách đều có fold xác định được, nên không page nào phải bỏ.

**Nếu không lấy được đủ 5 checkpoint:** báo lại project. FR-013a nói rõ một mapping không reproduce
được thì Method A **inadmissible** — báo là unavailable kèm lý do, **không** được lấy fold 0 chấm
thay. Đây là kết cục hợp lệ; ghi `fold_0` cho tất cả thì không.

### Ghi chú nhỏ

- `checkpoint.name` đang là `"model.pkl"` (tên file local). Nên ghi đúng tên asset đã pin:
  `fold.<n>.-.final.refined.model.2.pkl` — FR-044 nói tên file *là* identity.
- `provenance.json` phải ghi `"method": "manga-text-segmentation"` (khớp registry key).
- `checkpoint` trong `provenance.json` là checkpoint nào? Với LOFO, một hand-off dùng 5 checkpoint.
  Ghi checkpoint của fold được dùng nhiều nhất, và mô tả đủ 5 trong `checkpoint_load_evidence.detail`
  (kèm sha256 của từng cái bạn đã hash được). Project chỉ pin sẵn sha256 của fold 0; bốn cái còn lại
  ghi giá trị **quan sát được**, không đoán.
- File `outputs-20260915T160219Z-1-001.zip` trong thư mục A khớp rule `*.zip` của `.gitignore` nên
  không được track. Mask đã nằm rời trong `masks/` và đã được track — không cần zip.

---

## 2. Method B — `comic-text-detector`

**Không cần chạy lại model.** Mask, page set (390/390 khớp hoàn toàn), checkpoint và threshold đều
đúng. Chỉ cần sửa metadata và bổ sung `provenance.json`.

### Đã đúng (giữ nguyên)

- `checkpoint.sha256` = `1f90fa60aeeb1eb82e2ac1167a66bf139a8a61b8780acd351ead55268540cccb` — khớp pin
- `checkpoint.size_bytes` = `79948869` — khớp pin
- `code_license` = `GPL-3.0`, `threshold` = `0.235` — khớp
- `input_size` / `output_size` = `[1654, 1170]` — đúng
- 390 sidecar, 390 mask, không thừa không thiếu — đây là hand-off sạch nhất về page set
- Check 2 pass 390/390

### Phải sửa: repository và commit không khớp pin

| | Đang ghi | Đã pin trong `configs/dl.json` |
|---|---|---|
| URL | `https://github.com/Ajatt-Tools/comic_text_detector` | `https://github.com/dmMaze/comic-text-detector` |
| Revision | `e958d8b4a7461528e8df8aa8156e3928b57b2fc6` | `440b978563c71b758e31aaa315d100faba1efa2f` |

`Ajatt-Tools/comic_text_detector` là **fork**. FR-055 và `returned-result.md` rule 7 nói kết quả chạy
từ một revision khác **là một method khác**, không admissible dưới tên method đã pin. Hai lựa chọn:

1. **Chạy lại từ repo đã pin** (`dmMaze/comic-text-detector` @ `440b9785…`) — cách sạch, và vì mask
   không cần đổi gì khác nên chi phí thấp.
2. Nếu buộc phải dùng fork, **ghi vào `provenance.json` một object `deviation`** mô tả rõ:
   `{"applied": true, "what": "...", "why": "...", "changes_output": false}`. Chỉ hợp lệ nếu bản fork
   **không đổi output**; nếu có đổi thì kết quả là method khác và project sẽ từ chối.

Ngoài ra, `provenance.json` phải ghi `"method": "comic-text-detector"`.

### Ghi chú nhỏ

- `packages.torchvision` = `"default"` — không phải version. FR-044 đòi version thật của môi trường
  đã chạy. Ghi version cụ thể.
- `interpreter` = `"Python 3.10"` — nên ghi đủ, ví dụ `"Python 3.10.12"`.

---

## 3. Method C — `UNetPP_EfficientNetV2`

**Phải chạy lại inference.** Đây là hand-off lệch nhiều nhất.

### Phải sửa: tự dựng page list từ ground-truth (FR-054 / FR-054a)

`page_list_identity` = `DERIVED_FROM_LOCAL_GROUNDTRUTH`, và `errors.json` khai thẳng:

```json
"page_list_source": "ground-truth name-matched discovery"
```

Runner **không được** dựng manifest bằng cách tự khớp tên giữa `images/` và `post-processed/`.
Page list do project phát là **đầu vào**, không phải thứ để tái tạo. Hệ quả trên đĩa: **925 mask**
trong khi page list có **390 page** — thừa **545 mask**:

| Sách | Mask thừa | Khoảng page thừa |
|---|---|---|
| AosugiruHaru | 95 | 010–104 |
| AppareKappore | 87 | 010–096 |
| AisazuNihaIrarenai | 84 | 010–093 |
| AkkeraKanjinchou | 82 | 010–091 |
| Akuhamu | 71 | 010–080 |
| ARMS | 71 | 010–080 |
| Arisa | 45 | 010–054 |
| UchiNoNyan_sDiary | 10 | 000–009 |

**Việc phải làm:** chạy lại đúng **390 page** trong `benchmark/page-list.json`, và giao đúng 390
mask + 390 sidecar. Mask thừa không được chấp nhận, cũng không bị cắt hộ.

### Phải sửa: `input_size` / `output_size` bị đảo

Sidecar C ghi `[1170, 1654]`; hai method kia và page list đều dùng `[1654, 1170]` (width × height).
Đảo lại cho khớp.

### Phải sửa: `method` sai tên registry

Sidecar ghi `"method": "UNetPP_EfficientNetV2"`. Registry key của adapter là
**`unetpp-efficientnetv2`** (chữ thường, gạch nối) — cũng là tên trong `configs/dl.json`. `admit
--method` so trực tiếp với `provenance.method`, nên tên này phải khớp chính xác.

### Phải sửa: license

- `code_license` = `NOT_STATED_BY_UPSTREAM_REPOSITORY` → ghi `"none"` (upstream không công bố license
  — đây là *một vị trí được ghi nhận*, không phải trường thiếu).
- `weight_license` = `NOT_VERIFIED__VERIFY_HUGGING_FACE_MODEL_CARD` → phải tra model card trên
  Hugging Face và ghi kết quả thật, hoặc ghi `"none"` nếu không có. Không được để nguyên placeholder.

### Ghi chú nhỏ

- `repository` có đuôi `.git`: `https://github.com/ContemporaryCat/Manga-Text-Segmentation.git` —
  bỏ `.git` cho khớp pin.
- 10 page `UchiNoNyan'sDiary` đang thiếu (xem 0.3).

### Điểm làm tốt — nên giữ làm mẫu

`checkpoint.loaded_evidence` của C là bằng chứng dương đúng nghĩa FR-018a: 1396/1396 tensor key khớp
bitwise, kèm fingerprint và probe output hash. Đây chính xác là thứ project muốn, và nó quan trọng
nhất ở Method C — upstream của C **nuốt lỗi thiếu checkpoint, in warning rồi chạy tiếp với decoder
khởi tạo ngẫu nhiên**, vẫn ra mask trông hợp lệ. A và B nên nâng `checkpoint_load_evidence` lên mức
này.

---

## 4. Tóm tắt việc phải làm

| # | Việc | A | B | C |
|---|---|---|---|---|
| 1 | Sửa `page_list_identity` thành `b52aa60d…` (mọi sidecar) | ✅ | ✅ | ✅ |
| 2 | Thêm `provenance.json` ở gốc hand-off | ✅ | ✅ | ✅ |
| 3 | Đổi `UchiNoNyan_sDiary` → `UchiNoNyan'sDiary` | ✅ | — | ✅ |
| 4 | Chạy lại model | ✅ | — | ✅ |
| 5 | Sửa `fold_attribution` theo sách (FR-013a) | ✅ | — | — |
| 6 | Dùng repo/commit đã pin, hoặc ghi `deviation` | — | ✅ | — |
| 7 | Cắt về đúng 390 page | — | — | ✅ |
| 8 | Sửa `input_size`/`output_size` thành `[1654, 1170]` | — | — | ✅ |
| 9 | Sửa `method` → `unetpp-efficientnetv2` | — | — | ✅ |
| 10 | Giải quyết `code_license` / `weight_license` | — | — | ✅ |
| 11 | Ghi version thật vào `packages` | — | ✅ | — |

Mask **không** cần đổi ở bất kỳ đâu — convention đã đúng cả ba.

---

## 5. Tự soát trước khi gửi

Chạy đoạn này ở gốc repo trước khi bàn giao. Nó kiểm tra đúng những gì cửa FR-058 kiểm tra.

> Đánh số ở đây theo **code** (`receipt.validate_handoff`), không theo bảng trong
> `returned-result.md`. Code chạy 7 check: bảng contract liệt kê 6 và gộp `deviation` vào rule 7 dạng
> văn xuôi, nên check `fold_attribution` là **#7** trong code nhưng là **#6** trong bảng. Cùng một
> check, chỉ khác cách đếm — đừng để lệch số làm bạn tưởng thiếu bước.

```python
import json, pathlib
import numpy as np, cv2

pl  = json.loads(pathlib.Path("benchmark/page-list.json").read_text(encoding="utf-8"))
idr = json.loads(pathlib.Path("benchmark/image-identity.json").read_text(encoding="utf-8"))
W, H = pl["aligned_size"]
REQ = ("method", "repository", "checkpoint", "code_license", "weight_license",
       "checkpoint_load_evidence", "device", "interpreter", "packages", "seed", "run_timestamp")

# Bang fold o muc 1. Chi Method A can.
FOLDS = dict(zip(
    "AosugiruHaru BakuretsuKungFuGirl HanzaiKousyouninMinegishiEitarou HaruichibanNoFukukoro "
    "TouyouKidan YasasiiAkuma YumeNoKayoiji".split(), [0] * 7)) | dict(zip(
    "Akuhamu Arisa Count3DeKimeteAgeru Donburakokko EienNoWith EverydayOsakanaChan Hamlet "
    "ToutaMairimasu UnbalanceTokyo".split(), [1] * 9)) | dict(zip(
    "ARMS AppareKappore DualJustice HarukaRefrain UchiNoNyan'sDiary UchuKigekiM774 UltraEleven "
    "YoumaKourin".split(), [2] * 8)) | dict(zip(
    "AisazuNihaIrarenai AkkeraKanjinchou BurariTessenTorimonocho GakuenNoise GinNoChimera "
    "WarewareHaOniDearu YamatoNoHane YumeiroCooking".split(), [3] * 8)) | dict(zip(
    "BEMADER_P DollGun EvaLady GarakutayaManta HealingPlanet TsubasaNoKioku YukiNoFuruMachi".split(),
    [4] * 7))

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
for x in selfcheck("notebooks/deliverables/<method>"):
    print(x)
```

Không có dòng nào in ra = hand-off sẽ qua cửa.

Chạy trên ba hand-off hiện tại, đoạn này báo đúng: A 393 vấn đề, B 391, C 393 — khớp với những gì
mục 0–3 liệt kê. Bật `check_folds=True` cho Method A thì thành 703, trong đó 310 là vi phạm
`fold_attribution` (= 390 page trừ 70 page thuộc 7 sách vốn đúng ở fold 0).

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
- **Không nhúng ground-truth vào visualization.** Không GT overlay.
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

- **B và C chấm được ngay** khi hand-off hợp lệ — không phải tải weight nào cả. (C còn phải cắt về
  đúng 390 page trước đã, xem mục 3.)
- **A** vẫn cần đủ 5 checkpoint, nhưng chỉ vì phải **chạy lại inference**. Riêng check
  `fold_attribution` thì không mở checkpoint — nó suy bảng fold từ seed công bố rồi so với sidecar.

Nếu bạn có sẵn checkpoint, báo lại — nó giúp Method A chạy lại, **không** phải điều kiện để B/C được
chấm.
