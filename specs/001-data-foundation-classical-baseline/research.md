# Research: Data Foundation & Classical Baseline

**Date**: 2026-09-13 | **Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)

All unknowns resolved. No NEEDS CLARIFICATION remains.

## R1 — Binary normalization of multi-color ground truth

**Decision**: Two-class KMeans (k=2, cosine-space stable, fixed seed) on the mask's RGB pixel set
per file, applied in the loader (`normalize.py`). Which cluster is "text" is decided by
**rarity heuristic**: the text class is the minority cluster (dialogue text occupies < 50% of
any page; empty mask `EvaLady/006` is the degenerate case — see R3). Output: single-channel
uint8, text = 255, background = 0.

**Rationale**: GT masks use magenta `(255,1,255)` + black `(1,1,1)` (357 files), magenta-only
(89), near-black-only (3), all-white-empty (1). A fixed palette map would break on the next
encoding variant; KMeans adapts per file. Rarity heuristic is robust because even the densest
manga page is majority background.

**Alternatives considered**:
- Fixed RGB palette lookup: rejected — brittle against unseen encodings, violates Ponytail reuse.
- Otsu on grayscale GT: rejected — magenta and near-black collapse to similar gray levels.

## R2 — Mask ↔ raw alignment (1176×1656 → 1170×1654)

**Decision**: Default strategy `crop-topleft`: losslessly crop the GT mask's top-left
`1654×1170` window (no resampling). Verified across all 390 pairs: the GT canvas is a uniform
+6 px horizontal / +2 px vertical pad with zero text pixels in the padding band (max text pixel
x = 1653, y = 1169 in the 1176×1656 frame).

**Rationale**: Resizing a binary mask introduces resampling artifacts (gray fringes, shifted
edges) that corrupt IoU measurements. Cropping preserves every text pixel exactly.

**Alternatives considered**:
- `cv2.resize` to raw size: rejected as default (binary resampling artifacts).
- Center-crop: rejected — padding is asymmetric; top-left is the verified geometry.
- Configurable strategies: kept — `align.py` still exposes a strategy switch (`crop-topleft`
  default; resize explicitly opt-in and logged in sidecar metadata), satisfying FR-017 without
  making resize the default.

## R3 — Metric definitions and edge cases

**Decision** (`metrics.py`): per image, with TP/FP/FN computed on binary {0,255} arrays:
- Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2PR/(P+R), IoU = TP/(TP+FP+FN).
- Empty GT (`EvaLady/006`) and/or empty prediction: guard all divisions; define
  Precision = Recall = F1 = IoU = 1.0 when TP+FP+FN = 0 (perfect agreement on "no text"),
  0.0 when the denominator of a single metric is 0 but disagreement exists (e.g., FP > 0 with
  empty GT → Precision = 0, Recall = 1 by convention `R/(R)` with R undefined… fixed rule:
  if GT empty and prediction empty → all 1.0; if GT empty and prediction non-empty →
  P=0, R=1, F1=0, IoU=0; if prediction empty and GT non-empty → P=1, R=0, F1=0, IoU=0).
- No Pixel Accuracy anywhere (spec FR-027/FR-028, constitution Ponytail: no speculative metrics).

**Rationale**: The empty-mask page would otherwise crash every method sweep with ZeroDivision.
Explicit convention keeps the 390-pair sweep total (FR-003/FR-030).

## R4 — Classical baseline method configurations

**Decision**: Each method is a `name -> factory(config)` entry in the registry
(`methods/__init__.py`), all consuming single-channel grayscale input (FR-020 preprocessing:
grayscale + optional denoise) and emitting {0,255} uint8:
- `otsu`: Gaussian blur (configurable ksize) + `cv2.threshold(..., THRESH_BINARY+OTSU)`.
- `adaptive`: `cv2.adaptiveThreshold` (Gaussian, configurable blockSize/C).
- `mser`: `cv2.MSER_create` regions → filled convex hulls on blank canvas.
- `edges`: Canny (configurable thresholds) → dilate → fill contours.
- `components`: binarize (Otsu) → `connectedComponentsWithStats` → area/aspect filter.
- `morphology`: parameter set (kernel shape/size, open/close iterations) applied post-binarize.
- `pipeline`: ordered composition of the above stages from config (FR-020 last bullet).

**Rationale**: OpenCV provides all primitives — no new dependencies (Ponytail III). Registry
pattern means Spec 2 registers DL adapters without touching these files (SC-008).

## R5 — Configuration and CLI design

**Decision**: One JSON configuration file, schema-validated at load (`config.py`, stdlib `json` +
explicit key checks; no new dep like pydantic — Ponytail III). Contents: dataset paths, output
root, alignment strategy, method list + per-method params, metrics list, visualization
`{mode: first-N|best-N|worst-N, N}`, run id. CLI (`cli.py`): `discover`, `validate`, `run`,
`sweep`, `report`, `visualize` subcommands; `--config` selects file; `--method` overrides for
single-method runs. `report`/`visualize` reuse persisted metrics (FR-031: rerunnable without
re-running inference).

**Rationale**: FR-036 demands single-config control; subcommands map 1:1 to user stories
(US1 discovery → `discover`/`validate`; US2-3 → `run`/`sweep`; US4 → `report`; US5 →
`visualize`).

## R6 — Spec-2 forward-compatible method interface

**Decision** (`methods/__init__.py`, frozen in `contracts/method-interface.md`):
`BaseSegmentationMethod` with `name: str` property and
`segment(image: np.ndarray) -> np.ndarray` (grayscale in → {0,255} uint8 out). Registry:
`register(name, factory)`, `create(name, config)`. Spec 2 DL adapters implement the same two
members; `metrics.py`, output writer and sidecar schema are untouched.

**Rationale**: Satisfies FR-035 + SC-008 with the smallest possible surface (two members).
