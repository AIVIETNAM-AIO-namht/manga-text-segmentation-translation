"""Three-up visual: original | prediction mask | Spec-2 ground truth.

Red = mask says text. Green = GT says text. Yellow = both agree.
If the mask is honest, red should sit inside green.
"""

import glob
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

GT = Path("data/groundtruth/post-processed")
TARGETS = sys.argv[1:] or ["AkkeraKanjinchou/002", "ARMS/000", "AisazuNihaIrarenai/001"]
SCALE = 0.55

out_dir = Path("deliverables/diag/truth")
out_dir.mkdir(parents=True, exist_ok=True)

for page in sorted(glob.glob("outputs/translation/spec004-vilao/*/*/*/metadata.json")):
    record = json.load(io.open(page, encoding="utf-8"))
    image_id = record["image_id"]
    if image_id not in TARGETS:
        continue
    book, stem = image_id.split("/")
    original = cv2.imread(str(Path(page).parent / "original.png"))
    mask = cv2.imread(str(Path(page).parent / "mask.png"), cv2.IMREAD_GRAYSCALE)
    gt_rgb = cv2.imread(str(GT / book / f"{stem}.png"), cv2.IMREAD_UNCHANGED)[:1170, :1654]
    gt = ((gt_rgb[:, :, 0] != 255) | (gt_rgb[:, :, 1] != 255) | (gt_rgb[:, :, 2] != 255))

    m = mask > 127
    overlay = original.copy()
    overlay[m & ~gt] = (0, 0, 255)      # mask only  -> red   (over-cover)
    overlay[gt & ~m] = (0, 255, 0)      # GT only    -> green (missed)
    overlay[m & gt] = (0, 255, 255)     # both       -> yellow

    tiles = [original, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), overlay]
    tiles = [cv2.resize(t, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA) for t in tiles]
    sheet = np.hstack(tiles)
    out = out_dir / f"{image_id.replace('/', '_')}_truth.png"
    cv2.imwrite(str(out), sheet)
    print(f"{image_id:28s} mask={100*m.mean():5.1f}%  gt={100*gt.mean():5.1f}%"
          f"  both={100*(m&gt).mean():5.1f}%  mask-only={100*(m&~gt).mean():5.1f}%"
          f"  -> {out}")
