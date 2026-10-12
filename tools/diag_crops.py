"""Crop the worst regions side by side: original | inpainted | rendered, 3x zoom."""

import glob
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

IMAGE_ID = sys.argv[1] if len(sys.argv) > 1 else "AisazuNihaIrarenai/004"
PAD = 30
SCALE = 3
TOPN = 8

page = None
for p in glob.glob("outputs/translation/spec004-vilao/*/*/*/metadata.json"):
    r = json.load(io.open(p, encoding="utf-8"))
    if r["image_id"] == IMAGE_ID:
        page, rec = Path(p).parent, r
        break
assert page, IMAGE_ID

original = cv2.imread(str(page / "original.png"))
inpainted = cv2.imread(str(page / "inpainted.png"))
rendered = cv2.imread(str(page / "rendered.png"))
height, width = original.shape[:2]

out_dir = Path("deliverables/diag/crops")
out_dir.mkdir(parents=True, exist_ok=True)

# Worst = biggest boxes (the ones that visibly swallow artwork)
regions = sorted(rec["regions"], key=lambda r: -(r["bbox"][2] * r["bbox"][3]))[:TOPN]
for region in regions:
    x, y, w, h = region["bbox"]
    x0, y0 = max(0, x - PAD), max(0, y - PAD)
    x1, y1 = min(width, x + w + PAD), min(height, y + h + PAD)
    tiles = []
    for base in (original, inpainted, rendered):
        crop = base[y0:y1, x0:x1].copy()
        cv2.rectangle(crop, (x - x0, y - y0), (x - x0 + w, y - y0 + h), (0, 0, 255), 2)
        crop = cv2.resize(crop, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
        tiles.append(crop)
    sheet = np.hstack(tiles)
    out = out_dir / f"{IMAGE_ID.replace('/', '_')}_r{region['region_id']}.png"
    cv2.imwrite(str(out), sheet)
    print(
        f"r{region['region_id']:4d} bbox=({x},{y},{w},{h}) merged={region['merged_from']:3d}"
        f" status={region.get('render_status')} font={region.get('font_size_used')}"
        f"\n      ocr={region.get('ocr_text')!r}\n      vi ={region.get('translation')!r}"
    )
print(f"\n{out_dir}  ({len(regions)} sheets, original|inpainted|rendered)")
