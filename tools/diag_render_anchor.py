"""Does rendered text land ON the inpainted area?

The user's words: "chu no khong nam dung cho da inpaint ra" -- the text does not
sit where the inpaint is. Render centres text in the region BBOX and never looks
at the mask, so for a region built by merging (union bbox >> ink) the centre of
the box can be far from the ink that was actually erased.

Measure, per region: distance from the render anchor (bbox centre, what
render.py uses) to the mask ink centroid (what was actually inpainted), in units
of the region's own size.
"""

import glob
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def anchor_offset(mask_sub):
    """|bbox_centre - ink_centroid| normalised by bbox diagonal."""
    h, w = mask_sub.shape
    ys, xs = np.nonzero(mask_sub)
    if xs.size == 0:
        return None
    cx, cy = xs.mean(), ys.mean()
    bx, by = (w - 1) / 2.0, (h - 1) / 2.0
    return float(np.hypot(cx - bx, cy - by) / max(1.0, np.hypot(w, h)))


rows = []
for page in sorted(glob.glob("outputs/translation/spec004-vilao/*/*/*/metadata.json")):
    meta = json.load(io.open(page, encoding="utf-8"))
    page_dir = Path(page).parent
    mask = cv2.imread(str(page_dir / "mask.png"), cv2.IMREAD_GRAYSCALE) > 127
    regions = json.load(io.open(page_dir / "regions.json", encoding="utf-8"))["regions"]
    for r in regions:
        x, y, w, h = r["bbox"]
        sub = mask[y:y + h, x:x + w]
        ink = int(sub.sum())
        if ink == 0:
            continue
        off = anchor_offset(sub)
        if off is None:
            continue
        rows.append({"off": off, "ink": ink, "merged_from": r["merged_from"],
                     "fill": r["area"] / max(1, w * h), "w": w, "h": h})

off = np.array([r["off"] for r in rows])
ink = np.array([r["ink"] for r in rows], dtype=float)
print(f"regions: {len(rows)}")
print("anchor offset = |bbox centre - ink centroid| / bbox diagonal")
print("  0.00 = text lands dead centre of the ink; 0.25 = badly off\n")
for p in (50, 75, 90, 95, 99):
    print(f"  p{p:02d} = {np.percentile(off, p):.3f}")
print(f"  mean = {off.mean():.3f}   ink-weighted mean = "
      f"{float((off*ink).sum()/ink.sum()):.3f}")
print(f"  offset > 0.15: {100*(off > 0.15).mean():.1f}% of regions"
      f"  ({100*ink[off > 0.15].sum()/ink.sum():.1f}% of ink)")
print(f"  offset > 0.25: {100*(off > 0.25).mean():.1f}% of regions"
      f"  ({100*ink[off > 0.25].sum()/ink.sum():.1f}% of ink)")

print("\nby merged_from (does merging scatter the anchor?):")
for lo, hi in [(1, 2), (2, 5), (5, 15), (15, 10**9)]:
    sel = np.array([lo <= r["merged_from"] < hi for r in rows])
    if not sel.any():
        continue
    print(f"  merged_from {lo:2d}-{hi if hi < 10**8 else '+':>3}: {sel.sum():4d} regions"
          f"  median off={np.median(off[sel]):.3f}"
          f"  p90={np.percentile(off[sel], 90):.3f}"
          f"  >0.25: {100*(off[sel] > 0.25).mean():5.1f}%")
