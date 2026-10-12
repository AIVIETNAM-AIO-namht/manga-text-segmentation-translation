"""For one region: dump its components, the chain that links them, and the mask."""

import glob
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

IMAGE_ID = sys.argv[1]
REGION_ID = int(sys.argv[2])
MIN_AREA, MAX_AREA, MERGE_DISTANCE = 50, 12000, 8

for p in glob.glob("outputs/translation/spec004-vilao/*/*/*/metadata.json"):
    record = json.load(io.open(p, encoding="utf-8"))
    if record["image_id"] == IMAGE_ID:
        page = Path(p).parent
        break

region = next(r for r in record["regions"] if r["region_id"] == REGION_ID)
x, y, w, h = region["bbox"]
print(f"region {REGION_ID} bbox=({x},{y},{w},{h}) merged_from={region['merged_from']}")

mask = cv2.imread(str(page / "mask.png"), cv2.IMREAD_GRAYSCALE)
_, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
stats = stats[1:]  # drop background

# components overlapping the region box
inside = []
for index, s in enumerate(stats):
    sx, sy, sw, sh, area = (int(v) for v in s)
    if sx + sw <= x or sx >= x + w or sy + sh <= y or sy >= y + h:
        continue
    inside.append((index, sx, sy, sw, sh, area))
print(f"components overlapping the box: {len(inside)}")
print(f"  kept by area filter [{MIN_AREA},{MAX_AREA}]:"
      f" {sum(1 for c in inside if MIN_AREA <= c[5] <= MAX_AREA)}")
print(f"  dropped below min_area: {sum(1 for c in inside if c[5] < MIN_AREA)}")
print(f"  dropped above max_area: {sum(1 for c in inside if c[5] > MAX_AREA)}")

areas = sorted(c[5] for c in inside)
print(f"  areas: {areas[:5]} ... {areas[-5:]}")

kept = [(i, a, b, c, d, e) for (i, a, b, c, d, e) in inside if MIN_AREA <= e <= MAX_AREA]
kept.sort(key=lambda c: -c[5])
print("\n  largest kept components:")
for index, sx, sy, sw, sh, area in kept[:12]:
    print(f"    label={index:6d} bbox=({sx:4d},{sy:4d},{sw:4d},{sh:4d}) area={area}")

# Where do the kept components sit? Are they clustered or spread across the box?
if kept:
    xs = np.array([c[1] for c in kept])
    ys = np.array([c[2] for c in kept])
    print(f"\n  kept x range {xs.min()}..{xs.max()}   box x {x}..{x + w}")
    print(f"  kept y range {ys.min()}..{ys.max()}   box y {y}..{y + h}")
    # distance from the box centre
    cx, cy = x + w / 2, y + h / 2
    dist = np.hypot(xs + np.array([c[3] for c in kept]) / 2 - cx,
                    ys + np.array([c[4] for c in kept]) / 2 - cy)
    print(f"  distance from box centre: median {np.median(dist):.0f}"
          f"  max {dist.max():.0f}  box diagonal {np.hypot(w, h):.0f}")

# Is the ink spread out? This is what makes a union bbox useless.
group_mask = np.zeros_like(mask)
for index, *_ in kept:
    group_mask[labels == index] = 255
crop = group_mask[y : y + h, x : x + w]
print(f"\n  union ink fill: {crop.mean() / 255:.4f} of the box")
rows_with_ink = (crop > 0).any(axis=1).sum()
cols_with_ink = (crop > 0).any(axis=0).sum()
print(f"  box rows containing ink: {rows_with_ink}/{h}"
      f"   cols: {cols_with_ink}/{w}")

overlay = cv2.imread(str(page / "original.png"))[y : y + h, x : x + w].copy()
overlay[labels[y : y + h, x : x + w] > 0] = (0, 0, 255)
overlay = cv2.resize(overlay, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
out = Path("deliverables/diag/crops") / f"{IMAGE_ID.replace('/', '_')}_r{REGION_ID}_maskoverlay.png"
out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out), overlay)
print(f"\n  {out}")
