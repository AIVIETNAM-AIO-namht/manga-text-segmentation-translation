"""Split the giant-box problem into its two causes: mask connectivity vs proximity merge.

A group can be huge because (a) the mask itself has a page-spanning connected
component, or (b) proximity merge chained many components together.
"""

import glob
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

MIN_AREA, MAX_AREA, MERGE_DISTANCE = 50, 12000, 8
PAGE_PX = 1654 * 1170
MASK_PX = 1654 * 1170

pages = sorted(glob.glob("outputs/translation/spec004-vilao/*/*/*/metadata.json"))
print(f"{'image_id':32s} {'mask%':>6s} {'cc':>6s} {'biggestCC':>10s} {'cc%page':>8s}"
      f" {'>12:1':>6s} {'ncomp':>6s} {'kept':>5s}")
print("-" * 96)

tot_biggest, tot_line, tot_kept = [], [], []
for page in pages:
    record = __import__("json").loads(Path(page).read_text(encoding="utf-8"))
    mask = cv2.imread(str(Path(page).parent / "mask.png"), cv2.IMREAD_GRAYSCALE)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    stats = stats[1:]
    areas = stats[:, 4]
    biggest = int(areas.max())
    x, y, w, h = (int(v) for v in stats[areas.argmax()][:4])
    # a connected component that is line-like: one dimension dwarfs the other
    aspect = max(w, h) / max(1, min(w, h))
    kept = [(int(s[0]), int(s[1]), int(s[2]), int(s[3])) for s in stats
            if MIN_AREA <= s[4] <= MAX_AREA]
    line_kept = sum(1 for b in kept if max(b[2], b[3]) / max(1, min(b[2], b[3])) >= 12)
    tot_biggest.append(biggest / MASK_PX)
    tot_line.append(line_kept)
    tot_kept.append(len(kept))
    print(f"{record['image_id']:32s} {100 * mask.mean() / 255:5.1f}% {n - 1:6d}"
          f" {biggest:10d} {100 * biggest / MASK_PX:7.2f}% {aspect:6.1f}"
          f" {len(kept):6d} {line_kept:5d}")

print("-" * 96)
print(f"median biggest connected component: {100 * np.median(tot_biggest):.2f}% of the page")
print(f"pages where a single connected component exceeds 5% of the page:"
      f" {sum(1 for v in tot_biggest if v > 0.05)}/{len(tot_biggest)}")
print(f"median line-like components kept per page (aspect >= 12:1): {np.median(tot_line):.0f}")
