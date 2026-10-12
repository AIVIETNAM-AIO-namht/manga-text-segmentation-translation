"""Are the giant groups built from large, sparse, non-text components?

A merge can only inflate a box by ~merge_distance per axis (hg<8 and vg<8), so a
page-spanning union box requires page-spanning *components*. If those components
are sparse (ink << bbox), they are artwork, not glyphs -- and the right filter is
per-component, like min_area/max_area, not a gate on the union.
"""

import glob
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

MIN_AREA, MAX_AREA, MERGE_DISTANCE = 50, 12000, 8
PAGE_PX = 1654 * 1170

pages = sorted(glob.glob("outputs/translation/spec004-vilao/*/*/*/metadata.json"))
records = []
for page in pages:
    mask = cv2.imread(str(Path(page).parent / "mask.png"), cv2.IMREAD_GRAYSCALE)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = [s for s in stats[1:] if MIN_AREA <= s[4] <= MAX_AREA]
    boxes = [(int(s[0]), int(s[1]), int(s[2]), int(s[3])) for s in keep]
    areas = [int(s[4]) for s in keep]
    records.append((Path(page).parent.parent.parent.name + "/" + Path(page).parent.name,
                    boxes, areas))

fills = np.array([a / max(1, b[2] * b[3]) for _, boxes, areas in records
                  for b, a in zip(boxes, areas)])
print(f"components after area filter: {len(fills)}")
print(f"  own-fill percentiles: p1={np.percentile(fills,1):.3f}"
      f" p5={np.percentile(fills,5):.3f} p10={np.percentile(fills,10):.3f}"
      f" p25={np.percentile(fills,25):.3f} median={np.median(fills):.3f}")
for thr in (0.05, 0.10, 0.15, 0.20, 0.30):
    print(f"  own-fill < {thr:.2f}: {100*(fills < thr).mean():5.1f}% of components")

# What do the biggest, sparsest components actually look like?
print("\n  the 12 largest components by bbox, with their own fill:")
allc = sorted(((b, a, page) for page, boxes, areas in records
               for b, a in zip(boxes, areas)),
              key=lambda r: -(r[0][2] * r[0][3]))[:12]
for b, a, page in allc:
    print(f"    {page:26s} bbox={b[2]:5d}x{b[3]:<5d} area={a:6d}"
          f" fill={a/max(1,b[2]*b[3]):.3f} box={100*b[2]*b[3]/PAGE_PX:6.2f}% of page")


def groups_after(boxes, areas, min_fill):
    """Proximity merge, but components sparser than min_fill never participate."""
    idx = [i for i in range(len(boxes))
           if areas[i] / max(1, boxes[i][2] * boxes[i][3]) >= min_fill]
    n = len(idx)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a in range(n):
        for b in range(a + 1, n):
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            lx, ly, lw, lh = boxes[idx[a]]
            rx, ry, rw, rh = boxes[idx[b]]
            hg = max(lx - (rx + rw), rx - (lx + lw), 0)
            vg = max(ly - (ry + rh), ry - (ly + lh), 0)
            if hg < MERGE_DISTANCE and vg < MERGE_DISTANCE:
                parent[max(ra, rb)] = min(ra, rb)
    out = {}
    for i in range(n):
        out.setdefault(find(i), []).append(idx[i])
    return list(out.values())


print(f"\n{'min own-fill':>14s} {'kept':>7s} {'groups':>7s} {'>5%page':>8s} {'>20%':>6s}"
      f" {'biggest':>8s} {'dropped':>8s}")
for min_fill in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
    kept = groups = big5 = big20 = dropped = 0
    biggest = 0.0
    for _, boxes, areas in records:
        if min_fill > 0:
            dropped += sum(1 for a, b in zip(areas, boxes)
                           if a / max(1, b[2] * b[3]) < min_fill)
        for group in groups_after(boxes, areas, min_fill):
            kept += 1
            x0 = min(boxes[i][0] for i in group)
            y0 = min(boxes[i][1] for i in group)
            x1 = max(boxes[i][0] + boxes[i][2] for i in group)
            y1 = max(boxes[i][1] + boxes[i][3] for i in group)
            box_px = (x1 - x0) * (y1 - y0)
            groups += len(group) > 1
            big5 += box_px / PAGE_PX > 0.05
            big20 += box_px / PAGE_PX > 0.20
            biggest = max(biggest, box_px / PAGE_PX)
    print(f"{min_fill:14.2f} {kept:7d} {groups:7d} {big5:8d} {big20:6d}"
          f" {biggest*100:7.2f}% {dropped:8d}")
