"""Would a union-fill gate split legitimate text lines?

A gate that refuses to merge sparse glyphs would shatter a real text line into
one region per character -- OCR poison. So measure the fill of the multi-component
groups the CURRENT code forms: those are the merges we must not break.
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

rows = []
for page in pages:
    mask = cv2.imread(str(Path(page).parent / "mask.png"), cv2.IMREAD_GRAYSCALE)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = [s for s in stats[1:] if MIN_AREA <= s[4] <= MAX_AREA]
    boxes = [(int(s[0]), int(s[1]), int(s[2]), int(s[3])) for s in keep]
    areas = [int(s[4]) for s in keep]
    n = len(boxes)
    parent = list(range(n))
    members = {i: [i] for i in range(n)}

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
            lx, ly, lw, lh = boxes[a]
            rx, ry, rw, rh = boxes[b]
            hg = max(lx - (rx + rw), rx - (lx + lw), 0)
            vg = max(ly - (ry + rh), ry - (ly + lh), 0)
            if hg < MERGE_DISTANCE and vg < MERGE_DISTANCE:
                parent[max(ra, rb)] = min(ra, rb)
                members[min(ra, rb)] = members[ra] + members[rb]

    out = {}
    for i in range(n):
        out.setdefault(find(i), []).append(i)
    for group in out.values():
        if len(group) < 2:      # singletons: no merge decision, unaffected by any gate
            continue
        x0 = min(boxes[i][0] for i in group)
        y0 = min(boxes[i][1] for i in group)
        x1 = max(boxes[i][0] + boxes[i][2] for i in group)
        y1 = max(boxes[i][1] + boxes[i][3] for i in group)
        box_px = (x1 - x0) * (y1 - y0)
        rows.append((sum(areas[i] for i in group) / max(1, box_px),
                     len(group), x1 - x0, y1 - y0, box_px / PAGE_PX))

fill = np.array([r[0] for r in rows])
print(f"multi-component groups (the merges we must not break): {len(rows)}")
print(f"  fill percentiles: p1={np.percentile(fill,1):.3f} p5={np.percentile(fill,5):.3f}"
      f" p10={np.percentile(fill,10):.3f} p25={np.percentile(fill,25):.3f}"
      f" median={np.median(fill):.3f}")
for thr in (0.15, 0.20, 0.25, 0.35):
    below = fill < thr
    print(f"  gate {thr:.2f}: {below.sum():5d} of {len(rows)} merges refused"
          f" ({100*below.mean():4.1f}%)")

print("\n  the merges a 0.20 gate would refuse, worst-first (are they text?):")
bad = sorted((r for r in rows if r[0] < 0.20), key=lambda r: r[0])
for f, n, w, h, share in bad[:10]:
    print(f"    fill={f:.3f} merged_from={n:4d} box={w:5d}x{h:<5d}"
          f" = {100*share:6.2f}% of page")
print(f"\n  of the {len(bad)} refused merges, {sum(1 for r in bad if r[4] > 0.05)}"
      f" are >5% of the page (the giants);"
      f" {sum(1 for r in bad if r[4] <= 0.05)} are small boxes")
