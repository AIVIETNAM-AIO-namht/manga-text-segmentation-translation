"""Compare merge strategies on a structural metric: does the group box match its ink?

A correct text-line group has ink filling a decent share of its bbox. A group chained
across the page by a border strip has a box thousands of times larger than its ink.

Every variant is a *union predicate* — the decision is made when merging two
components, so no post-hoc group surgery.
"""

import glob
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

MIN_AREA, MAX_AREA, MERGE_DISTANCE = 50, 12000, 8
PAGE_PX = 1654 * 1170


def find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def gaps(a, b):
    lx, ly, lw, lh = a
    rx, ry, rw, rh = b
    hg = max(lx - (rx + rw), rx - (lx + lw), 0)
    vg = max(ly - (ry + rh), ry - (ly + lh), 0)
    return hg, vg


def boxes_of(boxes, group):
    x0 = min(boxes[i][0] for i in group)
    y0 = min(boxes[i][1] for i in group)
    x1 = max(boxes[i][0] + boxes[i][2] for i in group)
    y1 = max(boxes[i][1] + boxes[i][3] for i in group)
    return x0, y0, x1, y1


def build(boxes, areas, variant, param):
    """Union-find where each candidate merge is gated by `variant`."""
    n = len(boxes)
    parent = list(range(n))
    members = {i: [i] for i in range(n)}
    for a in range(n):
        for b in range(a + 1, n):
            ra, rb = find(parent, a), find(parent, b)
            if ra == rb:
                continue
            hg, vg = gaps(boxes[a], boxes[b])
            if not (hg < MERGE_DISTANCE and vg < MERGE_DISTANCE):
                continue
            if variant in ("nodiff", "both", "both2"):  # must overlap on at least one axis
                if hg > 0 and vg > 0:
                    continue
            if variant == "hub":  # line-like components never participate
                if any(param(boxes[i]) for i in (a, b)):
                    continue
            if variant in ("rule", "rule2"):  # thin long rules never bridge
                minor, major = param
                if any(min(b[2], b[3]) <= minor and max(b[2], b[3]) >= major
                       for b in (boxes[a], boxes[b])):
                    continue
            merged = members[ra] + members[rb]
            if variant in ("fill", "both"):  # refuse a merge that guts density
                x0, y0, x1, y1 = boxes_of(boxes, merged)
                ink = sum(areas[i] for i in merged)
                if ink / max(1, (x1 - x0) * (y1 - y0)) < param:
                    continue
            if variant in ("bboxfill", "both2") or (variant == "rule2" and param is not None):  # members must tile the union box
                x0, y0, x1, y1 = boxes_of(boxes, merged)
                own = sum(b[2] * b[3] for b in (boxes[i] for i in merged))
                gate = 0.50 if variant == "rule2" else param
                if own / max(1, (x1 - x0) * (y1 - y0)) < gate:
                    continue
            if variant == "grow":  # refuse a merge that inflates the box too far
                x0, y0, x1, y1 = boxes_of(boxes, merged)
                if (x1 - x0) * (y1 - y0) > param * sum(
                    b[2] * b[3] for b in (boxes[i] for i in merged)
                ):
                    continue
            parent[max(ra, rb)] = min(ra, rb)
            members[min(ra, rb)] = merged
    out = {}
    for i in range(n):
        out.setdefault(find(parent, i), []).append(i)
    return list(out.values())


def aspect(box):
    return max(box[2], box[3]) / max(1, min(box[2], box[3]))


VARIANTS = [
    ("current (proximity only)", "none", None),
    ("axis-overlap required", "nodiff", None),
    ("fill >= 0.20 (no axis gate)", "fill", 0.20),
    ("fill >= 0.35 (no axis gate)", "fill", 0.35),
    ("axis-overlap + fill >= 0.20", "both", 0.20),
    ("axis-overlap + fill >= 0.35", "both", 0.35),
]

pages = sorted(glob.glob("outputs/translation/spec004-vilao/*/*/*/metadata.json"))
records = []
for page in pages:
    mask = cv2.imread(str(Path(page).parent / "mask.png"), cv2.IMREAD_GRAYSCALE)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = [s for s in stats[1:] if MIN_AREA <= s[4] <= MAX_AREA]
    records.append((
        [(int(s[0]), int(s[1]), int(s[2]), int(s[3])) for s in keep],
        [int(s[4]) for s in keep],
    ))

print(f"pages={len(records)}  components/page median="
      f"{np.median([len(b) for b, _ in records]):.0f}")
print("  (reference: regions/page in metadata = 148 median)\n")

header = (f"{'variant':34s} {'groups':>7s} {'p1fill':>7s} {'p10fill':>8s}"
          f" {'<0.05':>7s} {'>5%page':>8s} {'>20%':>6s} {'biggest':>8s}")
print(header)
print("-" * len(header))
for label, variant, param in VARIANTS:
    fills, merged, big5, big20, biggest = [], 0, 0, 0, 0.0
    for boxes, areas in records:
        for group in build(boxes, areas, variant, param):
            x0, y0, x1, y1 = boxes_of(boxes, group)
            box_px = (x1 - x0) * (y1 - y0)
            ink = sum(areas[i] for i in group)
            fills.append(ink / max(1, box_px))
            merged += len(group) > 1
            share = box_px / PAGE_PX
            big5 += share > 0.05
            big20 += share > 0.20
            biggest = max(biggest, share)
    fills = np.array(fills)
    print(f"{label:34s} {len(fills):7d} {np.percentile(fills, 1):7.3f}"
          f" {np.percentile(fills, 10):8.3f} {100 * (fills < 0.05).mean():6.1f}%"
          f" {big5:8d} {big20:6d} {biggest * 100:7.2f}%")
