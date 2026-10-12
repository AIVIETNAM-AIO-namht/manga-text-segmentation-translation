"""What does the EXISTING min_area knob buy?

min_area filters connected components before merging, so it is a pure config
change -- no code, no test breakage. Simulate the real pipeline (component
filter -> single-linkage merge -> union bbox) at several min_area values and
score each against ground truth the same way as the other diagnostics.
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
MAX_AREA, MERGE_DISTANCE = 12000, 8


def load_gt(image_id):
    book, stem = image_id.split("/")
    p = GT / book / f"{stem}.png"
    if not p.is_file():
        return None
    rgb = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)[:1170, :1654]
    return (rgb[:, :, 0] != 255) | (rgb[:, :, 1] != 255) | (rgb[:, :, 2] != 255)


def merge_groups(boxes, min_area):
    """Component area filter + single-linkage proximity merge (mirrors regions.py)."""
    keep = [i for i, b in enumerate(boxes) if min_area <= b["area"] <= MAX_AREA]
    n = len(keep)
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
            lx, ly, lw, lh = (boxes[keep[a]][k] for k in ("x", "y", "w", "h"))
            rx, ry, rw, rh = (boxes[keep[b]][k] for k in ("x", "y", "w", "h"))
            hg = max(lx - (rx + rw), rx - (lx + lw), 0)
            vg = max(ly - (ry + rh), ry - (ly + lh), 0)
            if hg < MERGE_DISTANCE and vg < MERGE_DISTANCE:
                parent[max(ra, rb)] = min(ra, rb)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(keep[i])
    return list(groups.values())


pages = []
for page in sorted(glob.glob("outputs/translation/spec004-vilao/*/*/*/metadata.json")):
    meta = json.load(io.open(page, encoding="utf-8"))
    gt = load_gt(meta["image_id"])
    if gt is None:
        continue
    page_dir = Path(page).parent
    mask = cv2.imread(str(page_dir / "mask.png"), cv2.IMREAD_GRAYSCALE) > 127
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    boxes = [{"x": int(s[0]), "y": int(s[1]), "w": int(s[2]), "h": int(s[3]),
              "area": int(s[4])} for s in stats[1:]]
    pages.append((mask, gt, boxes))

# baseline totals (unfiltered) for normalisation
base_false = base_true = 0
for mask, gt, boxes in pages:
    for b in boxes:
        sub = mask[b["y"]:b["y"] + b["h"], b["x"]:b["x"] + b["w"]]
        ink = int(sub.sum())
        if not ink:
            continue
        hit = int((sub & gt[b["y"]:b["y"] + b["h"], b["x"]:b["x"] + b["w"]]).sum()) / ink
        if hit < 0.01:
            base_false += ink
        else:
            base_true += ink
total = base_false + base_true
print(f"baseline: {total:,} ink   false {100*base_false/total:.1f}%   true {100*base_true/total:.1f}%\n")

print(f"{'min_area':>9s} {'regions':>8s} {'false%all':>10s} {'removed':>8s} {'lostTrue':>9s}")
for min_area in (50, 100, 150, 200, 300, 400, 500, 700, 1000):
    false_ink = true_ink = kept = 0
    for mask, gt, boxes in pages:
        for group in merge_groups(boxes, min_area):
            x0 = min(boxes[i]["x"] for i in group)
            y0 = min(boxes[i]["y"] for i in group)
            x1 = max(boxes[i]["x"] + boxes[i]["w"] for i in group)
            y1 = max(boxes[i]["y"] + boxes[i]["h"] for i in group)
            sub = mask[y0:y1, x0:x1]
            ink = int(sub.sum())
            if not ink:
                continue
            hit = int((sub & gt[y0:y1, x0:x1]).sum()) / ink
            kept += 1
            if hit < 0.01:
                false_ink += ink
            else:
                true_ink += ink
    removed = 100 * (base_false - false_ink) / base_false
    lost = 100 * (base_true - true_ink) / base_true
    print(f"{min_area:9d} {kept:8d} {100*false_ink/total:9.1f}% {removed:7.1f}% {lost:8.1f}%")
