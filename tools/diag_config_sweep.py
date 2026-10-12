"""Can a config change reduce the symptom without losing real text?

min_area and merge_distance are both live config knobs (RegionExtractionConfig).
Sweep them jointly against ground truth and report, for each pair, how much of
the false ink (rendered text on artwork = the user's symptom) is removed and
how much real ink is lost. A pair DOMINATES the current (50, 8) when it removes
more false ink AND loses no more true ink.
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
MAX_AREA = 12000


def load_gt(image_id):
    book, stem = image_id.split("/")
    p = GT / book / f"{stem}.png"
    if not p.is_file():
        return None
    rgb = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)[:1170, :1654]
    return (rgb[:, :, 0] != 255) | (rgb[:, :, 1] != 255) | (rgb[:, :, 2] != 255)


def merge_groups(boxes, min_area, distance):
    """Mirror regions.py: area filter, then single-linkage proximity merge."""
    keep = [i for i, b in enumerate(boxes) if min_area <= b["area"] <= MAX_AREA]
    n = len(keep)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if distance > 0:
        for a in range(n):
            for b in range(a + 1, n):
                ra, rb = find(a), find(b)
                if ra == rb:
                    continue
                lx, ly, lw, lh = (boxes[keep[a]][k] for k in ("x", "y", "w", "h"))
                rx, ry, rw, rh = (boxes[keep[b]][k] for k in ("x", "y", "w", "h"))
                hg = max(lx - (rx + rw), rx - (lx + lw), 0)
                vg = max(ly - (ry + rh), ry - (ly + lh), 0)
                if hg < distance and vg < distance:
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


def score(min_area, distance):
    """Region-level: an isolated speckle region gets its OWN OCR call and its
    OWN render box on artwork -- that is the user's symptom. Count regions, not
    ink: once regions merge, the union box swallows GT and ink-weighting lies."""
    false_ink = true_ink = kept = false_regions = true_regions = 0
    bbox_areas = []
    for mask, gt, boxes in pages:
        for group in merge_groups(boxes, min_area, distance):
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
            bbox_areas.append((x1 - x0) * (y1 - y0))
            if hit < 0.01:
                false_ink += ink
                false_regions += 1
            else:
                true_ink += ink
                true_regions += 1
    bbox_areas = np.array(bbox_areas, dtype=float)
    return (kept, false_ink, true_ink, false_regions, true_regions,
            float(np.median(bbox_areas)), float(np.percentile(bbox_areas, 95)))


# Unfiltered baseline: every component is its own region (no area filter, no merge).
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
print(f"baseline (no filter, no merge): {total:,} ink"
      f"  false {100*base_false/total:.1f}%  true {100*base_true/total:.1f}%\n")

# Region-count baseline: every component is its own region.
base_false_regions = base_true_regions = 0
for mask, gt, boxes in pages:
    for b in boxes:
        sub = mask[b["y"]:b["y"] + b["h"], b["x"]:b["x"] + b["w"]]
        ink = int(sub.sum())
        if not ink:
            continue
        hit = int((sub & gt[b["y"]:b["y"] + b["h"], b["x"]:b["x"] + b["w"]]).sum()) / ink
        if hit < 0.01:
            base_false_regions += 1
        else:
            base_true_regions += 1
base_regions = base_false_regions + base_true_regions
print(f"baseline regions: {base_regions}  false {base_false_regions}"
      f" ({100*base_false_regions/base_regions:.1f}%)  true {base_true_regions}\n")

print(f"{'min_area':>8s} {'merge_d':>8s} {'regions':>8s} {'falseReg':>9s} {'falseReg%':>10s}"
      f" {'trueLost':>9s} {'removed':>8s} {'lostTrue':>9s} {'medBox':>9s} {'p95Box':>9s}")
rows = []
for min_area in (50, 80, 120):
    for distance in (0, 4, 8, 12, 16, 24, 32, 48):
        kept, false_ink, true_ink, fr, tr, med, p95 = score(min_area, distance)
        removed = 100 * (base_false - false_ink) / base_false
        lost = 100 * (base_true - true_ink) / base_true
        rows.append((min_area, distance, kept, removed, lost, fr, tr))
        print(f"{min_area:8d} {distance:8d} {kept:8d} {fr:9d} {100*fr/kept:9.1f}%"
              f" {base_true_regions - tr:9d} {removed:7.1f}% {lost:8.1f}%"
              f" {med:9.0f} {p95:9.0f}")

cur = next(r for r in rows if r[0] == 50 and r[1] == 8)
print(f"\ncurrent = min_area 50 / merge_distance 8:"
      f" {cur[5]} false regions of {cur[2]} ({100*cur[5]/cur[2]:.1f}%)"
      f"  ink removed {cur[3]:.1f}%  real-text regions lost {cur[6]}")
