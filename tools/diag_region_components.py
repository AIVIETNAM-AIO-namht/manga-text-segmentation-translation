"""Dump the connected components behind the worst-placed merged regions."""

import glob
import io
import json
from pathlib import Path

import cv2
import numpy as np

TARGETS = [
    ("AisazuNihaIrarenai/004", 137),
    ("AkkeraKanjinchou/003", 94),
    ("ARMS/002", 18),
    ("AisazuNihaIrarenai/003", 128),
]
MERGE_DISTANCE = 8
MIN_AREA = 50
MAX_AREA = 12000


def groups(components, distance):
    parent = list(range(len(components)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for left in range(len(components)):
        for right in range(left + 1, len(components)):
            lx, ly, lw, lh = components[left][:4]
            rx, ry, rw, rh = components[right][:4]
            hg = max(lx - (rx + rw), rx - (lx + lw), 0)
            vg = max(ly - (ry + rh), ry - (ly + lh), 0)
            if hg < distance and vg < distance:
                a, b = find(left), find(right)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    out = {}
    for index in range(len(components)):
        out.setdefault(find(index), []).append(index)
    return [out[key] for key in sorted(out)]


for image_id, region_id in TARGETS:
    page_dir = next(
        Path(p).parent
        for p in glob.glob("outputs/translation/spec004-vilao/*/*/*/metadata.json")
        if json.load(io.open(p, encoding="utf-8"))["image_id"] == image_id
    )
    record = json.load(io.open(page_dir / "metadata.json", encoding="utf-8"))
    region = next(r for r in record["regions"] if r["region_id"] == region_id)
    mask = cv2.imread(str(page_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = [
        (int(s[0]), int(s[1]), int(s[2]), int(s[3]), int(s[4]))
        for s in stats[1:]
    ]
    kept = [b for b in boxes if MIN_AREA <= b[4] <= MAX_AREA]
    print(f"\n=== {image_id} region {region_id}  recorded_box={region['bbox']}"
          f" merged_from={region['merged_from']} total_components={len(boxes)}"
          f" kept={len(kept)}")
    mine = None
    for group in groups(kept, MERGE_DISTANCE):
        x0 = min(kept[i][0] for i in group)
        y0 = min(kept[i][1] for i in group)
        x1 = max(kept[i][0] + kept[i][2] for i in group)
        y1 = max(kept[i][1] + kept[i][3] for i in group)
        if (x0, y0, x1 - x0, y1 - y0) == tuple(region["bbox"]):
            mine = group
            break
    if mine is None:
        print("  group not reproduced")
        continue
    members = sorted((kept[i] for i in mine), key=lambda b: (b[1], b[0]))
    for bx, by, bw, bh, area in members:
        print(f"    box=({bx:5d},{by:5d},{bw:5d},{bh:5d}) area={area}")
    # Which single merge step is the bridge? Report the largest pairwise gap
    # between consecutive members in reading order.
    print(f"  member count {len(members)}")
