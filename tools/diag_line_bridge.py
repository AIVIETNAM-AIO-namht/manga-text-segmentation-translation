"""Do line-like mask components explain the badly-placed regions?"""

import glob
import io
import json
import math
from pathlib import Path

import cv2
import numpy as np

PAGES = sorted(glob.glob("outputs/translation/spec004-vilao/*/*/*/metadata.json"))
MIN_AREA, MAX_AREA, MERGE_DISTANCE = 50, 12000, 8


def is_line(box):
    x, y, w, h, _ = box
    short, long_ = min(w, h), max(w, h)
    return short <= 8 and long_ >= 60


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


line_components = 0
total_components = 0
displaced_with_line = 0
displaced_total = 0
big_with_line = 0
big_total = 0
saved = []

for page in PAGES:
    record = json.load(io.open(page, encoding="utf-8"))
    page_dir = Path(page).parent
    rendered = cv2.imread(str(page_dir / "rendered.png"), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(page_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
    if rendered is None or mask is None:
        continue
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = [tuple(int(v) for v in s) for s in stats[1:]]
    total_components += len(boxes)
    line_components += sum(1 for b in boxes if is_line(b))
    kept = [b for b in boxes if MIN_AREA <= b[4] <= MAX_AREA]

    group_has_line = {}
    group_boxes = {}
    for group in groups(kept, MERGE_DISTANCE):
        x0 = min(kept[i][0] for i in group)
        y0 = min(kept[i][1] for i in group)
        x1 = max(kept[i][0] + kept[i][2] for i in group)
        y1 = max(kept[i][1] + kept[i][3] for i in group)
        key = (x0, y0, x1 - x0, y1 - y0)
        group_has_line[key] = any(is_line(kept[i]) for i in group)
        group_boxes[key] = (x0, y0, x1 - x0, y1 - y0)

    ink_new = rendered < 100
    ink_old = mask > 127
    for region in record["regions"]:
        box = tuple(region["bbox"])
        x, y, width, height = box
        new_window = ink_new[y : y + height, x : x + width]
        old_window = ink_old[y : y + height, x : x + width]
        if new_window.sum() < 5 or old_window.sum() < 5:
            continue
        ny, nx = np.nonzero(new_window)
        oy, ox = np.nonzero(old_window)
        offset = math.hypot(nx.mean() - ox.mean(), ny.mean() - oy.mean())
        relative = offset / math.hypot(width, height)
        bad = relative > 0.15
        has_line = group_has_line.get(box, False)
        big = region["merged_from"] > 1
        if bad:
            displaced_total += 1
            displaced_with_line += has_line
        if big:
            big_total += 1
            big_with_line += has_line
        if bad and has_line:
            saved.append((offset, relative, region["merged_from"], page_dir.parts[-2]))

print(f"mask components: {total_components}, line-like: {line_components}"
      f" ({100 * line_components / total_components:.1f}%)")
print(f"badly placed (>15% diag): {displaced_total}, of which contain a line:"
      f" {displaced_with_line} ({100 * displaced_with_line / max(1, displaced_total):.0f}%)")
print(f"merged regions: {big_total}, containing a line: {big_with_line}"
      f" ({100 * big_with_line / max(1, big_total):.0f}%)")
