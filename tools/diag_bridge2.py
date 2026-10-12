"""Re-test the bridge hypothesis with a thickness threshold that catches 663x12 rules."""

import glob
import io
import json
import math
from pathlib import Path

import cv2
import numpy as np

PAGES = sorted(glob.glob("outputs/translation/spec004-vilao/*/*/*/metadata.json"))
MIN_AREA, MAX_AREA, MERGE_DISTANCE = 50, 12000, 8


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


def stats(records, label):
    if not records:
        return
    n = len(records)
    bad = sum(r["bad"] for r in records)
    print(
        f"{label:34s} n={n:5d}  bad={100 * bad / n:5.1f}%"
        f"  med_fill={np.median([r['fill'] for r in records]):.3f}"
        f"  med_merged={np.median([r['merged'] for r in records]):.0f}"
    )


all_rows = []
for page in PAGES:
    record = json.load(io.open(page, encoding="utf-8"))
    page_dir = Path(page).parent
    rendered = cv2.imread(str(page_dir / "rendered.png"), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(page_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
    if rendered is None or mask is None:
        continue
    _, _, raw_stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = [tuple(int(v) for v in s) for s in raw_stats[1:]]
    kept = [b for b in boxes if MIN_AREA <= b[4] <= MAX_AREA]

    group_info = {}
    for group in groups(kept, MERGE_DISTANCE):
        x0 = min(kept[i][0] for i in group)
        y0 = min(kept[i][1] for i in group)
        x1 = max(kept[i][0] + kept[i][2] for i in group)
        y1 = max(kept[i][1] + kept[i][3] for i in group)
        key = (x0, y0, x1 - x0, y1 - y0)
        members = [kept[i] for i in group]
        # A component whose ink is a thin sliver but whose extent is long is a
        # rule or border: it bridges strokes that are nowhere near each other.
        thin = max(
            (
                min(b[2], b[3]) <= 16 and max(b[2], b[3]) >= 40
            )
            for b in members
        )
        group_info[key] = thin

    ink_new = rendered < 100
    ink_old = mask > 127
    for region in record["regions"]:
        x, y, width, height = region["bbox"]
        new_window = ink_new[y : y + height, x : x + width]
        old_window = ink_old[y : y + height, x : x + width]
        if new_window.sum() < 5 or old_window.sum() < 5:
            continue
        ny, nx = np.nonzero(new_window)
        oy, ox = np.nonzero(old_window)
        offset = math.hypot(nx.mean() - ox.mean(), ny.mean() - oy.mean())
        all_rows.append(
            {
                "bad": offset / math.hypot(width, height) > 0.15,
                "thin": group_info.get(tuple(region["bbox"]), False),
                "fill": old_window.sum() / (width * height),
                "merged": region["merged_from"],
                "page_frac": width * height / (1654 * 1170),
            }
        )

print(f"total {len(all_rows)}")
stats(all_rows, "ALL")
stats([r for r in all_rows if r["thin"]], "group contains thin rule")
stats([r for r in all_rows if not r["thin"]], "no thin rule")
print()
# Does chaining (sparse fill) rather than rules explain the tail?
stats([r for r in all_rows if r["fill"] < 0.15], "ink fill < 0.15")
stats([r for r in all_rows if r["fill"] >= 0.15], "ink fill >= 0.15")
stats([r for r in all_rows if r["page_frac"] > 0.01], "box > 1% of page")
stats([r for r in all_rows if r["merged"] > 1], "merged")
