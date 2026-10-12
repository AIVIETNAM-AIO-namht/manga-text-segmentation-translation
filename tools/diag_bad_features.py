"""Which feature actually separates the badly-placed regions from the good ones?"""

import glob
import io
import json
import math
from pathlib import Path

import cv2
import numpy as np

PAGES = sorted(glob.glob("outputs/translation/spec004-vilao/*/*/*/metadata.json"))
PAGE = 1654 * 1170
rows = []

for page in PAGES:
    record = json.load(io.open(page, encoding="utf-8"))
    page_dir = Path(page).parent
    rendered = cv2.imread(str(page_dir / "rendered.png"), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(page_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
    if rendered is None or mask is None:
        continue
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
        relative = offset / math.hypot(width, height)
        area = width * height
        rows.append(
            {
                "relative": relative,
                "bad": relative > 0.15,
                "merged": region["merged_from"],
                "area": area,
                "page_frac": area / PAGE,
                "fill": old_window.sum() / area,
                "aspect": max(width, height) / max(1, min(width, height)),
                "font": region.get("font_size_used") or 0,
                "overflow": bool(region.get("overflow_warning")),
                "status": region.get("render_status"),
            }
        )


def summarize(label, subset):
    if not subset:
        return
    n = len(subset)
    print(
        f"{label:16s} n={n:5d}  bad={100 * sum(r['bad'] for r in subset) / n:5.1f}%"
        f"  med_area={np.median([r['area'] for r in subset]):9.0f}"
        f"  med_page={np.median([r['page_frac'] for r in subset]):.3f}"
        f"  med_fill={np.median([r['fill'] for r in subset]):.3f}"
        f"  med_merged={np.median([r['merged'] for r in subset]):.0f}"
        f"  med_font={np.median([r['font'] for r in subset]):.0f}"
    )


print(f"total measured: {len(rows)}  bad: {sum(r['bad'] for r in rows)}")
summarize("ALL", rows)
print()
summarize("bad", [r for r in rows if r["bad"]])
summarize("good", [r for r in rows if not r["bad"]])
print()
summarize("merged_from=1", [r for r in rows if r["merged"] == 1])
summarize("merged_from>1", [r for r in rows if r["merged"] > 1])
summarize("merged_from>=10", [r for r in rows if r["merged"] >= 10])
print()
for threshold in (0.02, 0.05, 0.10, 0.20):
    big = [r for r in rows if r["page_frac"] >= threshold]
    summarize(f"box>={threshold:.2f}page", big)
print()
for threshold in (0.02, 0.05, 0.10):
    sparse = [r for r in rows if r["fill"] <= threshold]
    summarize(f"fill<={threshold:.2f}", sparse)
print()
for threshold in (2, 4, 8):
    tall = [r for r in rows if r["aspect"] >= threshold]
    summarize(f"aspect>={threshold}", tall)
print()
summarize("overflow", [r for r in rows if r["overflow"]])
summarize("no overflow", [r for r in rows if not r["overflow"]])
