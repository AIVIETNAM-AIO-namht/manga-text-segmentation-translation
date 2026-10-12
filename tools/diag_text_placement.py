"""Diagnostic: how far is rendered Vietnamese text from the original ink?"""

import glob
import io
import json
import math
from pathlib import Path

import cv2
import numpy as np

PAGES = sorted(glob.glob("outputs/translation/spec004-vilao/*/*/*/metadata.json"))
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
        new_y, new_x = np.nonzero(new_window)
        old_y, old_x = np.nonzero(old_window)
        offset = math.hypot(new_x.mean() - old_x.mean(), new_y.mean() - old_y.mean())
        rows.append(
            (
                offset,
                offset / math.hypot(width, height),
                region["merged_from"],
                width * height,
                region["region_id"],
                page_dir.parts[-2],
                record["image_id"],
            )
        )

rows.sort(reverse=True)
offsets = np.array([row[0] for row in rows])
relative = np.array([row[1] for row in rows])
print(f"regions measured: {len(rows)}")
for percentile in (50, 75, 90, 95, 99):
    print(
        f"  p{percentile}: offset {np.percentile(offsets, percentile):6.1f}px"
        f"   relative {np.percentile(relative, percentile):.3f}"
    )
print(f"  max: offset {offsets.max():.1f}px  relative {relative.max():.3f}")

big = [row for row in rows if row[1] > 0.15]
print(
    f"\ndisplaced >15% of box diagonal: {len(big)}"
    f" ({100 * len(big) / len(rows):.1f}%)"
)
print("worst 10:")
for row in rows[:10]:
    print(
        f"  {row[5]}/{row[6]} region {row[4]}: offset {row[0]:7.1f}px"
        f" rel {row[1]:.2f} merged_from={row[2]} box={row[3]}"
    )

# Does the box size predict the displacement? Split by merged_from.
for label, keep in (("single", lambda r: r[2] == 1), ("merged", lambda r: r[2] > 1)):
    subset = np.array([row[0] for row in rows if keep(row)])
    if subset.size:
        print(
            f"\n{label} (n={subset.size}): median offset {np.median(subset):.1f}px,"
            f" p90 {np.percentile(subset, 90):.1f}px"
        )
