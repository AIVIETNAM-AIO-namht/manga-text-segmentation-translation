"""Does rendered text ink actually land on the original text? Direct overlap measure."""

import glob
import io
import json
from pathlib import Path

import cv2
import numpy as np

PAGES = sorted(glob.glob("outputs/translation/spec004-vilao/*/*/*/metadata.json"))
NEAR = 12  # px tolerance: rendered ink within this of original ink counts as landed

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
    kernel = np.ones((2 * NEAR + 1, 2 * NEAR + 1), np.uint8)
    near_old = cv2.dilate(ink_old.astype(np.uint8), kernel).astype(bool)
    for region in record["regions"]:
        x, y, width, height = region["bbox"]
        new_window = ink_new[y : y + height, x : x + width]
        if new_window.sum() < 5:
            continue
        landed = (new_window & near_old[y : y + height, x : x + width]).sum()
        fraction = landed / new_window.sum()
        rows.append(
            {
                "fraction": fraction,
                "bad": fraction < 0.5,
                "page_frac": width * height / (1654 * 1170),
                "merged": region["merged_from"],
                "status": region.get("render_status"),
                "image": record["image_id"],
                "region_id": region["region_id"],
            }
        )

rows.sort(key=lambda r: r["fraction"])
n = len(rows)
print(f"regions measured: {n}")
print(f"rendered ink landing on original text:")
print(f"  <25%: {sum(r['fraction'] < 0.25 for r in rows):5d}"
      f" ({100 * sum(r['fraction'] < 0.25 for r in rows) / n:.1f}%)")
print(f"  <50%: {sum(r['bad'] for r in rows):5d} ({100 * sum(r['bad'] for r in rows) / n:.1f}%)")
print(f"  <75%: {sum(r['fraction'] < 0.75 for r in rows):5d}"
      f" ({100 * sum(r['fraction'] < 0.75 for r in rows) / n:.1f}%)")
fractions = np.array([r["fraction"] for r in rows])
for p in (5, 10, 25, 50, 75):
    print(f"  p{p}: {np.percentile(fractions, p):.3f}")
print("\nby box size:")
for lo, hi in ((0, 0.001), (0.001, 0.01), (0.01, 0.05), (0.05, 1.0)):
    subset = [r for r in rows if lo <= r["page_frac"] < hi]
    if subset:
        bad = sum(r["bad"] for r in subset)
        print(f"  page_frac [{lo:.3f},{hi:.3f}) n={len(subset):5d}"
              f"  median landing {np.median([r['fraction'] for r in subset]):.3f}"
              f"  bad {100 * bad / len(subset):5.1f}%")
print("\nworst 15:")
for r in rows[:15]:
    print(f"  {r['image']} region {r['region_id']:4d}: landing {r['fraction']:.3f}"
          f"  page_frac {r['page_frac']:.3f} merged={r['merged']} {r['status']}")
