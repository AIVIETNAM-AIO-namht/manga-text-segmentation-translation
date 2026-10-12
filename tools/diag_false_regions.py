"""How much of the RENDERED text sits where there is no real text?

GT (spec 002, line 1018) is glyph-level and complete for evaluated pages, so a
region whose mask pixels do not overlap GT is text the pipeline invented on
artwork -- it gets OCR'd, translated, and drawn over the picture.

Weight by MASK INK, not box area: ink is what actually becomes rendered glyphs.
Then ask which region property predicts a false region.
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


def load_gt(image_id):
    book, stem = image_id.split("/")
    path = GT / book / f"{stem}.png"
    if not path.is_file():
        return None
    rgb = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if rgb is None:
        return None
    rgb = rgb[:1170, :1654]
    return (rgb[:, :, 0] != 255) | (rgb[:, :, 1] != 255) | (rgb[:, :, 2] != 255)


rows = []
for page in sorted(glob.glob("outputs/translation/spec004-vilao/*/*/*/metadata.json")):
    meta = json.load(io.open(page, encoding="utf-8"))
    gt = load_gt(meta["image_id"])
    if gt is None:
        continue
    page_dir = Path(page).parent
    mask = cv2.imread(str(page_dir / "mask.png"), cv2.IMREAD_GRAYSCALE) > 127
    regions = json.load(io.open(page_dir / "regions.json", encoding="utf-8"))["regions"]
    for region in regions:
        x, y, w, h = region["bbox"]
        sub = mask[y : y + h, x : x + w]
        ink = int(sub.sum())
        if ink == 0:
            continue
        hit = int((sub & gt[y : y + h, x : x + w]).sum()) / ink
        rows.append({
            "image_id": meta["image_id"],
            "merged_from": region["merged_from"],
            "ink": ink,
            "fill": region["area"] / max(1, w * h),
            "hit": hit,
        })

ink = np.array([r["ink"] for r in rows], dtype=float)
hit = np.array([r["hit"] for r in rows])
total_ink = ink.sum()
print(f"regions with ink: {len(rows)}   total region ink: {total_ink:,.0f} px\n")

print("ink-weighted: share of all rendered text sitting on regions whose mask")
print("does not overlap real text:")
for thr in (0.01, 0.10, 0.30, 0.50):
    false = hit < thr
    print(f"  gt_hit < {thr:.2f}: {100 * ink[false].sum() / total_ink:5.1f}% of rendered text"
          f"   ({false.sum():4d} regions, {100 * false.mean():4.1f}% of regions)")

print("\nwhat predicts a false region? (ink-weighted median per bucket)")
print(f"{'gt_hit':>12s} {'regions':>8s} {'ink%':>7s} {'merged_from':>12s} {'fill':>7s}")
for lo, hi in [(0.0, 0.01), (0.01, 0.10), (0.10, 0.30), (0.30, 0.70), (0.70, 1.01)]:
    sel = hit >= lo
    sel &= hit < hi
    if not sel.any():
        continue
    print(f"{lo:5.2f}-{hi:5.2f} {sel.sum():8d} {100 * ink[sel].sum() / total_ink:6.1f}%"
          f" {np.median([r['merged_from'] for r, s in zip(rows, sel) if s]):12.1f}"
          f" {np.median([r['fill'] for r, s in zip(rows, sel) if s]):7.3f}")

print("\nby merged_from (does merging predict false?):")
for lo, hi in [(1, 2), (2, 5), (5, 15), (15, 10**9)]:
    sel = np.array([lo <= r["merged_from"] < hi for r in rows])
    if not sel.any():
        continue
    print(f"  merged_from {lo:2d}-{hi if hi < 10**8 else '+':>3}: {sel.sum():4d} regions"
          f"  ink%={100 * ink[sel].sum() / total_ink:5.1f}"
          f"  median hit={np.median(hit[sel]):.3f}"
          f"  false-ink(<0.01)={100 * ink[sel & (hit < 0.01)].sum() / max(1, ink[sel].sum()):5.1f}%")
