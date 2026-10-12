"""Is there ANY inference-time signal that separates false regions from real ones?

A usable filter must not peek at the ground truth. So score candidate features
(computed from the mask + OCR output alone) by how well they separate regions
whose mask overlaps real text from those that do not. AUC 0.5 = useless.
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


def auc(scores, labels):
    """Rank-based AUC (Mann-Whitney)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


rows = []
for page in sorted(glob.glob("outputs/translation/spec004-vilao/*/*/*/metadata.json")):
    meta = json.load(io.open(page, encoding="utf-8"))
    book, stem = meta["image_id"].split("/")
    gt_path = GT / book / f"{stem}.png"
    if not gt_path.is_file():
        continue
    rgb = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)[:1170, :1654]
    gt = (rgb[:, :, 0] != 255) | (rgb[:, :, 1] != 255) | (rgb[:, :, 2] != 255)

    page_dir = Path(page).parent
    mask = cv2.imread(str(page_dir / "mask.png"), cv2.IMREAD_GRAYSCALE) > 127
    # distance transform: distance from each ink pixel to background -> stroke radius
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    regions = json.load(io.open(page_dir / "regions.json", encoding="utf-8"))["regions"]

    for rec in meta["regions"]:
        r = regions[rec["region_id"] - 1]
        x, y, w, h = r["bbox"]
        sub = mask[y : y + h, x : x + w]
        ink = int(sub.sum())
        if ink == 0:
            continue
        hit = int((sub & gt[y : y + h, x : x + w]).sum()) / ink
        d = dist[y : y + h, x : x + w][sub]
        rows.append({
            "false": hit < 0.01,
            "ink": ink,
            "merged_from": r["merged_from"],
            "fill": r["area"] / max(1, w * h),
            "w": w,
            "h": h,
            "aspect": max(w, h) / max(1, min(w, h)),
            "stroke": float(d.max()) if d.size else 0.0,      # max half-thickness
            "stroke_med": float(np.median(d)) if d.size else 0.0,
            "compact": r["area"] / max(1, w * h),
            "ocr_len": len((rec.get("ocr_text") or "").strip()),
            "ocr_conf": rec.get("ocr_confidence"),
            "font": rec.get("font_size_used") or 0,
            "nlines": rec.get("line_count") or 0,
        })

labels = [r["false"] for r in rows]
print(f"regions: {len(rows)}   false: {sum(labels)} ({100*np.mean(labels):.1f}%)\n")
print("AUC of each inference-time feature for predicting a FALSE region")
print("(0.5 = no signal; >0.5 = higher value means more likely false):\n")
feats = ["ink", "merged_from", "fill", "w", "h", "aspect", "stroke",
         "stroke_med", "ocr_len", "ocr_conf", "font", "nlines"]
scored = []
for f in feats:
    vals = [r[f] for r in rows]
    if any(v is None for v in vals):
        vals = [(-1 if v is None else v) for v in vals]
    a = auc(vals, labels)
    scored.append((abs(a - 0.5), f, a))
for _, f, a in sorted(scored, reverse=True):
    verdict = "useless" if abs(a - 0.5) < 0.05 else ("weak" if abs(a - 0.5) < 0.12 else "SIGNAL")
    print(f"  {f:12s} AUC={a:.3f}   {verdict}")
