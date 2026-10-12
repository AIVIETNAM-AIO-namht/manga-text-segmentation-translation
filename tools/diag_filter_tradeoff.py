"""Trade-off of dropping low-evidence regions.

Ground truth tells us which regions are real. For each candidate filter we
report what it REMOVES (rendered text that was never real -- the user's
symptom) against what it LOSES (real text that would go untranslated).

Only features available at inference time are used. GT is the label only.
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
            "stroke": float(np.percentile(d, 90)) if d.size else 0.0,
            "w": w, "h": h,
        })

false_ink = sum(r["ink"] for r in rows if r["false"])
true_ink = sum(r["ink"] for r in rows if not r["false"])
total = false_ink + true_ink
print(f"total rendered ink: {total:,}")
print(f"  on real text : {true_ink:,} ({100*true_ink/total:.1f}%)")
print(f"  on artwork   : {false_ink:,} ({100*false_ink/total:.1f}%)  <- the user's symptom\n")

# A region is kept iff it has enough evidence. Score = a simple evidence count.
def report(name, keep):
    dropped = [r for r in rows if not keep(r)]
    kept = [r for r in rows if keep(r)]
    removed_false = sum(r["ink"] for r in dropped if r["false"])
    lost_true = sum(r["ink"] for r in dropped if not r["false"])
    # what remains
    rem_false = sum(r["ink"] for r in kept if r["false"])
    print(f"{name:38s} kept {len(kept):4d}/{len(rows)}")
    print(f"    removes false ink {100*removed_false/false_ink:5.1f}% of the symptom"
          f"   |  loses true ink {100*lost_true/true_ink:5.1f}% of real text"
          f"   |  residual false {100*rem_false/total:4.1f}% of all ink")


print("candidate filters (keeping = region survives):\n")
report("drop nothing (current)", lambda r: True)
report("drop merged_from == 1", lambda r: r["merged_from"] > 1)
report("drop ink < 200", lambda r: r["ink"] >= 200)
report("drop ink < 400", lambda r: r["ink"] >= 400)
report("drop merged_from==1 and ink<300",
       lambda r: not (r["merged_from"] == 1 and r["ink"] < 300))
report("drop merged_from==1 and ink<600",
       lambda r: not (r["merged_from"] == 1 and r["ink"] < 600))

print("\nevidence score = (merged_from>=2) + (ink>=300) + (stroke>=3) + (h>=25):")
for need in (1, 2, 3, 4):
    def keep(r, need=need):
        score = ((r["merged_from"] >= 2) + (r["ink"] >= 300)
                 + (r["stroke"] >= 3) + (r["h"] >= 25))
        return score >= need
    report(f"  keep if score >= {need}", keep)
