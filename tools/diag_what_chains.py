"""What ARE the components that chain groups into page-spanning boxes?

If they are panel borders / rules, they are not text and should not be merged
into a text region at all.
"""

import glob
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

MIN_AREA, MAX_AREA = 50, 12000
PAGE_W, PAGE_H = 1654, 1170

TARGETS = ["AisazuNihaIrarenai/004", "AisazuNihaIrarenai/009", "AkkeraKanjinchou/003"]

for page in sorted(glob.glob("outputs/translation/spec004-vilao/*/*/*/metadata.json")):
    record = __import__("json").loads(Path(page).read_text(encoding="utf-8"))
    if record["image_id"] not in TARGETS:
        continue
    mask = cv2.imread(str(Path(page).parent / "mask.png"), cv2.IMREAD_GRAYSCALE)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = [s for s in stats[1:] if MIN_AREA <= s[4] <= MAX_AREA]

    print(f"\n=== {record['image_id']}  kept={len(keep)} ===")
    # Components that are extreme in one dimension: candidates for rules/borders.
    line = [s for s in keep
            if max(s[2], s[3]) >= 60 and max(s[2], s[3]) / max(1, min(s[2], s[3])) >= 6]
    print(f"  long/thin components (>=60px, aspect>=6): {len(line)}")
    for s in sorted(line, key=lambda s: -max(s[2], s[3]))[:8]:
        fill = s[4] / max(1, s[2] * s[3])
        print(f"    ({s[0]:4d},{s[1]:4d},{s[2]:4d},{s[3]:4d}) area={s[4]:6d}"
              f" aspect={max(s[2], s[3]) / max(1, min(s[2], s[3])):5.1f} fill={fill:.2f}")

    # Big blobs (real text lines are blobs with decent fill)
    blob = [s for s in keep if s[2] * s[3] >= 400]
    print(f"  blobs (bbox>=400px2): {len(blob)}"
          f"  median fill {np.median([s[4] / max(1, s[2] * s[3]) for s in blob]):.2f}")
