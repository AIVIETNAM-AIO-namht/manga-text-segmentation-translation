"""How much longer is the Vietnamese text than the Japanese it replaces?"""

import glob
import io
import json
from pathlib import Path

import numpy as np

PAGES = sorted(glob.glob("outputs/translation/spec004-vilao/*/*/*/metadata.json"))
ratios = []
rows = []
for page in PAGES:
    record = json.load(io.open(page, encoding="utf-8"))
    for region in record["regions"]:
        jp = (region.get("ocr_text") or "").strip()
        vi = (region.get("translation") or "").strip()
        if not jp or not vi:
            continue
        rows.append((len(jp), len(vi), region["region_id"], record["image_id"]))

jp_len = np.array([r[0] for r in rows])
vi_len = np.array([r[1] for r in rows])
ratio = vi_len / np.maximum(jp_len, 1)
print(f"regions with both texts: {len(rows)}")
print(f"japanese chars: median {np.median(jp_len):.0f}  mean {jp_len.mean():.1f}")
print(f"vietnamese chars: median {np.median(vi_len):.0f}  mean {vi_len.mean():.1f}")
print(f"length ratio vi/jp: median {np.median(ratio):.2f}  p25 {np.percentile(ratio, 25):.2f}"
      f"  p75 {np.percentile(ratio, 75):.2f}  p90 {np.percentile(ratio, 90):.2f}")
print(f"regions where vietnamese is longer: {100 * (ratio > 1).mean():.1f}%")
print(f"regions where vietnamese is >2x longer: {100 * (ratio > 2).mean():.1f}%")

# A Japanese glyph is roughly square; a latin glyph is roughly half as wide.
# So visual width ratio ~= (vi_len * 0.5) / (jp_len * 1.0)
visual = (vi_len * 0.5) / np.maximum(jp_len, 1)
print(f"\nestimated visual width ratio (latin 0.5em vs CJK 1em): median {np.median(visual):.2f}"
      f"  p75 {np.percentile(visual, 75):.2f}  p90 {np.percentile(visual, 90):.2f}")
print(f"regions whose translation needs >1.5x the width: {100 * (visual > 1.5).mean():.1f}%")
