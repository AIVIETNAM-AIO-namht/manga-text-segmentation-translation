"""Whole-page contact sheets: original | inpainted | rendered, for eyeballing."""

import glob
import io
import json
from pathlib import Path

import cv2
import numpy as np

PAGES = sorted(glob.glob("outputs/translation/spec004-vilao/*/*/*/metadata.json"))
SCALE = 0.42
LABEL = 22


def label(panel, text):
    band = np.full((LABEL, panel.shape[1], 3), 255, np.uint8)
    cv2.putText(band, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return np.vstack([band, panel])


out_dir = Path("deliverables/diag/pages")
out_dir.mkdir(parents=True, exist_ok=True)

for page in PAGES:
    record = json.load(io.open(page, encoding="utf-8"))
    page_dir = Path(page).parent
    panels = []
    for name in ("original.png", "inpainted.png", "rendered.png"):
        image = cv2.imread(str(page_dir / name))
        if image is None:
            panels = []
            break
        small = cv2.resize(image, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        panels.append(label(small, name))
    if not panels:
        continue
    sheet = np.hstack(panels)
    name = f"{record['image_id'].replace('/', '_')}.png"
    cv2.imwrite(str(out_dir / name), sheet)

sheets = sorted(out_dir.glob("*.png"))
print(f"wrote {len(sheets)} sheets to {out_dir}")
for sheet in sheets:
    print(f"  {sheet.name}")
