"""Is the region-extraction mask the same as the inpainted hole? Measure per page."""

import glob
import io
import json
from pathlib import Path

import cv2
import numpy as np

PAGES = sorted(glob.glob("outputs/translation/spec004-vilao/*/*/*/metadata.json"))
for page in PAGES[:6]:
    record = json.load(io.open(page, encoding="utf-8"))
    page_dir = Path(page).parent
    mask = cv2.imread(str(page_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
    original = cv2.imread(str(page_dir / "original.png"), cv2.IMREAD_GRAYSCALE)
    inpainted = cv2.imread(str(page_dir / "inpainted.png"), cv2.IMREAD_GRAYSCALE)
    rendered = cv2.imread(str(page_dir / "rendered.png"), cv2.IMREAD_GRAYSCALE)
    hole = mask > 127
    # Pixels that changed between original and inpainted = the actual erase footprint
    changed = np.abs(original.astype(int) - inpainted.astype(int)) > 8
    print(f"\n{record['image_id']}  mask_px={hole.sum()}  changed_px={changed.sum()}")
    print(f"  mask covers changed: {(hole & changed).sum() / max(1, changed.sum()):.3f}")
    print(f"  changed inside mask: {(hole & changed).sum() / max(1, hole.sum()):.3f}")
    if rendered is not None:
        ink = rendered < 100
        print(f"  rendered ink px={ink.sum()}  inside mask:"
              f" {(ink & hole).sum() / max(1, ink.sum()):.3f}")
    # how much of the page did inpainting erase?
    print(f"  erase footprint = {100 * changed.sum() / changed.size:.2f}% of page")
    print(f"  mask = {100 * hole.sum() / hole.size:.2f}% of page")
