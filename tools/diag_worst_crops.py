"""Crop the worst-placed regions from original/mask/rendered for visual review."""

import glob
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np

TARGETS = [
    ("AisazuNihaIrarenai/004", 137),
    ("AisazuNihaIrarenai/003", 128),
    ("AkkeraKanjinchou/003", 94),
    ("ARMS/002", 18),
]
PAD = 40

for image_id, region_id in TARGETS:
    page_dir = next(
        Path(p).parent
        for p in glob.glob("outputs/translation/spec004-vilao/*/*/*/metadata.json")
        if json.load(io.open(p, encoding="utf-8"))["image_id"] == image_id
    )
    record = json.load(io.open(page_dir / "metadata.json", encoding="utf-8"))
    region = next(r for r in record["regions"] if r["region_id"] == region_id)
    x, y, width, height = region["bbox"]
    left, top = max(0, x - PAD), max(0, y - PAD)
    right, bottom = min(1654, x + width + PAD), min(1170, y + height + PAD)

    panels = []
    for name in ("original.png", "inpainted.png", "rendered.png"):
        image = cv2.imread(str(page_dir / name))
        panels.append(image[top:bottom, left:right])
    strip = np.concatenate(panels, axis=1)
    # Mark the region bbox on each panel so the render box is visible.
    for index in range(3):
        x0 = index * (right - left) + (x - left)
        y0 = y - top
        cv2.rectangle(strip, (x0, y0), (x0 + width, y0 + height), (0, 0, 255), 2)
    out = Path("deliverables/diag") / f"{image_id.replace('/', '_')}_{region_id}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), strip)
    print(
        f"{out}  box=({x},{y},{width},{height}) merged_from={region['merged_from']}"
        f" ocr={region.get('ocr_text', '')[:24]!r} font={region.get('font_size_used')}"
        f" render_status={region.get('render_status')}"
    )
