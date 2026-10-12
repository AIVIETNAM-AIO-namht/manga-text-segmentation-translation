"""Overlay the extracted region boxes on the original page to see the fragmentation."""

import glob
import io
import json
from pathlib import Path

import cv2
import numpy as np

TARGETS = ["AisazuNihaIrarenai/004", "AkkeraKanjinchou/000", "ARMS/002", "AisazuNihaIrarenai/000"]
COLORS = [(0, 0, 255), (0, 160, 0), (255, 0, 0), (0, 140, 255)]

out_dir = Path("deliverables/diag/boxes")
out_dir.mkdir(parents=True, exist_ok=True)

for image_id in TARGETS:
    page_dir = next(
        Path(p).parent
        for p in glob.glob("outputs/translation/spec004-vilao/*/*/*/metadata.json")
        if json.load(io.open(p, encoding="utf-8"))["image_id"] == image_id
    )
    record = json.load(io.open(page_dir / "metadata.json", encoding="utf-8"))
    original = cv2.imread(str(page_dir / "original.png"))
    rendered = cv2.imread(str(page_dir / "rendered.png"))
    panels = []
    for base, color in ((original, (0, 0, 255)), (rendered, (0, 0, 255))):
        canvas = base.copy()
        for region in record["regions"]:
            x, y, w, h = region["bbox"]
            thickness = 2 if region["merged_from"] > 1 else 1
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
        panels.append(canvas)
    sheet = np.hstack(panels)
    out = out_dir / f"{image_id.replace('/', '_')}.png"
    cv2.imwrite(str(out), sheet)
    merged = sum(1 for r in record["regions"] if r["merged_from"] > 1)
    print(f"{out}  regions={len(record['regions'])} merged={merged}")
