"""Five-panel, ground-truth-free comparison boards for Spec 003."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PANEL_LABELS = OrderedDict(
    (
        ("original_page", "Original page"),
        ("raw_prediction_mask", "Raw prediction mask"),
        ("dilated_mask", "Mask after dilation"),
        ("telea_result", "TELEA result"),
        ("ns_result", "NS result"),
    )
)


def render_board(sample: dict[str, Any]) -> dict[str, Any]:
    """Validate and describe exactly the five source/output panels for a page.

    This pure data step keeps board generation independent from metric scoring
    and makes it impossible for a caller to pass extra GT/overlay panels through.
    """
    required = {"image_id", "method", "algorithm", "mask_proc_config", "panels"}
    missing = required - set(sample)
    if missing:
        raise ValueError(f"Board sample missing fields: {sorted(missing)}")
    panels = sample["panels"]
    if set(panels) != set(PANEL_LABELS):
        raise ValueError(f"Board must contain exactly these panels: {tuple(PANEL_LABELS)}")
    ordered: OrderedDict[str, np.ndarray] = OrderedDict()
    expected_hw: tuple[int, int] | None = None
    for key in PANEL_LABELS:
        image = np.asarray(panels[key])
        if image.size == 0 or image.ndim not in (2, 3):
            raise ValueError(f"Panel {key!r} must be a non-empty image array")
        if expected_hw is None:
            expected_hw = image.shape[:2]
        elif image.shape[:2] != expected_hw:
            raise ValueError("All five board panels must have the same height and width")
        ordered[key] = image
    return {
        "image_id": str(sample["image_id"]),
        "method": str(sample["method"]),
        "algorithm": str(sample["algorithm"]),
        "mask_proc_config": sample["mask_proc_config"],
        "panels": ordered,
    }


def write_board(sample: dict[str, Any], path: Path, *, panel_width: int = 420) -> Path:
    """Render a labelled five-column PNG from a ``render_board`` sample."""
    board = render_board(sample)
    tiles: list[np.ndarray] = []
    height = None
    for key, label in PANEL_LABELS.items():
        image = board["panels"][key]
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        h, w = image.shape[:2]
        tile_height = max(1, int(h * panel_width / w))
        image = cv2.resize(image, (panel_width, tile_height), interpolation=cv2.INTER_AREA)
        if height is None:
            height = tile_height
        header = np.full((98, panel_width, 3), 255, dtype=np.uint8)
        cv2.putText(header, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(header, f"{board['image_id']} | {board['method']} | {board['algorithm']}", (10, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (20, 20, 20), 1, cv2.LINE_AA)
        dilation = board["mask_proc_config"].get("dilation", {})
        kernel = dilation.get("kernel_size") or []
        kernel_label = "x".join(str(value) for value in kernel) if kernel else "off"
        config_label = (
            f"dilation={kernel_label}/{dilation.get('kernel_shape', 'none')}"
            f"x{dilation.get('iterations') or 0}; radius="
            f"{board['mask_proc_config'].get('inpaint_radius', 'unknown')}"
        )
        cv2.putText(header, config_label[:54], (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (20, 20, 20), 1, cv2.LINE_AA)
        tiles.append(np.vstack((header, image)))
    # Keep each tile the same height if an input was resized with rounding.
    target_height = max(tile.shape[0] for tile in tiles)
    tiles = [cv2.copyMakeBorder(tile, 0, target_height - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for tile in tiles]
    output = np.hstack(tiles)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), output):
        raise OSError(f"Could not write board image to {path}")
    return path
