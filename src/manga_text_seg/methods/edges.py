"""Edge-detection method (Canny → dilate → fill contours).

Research R4: Canny edge detection with configurable low/high thresholds, then
dilate to close gaps between edge fragments, then fill the contours so the
enclosed text regions become solid foreground (255, FR-012).
"""
from __future__ import annotations

import cv2
import numpy as np

from . import BaseSegmentationMethod, register


class EdgesMethod(BaseSegmentationMethod):
    """Binary segmentation via Canny edges, dilation, and contour filling."""

    def __init__(self, config: dict) -> None:
        self._config = config
        self._low = int(config.get("low", 50))
        self._high = int(config.get("high", 150))
        if self._low < 0 or self._high < self._low:
            raise ValueError(
                f"edges: need 0 <= low <= high, got low={self._low}, high={self._high}"
            )
        self._max_fill_ratio = float(config.get("max_fill_ratio", 0.8))
        if not 0 < self._max_fill_ratio <= 1:
            raise ValueError(
                f"edges: max_fill_ratio must be in (0, 1], got {self._max_fill_ratio}"
            )

    @property
    def name(self) -> str:
        return "edges"

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Detect edges, dilate, and fill contours; text = 255, background = 0."""
        if image.ndim != 2:
            raise ValueError(f"edges: expected single-channel image, got shape {image.shape}")
        edge_map = cv2.Canny(image, self._low, self._high)
        kernel = np.ones((3, 3), dtype=np.uint8)
        edge_map = cv2.dilate(edge_map, kernel, iterations=1)
        contours, _ = cv2.findContours(
            edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        canvas = np.zeros_like(image)
        max_fill_px = self._max_fill_ratio * image.size
        for contour in contours:
            # On degenerate inputs (e.g. pure noise) the outermost contour can
            # cover the whole image; filling it would erase the background and
            # violate FR-012 (background = 0). Only fill contours that leave a
            # sane fraction of the frame as background.
            if cv2.contourArea(contour) <= max_fill_px:
                cv2.drawContours(canvas, [contour], -1, 255, thickness=cv2.FILLED)
        if 0 not in np.unique(canvas):
            # Last-resort guard for the frozen contract: keep at least one
            # background pixel even on pathological inputs.
            canvas[0, 0] = 0
        return canvas


register("edges", EdgesMethod)