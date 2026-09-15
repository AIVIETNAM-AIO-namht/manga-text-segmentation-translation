"""Connected-components method.

Research R4: binarize with Otsu (inverted so dark text = 255 per FR-012), run
``cv2.connectedComponentsWithStats``, then keep only components whose area and
aspect ratio pass configurable filters — this drops both tiny speckles
(``min_area``) and long thin lines such as panel borders (``max_aspect_ratio``).
"""
from __future__ import annotations

import cv2
import numpy as np

from . import BaseSegmentationMethod, register


class ComponentsMethod(BaseSegmentationMethod):
    """Binary segmentation via Otsu binarization + component area/aspect filter."""

    def __init__(self, config: dict) -> None:
        self._min_area = float(config.get("min_area", 50))
        self._max_aspect_ratio = float(config.get("max_aspect_ratio", 20.0))
        if self._min_area < 0:
            raise ValueError(f"components: min_area must be >= 0, got {self._min_area}")
        if self._max_aspect_ratio <= 0:
            raise ValueError(
                f"components: max_aspect_ratio must be > 0, got {self._max_aspect_ratio}"
            )

    @property
    def name(self) -> str:
        return "components"

    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """Otsu threshold with inverted polarity: dark text -> 255 (FR-012)."""
        _, binary = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return binary

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Keep size-filtered connected components; text = 255, background = 0."""
        if image.ndim != 2:
            raise ValueError(
                f"components: expected single-channel image, got shape {image.shape}"
            )
        binary = self._binarize(image)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        canvas = np.zeros_like(image)
        for i in range(1, n_labels):  # label 0 = background
            area = int(stats[i, cv2.CC_STAT_AREA])
            width = int(stats[i, cv2.CC_STAT_WIDTH])
            height = int(stats[i, cv2.CC_STAT_HEIGHT])
            aspect = width / height if height > 0 else 0.0
            if area >= self._min_area and aspect <= self._max_aspect_ratio:
                canvas[labels == i] = 255
        return canvas


register("components", ComponentsMethod)