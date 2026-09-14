"""Otsu global thresholding method.

Research R4: Otsu's method with an optional Gaussian pre-blur is the
classical baseline for manga text segmentation.  The threshold is computed
automatically from the image histogram (``cv2.THRESH_OTSU``).
"""
from __future__ import annotations

import cv2
import numpy as np

from . import BaseSegmentationMethod, register


class OtsuMethod(BaseSegmentationMethod):
    """Binary segmentation via Otsu thresholding, with optional Gaussian blur."""

    def __init__(self, config: dict) -> None:
        self._config = config
        self._blur_ksize = int(config.get("blur_ksize", 5))
        if self._blur_ksize % 2 == 0:
            raise ValueError(f"otsu: blur_ksize must be odd, got {self._blur_ksize}")

    @property
    def name(self) -> str:
        return "otsu"

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Apply Otsu thresholding; text = 255, background = 0."""
        if image.ndim != 2:
            raise ValueError(f"otsu: expected single-channel image, got shape {image.shape}")
        prepped = image
        if self._blur_ksize > 1:
            prepped = cv2.GaussianBlur(image, (self._blur_ksize, self._blur_ksize), 0)
        _, binary = cv2.threshold(prepped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary


register("otsu", OtsuMethod)