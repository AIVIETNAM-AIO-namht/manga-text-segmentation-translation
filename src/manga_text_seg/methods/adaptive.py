"""Adaptive thresholding method.

Research R4: ``cv2.adaptiveThreshold`` (Gaussian-weighted, configurable
blockSize/C) is the classical baseline for unevenly lit manga pages — a single
global threshold cannot separate text from background that varies across the
page.  Manga text is dark-on-light paper, so ``THRESH_BINARY_INV`` maps text to
255 (FR-012: text = 255, background = 0).
"""
from __future__ import annotations

import cv2
import numpy as np

from . import BaseSegmentationMethod, register


class AdaptiveMethod(BaseSegmentationMethod):
    """Binary segmentation via Gaussian adaptive thresholding."""

    def __init__(self, config: dict) -> None:
        self._block_size = int(config.get("block_size", 35))
        self._c = int(config.get("c", 5))
        if self._block_size % 2 == 0:
            raise ValueError(
                f"adaptive: block_size must be odd, got {self._block_size}"
            )
        if self._block_size < 3:
            raise ValueError(
                f"adaptive: block_size must be >= 3, got {self._block_size}"
            )

    @property
    def name(self) -> str:
        return "adaptive"

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian adaptive thresholding; text = 255, background = 0."""
        if image.ndim != 2:
            raise ValueError(
                f"adaptive: expected single-channel image, got shape {image.shape}"
            )
        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self._block_size,
            self._c,
        )


register("adaptive", AdaptiveMethod)