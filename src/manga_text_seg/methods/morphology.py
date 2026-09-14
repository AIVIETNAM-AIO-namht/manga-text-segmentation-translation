"""Morphological post-processing method.

Research R4: a parameter set (kernel shape/size, opening/closing iterations) is
applied to the Otsu binarized image (inverted so dark text = 255 per FR-012).
Opening removes isolated speckle noise, closing fills small holes — together
they denoise the threshold output before metrics are computed.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import BaseSegmentationMethod, register

_KERNEL_SHAPES = {
    "rect": cv2.MORPH_RECT,
    "ellipse": cv2.MORPH_ELLIPSE,
    "cross": cv2.MORPH_CROSS,
}


class MorphologyMethod(BaseSegmentationMethod):
    """Binary segmentation via Otsu binarization + opening/closing."""

    def __init__(self, config: dict) -> None:
        self._config = config
        self._kernel_size = int(config.get("kernel", 3))
        self._open_iter = int(config.get("open_iter", 1))
        self._close_iter = int(config.get("close_iter", 1))
        self._kernel_shape = str(config.get("kernel_shape", "rect"))
        if self._kernel_size < 1:
            raise ValueError(
                f"morphology: kernel size must be >= 1, got {self._kernel_size}"
            )
        if self._kernel_shape not in _KERNEL_SHAPES:
            raise ValueError(
                f"morphology: unknown kernel_shape '{self._kernel_shape}'; "
                f"expected one of {sorted(_KERNEL_SHAPES)}"
            )
        if self._open_iter < 0 or self._close_iter < 0:
            raise ValueError(
                "morphology: iterations must be >= 0 "
                f"(open={self._open_iter}, close={self._close_iter})"
            )

    @property
    def name(self) -> str:
        return "morphology"

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Binarize (Otsu INV), then apply opening and closing with the kernel."""
        if image.ndim != 2:
            raise ValueError(
                f"morphology: expected single-channel image, got shape {image.shape}"
            )
        _, binary = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        kernel = cv2.getStructuringElement(
            _KERNEL_SHAPES[self._kernel_shape], (self._kernel_size, self._kernel_size)
        )
        if self._open_iter > 0:
            binary = cv2.morphologyEx(
                binary, cv2.MORPH_OPEN, kernel, iterations=self._open_iter
            )
        if self._close_iter > 0:
            binary = cv2.morphologyEx(
                binary, cv2.MORPH_CLOSE, kernel, iterations=self._close_iter
            )
        return binary


register("morphology", MorphologyMethod)