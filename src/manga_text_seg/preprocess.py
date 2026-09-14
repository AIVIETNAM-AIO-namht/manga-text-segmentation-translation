"""Preprocessing pipeline: grayscale + denoise (FR-020).

Runs BEFORE ``method.segment`` (contract method-interface.md rule 4).
The loader already produces grayscale; this module keeps grayscale explicit
and adds optional denoising.

Supporting denoise modes:
  - "none"     : passthrough
  - "gaussian" : 3x3 Gaussian blur
  - "median"   : 3x3 median blur (edge-preserving)
"""
from __future__ import annotations

import cv2
import numpy as np

DENOISE_MODES = {"none", "gaussian", "median"}


class PreprocessError(ValueError):
    """Raised for invalid preprocessing parameters."""


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR or BGRA image to single-channel grayscale uint8."""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] in (3, 4):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise PreprocessError(f"Cannot convert image of shape {image.shape} to grayscale")


def denoise(image: np.ndarray, mode: str = "none") -> np.ndarray:
    """Apply an optional denoising step; return an unmodified array for 'none'."""
    if mode not in DENOISE_MODES:
        raise PreprocessError(f"Unknown denoise mode '{mode}'; expected one of {sorted(DENOISE_MODES)}")
    if mode == "none":
        return image
    if mode == "gaussian":
        return cv2.GaussianBlur(image, (3, 3), 0)
    return cv2.medianBlur(image, 3)  # median


def preprocess(image: np.ndarray, denoise_mode: str = "none") -> np.ndarray:
    """Full preprocessing: grayscale conversion then optional denoise."""
    gray = to_grayscale(image)
    return denoise(gray, denoise_mode)