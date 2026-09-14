"""Mask-image alignment via crop-topleft.

Research R2: GT masks (1176×1656) and raw pages (1170×1654) differ by 6px
horizontally and 2px vertically.  The default strategy is a lossless
crop-topleft (no padding, no resizing) because: (a) all text pixels fall
well within the crop region (max text y=1169, x=1653), and (b) zero text
pixels are lost.  An alternative "resize" strategy normalizes both to
mask dimensions for experimentation.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


class AlignmentError(RuntimeError):
    """Raised when alignment fails due to incompatible image dimensions."""


@dataclass(frozen=True)
class AlignedPair:
    mask: np.ndarray
    page: np.ndarray
    crop_height: int
    crop_width: int
    strategy: str


def _crop_topleft(mask: np.ndarray, page: np.ndarray) -> AlignedPair:
    """Crop both images to the minimum common dimensions from (0,0)."""
    h = min(mask.shape[0], page.shape[0])
    w = min(mask.shape[1], page.shape[1])
    return AlignedPair(
        mask=mask[:h, :w],
        page=page[:h, :w],
        crop_height=h,
        crop_width=w,
        strategy="crop-topleft",
    )


def _resize(mask: np.ndarray, page: np.ndarray) -> AlignedPair:
    """Resize page to match mask dimensions using INTER_NEAREST."""
    if page.shape[0] != mask.shape[0] or page.shape[1] != mask.shape[1]:
        import cv2
        resized = cv2.resize(
            page, (mask.shape[1], mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        resized = page
    return AlignedPair(
        mask=mask,
        page=resized,
        crop_height=mask.shape[0],
        crop_width=mask.shape[1],
        strategy="resize",
    )


def align(mask: np.ndarray, page: np.ndarray, strategy: str = "crop-topleft") -> AlignedPair:
    """Align mask and page using the given strategy.

    Strategies:
      - "crop-topleft" (default): lossless crop from (0,0) to min dims
      - "resize": interpolate page to match mask dims
    """
    if mask.ndim not in (2, 3) or page.ndim not in (2, 3):
        raise AlignmentError("Both mask and page must be 2D or 3D arrays")
    if mask.ndim == 3 and mask.shape[2] != 1:
        raise AlignmentError(f"Mask must be single-channel, got shape {mask.shape}")

    if strategy == "crop-topleft":
        return _crop_topleft(mask, page)
    elif strategy == "resize":
        return _resize(mask, page)
    else:
        raise AlignmentError(f"Unknown alignment strategy: '{strategy}'")