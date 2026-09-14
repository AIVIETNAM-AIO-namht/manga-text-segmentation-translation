"""Image I/O utilities — read-only loaders for GT masks and raw pages.

All loaders validate dtype and channel count at load time (fail fast).
GT masks are color PNGs (magenta-encoded text, Research R1); raw pages are
grayscale.  Only prediction masks are guaranteed strictly binary {0, 255}.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ImageError(RuntimeError):
    """Raised when an image file cannot be loaded or violates expected invariants."""


def load_mask(path: Path) -> np.ndarray:
    """Load a prediction mask as single-channel uint8 (0 or 255).

    Raises ``ImageError`` if the file cannot be read, has multiple channels,
    or contains values other than 0 and 255.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ImageError(f"Could not read mask file: {path}")
    if img.ndim != 2:
        raise ImageError(f"Mask {path} has {img.ndim} channels; expected single-channel")
    unique_vals = set(np.unique(img))
    valid = {0, 255}
    if not unique_vals.issubset(valid):
        raise ImageError(
            f"Mask {path} contains unexpected values: "
            f"{sorted(unique_vals)}; expected only {sorted(valid)}"
        )
    return img


def load_gt_mask(path: Path) -> np.ndarray:
    """Load a ground-truth mask as stored (color or grayscale, uint8).

    GT masks encode text as magenta (#FF00FF) pixels; the binary conversion
    is done later by ``normalize.normalize_mask``.  No value validation here.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ImageError(f"Could not read GT mask file: {path}")
    return img


MASK_ENCODINGS = ("magenta-black", "magenta-only", "near-black-only", "all-white-empty")


def detect_mask_encoding(path: Path) -> str:
    """Classify a GT mask's color encoding (manifest.schema.json enum).

    Magenta text + black background -> magenta-black; magenta on white ->
    magenta-only; unchecked dark text -> near-black-only; all-white (empty)
    -> all-white-empty.  Reads the stored file untouched (BGR or gray).
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ImageError(f"Could not read GT mask file: {path}")
    if img.ndim == 3:
        b, g, r = img[..., 0], img[..., 1], img[..., 2]
    else:
        b = g = r = img

    white = (b >= 230) & (g >= 230) & (r >= 230)
    black = (b <= 25) & (g <= 25) & (r <= 25)
    magenta = (r >= 180) & (g <= 80) & (b >= 180)

    if bool(white.all()):
        return "all-white-empty"
    has_magenta = bool(magenta.any())
    has_black = bool(black.any())
    if has_magenta and has_black:
        return "magenta-black"
    if has_magenta:
        return "magenta-only"
    if has_black:
        return "near-black-only"
    # Any other non-white content (e.g. mid-gray) is treated as dark text.
    return "near-black-only" if not bool(white.any()) else "all-white-empty"


def load_raw_page(path: Path) -> np.ndarray:
    """Load a raw manga page as single-channel grayscale uint8.

    Raises ``ImageError`` if the file cannot be read.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ImageError(f"Could not read raw page file: {path}")
    return img


def save_mask(path: Path, mask: np.ndarray) -> None:
    """Write a mask to disk as PNG (lossless)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask)


def save_raw(path: Path, img: np.ndarray) -> None:
    """Write a grayscale image to disk as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)