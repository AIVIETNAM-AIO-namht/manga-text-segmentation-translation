"""Mask normalization, alignment, emptiness check and dilation for Spec 003.

* T018 - ``normalize_mask``: background 0, text 255, anything else is refused
  (FR-014).
* T019 - ``align_mask``: pass-through at the aligned size, lossless top-left
  crop from the GT canvas, error otherwise (FR-015, FR-018).
* T020 - ``check_not_empty`` and ``dilate_mask`` (FR-010, FR-016, FR-018).

Every function returns a new array and leaves its input untouched. A refusal is
a ``MaskRejection`` carrying its FR-012 category; nothing here resizes,
re-thresholds or otherwise repairs an input that was refused (FR-013).

Sizes are ``(width, height)``, as in ``AlignmentRecord``; numpy shapes are
``(height, width)``.
"""

from __future__ import annotations

import cv2
import numpy as np

from manga_text_seg.config import DILATION_KERNEL_SHAPES, DilationConfig
from manga_text_seg.runs import AlignmentRecord, MaskRejection, RunError

#: The aligned page size (Spec 001). Fixed by the dataset, not tunable.
ALIGNED_SIZE = (1654, 1170)
#: The GT padded canvas, a multiple-of-8 size, 2 px wider and 6 px taller
#: (Spec 001). Its top-left ``ALIGNED_SIZE`` block is the page.
GT_CANVAS_SIZE = (1656, 1176)

BACKGROUND = 0
TEXT = 255

_KERNEL_SHAPES = {
    "rect": cv2.MORPH_RECT,
    "ellipse": cv2.MORPH_ELLIPSE,
    "cross": cv2.MORPH_CROSS,
}
assert set(_KERNEL_SHAPES) == DILATION_KERNEL_SHAPES  # config and code agree

#: How many offending values a rejection reason lists before it elides.
_MAX_LISTED_VALUES = 8


def normalize_mask(raw: np.ndarray) -> np.ndarray:
    """Return ``raw`` as a single-channel uint8 mask over ``{0, 255}``.

    ``raw`` is the array as stored (read with channel information intact).
    Channels that are identical collapse to one; channels that disagree are
    refused. Any value outside ``{0, 255}`` is refused, never coerced: ``{0, 1}``
    is not rescaled and 254 is not thresholded (FR-014).
    """
    mask = np.asarray(raw)
    if mask.ndim == 3:
        first = mask[..., 0]
        channels = range(1, mask.shape[2])
        if not all(np.array_equal(first, mask[..., i]) for i in channels):
            raise MaskRejection(
                "non_binary_mask",
                f"the {mask.shape[2]} channels of the mask disagree",
            )
        mask = first
    elif mask.ndim != 2:
        raise MaskRejection(
            "non_binary_mask", f"mask has {mask.ndim} dimensions; expected 2 or 3"
        )

    if mask.dtype != np.uint8:
        raise MaskRejection(
            "non_binary_mask", f"mask dtype is {mask.dtype}; expected uint8"
        )
    outside = sorted(set(np.unique(mask).tolist()) - {BACKGROUND, TEXT})
    if outside:
        shown = ", ".join(str(v) for v in outside[:_MAX_LISTED_VALUES])
        more = "" if len(outside) <= _MAX_LISTED_VALUES else ", ..."
        raise MaskRejection(
            "non_binary_mask",
            f"mask holds values outside {{0, 255}}: {shown}{more}",
        )
    return mask.copy()


def align_mask(mask: np.ndarray) -> tuple[np.ndarray, AlignmentRecord]:
    """Bring ``mask`` to the aligned page size and record how (FR-015, FR-018).

    At ``ALIGNED_SIZE`` it passes through unchanged. At ``GT_CANVAS_SIZE`` the
    top-left ``ALIGNED_SIZE`` block is kept, with no interpolation. Any other
    size is a ``wrong_size`` rejection.
    """
    height, width = mask.shape[:2]
    size = (width, height)
    if size == ALIGNED_SIZE:
        rule = "pass-through"
        aligned = mask.copy()
    elif size == GT_CANVAS_SIZE:
        rule = "crop-topleft"
        aligned = mask[: ALIGNED_SIZE[1], : ALIGNED_SIZE[0]].copy()
    else:
        raise MaskRejection(
            "wrong_size",
            f"mask is {width}x{height} (width x height, numpy shape "
            f"{height}x{width}); expected {ALIGNED_SIZE[0]}x{ALIGNED_SIZE[1]} "
            f"or {GT_CANVAS_SIZE[0]}x{GT_CANVAS_SIZE[1]}",
        )
    return aligned, AlignmentRecord(rule=rule, pre_size=size, post_size=ALIGNED_SIZE)


def check_not_empty(mask: np.ndarray) -> None:
    """Refuse a mask with no text pixel (FR-010)."""
    if not (mask == TEXT).any():
        raise MaskRejection("empty_mask", "mask contains no text pixel")


def dilate_mask(
    mask: np.ndarray, dilation: DilationConfig
) -> tuple[np.ndarray, DilationConfig]:
    """Dilate ``mask`` as ``dilation`` says and return the effective setting.

    Kernel shape, size and iterations come from the config alone (FR-016). The
    returned ``DilationConfig`` is what was actually applied, for the sample
    sidecar (FR-018). Disabled dilation returns an unchanged copy.
    """
    if not dilation.enabled:
        return mask.copy(), dilation
    if (
        dilation.kernel_shape is None
        or dilation.kernel_size is None
        or dilation.iterations is None
    ):
        raise RunError(
            "dilation is enabled but kernel_shape, kernel_size and iterations "
            "are not all set"
        )
    kernel = cv2.getStructuringElement(
        _KERNEL_SHAPES[dilation.kernel_shape], tuple(dilation.kernel_size)
    )
    dilated = cv2.dilate(mask, kernel, iterations=dilation.iterations)
    return dilated, dilation
