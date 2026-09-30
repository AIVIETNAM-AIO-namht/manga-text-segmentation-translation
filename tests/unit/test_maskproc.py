"""T013 - unit tests for mask normalization, alignment, emptiness and dilation.

Contract: contracts/intake-validation.md "Mask normalization and alignment"
(FR-014, FR-015, FR-016, FR-018). Written before ``maskproc.py`` exists, so
every test here must FAIL until T018-T020 land.

Sizes: ``AlignmentRecord`` and the tests' ``*_SIZE`` values are
``(width, height)``; numpy shapes are ``(height, width)``. The aligned shape
comes from the shared ``aligned_mask`` fixture rather than a second constant.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from manga_text_seg.maskproc import (
    align_mask,
    check_not_empty,
    dilate_mask,
    normalize_mask,
)

from manga_text_seg.config import DilationConfig
from manga_text_seg.runs import AlignmentRecord, MaskRejection

#: The known GT padded canvas (multiple-of-8), as a numpy shape (Spec 001).
GT_CANVAS_SHAPE = (1176, 1656)

_CV2_SHAPES = {
    "rect": cv2.MORPH_RECT,
    "ellipse": cv2.MORPH_ELLIPSE,
    "cross": cv2.MORPH_CROSS,
}


def _blank(shape: tuple[int, int]) -> np.ndarray:
    return np.zeros(shape, dtype=np.uint8)


# --------------------------------------------------------------------------- #
# normalize_mask (T018, FR-014)
# --------------------------------------------------------------------------- #
def test_normalize_accepts_binary_single_channel(aligned_mask):
    before = aligned_mask.copy()
    result = normalize_mask(aligned_mask)

    assert result.dtype == np.uint8
    assert result.ndim == 2
    assert np.array_equal(result, aligned_mask)
    assert np.array_equal(aligned_mask, before), "input must not be modified"


def test_normalize_collapses_identical_channels(aligned_mask):
    three_channel = np.stack([aligned_mask] * 3, axis=-1)

    result = normalize_mask(three_channel)

    assert result.ndim == 2
    assert np.array_equal(result, aligned_mask)


def test_normalize_rejects_channels_that_disagree(aligned_mask):
    other = np.zeros_like(aligned_mask)
    three_channel = np.stack([aligned_mask, other, other], axis=-1)

    with pytest.raises(MaskRejection) as exc:
        normalize_mask(three_channel)

    assert exc.value.category == "non_binary_mask"


@pytest.mark.parametrize(
    "bad_values",
    [(0, 128, 255), (0, 1), (0, 254), (1, 255)],
    ids=["gray-128", "zero-one", "antialiased-254", "no-background-zero"],
)
def test_normalize_rejects_values_outside_0_255(aligned_mask, bad_values):
    """A value outside {0, 255} is a rejection, never a coercion or a
    re-threshold (FR-014). {0, 1} is rejected too: it is not rescaled."""
    mask = aligned_mask.copy()
    mask[mask == 255] = bad_values[-1]
    if bad_values[0] != 0:
        mask[mask == 0] = bad_values[0]
    if len(bad_values) == 3:
        mask[0, 0] = bad_values[1]
    before = mask.copy()

    with pytest.raises(MaskRejection) as exc:
        normalize_mask(mask)

    assert exc.value.category == "non_binary_mask"
    assert exc.value.reason
    assert np.array_equal(mask, before), "a rejected input is never altered"


def test_normalize_rejection_names_the_offending_value(aligned_mask):
    mask = aligned_mask.copy()
    mask[5, 5] = 128

    with pytest.raises(MaskRejection) as exc:
        normalize_mask(mask)

    assert "128" in exc.value.reason


# --------------------------------------------------------------------------- #
# align_mask (T019, FR-015, FR-018)
# --------------------------------------------------------------------------- #
def test_align_passes_through_at_aligned_size(aligned_mask):
    aligned, record = align_mask(aligned_mask)

    assert np.array_equal(aligned, aligned_mask)
    assert record == AlignmentRecord(
        rule="pass-through", pre_size=(1654, 1170), post_size=(1654, 1170)
    )


def test_align_crops_top_left_losslessly_from_gt_canvas():
    canvas = _blank(GT_CANVAS_SHAPE)
    canvas[10:20, 10:30] = 255  # inside the page area: must survive, in place
    canvas[1172:1175, 1655] = 255  # inside the padding: must be cut away
    before = canvas.copy()

    aligned, record = align_mask(canvas)

    assert aligned.shape == (1170, 1654)
    assert np.array_equal(aligned, canvas[:1170, :1654]), "pure top-left slice"
    assert (aligned[10:20, 10:30] == 255).all()
    assert record == AlignmentRecord(
        rule="crop-topleft", pre_size=(1656, 1176), post_size=(1654, 1170)
    )
    assert np.array_equal(canvas, before), "input must not be modified"


@pytest.mark.parametrize(
    "shape",
    [
        (1170, 1656),  # only the width is padded
        (1176, 1654),  # only the height is padded
        (1000, 1000),
        (2340, 3308),
        (1654, 1170),  # width and height swapped
    ],
    ids=["width-only", "height-only", "square", "double", "transposed"],
)
def test_align_rejects_any_other_size(shape):
    mask = _blank(shape)
    mask[10:20, 10:30] = 255

    with pytest.raises(MaskRejection) as exc:
        align_mask(mask)

    assert exc.value.category == "wrong_size"
    assert str(shape[0]) in exc.value.reason and str(shape[1]) in exc.value.reason


# --------------------------------------------------------------------------- #
# check_not_empty (T020, FR-010)
# --------------------------------------------------------------------------- #
def test_check_not_empty_rejects_an_all_zero_mask(aligned_mask):
    with pytest.raises(MaskRejection) as exc:
        check_not_empty(_blank(aligned_mask.shape))

    assert exc.value.category == "empty_mask"


def test_check_not_empty_accepts_a_single_text_pixel(aligned_mask):
    mask = _blank(aligned_mask.shape)
    mask[600, 600] = 255

    check_not_empty(mask)  # must not raise


# --------------------------------------------------------------------------- #
# dilate_mask (T020, FR-016, FR-018)
# --------------------------------------------------------------------------- #
def test_dilate_disabled_returns_the_mask_unchanged(aligned_mask):
    config = DilationConfig(enabled=False)

    dilated, effective = dilate_mask(aligned_mask, config)

    assert np.array_equal(dilated, aligned_mask)
    assert effective.enabled is False


@pytest.mark.parametrize(
    ("shape", "size", "iterations"),
    [
        ("ellipse", (3, 3), 1),
        ("rect", (5, 5), 2),
        ("cross", (3, 3), 1),
    ],
)
def test_dilate_follows_the_config(aligned_mask, shape, size, iterations):
    """Shape, size and iterations come from the config, never from the code."""
    config = DilationConfig(
        enabled=True, kernel_shape=shape, kernel_size=size, iterations=iterations
    )
    before = aligned_mask.copy()
    kernel = cv2.getStructuringElement(_CV2_SHAPES[shape], size)
    expected = cv2.dilate(aligned_mask, kernel, iterations=iterations)

    dilated, effective = dilate_mask(aligned_mask, config)

    assert np.array_equal(dilated, expected)
    assert dilated.dtype == np.uint8
    assert dilated.shape == aligned_mask.shape
    assert set(np.unique(dilated)) <= {0, 255}
    assert (dilated >= aligned_mask).all(), "dilation only ever adds text pixels"
    assert (dilated != aligned_mask).any(), "the mask actually grew"
    assert np.array_equal(aligned_mask, before), "input must not be modified"
    assert effective == config


def test_dilate_records_the_effective_values(aligned_mask):
    config = DilationConfig(
        enabled=True, kernel_shape="rect", kernel_size=(7, 7), iterations=3
    )

    _, effective = dilate_mask(aligned_mask, config)

    assert effective.kernel_shape == "rect"
    assert tuple(effective.kernel_size) == (7, 7)
    assert effective.iterations == 3
