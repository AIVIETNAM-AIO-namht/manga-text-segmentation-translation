"""Method B's geometry: the letterbox inverted, and the threshold that scored.

T036 (FR-024a, FR-025, FR-026, FR-030; edge cases FR-026/FR-030).

Two things are checked here, and neither of them needs a checkpoint, a
download, or torch — which is the point, because FR-051 says the default suite
must run without any of those.

**The geometry.**  Method B letterboxes the page onto a 1024×1024 canvas with
stride-64 bottom-right zero padding.  FR-024a forbids treating that as the same
transformation as the multiple-of-8 padding the ground truth already carries —
the aligned page is a multiple of 8 but not of 32, let alone 64 — so the
inverse has to be Method B's own.  The round-trip test is what makes that more
than a claim: a block drawn in the aligned space, letterboxed forward through
the method's own transform and inverted back, must land where it started.

**The threshold.**  Upstream accepts a configurable mask threshold and never
uses it, hardcoding its own cutoff instead.  FR-030 exists for exactly this:
the recorded effective threshold has to be the one that explains the mask, not
the configured one.  :func:`threshold_verification` recovers the cutoff from
the probability map and the produced mask, so a configuration that disagrees
with the implementation is caught rather than transcribed.

Every shape in this file is ``(height, width)``, numpy's convention.
"""
from __future__ import annotations

import numpy as np
import pytest

from manga_text_seg.adapters.comic_text_detector import (
    CANVAS_SIZE,
    PADDING_MULTIPLE,
    apply_threshold,
    invert_letterbox,
    letterbox,
    letterbox_geometry,
    threshold_verification,
)

#: The aligned page as ``(height, width)`` — conftest's ``ALIGNED_SHAPE``,
#: restated so this file reads on its own.
ALIGNED_SHAPE = (1170, 1654)

#: A sigmoid map with three distinct levels, so a configured cutoff and an
#: applied cutoff can be told apart.  The middle level is what does it: it sits
#: above Method B's hardcoded cutoff and below the configured 0.5, so a
#: verification that trusts the configuration produces a visibly different mask.
BACKGROUND, MIDLEVEL, TEXT = 0.05, 0.35, 0.90


def _probability_map(shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    """The refined-mask probability map Method B's segmentation head emits."""
    probs = np.full(shape, BACKGROUND, dtype=np.float32)
    probs[10:30, 10:30] = MIDLEVEL
    probs[40:50, 40:60] = TEXT
    return probs


def _block(shape: tuple[int, int], rows: slice, cols: slice) -> np.ndarray:
    """A binary mask with one block set, in the FR-004 convention."""
    mask = np.zeros(shape, dtype=np.uint8)
    mask[rows, cols] = 255
    return mask


def test_the_letterbox_geometry_is_method_bs_own():
    """FR-024a: recorded per method, and not the ground truth's multiple-of-8."""
    geometry = letterbox_geometry(*ALIGNED_SHAPE)

    assert tuple(geometry["canvas"]) == (CANVAS_SIZE, CANVAS_SIZE)
    # The stride-64 requirement — which is what makes this not the GT's rule.
    assert geometry["padding_multiple"] == PADDING_MULTIPLE
    assert geometry["padding_multiple"] != 8

    scaled_h, scaled_w = geometry["scaled_size"]
    padded_h, padded_w = geometry["padded_size"]
    pad_h, pad_w = geometry["pad"]

    # Aspect is preserved on the way in, and the canvas is not overflowed.
    assert 0 < scaled_h <= CANVAS_SIZE and 0 < scaled_w <= CANVAS_SIZE
    assert max(scaled_h, scaled_w) == CANVAS_SIZE
    assert scaled_w / scaled_h == pytest.approx(ALIGNED_SHAPE[1] / ALIGNED_SHAPE[0], rel=0.02)
    assert geometry["scale"] == pytest.approx(
        CANVAS_SIZE / max(ALIGNED_SHAPE), rel=0.02
    )

    # Bottom-right zero padding to the stride, not a crop and not a squeeze.
    assert padded_h % PADDING_MULTIPLE == 0
    assert padded_w % PADDING_MULTIPLE == 0
    assert padded_h >= scaled_h and padded_w >= scaled_w
    assert pad_h == padded_h - scaled_h
    assert pad_w == padded_w - scaled_w
    assert pad_h >= 0 and pad_w >= 0


def test_the_inverse_returns_exactly_the_aligned_page_size():
    """FR-024/FR-028: the prediction lands in the aligned space, unresized."""
    geometry = letterbox_geometry(*ALIGNED_SHAPE)
    canvas_mask = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    canvas_mask[100:300, 100:400] = 255

    inverted = invert_letterbox(canvas_mask, geometry)

    assert inverted.shape == ALIGNED_SHAPE
    assert inverted.dtype == np.uint8
    assert set(np.unique(inverted)) <= {0, 255}


def test_the_letterbox_inverts_back_to_where_it_started():
    """FR-024a: the mapping is inverted, not approximated by a shared rule.

    A block drawn in the aligned space survives the round trip through the
    method's own transform to within the stride-64 quantisation the method
    itself imposes — which is the residual FR-025 requires be recorded rather
    than assumed away.
    """
    geometry = letterbox_geometry(*ALIGNED_SHAPE)
    original = _block(ALIGNED_SHAPE, slice(400, 700), slice(500, 1100))

    restored = invert_letterbox(letterbox(original, geometry), geometry)

    assert restored.shape == ALIGNED_SHAPE
    rows, cols = np.nonzero(original)
    back_rows, back_cols = np.nonzero(restored)
    assert back_rows.size and back_cols.size

    # The block's centre is where it was, to within a few percent of the page.
    assert abs(float(back_rows.mean()) - float(rows.mean())) < 0.03 * ALIGNED_SHAPE[0]
    assert abs(float(back_cols.mean()) - float(cols.mean())) < 0.03 * ALIGNED_SHAPE[1]

    # And it did not smear across the page: the area is in the same ballpark.
    assert 0.7 < back_rows.size / rows.size < 1.4


def test_the_configured_threshold_is_verified_against_the_mask():
    """FR-026/FR-030: the applied cutoff is the one that scored the mask."""
    probs = _probability_map()
    mask = apply_threshold(probs, 0.5)

    verification = threshold_verification(probs, mask, configured=0.5)

    assert verification["verified"] is True
    assert verification["configured"] == pytest.approx(0.5)
    assert verification["applied"] == pytest.approx(0.5)
    # The recorded threshold reproduces the mask exactly.
    assert np.array_equal(apply_threshold(probs, verification["applied"]), mask)


def test_a_configured_threshold_the_code_ignores_is_reported_as_unused():
    """FR-030: the effective threshold is recorded, not the configured one.

    This is Method B's known breakage: the configuration exposes a cutoff and
    the implementation hardcodes its own.  The record must carry what actually
    scored the mask, and must say the configured value was not it.
    """
    probs = _probability_map()
    mask = apply_threshold(probs, 0.5)

    verification = threshold_verification(probs, mask, configured=0.235)

    assert verification["verified"] is False
    assert verification["configured"] == pytest.approx(0.235)
    # The configured cutoff does not reproduce the mask; the applied one does.
    assert not np.array_equal(apply_threshold(probs, 0.235), mask)
    assert verification["applied"] != pytest.approx(0.235, abs=0.05)
    assert verification["applied"] > verification["configured"]
    assert np.array_equal(apply_threshold(probs, verification["applied"]), mask)
    assert verification["note"], "an ignored configured threshold must be explained"


def test_the_threshold_helper_emits_the_frozen_binary_convention():
    """FR-028: single-channel, binary, values ⊆ {0, 255}, uint8."""
    probs = _probability_map()
    mask = apply_threshold(probs, 0.5)

    assert mask.shape == probs.shape
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 255}
    assert mask.max() == 255, "a map with text above the cutoff must produce text"
    assert mask.min() == 0, "and a map with background below it must produce background"
