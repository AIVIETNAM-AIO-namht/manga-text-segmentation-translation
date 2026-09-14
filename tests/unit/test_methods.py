"""Unit tests for the FR-020 classical segmentation methods and the pipeline.

TDD Principle II: tests written first (RED), validated against the
implementations in ``src/manga_text_seg/methods/``.  Text = 255, background = 0
(FR-012); note only otsu.py keeps THRESH_BINARY polarity — the new methods use
THRESH_BINARY_INV so dark-on-light manga text maps to 255.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from manga_text_seg.config import DEFAULT_METHODS
from manga_text_seg.methods import UnknownMethodError, available, create


def _test_image() -> np.ndarray:
    """64×64 grayscale: light page (200) with a dark text band (30)."""
    img = np.full((64, 64), 200, dtype=np.uint8)
    img[:, 10:13] = 30
    return img


# =====================================================================
# Registry completeness (FR-020 mandate)
# =====================================================================

class TestRegistry:
    """All FR-020 methods plus the pipeline are registered."""

    def test_all_fr020_methods_registered(self) -> None:
        """The seven expected registry keys exist (config defaults parity)."""
        assert set(available()) == {
            "adaptive",
            "components",
            "edges",
            "morphology",
            "mser",
            "otsu",
            "pipeline",
        }

    def test_every_default_method_is_creatable(self) -> None:
        """Every method in config.DEFAULT_METHODS runs on a real image."""
        img = _test_image()
        for entry in DEFAULT_METHODS:
            method = create(entry["name"], entry.get("params") or {})
            result = method.segment(img)
            assert result.shape == img.shape
            assert result.dtype == np.uint8
            assert set(np.unique(result)) <= {0, 255}


# =====================================================================
# adaptive
# =====================================================================

class TestAdaptive:
    """Gaussian adaptive thresholding (THRESH_BINARY_INV polarity)."""

    def test_dark_stroke_is_text_white(self) -> None:
        """A dark stroke on a mid-gray page comes out 255, page stays 0."""
        # Arrange
        img = _test_image()
        # Act
        result = create("adaptive").segment(img)
        # Assert
        assert result.shape == img.shape
        assert result.dtype == np.uint8
        assert (result[:, 10:13] == 255).all()
        assert result[32, 40] == 0

    def test_even_block_size_rejected(self) -> None:
        """block_size must be odd (local-window requirement)."""
        with pytest.raises(ValueError):
            create("adaptive", {"block_size": 8})

    def test_block_size_below_3_rejected(self) -> None:
        """block_size must be >= 3."""
        with pytest.raises(ValueError):
            create("adaptive", {"block_size": 1})


# =====================================================================
# mser
# =====================================================================

class TestMser:
    """MSER region detection → convex-hull filling (text = 255)."""

    def test_dark_disc_region_filled_white(self) -> None:
        """A dark gradient disc on white is detected and its hull filled 255.

        MSER looks for extremal regions bounded by intensity *gradients* (like
        real text strokes). A perfectly flat synthetic disc (solid 0 on solid
        255) has no gradient, so OpenCV returns no stable regions for it — the
        test premise must match the algorithm's operating domain. Blurring gives
        the disc a smooth intensity ramp while keeping the center darkest.
        """
        # Arrange
        img = np.full((64, 64), 255, dtype=np.uint8)
        cv2.circle(img, (32, 32), 14, 0, -1)  # dark disc, radius 14
        img = cv2.GaussianBlur(img, (5, 5), 0)  # smooth gradient (real-text-like)
        # Act
        result = create("mser").segment(img)
        # Assert
        assert result.shape == img.shape
        assert result.dtype == np.uint8
        assert result[32, 32] == 255  # disc center within a detected hull
        assert 0 in np.unique(result)  # background pixels remain 0

    def test_tuning_params_are_tolerated(self) -> None:
        """Extra MSER tuning keys do not break empty/default construction."""
        method = create("mser", {"delta": 1, "min_area": 10})
        assert method.name == "mser"
        result = method.segment(_test_image())
        assert result.dtype == np.uint8 and result.shape == (64, 64)


# =====================================================================
# edges
# =====================================================================

class TestEdges:
    """Canny → dilate → fill contours (text = 255)."""

    def test_rect_interior_filled_white(self) -> None:
        """Boundary edges of a bright square are filled solid."""
        # Arrange
        img = np.zeros((64, 64), dtype=np.uint8)
        img[10:41, 10:41] = 255
        # Act
        result = create("edges").segment(img)
        # Assert
        assert result.shape == img.shape
        assert result.dtype == np.uint8
        assert result[25, 25] == 255  # interior filled by contour
        assert result[2, 2] == 0  # far background untouched
        assert result[62, 62] == 0

    def test_invalid_thresholds_rejected(self) -> None:
        """low < 0 or low > high is a configuration error."""
        with pytest.raises(ValueError):
            create("edges", {"low": -1})
        with pytest.raises(ValueError):
            create("edges", {"low": 200, "high": 50})


# =====================================================================
# components
# =====================================================================

class TestComponents:
    """Otsu-INV binarization + area/aspect component filter."""

    def test_keeps_large_compact_and_drops_items(self) -> None:
        """Large square kept; speckle and wide-thin bars filtered out."""
        # Arrange
        img = np.full((64, 64), 255, dtype=np.uint8)
        img[5:26, 5:26] = 0  # 21×21 square (area 441, aspect 1) → keep
        img[40, 40] = 0  # 1 px speckle (area 1 < min_area 50) → drop
        img[30, 5:63] = 0  # 58×1 bar (aspect 58 > 20) → drop
        # Act
        result = create("components", {"min_area": 50, "max_aspect_ratio": 20.0}).segment(img)
        # Assert
        assert result.shape == img.shape
        assert result[15, 15] == 255  # kept component
        assert result[40, 40] == 0  # speckle dropped
        assert result[30, 30] == 0  # wide-thin bar dropped

    def test_negative_min_area_rejected(self) -> None:
        with pytest.raises(ValueError):
            create("components", {"min_area": -1})

    def test_non_positive_aspect_ratio_rejected(self) -> None:
        with pytest.raises(ValueError):
            create("components", {"max_aspect_ratio": 0})


# =====================================================================
# morphology
# =====================================================================

class TestMorphology:
    """Otsu-INV binarization + opening/closing with a configurable kernel."""

    def test_open_removes_isolated_speckle(self) -> None:
        """Opening wipes a lone speckle but keeps the solid text block."""
        # Arrange
        img = np.full((64, 64), 255, dtype=np.uint8)
        img[10:31, 10:31] = 0  # solid dark block
        img[45, 45] = 0  # isolated 1 px speckle
        # Act
        result = create("morphology").segment(img)
        # Assert
        assert result.shape == img.shape
        assert result[20, 20] == 255  # block survives opening
        assert result[45, 45] == 0  # speckle removed
        assert result[2, 2] == 0  # background remains 0

    def test_close_fills_small_hole_inside_text(self) -> None:
        """Closing fills a 1 px bright hole inside a dark text region."""
        # Arrange
        img = np.full((64, 64), 255, dtype=np.uint8)
        img[10:41, 10:41] = 0  # solid dark block
        img[30, 30] = 255  # 1 px hole inside the block
        # Act
        result = create("morphology").segment(img)
        # Assert
        assert result[30, 30] == 255  # hole filled by closing
        assert result[20, 20] == 255  # block interior preserved
        assert result[2, 2] == 0

    def test_invalid_kernel_rejected(self) -> None:
        with pytest.raises(ValueError):
            create("morphology", {"kernel": 0})

    def test_unknown_kernel_shape_rejected(self) -> None:
        with pytest.raises(ValueError):
            create("morphology", {"kernel_shape": "star"})

    def test_negative_iterations_rejected(self) -> None:
        with pytest.raises(ValueError):
            create("morphology", {"open_iter": -1})
        with pytest.raises(ValueError):
            create("morphology", {"close_iter": -1})


# =====================================================================
# pipeline
# =====================================================================

class TestPipeline:
    """Composable multi-stage pipelines (FR-020 last bullet)."""

    def test_default_pipeline_matches_otsu(self) -> None:
        """No-config pipeline ≡ a single Otsu stage."""
        # Arrange
        img = _test_image()
        # Act
        result = create("pipeline").segment(img)
        expected = create("otsu").segment(img)
        # Assert
        assert np.array_equal(result, expected)

    def test_stages_chain_in_order_to_binary(self) -> None:
        """otsu → morphology runs end-to-end and stays binary."""
        # Arrange
        img = _test_image()
        stages = [{"method": "otsu"}, {"method": "morphology"}]
        # Act
        result = create("pipeline", {"stages": stages}).segment(img)
        # Assert
        assert result.shape == img.shape
        assert result.dtype == np.uint8
        assert set(np.unique(result)) <= {0, 255}

    def test_non_list_stages_rejected(self) -> None:
        with pytest.raises(ValueError):
            create("pipeline", {"stages": "otsu"})

    def test_stage_without_method_key_rejected(self) -> None:
        with pytest.raises(ValueError):
            create("pipeline", {"stages": [{"params": {}}]})

    def test_unknown_stage_method_rejected(self) -> None:
        with pytest.raises(UnknownMethodError):
            create("pipeline", {"stages": [{"method": "does-not-exist"}]})