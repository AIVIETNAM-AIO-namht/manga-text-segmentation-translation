"""Unit tests for normalization and alignment (imaging, normalize, align).

TDD Principle II: tests written first (RED), validated against implementation.
Covers normalize_mask (KMeans-2), align (crop-topleft, resize), and imaging
loaders/validators.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from manga_text_seg.align import AlignedPair, align, AlignmentError
from manga_text_seg.imaging import (
    ImageError,
    detect_mask_encoding,
    load_gt_mask,
    load_mask,
    load_raw_page,
    MASK_ENCODINGS,
    save_mask,
)
from manga_text_seg.normalize import NormalizeError, normalize_mask


# =====================================================================
# normalize_mask tests
# =====================================================================

class TestNormalizeMask:
    """KMeans-2 normalization: color GT → binary {0, 255} uint8."""

    def test_all_zero_output_for_single_intensity(self) -> None:
        """All-zero input → all-zero output (no text)."""
        img = np.zeros((10, 10), dtype=np.uint8)
        result = normalize_mask(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8
        assert set(np.unique(result)) == {0}

    def test_all_255_input(self) -> None:
        """All-255 input → all-zero output (single cluster, no foreground)."""
        img = np.full((10, 10), 255, dtype=np.uint8)
        result = normalize_mask(img)
        assert set(np.unique(result)) == {0}

    def test_bicolor_magenta_black(self) -> None:
        """Magenta (255,0,255) + black (0,0,0) → binary output."""
        img = np.zeros((20, 20, 3), dtype=np.uint8)
        img[:10, :, :] = (255, 0, 255)  # magenta in BGR
        img[10:, :, :] = (0, 0, 0)      # black
        result = normalize_mask(img)
        assert result.dtype == np.uint8
        assert result.ndim == 2
        assert set(np.unique(result)).issubset({0, 255})

    def test_output_shape_preserved(self) -> None:
        """Output shape matches input shape."""
        img = np.random.randint(0, 256, (30, 50), dtype=np.uint8)
        result = normalize_mask(img)
        assert result.shape == img.shape

    def test_output_is_uint8(self) -> None:
        """Output dtype is uint8."""
        img = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
        result = normalize_mask(img)
        assert result.dtype == np.uint8

    def test_binary_values_only(self) -> None:
        """Output contains only 0 and 255."""
        img = np.random.randint(0, 256, (20, 20), dtype=np.uint8)
        result = normalize_mask(img)
        unique = set(np.unique(result))
        assert unique.issubset({0, 255})

    def test_three_channel_grayscale(self) -> None:
        """3-channel input (all same value per pixel) normalizes correctly."""
        gray = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
        img3 = np.stack([gray, gray, gray], axis=-1)
        result = normalize_mask(img3)
        assert result.ndim == 2
        assert result.shape == (10, 10)

    def test_deterministic(self) -> None:
        """Same input always produces same output."""
        img = np.random.randint(0, 256, (15, 15), dtype=np.uint8)
        r1 = normalize_mask(img)
        r2 = normalize_mask(img)
        np.testing.assert_array_equal(r1, r2)


# =====================================================================
# align tests
# =====================================================================

class TestAlignCropTopleft:
    """crop-topleft alignment: both images cropped to min common dims."""

    def test_same_size_no_crop(self) -> None:
        """Same-size inputs → output shape unchanged."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        page = np.ones((100, 100), dtype=np.uint8)
        result = align(mask, page, "crop-topleft")
        assert result.mask.shape == (100, 100)
        assert result.page.shape == (100, 100)

    def test_mask_larger_than_page(self) -> None:
        """Mask larger → both cropped to page dims."""
        mask = np.zeros((120, 110), dtype=np.uint8)
        page = np.ones((100, 100), dtype=np.uint8)
        result = align(mask, page, "crop-topleft")
        assert result.mask.shape == (100, 100)
        assert result.page.shape == (100, 100)

    def test_page_larger_than_mask(self) -> None:
        """Page larger → both cropped to mask dims."""
        mask = np.zeros((80, 80), dtype=np.uint8)
        page = np.ones((120, 120), dtype=np.uint8)
        result = align(mask, page, "crop-topleft")
        assert result.mask.shape == (80, 80)
        assert result.page.shape == (80, 80)

    def test_crop_topleft_preserves_origin(self) -> None:
        """Cropped content comes from (0,0) — top-left region preserved."""
        mask = np.arange(100).reshape(10, 10).astype(np.uint8)
        page = np.arange(100).reshape(10, 10).astype(np.uint8) + 100
        result = align(mask, page, "crop-topleft")
        np.testing.assert_array_equal(result.mask, mask)
        np.testing.assert_array_equal(result.page, page)

    def test_returns_aligned_pair(self) -> None:
        """Result is an AlignedPair dataclass."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        page = np.ones((50, 50), dtype=np.uint8)
        result = align(mask, page)
        assert isinstance(result, AlignedPair)
        assert result.strategy == "crop-topleft"

    def test_crop_dimensions_recorded(self) -> None:
        """AlignedPair records crop_height and crop_width."""
        mask = np.zeros((100, 120), dtype=np.uint8)
        page = np.ones((90, 100), dtype=np.uint8)
        result = align(mask, page, "crop-topleft")
        assert result.crop_height == 90
        assert result.crop_width == 100


class TestAlignResize:
    """resize alignment: page resized to match mask dims."""

    def test_resize_page_to_mask_dims(self) -> None:
        """Page resized to match mask dimensions."""
        mask = np.zeros((100, 120), dtype=np.uint8)
        page = np.ones((80, 90), dtype=np.uint8)
        result = align(mask, page, "resize")
        assert result.page.shape == (100, 120)
        assert result.mask.shape == (100, 120)

    def test_same_size_no_resize(self) -> None:
        """Same-size inputs → no resizing."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        page = np.ones((50, 50), dtype=np.uint8)
        result = align(mask, page, "resize")
        assert result.page.shape == (50, 50)

    def test_resize_strategy_name(self) -> None:
        """AlignedPair.strategy == 'resize'."""
        mask = np.zeros((60, 60), dtype=np.uint8)
        page = np.ones((40, 40), dtype=np.uint8)
        result = align(mask, page, "resize")
        assert result.strategy == "resize"


class TestAlignEdgeCases:
    """Edge cases and error handling."""

    def test_invalid_strategy_raises(self) -> None:
        """Unknown strategy raises AlignmentError."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        page = np.ones((10, 10), dtype=np.uint8)
        with pytest.raises(AlignmentError):
            align(mask, page, "invalid_strategy")

    def test_non_2d_input_raises(self) -> None:
        """1D array raises AlignmentError."""
        mask = np.zeros((10,), dtype=np.uint8)
        page = np.ones((10,), dtype=np.uint8)
        with pytest.raises(AlignmentError):
            align(mask, page)

    def test_3d_mask_with_multiple_channels_raises(self) -> None:
        """3-channel mask raises AlignmentError (must be single-channel)."""
        mask = np.zeros((10, 10, 3), dtype=np.uint8)
        page = np.ones((10, 10), dtype=np.uint8)
        with pytest.raises(AlignmentError):
            align(mask, page)

    def test_3d_single_channel_mask_accepted(self) -> None:
        """(H, W, 1) mask is accepted."""
        mask = np.zeros((10, 10, 1), dtype=np.uint8)
        page = np.ones((10, 10), dtype=np.uint8)
        result = align(mask, page)
        assert result.mask.shape[0] == 10


# =====================================================================
# imaging loader tests
# =====================================================================

class TestLoadMask:
    """load_mask: strict binary {0, 255} single-channel validator."""

    def test_valid_binary_mask(self, tmp_path: Path) -> None:
        """Valid binary mask loads correctly."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[:5, :] = 255
        path = tmp_path / "mask.png"
        cv2.imwrite(str(path), mask)
        result = load_mask(path)
        np.testing.assert_array_equal(result, mask)

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Missing file raises ImageError."""
        with pytest.raises(ImageError):
            load_mask(tmp_path / "nonexistent.png")

    def test_multichannel_rejected(self, tmp_path: Path) -> None:
        """3-channel image raises ImageError."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        path = tmp_path / "color.png"
        cv2.imwrite(str(path), img)
        with pytest.raises(ImageError, match="single-channel"):
            load_mask(path)

    def test_non_binary_values_rejected(self, tmp_path: Path) -> None:
        """Image with values other than {0, 255} raises ImageError."""
        img = np.full((10, 10), 128, dtype=np.uint8)
        path = tmp_path / "gray.png"
        cv2.imwrite(str(path), img)
        with pytest.raises(ImageError, match="unexpected values"):
            load_mask(path)


class TestLoadGtMask:
    """load_gt_mask: permissive loader for color GT masks."""

    def test_loads_color_image(self, tmp_path: Path) -> None:
        """Color (3-channel) GT mask loads without error."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:5, :, :] = (255, 0, 255)  # magenta BGR
        path = tmp_path / "gt.png"
        cv2.imwrite(str(path), img)
        result = load_gt_mask(path)
        assert result is not None
        assert result.ndim == 3

    def test_loads_grayscale_image(self, tmp_path: Path) -> None:
        """Grayscale GT mask loads without error."""
        img = np.zeros((10, 10), dtype=np.uint8)
        path = tmp_path / "gt_gray.png"
        cv2.imwrite(str(path), img)
        result = load_gt_mask(path)
        assert result is not None

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Missing file raises ImageError."""
        with pytest.raises(ImageError):
            load_gt_mask(tmp_path / "nonexistent.png")


class TestLoadRawPage:
    """load_raw_page: grayscale loader for raw manga pages."""

    def test_loads_grayscale(self, tmp_path: Path) -> None:
        """Grayscale page loads correctly."""
        img = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
        path = tmp_path / "page.jpg"
        cv2.imwrite(str(path), img)
        result = load_raw_page(path)
        assert result.ndim == 2
        assert result.dtype == np.uint8

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Missing file raises ImageError."""
        with pytest.raises(ImageError):
            load_raw_page(tmp_path / "nonexistent.jpg")


class TestSaveMask:
    """save_mask: write mask to disk as PNG."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        """Save and reload a mask — values preserved."""
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[5:15, 5:25] = 255
        path = tmp_path / "out.png"
        save_mask(path, mask)
        loaded = load_mask(path)
        np.testing.assert_array_equal(loaded, mask)

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save_mask creates intermediate directories."""
        path = tmp_path / "a" / "b" / "c" / "mask.png"
        mask = np.zeros((5, 5), dtype=np.uint8)
        save_mask(path, mask)
        assert path.exists()


# =====================================================================
# detect_mask_encoding tests
# =====================================================================

class TestDetectMaskEncoding:
    """classify_mask_encoding: color → enum mapping."""

    VALID_ENCODINGS = set(MASK_ENCODINGS)

    def test_magenta_black(self, tmp_path: Path) -> None:
        """Magenta + black pixels → 'magenta-black'."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:5, :, :] = (255, 0, 255)  # magenta BGR
        img[5:, :, :] = (0, 0, 0)      # black
        path = tmp_path / "mb.png"
        cv2.imwrite(str(path), img)
        assert detect_mask_encoding(path) == "magenta-black"

    def test_magenta_only(self, tmp_path: Path) -> None:
        """Magenta on white → 'magenta-only'."""
        img = np.full((10, 10, 3), 255, dtype=np.uint8)
        img[:5, :, :] = (255, 0, 255)  # magenta BGR
        path = tmp_path / "mo.png"
        cv2.imwrite(str(path), img)
        assert detect_mask_encoding(path) == "magenta-only"

    def test_near_black_only(self, tmp_path: Path) -> None:
        """Dark text only → 'near-black-only'."""
        img = np.full((10, 10, 3), 255, dtype=np.uint8)
        img[:5, :, :] = (0, 0, 0)  # black
        path = tmp_path / "nb.png"
        cv2.imwrite(str(path), img)
        assert detect_mask_encoding(path) == "near-black-only"

    def test_all_white_empty(self, tmp_path: Path) -> None:
        """All-white image → 'all-white-empty'."""
        img = np.full((10, 10, 3), 255, dtype=np.uint8)
        path = tmp_path / "we.png"
        cv2.imwrite(str(path), img)
        assert detect_mask_encoding(path) == "all-white-empty"

    def test_result_in_enum(self, tmp_path: Path) -> None:
        """Any detected encoding is a valid enum value."""
        img = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        path = tmp_path / "rand.png"
        cv2.imwrite(str(path), img)
        result = detect_mask_encoding(path)
        assert result in self.VALID_ENCODINGS

    def test_grayscale_input(self, tmp_path: Path) -> None:
        """Grayscale input is handled without error."""
        img = np.zeros((10, 10), dtype=np.uint8)
        img[:5, :] = 255
        path = tmp_path / "gray.png"
        cv2.imwrite(str(path), img)
        result = detect_mask_encoding(path)
        assert result in self.VALID_ENCODINGS

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Missing file raises ImageError."""
        with pytest.raises(ImageError):
            detect_mask_encoding(tmp_path / "nope.png")
