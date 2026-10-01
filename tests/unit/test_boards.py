"""T041 [P] [US4] Five-panel comparison board — no GT, no GT overlay (FR-033, SC-007).

**TDD status**: This test file is written FIRST and MUST FAIL until
``src/manga_text_seg/boards.py`` is implemented (Constitution Principle II).

## What a comparison board contains (FR-033, SC-007)

For each selected representative sample, the system renders a composite image
(the "board") containing **exactly five panels**, each labelled:

    1. original_page       — the raw manga page
    2. raw_prediction_mask — the binary mask from Spec 2 (before any processing)
    3. dilated_mask         — the mask after morphological dilation
    4. telea_result         — inpainted image using ``INPAINT_TELEA``
    5. ns_result            — inpainted image using ``INPAINT_NS``

## Prohibited content (FR-033, SC-007, TEAM-ASSIGNMENT T041)

- **No panel** may be a ground-truth mask.
- **No panel** may be a prediction-vs-GT overlay.

The board's purpose is to show what the *system* produces, not to compare
against a human-annotated gold standard.

## Labels on each board (SC-007)

Every board must carry visible identity labels:
- segmentation method identity
- inpainting algorithm
- mask-processing configuration (at minimum: dilation kernel size, iterations)
- ``image_id``

## Module under test (not yet implemented)

    from manga_text_seg.boards import render_board, BoardResult

All test imports are inside test functions so that the file can be *collected*
by pytest even before ``boards.py`` exists (the import error surfaces as
a FAIL only when the test runs, not at collection time).

Spec file refs: specs/003-text-removal-inpainting/tasks.md T041,
spec.md FR-033, spec.md SC-007, data-model.md §ComparisonBoard.
"""
from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The five panels mandated by FR-033 (exact keys).
FIVE_PANEL_KEYS: list[str] = [
    "original_page",
    "raw_prediction_mask",
    "dilated_mask",
    "telea_result",
    "ns_result",
]

#: Four method identities (FR-026).
METHODS: list[str] = [
    "classical_baseline",
    "manga_text_segmentation",
    "comic_text_detector",
    "unetpp_efficientnetv2",
]

#: Terms that must NEVER appear in any board label or metadata because
#: they imply ground-truth involvement (TEAM-ASSIGNMENT T041 §3–§4).
GT_PROHIBITED_TERMS: list[str] = [
    "ground_truth",
    "ground-truth",
    "groundtruth",
    "gt_mask",
    "gt_overlay",
    "overlay",
]

# ---------------------------------------------------------------------------
# Synthetic fixture data
# ---------------------------------------------------------------------------

#: Minimal page geometry for fixture images (small for speed).
PAGE_H, PAGE_W = 64, 48


def _make_dummy_image(h: int = PAGE_H, w: int = PAGE_W,
                      channels: int = 3) -> np.ndarray:
    """Create a small random uint8 image for fixture use."""
    if channels == 1:
        return np.random.randint(0, 256, (h, w), dtype=np.uint8)
    return np.random.randint(0, 256, (h, w, channels), dtype=np.uint8)


def _make_dummy_mask(h: int = PAGE_H, w: int = PAGE_W) -> np.ndarray:
    """Create a small binary mask (single channel, uint8)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[: h // 3, : w // 3] = 255
    return mask


#: A sample descriptor — the input to ``render_board``.
FIXTURE_SAMPLES: list[dict[str, Any]] = [
    {
        "image_id": "ARMS/000",
        "method": "manga_text_segmentation",
        "algorithm": "telea",
        "mask_proc_config": {
            "dilation": {"enabled": True, "kernel_shape": "ellipse",
                         "kernel_size": [3, 3], "iterations": 1},
            "inpaint_radius": 3,
        },
        "panels": {
            "original_page": _make_dummy_image(),
            "raw_prediction_mask": _make_dummy_mask(),
            "dilated_mask": _make_dummy_mask(),
            "telea_result": _make_dummy_image(),
            "ns_result": _make_dummy_image(),
        },
    },
    {
        "image_id": "Arisa/003",
        "method": "comic_text_detector",
        "algorithm": "ns",
        "mask_proc_config": {
            "dilation": {"enabled": True, "kernel_shape": "ellipse",
                         "kernel_size": [5, 5], "iterations": 2},
            "inpaint_radius": 5,
        },
        "panels": {
            "original_page": _make_dummy_image(),
            "raw_prediction_mask": _make_dummy_mask(),
            "dilated_mask": _make_dummy_mask(),
            "telea_result": _make_dummy_image(),
            "ns_result": _make_dummy_image(),
        },
    },
    {
        "image_id": "DollGun/002",
        "method": "unetpp_efficientnetv2",
        "algorithm": "telea",
        "mask_proc_config": {
            "dilation": {"enabled": True, "kernel_shape": "ellipse",
                         "kernel_size": [3, 3], "iterations": 1},
            "inpaint_radius": 3,
        },
        "panels": {
            "original_page": _make_dummy_image(),
            "raw_prediction_mask": _make_dummy_mask(),
            "dilated_mask": _make_dummy_mask(),
            "telea_result": _make_dummy_image(),
            "ns_result": _make_dummy_image(),
        },
    },
    {
        "image_id": "EvaLady/001",
        "method": "classical_baseline",
        "algorithm": "ns",
        "mask_proc_config": {
            "dilation": {"enabled": True, "kernel_shape": "ellipse",
                         "kernel_size": [3, 3], "iterations": 1},
            "inpaint_radius": 3,
        },
        "panels": {
            "original_page": _make_dummy_image(),
            "raw_prediction_mask": _make_dummy_mask(),
            "dilated_mask": _make_dummy_mask(),
            "telea_result": _make_dummy_image(),
            "ns_result": _make_dummy_image(),
        },
    },
]


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

def _import_render_board():
    """Import the module under test; fail with a clear TDD message if absent."""
    try:
        from manga_text_seg.boards import render_board  # noqa: PLC0415
        return render_board
    except ImportError as exc:
        pytest.fail(
            f"boards.py is not implemented yet — T041 MUST FAIL (TDD): {exc}"
        )


# ---------------------------------------------------------------------------
# Tests: Exactly five panels (FR-033)
# ---------------------------------------------------------------------------

class TestFivePanelStructure:
    """Every board contains exactly five panels — no more, no fewer (FR-033)."""

    def test_board_result_has_five_panels(self) -> None:
        """The ``render_board`` return value for each sample has five panel entries."""
        render_board = _import_render_board()
        for sample in FIXTURE_SAMPLES:
            result = render_board(sample)
            panels = result.get("panels", {})
            assert len(panels) == 5, (
                f"Sample '{sample['image_id']}': expected 5 panels, "
                f"got {len(panels)}: {sorted(panels.keys())}"
            )

    def test_all_five_panel_keys_present(self) -> None:
        """Exactly the five mandated panel keys appear in every board."""
        render_board = _import_render_board()
        for sample in FIXTURE_SAMPLES:
            result = render_board(sample)
            panels = result.get("panels", {})
            actual_keys = set(panels.keys())
            expected_keys = set(FIVE_PANEL_KEYS)
            assert actual_keys == expected_keys, (
                f"Sample '{sample['image_id']}': panel key mismatch. "
                f"Missing: {expected_keys - actual_keys}, "
                f"Extra: {actual_keys - expected_keys}"
            )

    def test_panel_order_matches_spec(self) -> None:
        """Panel keys, when ordered, follow the spec sequence (FR-033).

        The order is: original page → raw prediction mask → dilated mask →
        TELEA result → NS result. This tests that the keys are at minimum
        iterable in the mandated order.
        """
        render_board = _import_render_board()
        for sample in FIXTURE_SAMPLES:
            result = render_board(sample)
            panels = result.get("panels", {})
            actual_order = list(panels.keys())
            assert actual_order == FIVE_PANEL_KEYS, (
                f"Sample '{sample['image_id']}': panel order mismatch. "
                f"Expected {FIVE_PANEL_KEYS}, got {actual_order}"
            )


# ---------------------------------------------------------------------------
# Tests: Each panel is a valid image array
# ---------------------------------------------------------------------------

class TestPanelContent:
    """Each panel holds a numpy array representing an image."""

    def test_panels_are_numpy_arrays(self) -> None:
        """Every panel value is a numpy ndarray (not None, not a path string)."""
        render_board = _import_render_board()
        for sample in FIXTURE_SAMPLES:
            result = render_board(sample)
            panels = result.get("panels", {})
            for key, panel in panels.items():
                assert isinstance(panel, np.ndarray), (
                    f"Sample '{sample['image_id']}' / panel '{key}': "
                    f"expected numpy ndarray, got {type(panel).__name__}"
                )

    def test_panels_share_height_width(self) -> None:
        """All five panels share the same spatial dimensions (H, W)."""
        render_board = _import_render_board()
        for sample in FIXTURE_SAMPLES:
            result = render_board(sample)
            panels = result.get("panels", {})
            shapes = {key: panel.shape[:2] for key, panel in panels.items()}
            unique_shapes = set(shapes.values())
            assert len(unique_shapes) == 1, (
                f"Sample '{sample['image_id']}': panels have different "
                f"spatial dimensions: {shapes}"
            )

    def test_panels_are_non_empty(self) -> None:
        """No panel is a zero-size array."""
        render_board = _import_render_board()
        for sample in FIXTURE_SAMPLES:
            result = render_board(sample)
            panels = result.get("panels", {})
            for key, panel in panels.items():
                assert panel.size > 0, (
                    f"Sample '{sample['image_id']}' / panel '{key}': "
                    "panel image has zero size"
                )


# ---------------------------------------------------------------------------
# Tests: Identity labels on the board (SC-007)
# ---------------------------------------------------------------------------

class TestBoardLabels:
    """The board carries identity labels: method, algorithm, config, image_id."""

    def test_board_has_image_id(self) -> None:
        """The board result includes the ``image_id`` that produced it."""
        render_board = _import_render_board()
        for sample in FIXTURE_SAMPLES:
            result = render_board(sample)
            assert "image_id" in result, (
                f"Sample '{sample['image_id']}': 'image_id' label missing"
            )
            assert result["image_id"] == sample["image_id"]

    def test_board_has_method_label(self) -> None:
        """The board result includes the segmentation method identity."""
        render_board = _import_render_board()
        for sample in FIXTURE_SAMPLES:
            result = render_board(sample)
            assert "method" in result, (
                f"Sample '{sample['image_id']}': 'method' label missing"
            )
            assert result["method"] == sample["method"]

    def test_board_has_algorithm_label(self) -> None:
        """The board result includes the inpainting algorithm identity."""
        render_board = _import_render_board()
        for sample in FIXTURE_SAMPLES:
            result = render_board(sample)
            assert "algorithm" in result, (
                f"Sample '{sample['image_id']}': 'algorithm' label missing"
            )
            assert result["algorithm"] in ("telea", "ns"), (
                f"Sample '{sample['image_id']}': algorithm "
                f"'{result['algorithm']}' not in {{telea, ns}}"
            )
            assert result["algorithm"] == sample["algorithm"]

    def test_board_has_mask_proc_config(self) -> None:
        """The board result carries the mask-processing configuration."""
        render_board = _import_render_board()
        for sample in FIXTURE_SAMPLES:
            result = render_board(sample)
            assert "mask_proc_config" in result, (
                f"Sample '{sample['image_id']}': "
                "'mask_proc_config' label missing"
            )


# ---------------------------------------------------------------------------
# Tests: Absolute prohibition of ground-truth content (T041 §3–§4)
# ---------------------------------------------------------------------------

class TestNoGroundTruth:
    """No panel is a GT mask and no panel is a prediction-vs-GT overlay."""

    def test_no_gt_mask_panel_key(self) -> None:
        """No panel key matches any ground-truth related term."""
        render_board = _import_render_board()
        for sample in FIXTURE_SAMPLES:
            result = render_board(sample)
            panels = result.get("panels", {})
            for key in panels:
                for term in GT_PROHIBITED_TERMS:
                    assert term not in key.lower(), (
                        f"Sample '{sample['image_id']}': panel key '{key}' "
                        f"contains prohibited GT term '{term}'. "
                        "No ground-truth panel is allowed on the board "
                        "(FR-033, TEAM-ASSIGNMENT T041)."
                    )

    def test_no_gt_overlay_panel_key(self) -> None:
        """No panel key suggests a prediction-vs-GT overlay."""
        render_board = _import_render_board()
        gt_overlay_patterns = [
            re.compile(r"(gt|ground.?truth).*(overlay|compar)", re.IGNORECASE),
            re.compile(r"(overlay|compar).*(gt|ground.?truth)", re.IGNORECASE),
            re.compile(r"pred.*vs.*gt", re.IGNORECASE),
            re.compile(r"gt.*vs.*pred", re.IGNORECASE),
        ]
        for sample in FIXTURE_SAMPLES:
            result = render_board(sample)
            panels = result.get("panels", {})
            for key in panels:
                for pattern in gt_overlay_patterns:
                    assert not pattern.search(key), (
                        f"Sample '{sample['image_id']}': panel key '{key}' "
                        f"matches GT-overlay pattern. No prediction-vs-GT "
                        "overlay is allowed on the board (T041 §4)."
                    )

    def test_board_metadata_has_no_gt_references(self) -> None:
        """Serialised board metadata must not reference ground truth."""
        render_board = _import_render_board()
        for sample in FIXTURE_SAMPLES:
            result = render_board(sample)
            # Serialise all non-panel values to check for GT terms
            metadata = {k: v for k, v in result.items() if k != "panels"}
            text = json.dumps(metadata, default=str)
            for term in GT_PROHIBITED_TERMS:
                assert term not in text.lower(), (
                    f"Sample '{sample['image_id']}': board metadata contains "
                    f"prohibited GT term '{term}'. No ground-truth reference "
                    "may appear in the board output."
                )

    def test_exactly_five_panels_no_extra(self) -> None:
        """There are exactly 5 panel keys — no sixth GT panel sneaked in."""
        render_board = _import_render_board()
        for sample in FIXTURE_SAMPLES:
            result = render_board(sample)
            panels = result.get("panels", {})
            extra = set(panels.keys()) - set(FIVE_PANEL_KEYS)
            assert not extra, (
                f"Sample '{sample['image_id']}': unexpected extra panel(s) "
                f"{extra}. Only the five mandated panels are allowed (FR-033)."
            )


# ---------------------------------------------------------------------------
# Tests: Cross-sample invariants
# ---------------------------------------------------------------------------

class TestCrossSampleInvariants:
    """Invariants that apply across multiple board generations."""

    def test_different_methods_produce_boards(self) -> None:
        """All four method identities can each produce a board without error."""
        render_board = _import_render_board()
        methods_seen = set()
        for sample in FIXTURE_SAMPLES:
            result = render_board(sample)
            methods_seen.add(result.get("method"))
        assert len(methods_seen) >= 2, (
            f"Expected boards from multiple methods, but only saw: {methods_seen}"
        )

    def test_both_algorithms_produce_boards(self) -> None:
        """Both 'telea' and 'ns' algorithms produce valid boards."""
        render_board = _import_render_board()
        algorithms_seen = set()
        for sample in FIXTURE_SAMPLES:
            result = render_board(sample)
            algorithms_seen.add(result.get("algorithm"))
        assert "telea" in algorithms_seen, "No TELEA board produced"
        assert "ns" in algorithms_seen, "No NS board produced"

    def test_each_board_is_independent(self) -> None:
        """Each board result carries its own distinct image_id."""
        render_board = _import_render_board()
        image_ids = []
        for sample in FIXTURE_SAMPLES:
            result = render_board(sample)
            image_ids.append(result.get("image_id"))
        assert len(set(image_ids)) == len(image_ids), (
            f"Board image_ids are not all unique: {image_ids}"
        )
