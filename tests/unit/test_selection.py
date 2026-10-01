"""T040 [P] [US4] Five sample-selection rules, N = 5 per method (FR-034, SC-007, research R2).

**TDD status**: This test file is written FIRST and MUST FAIL until
``src/manga_text_seg/selection.py`` is implemented (Constitution Principle II).

## What the five rules do (FR-034)

Given Spec 2's per-page metrics (IoU, F1, FP count, FN count) for each of the
four segmentation method identities, select up to N = 5 representative pages
**per rule per method**:

    1. highest_iou_f1   — top N by mean(IoU, F1)
    2. lowest_iou_f1    — bottom N by mean(IoU, F1)
    3. most_false_positives  — top N by FP count
    4. most_false_negatives  — top N by FN count
    5. manual_artifact_flags — at most N from a human-supplied list

## Shortfall rule (FR-034, research R2)

When a rule yields fewer than N candidates, the selection report records the
shortfall (``"shortfall": k``) and the list is returned as-is.  The system
MUST NOT pad with arbitrary pages.

## classical_baseline source (research R1)

``classical_baseline``'s metrics come from Spec 1's ``metrics.csv`` for the
method it resolved to (config-pinned to ``adaptive``).  The selection function
must accept the same metric schema for all four identities.

## Module under test (not yet implemented)

    from manga_text_seg.selection import select_samples, SelectionResult

All test imports are inside test functions so that the file can be *collected*
by pytest even before ``selection.py`` exists (the import error surfaces as
an XFAIL or ERROR only when the test runs, not at collection time).

Spec file refs: specs/003-text-removal-inpainting/tasks.md T040,
data-model.md §SampleSelectionRule, research.md §R2.
"""
from __future__ import annotations

from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Synthetic fixture data
# ---------------------------------------------------------------------------

#: Four method identities (FR-026).
METHODS = [
    "classical_baseline",
    "manga_text_segmentation",
    "comic_text_detector",
    "unetpp_efficientnetv2",
]

#: Simulated per-page metrics table.
#: Each entry: (image_id, iou, f1, fp, fn)
#: Crafted so the expected selection for each rule is unambiguous:
#:   - Highest IoU/F1 candidates are pages 000–004 (top 5 are deterministic).
#:   - Lowest IoU/F1 candidates are pages 005–009.
#:   - Most-FP candidates are pages 010–014.
#:   - Most-FN candidates are pages 015–019.
#: Pages 020–024 are mid-range filler.
_RAW_METRICS: list[tuple[str, float, float, int, int]] = [
    # (image_id,                   iou,  f1,   fp,   fn )
    ("ARMS/000",                   0.90, 0.92,  500, 100),  # high IoU/F1
    ("ARMS/001",                   0.88, 0.90,  480, 110),
    ("ARMS/002",                   0.86, 0.88,  460, 120),
    ("ARMS/003",                   0.84, 0.86,  440, 130),
    ("ARMS/004",                   0.82, 0.84,  420, 140),
    ("ARMS/005",                   0.10, 0.11,  200, 300),  # low IoU/F1
    ("ARMS/006",                   0.09, 0.10,  210, 310),
    ("ARMS/007",                   0.08, 0.09,  220, 320),
    ("ARMS/008",                   0.07, 0.08,  230, 330),
    ("ARMS/009",                   0.06, 0.07,  240, 340),
    ("Arisa/000",                  0.50, 0.52, 2000,  50),  # high FP
    ("Arisa/001",                  0.51, 0.53, 1900,  55),
    ("Arisa/002",                  0.52, 0.54, 1800,  60),
    ("Arisa/003",                  0.53, 0.55, 1700,  65),
    ("Arisa/004",                  0.54, 0.56, 1600,  70),
    ("Arisa/005",                  0.55, 0.57,  100, 3000),  # high FN
    ("Arisa/006",                  0.56, 0.58,  105, 2900),
    ("Arisa/007",                  0.57, 0.59,  110, 2800),
    ("Arisa/008",                  0.58, 0.60,  115, 2700),
    ("Arisa/009",                  0.59, 0.61,  120, 2600),
    ("DollGun/000",                0.45, 0.47,  300, 300),  # filler
    ("DollGun/001",                0.46, 0.48,  310, 310),
    ("DollGun/002",                0.47, 0.49,  320, 320),
    ("DollGun/003",                0.48, 0.50,  330, 330),
    ("DollGun/004",                0.49, 0.51,  340, 340),
]

#: Human-configured manual artifact flags (5 entries = exactly N).
MANUAL_FLAGS_FULL: list[str] = [
    "ARMS/000", "ARMS/001", "ARMS/002", "ARMS/003", "ARMS/004"
]

#: Human-configured manual artifact flags (3 entries < N → shortfall = 2).
MANUAL_FLAGS_SHORT: list[str] = ["ARMS/005", "ARMS/006", "ARMS/007"]


def _make_metrics(
    methods: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build a synthetic metric dict keyed by method identity.

    All four methods receive the same pages so cross-method assertions are easy.
    ``classical_baseline`` uses the same schema — its metrics come from Spec 1's
    ``metrics.csv`` for the ``adaptive`` method it resolved to (research R1).
    """
    methods = methods or METHODS
    return {
        method: [
            {
                "image_id": iid,
                "iou": iou,
                "f1": f1,
                "fp": fp,
                "fn": fn,
            }
            for iid, iou, f1, fp, fn in _RAW_METRICS
        ]
        for method in methods
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ids(results: list[dict]) -> list[str]:
    """Extract ``image_id`` values from a list of selection entries."""
    return [r["image_id"] for r in results]


# ---------------------------------------------------------------------------
# Import guard: all imports inside tests so collection doesn't fail early
# ---------------------------------------------------------------------------

def _import_select_samples():
    """Import the module under test, skip gracefully if not yet implemented."""
    try:
        from manga_text_seg.selection import select_samples  # noqa: PLC0415
        return select_samples
    except ImportError as exc:
        pytest.fail(
            f"selection.py is not implemented yet — T040 MUST FAIL (TDD): {exc}"
        )


# ---------------------------------------------------------------------------
# Tests: Rule 1 — Highest IoU/F1
# ---------------------------------------------------------------------------

class TestHighestIouF1:
    """Each method's top-N pages by mean(IoU, F1) are selected (FR-034 rule 1)."""

    def test_selects_top_n_per_method(self) -> None:
        """Rule 'highest_iou_f1' selects exactly N = 5 pages per method."""
        select_samples = _import_select_samples()
        metrics = _make_metrics()
        result = select_samples(
            metrics=metrics,
            rule="highest_iou_f1",
            n=5,
        )
        for method in METHODS:
            assert method in result, f"Method '{method}' missing from result"
            assert len(result[method]["selected"]) == 5, (
                f"Expected 5 selected pages for '{method}', "
                f"got {len(result[method]['selected'])}"
            )

    def test_selected_pages_are_highest_scoring(self) -> None:
        """Pages ARMS/000–004 (highest IoU/F1 in fixture) appear in every method's selection."""
        select_samples = _import_select_samples()
        metrics = _make_metrics()
        result = select_samples(metrics=metrics, rule="highest_iou_f1", n=5)
        expected_top = {"ARMS/000", "ARMS/001", "ARMS/002", "ARMS/003", "ARMS/004"}
        for method in METHODS:
            selected = set(_ids(result[method]["selected"]))
            assert selected == expected_top, (
                f"Method '{method}': expected top-5 pages {expected_top}, "
                f"got {selected}"
            )

    def test_no_shortfall_recorded_when_enough_candidates(self) -> None:
        """No shortfall is reported when the pool has >= N candidates."""
        select_samples = _import_select_samples()
        metrics = _make_metrics()
        result = select_samples(metrics=metrics, rule="highest_iou_f1", n=5)
        for method in METHODS:
            assert result[method].get("shortfall", 0) == 0, (
                f"Method '{method}': unexpected shortfall when N candidates >= 5"
            )

    def test_result_carries_required_metadata_fields(self) -> None:
        """Each result carries 'method', 'rule', 'selected', 'shortfall'."""
        select_samples = _import_select_samples()
        metrics = _make_metrics()
        result = select_samples(metrics=metrics, rule="highest_iou_f1", n=5)
        for method in METHODS:
            entry = result[method]
            assert "method" in entry, f"'method' key missing for {method}"
            assert "rule" in entry, f"'rule' key missing for {method}"
            assert "selected" in entry, f"'selected' key missing for {method}"
            assert "shortfall" in entry, f"'shortfall' key missing for {method}"
            assert entry["rule"] == "highest_iou_f1"
            assert entry["method"] == method


# ---------------------------------------------------------------------------
# Tests: Rule 2 — Lowest IoU/F1
# ---------------------------------------------------------------------------

class TestLowestIouF1:
    """Each method's bottom-N pages are selected (FR-034 rule 2)."""

    def test_selected_pages_are_lowest_scoring(self) -> None:
        """Pages ARMS/005–009 (lowest IoU/F1 in fixture) appear for every method."""
        select_samples = _import_select_samples()
        metrics = _make_metrics()
        result = select_samples(metrics=metrics, rule="lowest_iou_f1", n=5)
        expected_bottom = {"ARMS/005", "ARMS/006", "ARMS/007", "ARMS/008", "ARMS/009"}
        for method in METHODS:
            selected = set(_ids(result[method]["selected"]))
            assert selected == expected_bottom, (
                f"Method '{method}': expected bottom-5 pages {expected_bottom}, "
                f"got {selected}"
            )


# ---------------------------------------------------------------------------
# Tests: Rule 3 — Most False Positives
# ---------------------------------------------------------------------------

class TestMostFalsePositives:
    """Top-N by FP count (FR-034 rule 3)."""

    def test_selected_pages_have_most_fp(self) -> None:
        """Arisa/000–004 have the highest FP counts in the fixture."""
        select_samples = _import_select_samples()
        metrics = _make_metrics()
        result = select_samples(metrics=metrics, rule="most_false_positives", n=5)
        expected = {"Arisa/000", "Arisa/001", "Arisa/002", "Arisa/003", "Arisa/004"}
        for method in METHODS:
            selected = set(_ids(result[method]["selected"]))
            assert selected == expected, (
                f"Method '{method}': expected most-FP pages {expected}, got {selected}"
            )


# ---------------------------------------------------------------------------
# Tests: Rule 4 — Most False Negatives
# ---------------------------------------------------------------------------

class TestMostFalseNegatives:
    """Top-N by FN count (FR-034 rule 4)."""

    def test_selected_pages_have_most_fn(self) -> None:
        """Arisa/005–009 have the highest FN counts in the fixture."""
        select_samples = _import_select_samples()
        metrics = _make_metrics()
        result = select_samples(metrics=metrics, rule="most_false_negatives", n=5)
        expected = {"Arisa/005", "Arisa/006", "Arisa/007", "Arisa/008", "Arisa/009"}
        for method in METHODS:
            selected = set(_ids(result[method]["selected"]))
            assert selected == expected, (
                f"Method '{method}': expected most-FN pages {expected}, got {selected}"
            )


# ---------------------------------------------------------------------------
# Tests: Rule 5 — Manual Artifact Flags
# ---------------------------------------------------------------------------

class TestManualArtifactFlags:
    """Human-supplied flag list, at most N items (FR-034 rule 5)."""

    def test_full_flag_list_returns_all_five_no_shortfall(self) -> None:
        """When exactly N flags are supplied, all N are returned and shortfall = 0."""
        select_samples = _import_select_samples()
        metrics = _make_metrics()
        result = select_samples(
            metrics=metrics,
            rule="manual_artifact_flags",
            n=5,
            manual_flags=MANUAL_FLAGS_FULL,
        )
        for method in METHODS:
            entry = result[method]
            selected = _ids(entry["selected"])
            assert len(selected) == 5, (
                f"Method '{method}': expected 5 flagged pages, got {len(selected)}"
            )
            assert entry.get("shortfall", 0) == 0
            assert set(selected) == set(MANUAL_FLAGS_FULL)

    def test_short_flag_list_records_shortfall_not_padded(self) -> None:
        """When only 3 flags are supplied (< N=5), shortfall = 2 is recorded.

        This is the core invariant: the system MUST NOT pad the list with
        arbitrary pages that do not satisfy the manual-flag rule (FR-034,
        research R2).
        """
        select_samples = _import_select_samples()
        metrics = _make_metrics()
        result = select_samples(
            metrics=metrics,
            rule="manual_artifact_flags",
            n=5,
            manual_flags=MANUAL_FLAGS_SHORT,
        )
        for method in METHODS:
            entry = result[method]
            selected = _ids(entry["selected"])
            assert len(selected) == 3, (
                f"Method '{method}': expected exactly 3 selected (no padding), "
                f"got {len(selected)}"
            )
            assert entry["shortfall"] == 2, (
                f"Method '{method}': expected shortfall=2, "
                f"got {entry.get('shortfall')}"
            )
            assert set(selected) == set(MANUAL_FLAGS_SHORT)

    def test_empty_flag_list_records_full_shortfall(self) -> None:
        """When no flags are supplied at all, shortfall = N and selected = []."""
        select_samples = _import_select_samples()
        metrics = _make_metrics()
        result = select_samples(
            metrics=metrics,
            rule="manual_artifact_flags",
            n=5,
            manual_flags=[],
        )
        for method in METHODS:
            entry = result[method]
            assert entry["selected"] == [], (
                f"Method '{method}': expected empty selection, got {entry['selected']}"
            )
            assert entry["shortfall"] == 5, (
                f"Method '{method}': expected shortfall=5, "
                f"got {entry.get('shortfall')}"
            )


# ---------------------------------------------------------------------------
# Tests: Cross-method invariants
# ---------------------------------------------------------------------------

class TestCrossMethodInvariants:
    """Invariants that apply across all four method identities (FR-034, SC-007)."""

    def test_all_four_methods_present_in_result(self) -> None:
        """Result dict always contains exactly the four configured method identities."""
        select_samples = _import_select_samples()
        metrics = _make_metrics()
        for rule in ("highest_iou_f1", "lowest_iou_f1",
                     "most_false_positives", "most_false_negatives"):
            result = select_samples(metrics=metrics, rule=rule, n=5)
            assert set(result.keys()) == set(METHODS), (
                f"Rule '{rule}': expected methods {set(METHODS)}, "
                f"got {set(result.keys())}"
            )

    def test_n_is_applied_identically_per_method(self) -> None:
        """Each method receives the same upper bound N (no per-method override)."""
        select_samples = _import_select_samples()
        metrics = _make_metrics()
        result = select_samples(metrics=metrics, rule="highest_iou_f1", n=3)
        for method in METHODS:
            count = len(result[method]["selected"])
            assert count == 3, (
                f"Method '{method}': N=3 should yield exactly 3 samples, "
                f"got {count}"
            )

    def test_classical_baseline_uses_same_metric_schema(self) -> None:
        """``classical_baseline`` accepts the identical metric record schema (research R1).

        Its metrics come from Spec 1's ``metrics.csv`` for the resolved classical
        method (e.g. ``adaptive``). No special-casing for ``classical_baseline`` is
        needed on the caller side: the selection function treats all four identities
        the same way.
        """
        select_samples = _import_select_samples()
        # Only classical_baseline in the metrics dict
        metrics = {"classical_baseline": _make_metrics()["classical_baseline"]}
        result = select_samples(metrics=metrics, rule="highest_iou_f1", n=5)
        assert "classical_baseline" in result
        assert len(result["classical_baseline"]["selected"]) == 5

    def test_unknown_rule_raises_value_error(self) -> None:
        """Passing an unrecognised rule name raises a clear ValueError."""
        select_samples = _import_select_samples()
        metrics = _make_metrics()
        with pytest.raises((ValueError, KeyError)):
            select_samples(metrics=metrics, rule="best_vibes", n=5)

    def test_n_zero_returns_empty_selection(self) -> None:
        """N = 0 is an edge case: every method gets an empty list, shortfall = 0."""
        select_samples = _import_select_samples()
        metrics = _make_metrics()
        result = select_samples(metrics=metrics, rule="highest_iou_f1", n=0)
        for method in METHODS:
            assert result[method]["selected"] == []
            assert result[method].get("shortfall", 0) == 0

    def test_pool_smaller_than_n_records_shortfall(self) -> None:
        """When a method has only 2 pages and N = 5, shortfall = 3."""
        select_samples = _import_select_samples()
        tiny_metrics = {
            method: [
                {"image_id": f"EvaLady/00{i}", "iou": 0.5 + i * 0.01,
                 "f1": 0.5 + i * 0.01, "fp": 100 + i, "fn": 100 + i}
                for i in range(2)
            ]
            for method in METHODS
        }
        result = select_samples(metrics=tiny_metrics, rule="highest_iou_f1", n=5)
        for method in METHODS:
            entry = result[method]
            assert len(entry["selected"]) == 2, (
                f"Method '{method}': expected 2 selected (pool size), "
                f"got {len(entry['selected'])}"
            )
            assert entry["shortfall"] == 3, (
                f"Method '{method}': expected shortfall=3, got {entry.get('shortfall')}"
            )
