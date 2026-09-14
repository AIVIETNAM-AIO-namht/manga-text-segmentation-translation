"""Contract tests for metrics schema (metrics.schema.json).

Hand-rolled validators — no jsonschema dependency (Constitution Principle III).
Validates metrics.csv rows and summaries.json summaries against the frozen
contract. No Pixel Accuracy field is checked for absence.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from manga_text_seg.report import (
    METRIC_COLUMNS,
    SCORECARD_COLUMNS,
    write_metrics_csv,
    write_summaries_json,
    load_metrics_csv,
)

# --- Contract constants (from contracts/metrics.schema.json) ---

ROW_REQUIRED_FIELDS = {
    "manga", "stem", "method", "iou", "precision", "recall", "f1",
    "tp", "fp", "fn", "inference_time_seconds",
}
SUMMARY_REQUIRED_FIELDS = {
    "method", "n_images", "mean_iou", "std_iou",
    "mean_precision", "mean_recall", "mean_f1", "mean_time_seconds",
}
# Row-level fields (each CSV row) vs summary-only fields (summaries.json).
# Mixed row/summary field sets caused repeated KeyError bugs (rows have no
# n_images, summaries have no tp/fp/fn) — split deliberately so each validator
# loops only its own set.
ROW_NON_NEGATIVE_INT_FIELDS = {"tp", "fp", "fn"}
SUMMARY_NON_NEGATIVE_INT_FIELDS = {"n_images"}
ROW_RANGE_01_FIELDS = {"iou", "precision", "recall", "f1"}
SUMMARY_RANGE_01_FIELDS = {"mean_iou", "mean_precision", "mean_recall", "mean_f1"}
BANNED_FIELDS = {"pixel_accuracy", "pixel_accuracy_mean", "accuracy"}


# --- Helpers ---

def _make_row(manga: str = "Test", stem: str = "001", method: str = "otsu",
              iou: float = 0.5, precision: float = 0.6, recall: float = 0.7,
              f1: float = 0.65, tp: int = 10, fp: int = 3, fn: int = 4,
              inference_time_seconds: float = 0.12) -> dict:
    """Create a minimal valid metrics row."""
    return {
        "manga": manga, "stem": stem, "method": method,
        "iou": iou, "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "inference_time_seconds": inference_time_seconds,
    }


def _validate_row(row: dict, idx: int) -> None:
    """Validate a single metrics row against the schema contract."""
    missing = ROW_REQUIRED_FIELDS - set(row.keys())
    assert not missing, f"Row {idx} missing required fields: {missing}"

    for field in ROW_RANGE_01_FIELDS:
        val = row[field]
        assert isinstance(val, (int, float)), f"Row {idx}: {field} must be numeric"
        assert 0 <= val <= 1, f"Row {idx}: {field}={val} out of range [0, 1]"

    for field in ROW_NON_NEGATIVE_INT_FIELDS:
        val = row[field]
        assert isinstance(val, int), f"Row {idx}: {field} must be integer"
        assert val >= 0, f"Row {idx}: {field}={val} < 0"

    assert isinstance(row["inference_time_seconds"], (int, float))
    assert row["inference_time_seconds"] >= 0

    # Ban Pixel Accuracy
    banned = BANNED_FIELDS & set(row.keys())
    assert not banned, f"Row {idx}: banned fields found: {banned}"


def _validate_summary(summary: dict, idx: int) -> None:
    """Validate a single summary entry against the schema contract."""
    missing = SUMMARY_REQUIRED_FIELDS - set(summary.keys())
    assert not missing, f"Summary {idx} missing required fields: {missing}"

    assert isinstance(summary["n_images"], int)
    assert summary["n_images"] >= 0

    for field in SUMMARY_RANGE_01_FIELDS:
        val = summary[field]
        assert isinstance(val, (int, float)), f"Summary {idx}: {field} must be numeric"
        assert 0 <= val <= 1, f"Summary {idx}: {field}={val} out of range [0, 1]"

    assert isinstance(summary["std_iou"], (int, float))
    assert summary["std_iou"] >= 0
    assert isinstance(summary["mean_time_seconds"], (int, float))
    assert summary["mean_time_seconds"] >= 0

    banned = BANNED_FIELDS & set(summary.keys())
    assert not banned, f"Summary {idx}: banned fields found: {banned}"


# --- Tests ---

class TestMetricsRowContract:
    """Validate rows written by write_metrics_csv against the contract."""

    def test_row_required_fields_present(self) -> None:
        """Contract: each row has exactly the 11 required fields."""
        row = _make_row()
        missing = ROW_REQUIRED_FIELDS - set(row.keys())
        assert not missing

    def test_iou_range(self) -> None:
        """Contract: iou is number in [0, 1]."""
        for val in [0.0, 0.5, 1.0]:
            row = _make_row(iou=val)
            assert 0 <= row["iou"] <= 1

    def test_precision_range(self) -> None:
        """Contract: precision is number in [0, 1]."""
        row = _make_row(precision=0.0)
        assert 0 <= row["precision"] <= 1
        row = _make_row(precision=1.0)
        assert 0 <= row["precision"] <= 1

    def test_recall_range(self) -> None:
        """Contract: recall is number in [0, 1]."""
        row = _make_row(recall=0.0)
        assert 0 <= row["recall"] <= 1

    def test_f1_range(self) -> None:
        """Contract: f1 is number in [0, 1]."""
        row = _make_row(f1=1.0)
        assert 0 <= row["f1"] <= 1

    def test_tp_fp_fn_non_negative_int(self) -> None:
        """Contract: tp, fp, fn are integers >= 0."""
        row = _make_row(tp=0, fp=0, fn=0)
        for field in ("tp", "fp", "fn"):
            assert isinstance(row[field], int)
            assert row[field] >= 0

    def test_inference_time_non_negative(self) -> None:
        """Contract: inference_time_seconds >= 0."""
        row = _make_row(inference_time_seconds=0.0)
        assert row["inference_time_seconds"] >= 0

    def test_pixel_accuracy_absent(self) -> None:
        """Contract: no Pixel Accuracy field anywhere in rows."""
        row = _make_row()
        for banned in BANNED_FIELDS:
            assert banned not in row


class TestMetricsCSVContract:
    """Write metrics.csv via report module and validate the persisted file."""

    def test_csv_roundtrip(self, tmp_path: Path) -> None:
        """Written CSV has correct header and valid rows."""
        rows = [
            _make_row(manga="A", stem="001", iou=0.8, tp=20, fp=5, fn=3),
            _make_row(manga="B", stem="002", iou=0.3, tp=5, fp=10, fn=15),
        ]
        csv_path = tmp_path / "metrics.csv"
        write_metrics_csv(rows, csv_path)
        loaded = load_metrics_csv(csv_path)

        assert len(loaded) == 2
        for i, row in enumerate(loaded):
            _validate_row(row, i)

    def test_csv_header_matches_contract(self, tmp_path: Path) -> None:
        """CSV header columns match METRIC_COLUMNS from report.py."""
        csv_path = tmp_path / "metrics.csv"
        write_metrics_csv([_make_row()], csv_path)
        with open(csv_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames or []
        # METRIC_COLUMNS should contain the 11 required fields
        for field in ROW_REQUIRED_FIELDS:
            assert field in header, f"Required field '{field}' not in CSV header"

    def test_no_pixel_accuracy_in_csv(self, tmp_path: Path) -> None:
        """Contract: no Pixel Accuracy column in CSV."""
        csv_path = tmp_path / "metrics.csv"
        write_metrics_csv([_make_row()], csv_path)
        with open(csv_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            header = set(reader.fieldnames or [])
        for banned in BANNED_FIELDS:
            assert banned not in header


class TestSummariesContract:
    """Write summaries.json via report module and validate."""

    def test_summary_required_fields(self) -> None:
        """Contract: each summary has the 8 required fields."""
        summary = {
            "method": "otsu", "n_images": 10,
            "mean_iou": 0.5, "std_iou": 0.1,
            "mean_precision": 0.6, "mean_recall": 0.7,
            "mean_f1": 0.65, "mean_time_seconds": 0.12,
        }
        _validate_summary(summary, 0)

    def test_summaries_json_roundtrip(self, tmp_path: Path) -> None:
        """Written summaries.json has correct structure."""
        rows = [_make_row(method="otsu"), _make_row(method="otsu", iou=0.9)]
        summary_path = tmp_path / "summaries.json"
        write_summaries_json(rows, summary_path, run_id="test")

        with open(summary_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        assert "summaries" in data
        assert isinstance(data["summaries"], list)
        assert len(data["summaries"]) >= 1

        for i, s in enumerate(data["summaries"]):
            _validate_summary(s, i)

    def test_no_pixel_accuracy_in_summaries(self, tmp_path: Path) -> None:
        """Contract: no Pixel Accuracy field in summaries."""
        rows = [_make_row(method="otsu")]
        summary_path = tmp_path / "summaries.json"
        write_summaries_json(rows, summary_path, run_id="test")

        with open(summary_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        for s in data.get("summaries", []):
            for banned in BANNED_FIELDS:
                assert banned not in s


class TestReportModuleColumns:
    """Verify report.py METRIC_COLUMNS matches the contract."""

    def test_metric_columns_match_contract(self) -> None:
        """METRIC_COLUMNS contains exactly the 11 required row fields."""
        for field in ROW_REQUIRED_FIELDS:
            assert field in METRIC_COLUMNS, f"METRIC_COLUMNS missing '{field}'"

    def test_no_banned_columns(self) -> None:
        """METRIC_COLUMNS must not contain Pixel Accuracy."""
        metric_names_lower = {c.lower() for c in METRIC_COLUMNS}
        for banned in BANNED_FIELDS:
            assert banned not in metric_names_lower, (
                f"METRIC_COLUMNS contains banned field '{banned}'"
            )
