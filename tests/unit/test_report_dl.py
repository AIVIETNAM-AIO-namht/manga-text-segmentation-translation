"""The cross-method report: one table and one chart set.

T035 (US3 scenarios 4, 5, 6 and 7; FR-036, FR-038, FR-060, SC-011).

Everything here is asserted against the written artefacts rather than against
the objects that produced them, because every one of these requirements is a
property of what a reader sees: a timing figure without its device, a method
ordered by its own speed, a score with no contamination disclosure, or a Pixel
Accuracy column.  A report that got the objects right and the table wrong is
exactly the failure these four scenarios describe.

The fixture run is deliberately awkward in two ways.  Its two stand-ins sit on
differently named devices, and the alphabetically first of them is the slower —
so a table that quietly sorts by time, or that labels every bar with one
device, fails visibly instead of passing by coincidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from manga_text_seg.benchmark import assemble_comparison
from manga_text_seg.imaging import load_raw_page
from manga_text_seg.receipt import admit
from manga_text_seg.report import (
    write_comparison_charts,
    write_comparison_csv,
    write_metrics_csv,
)

#: The four accuracy metrics, and nothing else (FR-027: the metric set is
#: frozen; FR-035: no OCR, translation or inpainting measure ranks a method).
ACCURACY_METRICS = ("iou", "precision", "recall", "f1")

#: Every column the one summary table must carry (FR-037, SC-011).
REQUIRED_COLUMNS = (
    "method",
    "status",
    "iou_mean",
    "iou_std",
    "precision_mean",
    "precision_std",
    "recall_mean",
    "recall_std",
    "f1_mean",
    "f1_std",
    "timing_mean_seconds",
    "timing_total_seconds",
    "device_name",
    "disclosure",
    "reason",
)

#: The same ban Spec 1's metrics contract enforces, so the two cannot drift.
BANNED_FIELDS = {"pixel_accuracy", "pixel_accuracy_mean", "accuracy"}

#: Two producing devices, named (FR-060: a bare ``"cpu"`` is not a name).
DEVICES = {
    "standin-a": {"type": "cuda", "name": "NVIDIA GeForce RTX 4090"},
    "standin-b": {"type": "cpu", "name": "12th Gen Intel(R) Core(TM) i5-12500H"},
}

#: Seconds per page.  ``standin-a`` sorts first and is the slower of the two,
#: so method order and time order disagree — which is the point.
SECONDS = {"standin-a": 9.0, "standin-b": 0.5}


def _baseline(tmp_path: Path) -> Path:
    """A two-method classical baseline, in the shape Spec 1's sweep writes."""
    rows = [
        {
            "manga": "ARMS",
            "stem": f"{index:03d}",
            "method": method,
            "iou": iou,
            "precision": iou + 0.10,
            "recall": iou + 0.20,
            "f1": iou + 0.15,
            "tp": 10 + index,
            "fp": 5,
            "fn": 3,
            "inference_time_seconds": 0.01 * (index + 1),
        }
        for method, iou in (("otsu", 0.10), ("mser", 0.40))
        for index in range(2)
    ]
    path = tmp_path / "classical" / "metrics.csv"
    write_metrics_csv(rows, path)
    return path


def _provenance(default_provenance: dict, method: str) -> dict:
    record = json.loads(json.dumps(default_provenance))
    record["method"] = method
    record["device"] = DEVICES[method]
    return record


def _timed(method: str):
    """A per-page inference time, which the hand-off layout does not carry."""

    def hook(image_id: str, doc: dict) -> dict:
        doc["inference_time_seconds"] = SECONDS[method]
        return doc

    return hook


@pytest.fixture()
def report(exported, make_handoff, default_provenance, tmp_path) -> dict:
    """A run with two stand-ins admitted, plus its assembled comparison."""
    run_dir = tmp_path / "run"
    masks = {
        pair.image_id: (load_raw_page(pair.raw_path) > 127).astype("uint8") * 255
        for pair in exported["manifest"].pairs
    }
    for method in ("standin-a", "standin-b"):
        admit(
            make_handoff(
                exported["page_list"],
                masks,
                method=method,
                provenance=_provenance(default_provenance, method),
                sidecar=_timed(method),
            ),
            page_list=exported["page_list"],
            identity_record=exported["identity_record"],
            run_dir=run_dir,
            manifest=exported["manifest"],
        )
    comparison = assemble_comparison(
        baseline_metrics=_baseline(tmp_path), run_dir=run_dir
    )
    return {"comparison": comparison, "run_dir": run_dir}


def _table(report: dict, tmp_path: Path) -> pd.DataFrame:
    """The one summary table, as a reader receives it."""
    path = write_comparison_csv(report["comparison"], tmp_path / "comparison.csv")
    return pd.read_csv(path).set_index("method")


def test_the_table_carries_every_required_column(report, tmp_path):
    """FR-037: one table, four metrics with mean and std, time, device, status."""
    path = write_comparison_csv(report["comparison"], tmp_path / "comparison.csv")
    header = set(pd.read_csv(path).columns)
    missing = sorted(set(REQUIRED_COLUMNS) - header)
    assert not missing, f"the summary table is missing {missing}"


def test_every_timing_figure_carries_its_producing_device(report, tmp_path):
    """FR-060: a timing figure that does not say what produced it is unusable."""
    table = _table(report, tmp_path)
    scored = table[table["status"] == "scored"]
    assert set(scored.index) == {"otsu", "mser", "standin-a", "standin-b"}

    for method, row in scored.iterrows():
        name = row["device_name"]
        assert isinstance(name, str) and name.strip(), f"{method} has no device"
        assert name.strip().lower() != "cpu", f"{method} names a type, not a device"

    # Two devices in one run: each keeps its own name.
    assert (
        scored.loc["standin-a", "device_name"]
        != scored.loc["standin-b", "device_name"]
    )
    assert "RTX 4090" in scored.loc["standin-a", "device_name"]


def test_no_cross_device_timing_ranking_is_presented(report, tmp_path):
    """FR-037/FR-060: timing is reported per device, never ordered across them."""
    table = _table(report, tmp_path)

    # The two stand-ins disagree on speed and on sort order, so a table that
    # ranked by time would reorder them and this would catch it.
    assert (
        table.loc["standin-a", "timing_mean_seconds"]
        > table.loc["standin-b", "timing_mean_seconds"]
    )
    assert list(table.index) == sorted(table.index)

    assert not [c for c in table.columns if "rank" in c.lower()]


def test_an_outstanding_method_is_named_with_a_reason_and_no_number(
    report, tmp_path
):
    """FR-036: present, explained, and carrying no placeholder value at all."""
    table = _table(report, tmp_path)
    record = json.loads(
        (report["run_dir"] / "run.json").read_text(encoding="utf-8")
    )
    assert record["methods_awaited"], "the fixture run must still await something"

    not_run = table[table["status"] == "not_run"]
    assert set(not_run.index) == set(record["methods_awaited"])

    for method, row in not_run.iterrows():
        assert isinstance(row["reason"], str) and row["reason"].strip()
        for metric in ACCURACY_METRICS:
            assert pd.isna(row[f"{metric}_mean"]), f"{method} invented {metric}"
            assert pd.isna(row[f"{metric}_std"]), f"{method} invented {metric} std"
        assert pd.isna(row["timing_mean_seconds"])
        assert pd.isna(row["timing_total_seconds"])


def test_the_contamination_disclosure_sits_beside_every_score(report, tmp_path):
    """SC-011: no method's scores appear without their contamination status."""
    table = _table(report, tmp_path)
    scored = table[table["status"] == "scored"]
    assert not scored.empty

    for method, row in scored.iterrows():
        disclosure = row["disclosure"]
        assert isinstance(disclosure, str) and disclosure.strip(), (
            f"{method} reports scores with no contamination disclosure"
        )

    # Spec 1's baseline is the one method with nothing to disclose.
    assert "overlap" in scored.loc["otsu", "disclosure"].lower()


def test_pixel_accuracy_is_not_a_ranking_metric(report, tmp_path):
    """US3 scenario 7: the frozen metric set, in the table and in the rows."""
    path = write_comparison_csv(report["comparison"], tmp_path / "comparison.csv")
    lowered = {column.lower() for column in pd.read_csv(path).columns}
    assert not (lowered & BANNED_FIELDS)

    for row in report["comparison"]["rows"]:
        if row["metrics"] is not None:
            assert set(row["metrics"]) == set(ACCURACY_METRICS)


def test_the_accuracy_chart_set_is_written(report, tmp_path):
    """FR-038: a chart comparing IoU, Precision, Recall and F1 across methods."""
    paths = write_comparison_charts(report["comparison"], tmp_path / "charts")

    assert paths, "no charts were written"
    for path in paths:
        path = Path(path)
        assert path.is_file()
        assert path.stat().st_size > 1024, f"{path} is not a rendered chart"
