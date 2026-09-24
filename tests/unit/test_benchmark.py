"""The combined comparison: the classical baseline plus every admitted method.

T034 (US3 scenarios 1 and 2; FR-034, FR-037, FR-059).

The comparison is assembled from results that already exist — Spec 1's
``metrics.csv`` for the classical baseline and each admitted method's
``metrics/per-page.csv`` — so nothing here re-runs inference, and admitting a
later result cannot move an earlier one.  US3 scenario 2 is exactly that
promise, and it is why the third admission in the last test is compared byte
for byte rather than by value: FR-059 says the earlier results are untouched,
not that they are recomputed to the same numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from manga_text_seg.benchmark import assemble_comparison, write_comparison
from manga_text_seg.imaging import load_raw_page
from manga_text_seg.receipt import admit
from manga_text_seg.report import write_metrics_csv

#: The four accuracy metrics every row carries, mean and standard deviation
#: alike (FR-037).  Named once so the test says what it is checking.
ACCURACY_METRICS = ("iou", "precision", "recall", "f1")

#: The methods the fixture baseline contributes, with the IoU each page scores.
BASELINE_METHODS = (("otsu", 0.10), ("mser", 0.40))


def _baseline(tmp_path: Path) -> Path:
    """A two-method classical baseline, in the shape Spec 1's sweep writes."""
    rows = [
        {
            "manga": "ARMS",
            "stem": f"{index:03d}",
            "method": method,
            # Symmetric per-page spread, so the mean stays exactly ``iou`` while
            # the std is genuinely non-zero — a constant fixture could not tell a
            # computed std from a hardcoded zero.
            "iou": iou + 0.02 * (2 * index - 1),
            "precision": iou + 0.10,
            "recall": iou + 0.20,
            "f1": iou + 0.15,
            "tp": 10 + index,
            "fp": 5,
            "fn": 3,
            "inference_time_seconds": 0.01 * (index + 1),
        }
        for method, iou in BASELINE_METHODS
        for index in range(2)
    ]
    path = tmp_path / "classical" / "metrics.csv"
    write_metrics_csv(rows, path)
    return path


def _masks(manifest) -> dict:
    """A deterministic binary mask per page: no adapter, no checkpoint (FR-050)."""
    return {
        pair.image_id: (load_raw_page(pair.raw_path) > 127).astype("uint8") * 255
        for pair in manifest.pairs
    }


def _timed(image_id: str, doc: dict) -> dict:
    """A per-page inference time, which the hand-off layout itself does not carry."""
    doc["inference_time_seconds"] = 2.5
    return doc


def _provenance_for(default_provenance: dict, method: str) -> dict:
    record = json.loads(json.dumps(default_provenance))
    record["method"] = method
    return record


def _admit(exported, make_handoff, default_provenance, run_dir, method, masks) -> list:
    """Admit one separately constructed hand-off, as a runner's return would be."""
    return admit(
        make_handoff(
            exported["page_list"],
            masks,
            method=method,
            provenance=_provenance_for(default_provenance, method),
            sidecar=_timed,
        ),
        page_list=exported["page_list"],
        identity_record=exported["identity_record"],
        run_dir=run_dir,
        manifest=exported["manifest"],
    )


def _rows(comparison: dict) -> dict:
    return {row["method"]: row for row in comparison["rows"]}


def test_one_row_per_method_with_mean_std_and_timing(
    exported, make_handoff, default_provenance, tmp_path
):
    """US3 scenario 1: every method, all four metrics, mean, std and timing."""
    run_dir = tmp_path / "run"
    masks = _masks(exported["manifest"])
    for method in ("standin-a", "standin-b"):
        _admit(exported, make_handoff, default_provenance, run_dir, method, masks)

    comparison = assemble_comparison(
        baseline_metrics=_baseline(tmp_path), run_dir=run_dir
    )

    assert comparison["run_id"] == run_dir.name
    assert comparison["page_list_identity"] == exported["page_list"]["page_list_identity"]

    rows = _rows(comparison)
    # The baseline's two methods and the two admitted ones are all scored; the
    # methods the run still awaits are rows too, but they are scenario 2's
    # business and carry no numbers.
    scored = {name: row for name, row in rows.items() if row["status"] == "scored"}
    assert sorted(scored) == ["mser", "otsu", "standin-a", "standin-b"]
    for row in scored.values():
        for metric in ACCURACY_METRICS:
            cell = row["metrics"][metric]
            assert 0.0 <= cell["mean"] <= 1.0
            assert cell["std"] >= 0.0
        assert row["timing"]["mean_seconds"] >= 0.0
        assert row["timing"]["total_seconds"] >= row["timing"]["mean_seconds"]
        assert row["disclosure"]

    # The baseline's own numbers survive the composition unchanged.
    assert rows["otsu"]["metrics"]["iou"]["mean"] == pytest.approx(0.10)
    assert rows["mser"]["metrics"]["iou"]["mean"] == pytest.approx(0.40)
    assert rows["otsu"]["metrics"]["iou"]["std"] > 0.0


def test_an_outstanding_method_is_reported_as_not_run(
    exported, make_handoff, default_provenance, tmp_path
):
    """US3 scenario 2: a run with a method still outstanding is reportable.

    FR-059 forbids deferring the table until everyone returns, and FR-036
    forbids filling the gap with a number: the row is present, says it did not
    run, and carries no metric or timing value at all.
    """
    run_dir = tmp_path / "run"
    masks = _masks(exported["manifest"])
    _admit(exported, make_handoff, default_provenance, run_dir, "standin-a", masks)

    comparison = assemble_comparison(
        baseline_metrics=_baseline(tmp_path), run_dir=run_dir
    )

    rows = _rows(comparison)
    assert rows["standin-a"]["status"] == "scored"

    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["methods_awaited"], "the fixture run must still await something"
    for method in record["methods_awaited"]:
        row = rows[method]
        assert row["status"] == "not_run"
        assert row["reason"]
        assert row["metrics"] is None
        assert row["timing"] is None


def test_a_later_admission_leaves_the_earlier_rows_untouched(
    exported, make_handoff, default_provenance, tmp_path
):
    """FR-059: a result arriving later must not disturb the ones already in."""
    run_dir = tmp_path / "run"
    masks = _masks(exported["manifest"])
    for method in ("standin-a", "standin-b"):
        _admit(exported, make_handoff, default_provenance, run_dir, method, masks)

    before = assemble_comparison(baseline_metrics=_baseline(tmp_path), run_dir=run_dir)
    admitted_masks = run_dir / "standin-a" / "masks"
    snapshot = {p: p.read_bytes() for p in sorted(admitted_masks.rglob("*.png"))}

    _admit(exported, make_handoff, default_provenance, run_dir, "standin-c", masks)

    after = assemble_comparison(baseline_metrics=_baseline(tmp_path), run_dir=run_dir)

    assert {p: p.read_bytes() for p in sorted(admitted_masks.rglob("*.png"))} == snapshot

    before_rows, after_rows = _rows(before), _rows(after)
    for method in ("otsu", "mser", "standin-a", "standin-b"):
        assert after_rows[method] == before_rows[method]
    assert after_rows["standin-c"]["status"] == "scored"


def test_the_comparison_round_trips_through_its_own_file(
    exported, make_handoff, default_provenance, tmp_path
):
    """The run persists one comparison; reading it back yields the same table."""
    run_dir = tmp_path / "run"
    masks = _masks(exported["manifest"])
    _admit(exported, make_handoff, default_provenance, run_dir, "standin-a", masks)

    comparison = assemble_comparison(
        baseline_metrics=_baseline(tmp_path), run_dir=run_dir
    )
    path = write_comparison(comparison, run_dir / "comparison.json")

    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == comparison
