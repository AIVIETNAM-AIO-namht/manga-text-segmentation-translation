"""T034: performance aggregates are grouped by method and algorithm."""

import json

from manga_text_seg.runs import SampleCounts, write_performance_summary


def test_summary_reports_mean_samples_failures_and_runtime(tmp_path):
    path = write_performance_summary(
        tmp_path,
        {"classical_baseline": {"telea": [1.0, 3.0], "ns": [2.0]}},
        {"classical_baseline": {
            "telea": SampleCounts(3, 2, 1),
            "ns": SampleCounts(2, 1, 1),
        }},
        runtime={"device": {"type": "cpu", "name": "test"}},
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    telea = data["groups"]["classical_baseline"]["telea"]
    assert telea == {
        "mean_processing_time_seconds": 2.0,
        "sample_count": 2,
        "failure_count": 1,
    }
    assert data["groups"]["classical_baseline"]["ns"]["sample_count"] == 1
    assert data["runtime"]["device"]["type"] == "cpu"


def test_empty_success_group_still_reports_failures(tmp_path):
    path = write_performance_summary(
        tmp_path, {}, {"classical_baseline": {"telea": SampleCounts(1, 0, 1)}},
        runtime={},
    )
    group = json.loads(path.read_text())["groups"]["classical_baseline"]["telea"]
    assert group["mean_processing_time_seconds"] is None
    assert group["sample_count"] == 0
    assert group["failure_count"] == 1
