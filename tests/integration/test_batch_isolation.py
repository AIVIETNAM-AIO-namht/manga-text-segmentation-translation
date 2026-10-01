"""T032: individual sample failures are recorded without stopping a batch."""

import pytest

from manga_text_seg.inpaint import process_batch
from manga_text_seg.runs import ERROR_CATEGORIES, MaskRejection


@pytest.mark.parametrize("category", ERROR_CATEGORIES)
def test_each_rejection_category_is_isolated(category, monkeypatch, inpaint_config):
    config = inpaint_config({"classical_baseline": "/unused"})
    pages = [("ARMS", f"{i:03}") for i in range(10)]

    def fake_process(manifest, manga, stem, **kwargs):
        if stem == "004":
            raise MaskRejection(category, f"induced {category}", f"{manga}/{stem}")
        return {
            "telea": type("Metadata", (), {"processing_time_seconds": 0.01})(),
            "ns": type("Metadata", (), {"processing_time_seconds": 0.02})(),
        }

    monkeypatch.setattr("manga_text_seg.inpaint.process_sample", fake_process)
    result = process_batch(
        object(), pages, methods=("classical_baseline",), run_id="batch",
        config=config, origins={"classical_baseline": ("exp", None)},
    )
    assert result["errors"].to_dict()["total"] == 1
    item = result["errors"].to_dict()["by_category"][category]["entries"][0]
    assert item["image_id"] == "ARMS/004"
    assert "induced" in item["reason"]
    assert result["counts"]["classical_baseline"]["telea"].succeeded == 9
    assert result["counts"]["classical_baseline"]["telea"].failed == 1
