"""T030–T038 fixture-only OCR behavior; no OCR engine or model is required."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from manga_text_seg.config import load_translation_config
from manga_text_seg.translation.ocr import (
    OcrAdapter,
    OcrEngineUnavailable,
    apply_manual_edit,
    load_ocr_results,
)
from manga_text_seg.translation.regions import TextRegion, extract_regions
from manga_text_seg.translation.runs import ErrorReport


@pytest.fixture()
def translation_config():
    path = Path(__file__).resolve().parents[3] / "configs" / "translation.json"
    return load_translation_config(path)


@pytest.fixture()
def image_id() -> str:
    path = Path(__file__).resolve().parents[3] / "benchmark" / "page-list.json"
    return json.loads(path.read_text(encoding="utf-8"))["pages"][0]["image_id"]


@pytest.fixture()
def aligned_pages():
    original = np.zeros((1170, 1654, 3), dtype=np.uint8)
    original[:] = (11, 22, 33)
    original[80:100, 80:110] = (40, 40, 40)
    original[30:50, 20:50] = (80, 80, 80)
    original[30:50, 120:150] = (50, 50, 50)
    inpainted = np.zeros_like(original)
    inpainted[:] = (201, 202, 203)
    return original, inpainted


def _regions(config, *, shifted: bool = False) -> tuple[TextRegion, ...]:
    mask = np.zeros((1170, 1654), dtype=np.uint8)
    x = 120 if shifted else 20
    mask[30:50, x : x + 30] = 255
    mask[80:100, 80:110] = 255
    region_config = replace(
        config.region_extraction,
        min_area=1,
        max_area=10_000,
        merge_distance=0,
        crop_padding=3,
    )
    return extract_regions(mask, region_config).regions


def _stub(text_by_region, confidence_by_region=None, calls=None):
    confidence_by_region = confidence_by_region or {}

    def recognize(image: Image.Image):
        if calls is not None:
            calls.append(image)
        marker = image.getpixel((3, 3))
        marker = marker if isinstance(marker, int) else marker[0]
        region_id = 1 if marker in {40, 50} else 2
        result = text_by_region[region_id]
        if isinstance(result, Exception):
            raise result
        confidence = confidence_by_region.get(region_id)
        return result if confidence is None else (result, confidence)

    return recognize


def test_stub_writes_one_complete_ocr_record_per_region(
    tmp_path, translation_config, aligned_pages
):
    original, _ = aligned_pages
    regions = _regions(translation_config)
    adapter = OcrAdapter(
        _stub({1: "日本語", 2: "テキスト"}, {1: 0.91}),
        engine="stub-ocr",
        engine_version="stub-1",
    )

    run = adapter.run(original, regions, tmp_path, config=translation_config.ocr)

    assert run.path == tmp_path / "ocr.json"
    assert run.path.is_file()
    assert [record.region_id for record in run.results] == [1, 2]
    assert run.results[0].crop_path == "crops/1.png"
    assert run.results[0].text == "日本語"
    assert run.results[0].confidence == pytest.approx(0.91)
    assert run.results[1].confidence is None
    assert all(record.processing_time_seconds >= 0 for record in run.results)
    assert [record.status for record in run.results] == ["ok", "ok"]
    document = json.loads(run.path.read_text(encoding="utf-8"))
    assert len(document["results"]) == 2
    assert document["engine"] == "stub-ocr"
    assert document["engine_version"] == "stub-1"


def test_crop_is_taken_from_original_pre_inpaint_image(
    tmp_path, translation_config, aligned_pages
):
    original, inpainted = aligned_pages
    regions = _regions(translation_config)
    run = OcrAdapter(_stub({1: "one", 2: "two"}), engine="fixture").run(
        original, regions, tmp_path, config=translation_config.ocr
    )

    saved = cv2.imread(str(tmp_path / run.results[0].crop_path), cv2.IMREAD_COLOR)
    assert saved is not None
    assert tuple(saved[3, 3]) == tuple(original[80, 80])
    assert not np.array_equal(saved[3, 3], inpainted[80, 80])


def test_region_failure_isolated_and_reported(
    tmp_path, translation_config, aligned_pages, image_id
):
    original, _ = aligned_pages
    report = ErrorReport(run_id="ocr-isolation")
    adapter = OcrAdapter(
        _stub({1: "recognized", 2: RuntimeError("fixture failure")}),
        engine="fixture",
        error_report=report,
    )

    run = adapter.run(
        original,
        _regions(translation_config),
        tmp_path,
        config=translation_config.ocr,
        image_id=image_id,
    )

    assert [record.status for record in run.results] == ["ok", "failed"]
    assert run.results[0].text == "recognized"
    assert run.results[1].text == ""
    assert "fixture failure" in run.results[1].error
    assert report.counts["ocr_failure"] == 1
    assert report.categories["ocr_failure"][0]["region_id"] == 2


def test_manual_edit_is_honored_by_next_ocr_run_and_loader(
    tmp_path, translation_config, aligned_pages
):
    original, _ = aligned_pages
    regions = _regions(translation_config)
    first = OcrAdapter(_stub({1: "raw one", 2: "raw two"}), engine="fixture")
    first.run(original, regions, tmp_path, config=translation_config.ocr)
    ocr_path = tmp_path / "ocr.json"
    document = json.loads(ocr_path.read_text(encoding="utf-8"))
    document["results"][0]["text"] = "edited text"  # emulate a human editing ocr.json
    ocr_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    loaded_before_rerun = load_ocr_results(ocr_path)

    second = OcrAdapter(_stub({1: "replacement", 2: "new two"}), engine="fixture")
    run = second.run(original, regions, tmp_path, config=translation_config.ocr)
    loaded = load_ocr_results(ocr_path)

    assert loaded_before_rerun[0].manually_edited is True
    assert loaded_before_rerun[0].text == "edited text"
    assert run.results[0].text == "edited text"
    assert run.results[0].manually_edited is True
    assert run.results[1].text == "new two"
    assert loaded[0].text == "edited text"
    assert loaded[0].manually_edited is True
    assert (
        json.loads(ocr_path.read_text(encoding="utf-8"))["manual_edits"][0][
            "original_text"
        ]
        == "raw one"
    )


def test_changed_region_set_invalidates_and_reports_manual_edit(
    tmp_path, translation_config, aligned_pages
):
    original, _ = aligned_pages
    regions = _regions(translation_config)
    OcrAdapter(_stub({1: "one", 2: "two"}), engine="fixture").run(
        original, regions, tmp_path, config=translation_config.ocr
    )
    changed_region_id = next(
        region.region_id for region in regions if region.bbox[1] < 60
    )
    unchanged_region_id = next(
        region.region_id for region in regions if region.bbox[1] >= 60
    )
    apply_manual_edit(tmp_path / "ocr.json", changed_region_id, "manual")
    apply_manual_edit(tmp_path / "ocr.json", unchanged_region_id, "unchanged manual")
    changed_mask = np.zeros((1170, 1654), dtype=np.uint8)
    changed_mask[30:50, 25:55] = 255  # region 1 geometry changes
    changed_mask[80:100, 80:110] = 255  # region 2 remains identical
    changed_regions = extract_regions(
        changed_mask,
        replace(
            translation_config.region_extraction,
            min_area=1,
            max_area=10_000,
            merge_distance=0,
            crop_padding=3,
        ),
    ).regions

    rerun = OcrAdapter(
        _stub({1: "fresh recognition", 2: "fresh recognition"}), engine="fixture"
    ).run(original, changed_regions, tmp_path, config=translation_config.ocr)
    document = json.loads((tmp_path / "ocr.json").read_text(encoding="utf-8"))

    by_id = {result.region_id: result for result in rerun.results}
    assert by_id[changed_region_id].text == "fresh recognition"
    assert by_id[changed_region_id].manually_edited is False
    assert by_id[unchanged_region_id].text == "unchanged manual"
    assert by_id[unchanged_region_id].manually_edited is True
    invalidated = next(
        edit for edit in rerun.manual_edits if edit.region_id == changed_region_id
    )
    preserved = next(
        edit for edit in rerun.manual_edits if edit.region_id == unchanged_region_id
    )
    assert invalidated.invalidated_by_reextraction is True
    assert preserved.invalidated_by_reextraction is False
    assert rerun.invalidation_reports == (
        {
            "region_id": changed_region_id,
            "reason": "region set changed during re-extraction",
        },
    )
    invalidated_doc_edit = next(
        edit
        for edit in document["manual_edits"]
        if edit["region_id"] == changed_region_id
    )
    assert invalidated_doc_edit["invalidated_by_reextraction"] is True
    assert document["invalidation_reports"]


def test_optional_preprocessing_is_applied_and_recorded(
    tmp_path, translation_config, aligned_pages
):
    original, _ = aligned_pages
    observed = []
    config = replace(
        translation_config.ocr,
        preprocessing_enabled=True,
        grayscale=True,
        upscale_factor=2.0,
    )
    run = OcrAdapter(_stub({1: "one", 2: "two"}, calls=observed), engine="fixture").run(
        original, _regions(translation_config), tmp_path, config=config
    )

    assert all(image.mode == "L" for image in observed)
    assert observed[0].size == (72, 52)
    assert run.preprocessing == {
        "enabled": True,
        "grayscale": True,
        "upscale_factor": 2.0,
    }
    assert all(record.preprocessing == run.preprocessing for record in run.results)


def test_empty_text_is_distinct_from_engine_failure(
    tmp_path, translation_config, aligned_pages
):
    original, _ = aligned_pages
    run = OcrAdapter(_stub({1: "", 2: "usable"}), engine="fixture").run(
        original, _regions(translation_config), tmp_path, config=translation_config.ocr
    )

    assert run.results[0].status == "empty"
    assert run.results[0].text == ""
    assert run.results[0].error
    assert run.results[1].status == "ok"


def test_lazy_engine_resolution_and_missing_engine_are_recorded(
    tmp_path, translation_config, aligned_pages, image_id, monkeypatch
):
    import manga_text_seg.translation.ocr as ocr_module

    calls = []

    def missing_engine():
        calls.append("load")
        raise OcrEngineUnavailable("manga-ocr is not installed")

    monkeypatch.setattr(ocr_module, "_load_manga_ocr", missing_engine)
    adapter = OcrAdapter(error_report=ErrorReport(run_id="missing-ocr"))
    assert calls == []  # constructing/importing the adapter does not load manga-ocr

    original, _ = aligned_pages
    run = adapter.run(
        original,
        _regions(translation_config),
        tmp_path,
        config=translation_config.ocr,
        image_id=image_id,
    )

    assert calls == ["load"]
    assert [record.status for record in run.results] == ["failed", "failed"]
    assert all(
        record.text == "" and record.confidence is None for record in run.results
    )
    assert all("manga-ocr is not installed" in record.error for record in run.results)
    assert adapter.error_report.counts["ocr_failure"] == 2
