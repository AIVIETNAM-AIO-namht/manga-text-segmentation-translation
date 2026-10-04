"""Direct contract tests for T009–T012's run module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from manga_text_seg.translation.runs import (
    ERROR_CATEGORIES,
    ErrorReport,
    TranslationRun,
    create_run_directory,
    page_directory,
    run_root,
    write_run_json,
)


def _run() -> TranslationRun:
    inpainting_source = {"run_id": "inpaint-a", "algorithm": "telea"}
    region_config = {
        "min_area": 1,
        "max_area": 1000000,
        "merge_distance": 0,
        "crop_padding": 0,
        "reading_order": "manga",
        "polygon_epsilon": 1.0,
    }
    ocr_config = {
        "engine": "manga-ocr",
        "preprocessing": {"enabled": False, "grayscale": False, "upscale_factor": 1.0},
    }
    provider = {
        "name": "openai",
        "model": "model-example",
        "version": None,
        "deterministic": False,
        "secret_env_var": "MANGA_TRANSLATE_API_KEY",
    }
    config_provider = {**provider, "base_url": "https://api.example.test/v1"}
    rendering = {
        "font_family": "DejaVu Sans",
        "font_path": "assets/fonts/DejaVuSans.ttf",
        "fallback_font_path": None,
        "font_size": 32,
        "min_font_size": 12,
        "color": [0, 0, 0, 255],
        "stroke_width": 0,
        "stroke_fill": [255, 255, 255, 255],
        "background": None,
        "wrap": True,
        "line_spacing": 1.0,
        "align_h": "center",
        "align_v": "middle",
        "text_direction": "horizontal",
        "expansion_allowance": 0,
    }
    resolved_config = {
        "manifest_path": "outputs/segmentation/default/manifest.json",
        "segmentation_method": "classical_baseline",
        "inpainting_source": inpainting_source,
        "target_language": "vi",
        "region_extraction": region_config,
        "ocr": ocr_config,
        "provider": config_provider,
        "rendering": rendering,
        "output_roots": {
            "translation": "outputs/translation",
            "cache": "cache/translation",
        },
    }
    return TranslationRun(
        run_id="run-a",
        segmentation_method="classical_baseline",
        inpainting_source=inpainting_source,
        target_language="vi",
        ocr_engine={"name": "manga-ocr", "version": "1.0"},
        provider=provider,
        rendering=rendering,
        resolved_config=resolved_config,
        input_image_ids=("ARMS/000",),
        page_statuses={"ARMS/000": "accepted"},
        # Valid source digests are carried verbatim, without recomputation.
        page_list_identity="a" * 64,
        input_image_identity={"ARMS/000": "b" * 64},
        region_extraction_config=region_config,
        counts={"pages_attempted": 1, "pages_succeeded": 0, "pages_failed": 0},
    )


def test_run_and_page_addressing_uses_run_method_image_id(tmp_path: Path) -> None:
    assert run_root(tmp_path, "run-a") == tmp_path / "run-a"
    assert page_directory(tmp_path, "run-a", "classical_baseline", "ARMS/000") == (
        tmp_path / "run-a" / "classical_baseline" / "ARMS" / "000"
    )


def test_addressing_rejects_untrusted_path_parts(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_root(tmp_path, "../escape")
    with pytest.raises(ValueError):
        page_directory(tmp_path, "run-a", "classical_baseline", "../000")
    with pytest.raises(ValueError):
        page_directory(tmp_path, "run-a", "bad-method", "ARMS/000")


def test_nonoverwrite_and_overwrite_are_limited_to_target_run(tmp_path: Path) -> None:
    root = tmp_path / "outputs" / "translation"
    run_a = create_run_directory(root, "run-a")
    run_b = create_run_directory(root, "run-b")
    (run_a / "marker.txt").write_text("old", encoding="utf-8")
    (run_b / "marker.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        create_run_directory(root, "run-a")
    replaced = create_run_directory(root, "run-a", overwrite=True)
    assert replaced == run_a
    assert not (run_a / "marker.txt").exists()
    assert (run_b / "marker.txt").read_text(encoding="utf-8") == "keep"


def test_error_report_has_all_eleven_categories_even_at_zero(tmp_path: Path) -> None:
    report = ErrorReport("run-a")
    assert set(report.counts) == set(ERROR_CATEGORIES)
    assert len(report.counts) == 11
    assert set(report.to_dict()["categories"]) == set(ERROR_CATEGORIES)
    report.add(
        "ocr_failure",
        image_id="ARMS/000",
        region_id=1,
        stage="ocr",
        reason="fixture OCR failure",
    )
    path = report.write(tmp_path / "run-a")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["counts"]["ocr_failure"] == 1
    assert saved["categories"]["ocr_failure"][0] == {
        "image_id": "ARMS/000",
        "region_id": 1,
        "stage": "ocr",
        "reason": "fixture OCR failure",
    }
    assert all(
        saved["counts"][key] == 0 for key in ERROR_CATEGORIES if key != "ocr_failure"
    )


def test_error_report_rejects_invalid_category_stage_and_entry_fields() -> None:
    report = ErrorReport("run-a")
    with pytest.raises(ValueError, match="unknown error category"):
        report.add(
            "invented", image_id="ARMS/000", region_id=None, stage="intake", reason="x"
        )
    with pytest.raises(ValueError, match="stage"):
        report.add(
            "ocr_failure",
            image_id="ARMS/000",
            region_id=1,
            stage="segmentation",
            reason="x",
        )
    with pytest.raises(ValueError, match="reason"):
        report.add(
            "ocr_failure", image_id="ARMS/000", region_id=1, stage="ocr", reason=" "
        )


def test_translation_run_serializes_identities_verbatim(tmp_path: Path) -> None:
    record = _run()
    path = write_run_json(tmp_path / "run-a", record)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["page_list_identity"] == "a" * 64
    assert saved["input_image_identity"] == {"ARMS/000": "b" * 64}
    assert saved["run_id"] == "run-a"
    assert saved["translation_provider"]["deterministic"] is False
    assert saved["region_extraction_config"]["reading_order"] == "manga"
    assert saved["page_statuses"] == {"ARMS/000": "accepted"}


def test_translation_run_rejects_missing_or_malformed_identity_values() -> None:
    fields = _run().__dict__
    with pytest.raises(ValueError, match="Spec 2 sha256"):
        TranslationRun(**{**fields, "page_list_identity": "unavailable"})
    with pytest.raises(ValueError, match="Spec 2 sha256"):
        TranslationRun(
            **{
                **fields,
                "input_image_identity": {"ARMS/000": "not-an-identity"},
            }
        )


def test_run_writer_requires_validated_translation_run(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="TranslationRun"):
        write_run_json(tmp_path / "run-a", {"run_id": "run-a"})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="directory name"):
        write_run_json(tmp_path / "other", _run())


def test_translation_run_rejects_missing_identity_and_invalid_provenance() -> None:
    with pytest.raises(ValueError, match="page_list_identity"):
        TranslationRun(
            **{**_run().__dict__, "page_list_identity": ""},
        )
    with pytest.raises(ValueError, match="inpainting_source.algorithm"):
        TranslationRun(
            **{
                **_run().__dict__,
                "inpainting_source": {"run_id": "inpaint-a", "algorithm": "fake"},
            },
        )
