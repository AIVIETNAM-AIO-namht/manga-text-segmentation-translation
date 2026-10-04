"""T041 secret-value audit across Spec 004 translation artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from manga_text_seg.config import load_translation_config
from manga_text_seg.translation.runs import (
    ErrorReport,
    TranslationRun,
    write_run_json,
)
from manga_text_seg.translation.translate import (
    MissingProviderSecretError,
    TranslationRequest,
    TranslationService,
    write_translations_json,
)

SECRET_ENV = "SPEC004_TRANSLATION_TEST_SECRET"
SECRET_VALUE = "unit-test-sentinel-secret-value-do-not-log"


class SecretEchoFailure:
    deterministic = False

    def __init__(self):
        self.calls = []

    def translate(self, text, target_language):
        self.calls.append((text, target_language))
        if text == "safe source":
            return "safe translation"
        raise RuntimeError(f"provider transport error: {SECRET_VALUE}")


def _config(tmp_path):
    path = Path(__file__).resolve().parents[3] / "configs" / "translation.json"
    base = load_translation_config(path)
    provider = replace(
        base.provider,
        name="fixture-provider",
        model="fixture-model",
        version="fixture-v1",
        secret_env_var=SECRET_ENV,
        cache=replace(base.provider.cache, root=tmp_path / "cache" / "translation"),
        retry=replace(base.provider.retry, max_attempts=1),
    )
    return replace(base, provider=provider)


def _run_record(config, service) -> TranslationRun:
    provider = service.provenance
    rendering = {
        "font_family": config.rendering.font_family,
        "font_path": str(config.rendering.font_path),
        "fallback_font_path": (
            str(config.rendering.fallback_font_path)
            if config.rendering.fallback_font_path
            else None
        ),
        "font_size": config.rendering.font_size,
        "min_font_size": config.rendering.min_font_size,
        "color": list(config.rendering.color),
        "stroke_width": config.rendering.stroke_width,
        "stroke_fill": list(config.rendering.stroke_fill),
        "background": config.rendering.background,
        "wrap": config.rendering.wrap,
        "line_spacing": config.rendering.line_spacing,
        "align_h": config.rendering.align_h,
        "align_v": config.rendering.align_v,
        "text_direction": config.rendering.text_direction,
        "expansion_allowance": config.rendering.expansion_allowance,
    }
    provider_config = {
        **provider,
        "base_url": config.provider.base_url,
        "timeout_seconds": config.provider.timeout_seconds,
        "secret_env_var": SECRET_ENV,
        "retry": asdict(config.provider.retry),
        "failure_policy": config.provider.failure_policy,
        "cache": {
            "enabled": config.provider.cache.enabled,
            "bypass": config.provider.cache.bypass,
            "root": str(config.provider.cache.root),
        },
    }
    region = asdict(config.region_extraction)
    ocr = {
        "engine": config.ocr.engine,
        "preprocessing": {
            "enabled": config.ocr.preprocessing_enabled,
            "grayscale": config.ocr.grayscale,
            "upscale_factor": config.ocr.upscale_factor,
        },
    }
    resolved = {
        "manifest_path": str(config.manifest_path),
        "segmentation_method": config.segmentation_method,
        "inpainting_source": {
            "run_id": config.inpainting_run_id,
            "algorithm": config.inpainting_algorithm,
        },
        "target_language": config.target_language,
        "region_extraction": region,
        "ocr": ocr,
        "provider": provider_config,
        "rendering": rendering,
        "output_roots": {
            "translation": str(config.output_roots.translation),
            "cache": str(config.output_roots.cache),
        },
    }
    return TranslationRun(
        run_id="secret-audit",
        segmentation_method=config.segmentation_method,
        inpainting_source={
            "run_id": config.inpainting_run_id,
            "algorithm": config.inpainting_algorithm,
        },
        target_language=config.target_language,
        ocr_engine={"name": config.ocr.engine, "version": None},
        provider=provider,
        rendering=rendering,
        resolved_config=resolved,
        input_image_ids=("ARMS/000",),
        page_statuses={"ARMS/000": "accepted"},
        page_list_identity="a" * 64,
        input_image_identity={"ARMS/000": "b" * 64},
        region_extraction_config=region,
        counts={"pages_attempted": 1, "pages_succeeded": 1, "pages_failed": 0},
    )


def test_secret_value_is_absent_from_all_translation_outputs(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setenv(SECRET_ENV, SECRET_VALUE)
    config = _config(tmp_path)
    provider = SecretEchoFailure()
    errors = ErrorReport(run_id="secret-audit")
    service = TranslationService(config, provider, error_report=errors)
    results = service.translate_regions(
        [
            TranslationRequest(1, "safe source"),
            TranslationRequest(2, "provider failure"),
        ],
        image_id="ARMS/000",
    )
    run_dir = tmp_path / "secret-audit"
    run_dir.mkdir()
    write_run_json(run_dir, _run_record(config, service))
    errors.write(run_dir)
    write_translations_json(run_dir / "ARMS" / "000" / "translations.json", results)
    page_dir = run_dir / "ARMS" / "000"
    (page_dir / "ocr.json").write_text(
        json.dumps({"results": [{"region_id": 1, "text": "safe OCR text"}]}),
        encoding="utf-8",
    )
    (page_dir / "metadata.json").write_text(
        json.dumps(
            {
                "translation_provider": service.provenance,
                "regions": [asdict(result) for result in results],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    artifacts = [path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()]
    assert provider.calls == [
        ("safe source", config.target_language),
        ("provider failure", config.target_language),
    ]
    assert results[1].error == "provider transport error: [REDACTED]"
    assert SECRET_VALUE.encode() not in b"\n".join(artifacts)
    assert SECRET_VALUE not in caplog.text


def test_missing_configured_secret_fails_before_provider_request(tmp_path, monkeypatch):
    monkeypatch.delenv(SECRET_ENV, raising=False)
    provider = SecretEchoFailure()

    with pytest.raises(MissingProviderSecretError, match=SECRET_ENV):
        TranslationService(_config(tmp_path), provider)

    assert provider.calls == []
