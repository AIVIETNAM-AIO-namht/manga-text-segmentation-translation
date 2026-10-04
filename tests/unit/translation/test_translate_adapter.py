"""T039 provider-adapter tests using deterministic in-process fakes."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from manga_text_seg.config import load_translation_config
from manga_text_seg.translation.runs import ErrorReport
from manga_text_seg.translation.translate import (
    FixtureTranslationProvider,
    ProviderHttpError,
    TranslationRequest,
    TranslationService,
)


@pytest.fixture()
def config(tmp_path):
    path = Path(__file__).resolve().parents[3] / "configs" / "translation.json"
    base = load_translation_config(path)
    provider = replace(
        base.provider,
        name="fixture-provider",
        model="fixture-model",
        version="fixture-v1",
        secret_env_var=None,
        retry=replace(
            base.provider.retry,
            max_attempts=3,
            initial_backoff_seconds=0.25,
            max_backoff_seconds=0.5,
            retry_status_codes=(429, 500, 502, 503, 504),
        ),
        cache=replace(base.provider.cache, root=tmp_path / "cache", enabled=False),
    )
    return replace(base, provider=provider)


class ScriptedProvider:
    deterministic = True

    def __init__(self, script):
        self.script = {text: list(outcomes) for text, outcomes in script.items()}
        self.calls = []

    def translate(self, text, target_language):
        self.calls.append((text, target_language))
        outcome = self.script[text].pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def test_success_records_per_region_text_and_provider_provenance(config, tmp_path):
    provider = FixtureTranslationProvider(
        {"日本語": "Vietnamese", "broken": RuntimeError("isolated failure")}
    )
    service = TranslationService(config, provider)

    results = service.translate_regions(
        [
            TranslationRequest(1, "日本語"),
            TranslationRequest(2, "broken"),
        ],
        output_path=tmp_path / "translations.json",
    )

    assert provider.calls == [
        ("日本語", config.target_language),
        ("broken", config.target_language),
    ]
    result = results[0]
    assert (result.region_id, result.source_text, result.translated_text) == (
        1,
        "日本語",
        "Vietnamese",
    )
    assert result.status == "ok"
    assert result.cache_status == "miss"
    assert result.failure_policy == config.provider.failure_policy
    assert result.provider == "fixture-provider"
    assert result.model == "fixture-model"
    assert result.version == "fixture-v1"
    assert result.processing_time_seconds >= 0
    assert results[1].status == "failed"
    assert results[1].translated_text == "broken"
    assert (tmp_path / "translations.json").is_file()
    saved = json.loads(
        (tmp_path / "translations.json").read_text(encoding="utf-8")
    )
    assert all(
        (item["provider"], item["model"], item["version"])
        == ("fixture-provider", "fixture-model", "fixture-v1")
        for item in saved
    )


def test_timeout_and_rate_limit_retry_with_capped_backoff(config, tmp_path):
    provider = ScriptedProvider(
        {
            "timeout": [
                TimeoutError("slow"),
                TimeoutError("still slow"),
                "ok after timeout",
            ],
            "rate": [ProviderHttpError(429, "busy"), "ok after rate limit"],
        }
    )
    sleeps = []
    report = ErrorReport(run_id="retry-test")
    service = TranslationService(
        config, provider, error_report=report, sleeper=sleeps.append
    )

    results = service.translate_regions(
        [TranslationRequest(1, "timeout"), TranslationRequest(2, "rate")],
        image_id="ARMS/000",
    )

    assert [result.translated_text for result in results] == [
        "ok after timeout",
        "ok after rate limit",
    ]
    assert len(provider.calls) == 5
    assert sleeps == [0.25, 0.5, 0.25]
    assert report.counts["api_timeout_or_rate_limit"] == 3
    assert report.counts["translation_failure"] == 0
    assert [
        entry["region_id"]
        for entry in report.categories["api_timeout_or_rate_limit"]
    ] == [1, 1, 2]


def test_hard_failure_is_not_retried_and_keeps_ocr_text(config):
    provider = ScriptedProvider(
        {"source": [RuntimeError("permanent provider failure")]}
    )
    report = ErrorReport(run_id="hard-fail")
    service = TranslationService(config, provider, error_report=report)

    result = service.translate_one(
        TranslationRequest(7, "source"), image_id="ARMS/000"
    )

    assert provider.calls == [("source", config.target_language)]
    assert result.status == "failed"
    assert result.translated_text == "source"
    assert result.error == "permanent provider failure"
    assert report.counts["translation_failure"] == 1


@pytest.mark.parametrize(
    ("policy", "expected_status", "expected_text"),
    [
        ("keep_ocr_text", "failed", "source"),
        ("exclude_from_rendering", "skipped", ""),
    ],
)
def test_failure_policy_is_applied_and_recorded(
    config, policy, expected_status, expected_text
):
    provider_config = replace(config.provider, failure_policy=policy)
    provider = ScriptedProvider({"source": [RuntimeError("provider unavailable")]})
    service = TranslationService(replace(config, provider=provider_config), provider)

    result = service.translate_one(TranslationRequest(1, "source"))

    assert result.status == expected_status
    assert result.translated_text == expected_text
    assert result.failure_policy == policy
