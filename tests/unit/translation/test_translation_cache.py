"""T040 shared content-addressed cache behavior."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from manga_text_seg.config import load_translation_config
from manga_text_seg.translation.translate import (
    FixtureTranslationProvider,
    TranslationRequest,
    TranslationService,
    translation_cache_key,
)


def _config(tmp_path, *, bypass=False):
    path = Path(__file__).resolve().parents[3] / "configs" / "translation.json"
    base = load_translation_config(path)
    provider = replace(
        base.provider,
        name="fixture-provider",
        model="fixture-model",
        version="fixture-v1",
        secret_env_var=None,
        cache=replace(
            base.provider.cache,
            root=tmp_path / "cache" / "translation",
            enabled=True,
            bypass=bypass,
        ),
    )
    return replace(base, provider=provider)


def test_first_translation_misses_and_second_identical_translation_hits(tmp_path):
    config = _config(tmp_path)
    first_provider = FixtureTranslationProvider({"同じ": "same"})
    first = TranslationService(config, first_provider).translate_one(
        TranslationRequest(1, "同じ")
    )
    second_provider = FixtureTranslationProvider({})
    second = TranslationService(config, second_provider).translate_one(
        TranslationRequest(8, "同じ")
    )
    key = translation_cache_key(
        "同じ", config.target_language, config.provider.name, config.provider.model
    )
    canonical = json.dumps(
        ["同じ", config.target_language, config.provider.name, config.provider.model],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    assert key == hashlib.sha256(canonical).hexdigest()
    assert first.cache_status == "miss"
    assert second.cache_status == "hit"
    assert first.translated_text == second.translated_text == "same"
    assert first_provider.calls == [("同じ", config.target_language)]
    assert second_provider.calls == []
    assert (config.provider.cache.root / f"{key}.json").is_file()


def test_bypass_forces_provider_call_without_deleting_or_changing_cache(tmp_path):
    config = _config(tmp_path)
    initial_provider = FixtureTranslationProvider({"text": "cached value"})
    TranslationService(config, initial_provider).translate_one(
        TranslationRequest(1, "text")
    )
    key = translation_cache_key(
        "text", config.target_language, config.provider.name, config.provider.model
    )
    cache_path = config.provider.cache.root / f"{key}.json"
    original_bytes = cache_path.read_bytes()

    bypass_config = replace(
        config,
        provider=replace(
            config.provider,
            cache=replace(config.provider.cache, bypass=True),
        ),
    )
    fresh_provider = FixtureTranslationProvider({"text": "fresh value"})
    result = TranslationService(bypass_config, fresh_provider).translate_one(
        TranslationRequest(2, "text")
    )

    assert result.cache_status == "miss"
    assert result.translated_text == "fresh value"
    assert fresh_provider.calls == [("text", config.target_language)]
    assert cache_path.is_file()
    assert cache_path.read_bytes() == original_bytes
