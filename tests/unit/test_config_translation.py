"""T006 configuration contract tests for Spec 004's typed loader."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from manga_text_seg.config import ConfigError, load_translation_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "translation.json"


@pytest.fixture()
def config_doc() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture()
def write_config(tmp_path: Path, config_doc):
    def write(doc: dict) -> Path:
        path = tmp_path / "translation.json"
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return path

    return write


def test_shipped_config_loads_and_resolves_paths(config_doc) -> None:
    config = load_translation_config(CONFIG_PATH, repo_root=REPO_ROOT)
    assert config.manifest_path == (REPO_ROOT / config_doc["manifest_path"]).resolve()
    assert config.rendering.font_path == (REPO_ROOT / config_doc["rendering"]["font_path"]).resolve()
    assert config.provider.cache.root == (REPO_ROOT / "cache" / "translation").resolve()
    assert config.output_roots.translation == (REPO_ROOT / "outputs" / "translation").resolve()
    assert config.output_roots.cache == config.provider.cache.root
    assert config.provider.failure_policy == "keep_ocr_text"


@pytest.mark.parametrize(
    "key",
    [
        "manifest_path", "segmentation_method", "inpainting_source", "target_language",
        "region_extraction", "ocr", "provider", "rendering", "output_roots",
    ],
)
def test_missing_top_level_required_key_is_rejected(config_doc, write_config, key) -> None:
    doc = copy.deepcopy(config_doc)
    del doc[key]
    with pytest.raises(ConfigError, match=key):
        load_translation_config(write_config(doc), repo_root=REPO_ROOT)


@pytest.mark.parametrize(
    "path",
    [
        ("inpainting_source", "algorithm"),
        ("region_extraction", "crop_padding"),
        ("ocr", "preprocessing"),
        ("provider", "retry"),
        ("provider", "base_url"),
        ("provider", "cache"),
        ("rendering", "font_path"),
        ("output_roots", "translation"),
    ],
)
def test_missing_nested_required_key_is_rejected(config_doc, write_config, path) -> None:
    doc = copy.deepcopy(config_doc)
    del doc[path[0]][path[1]]
    with pytest.raises(ConfigError, match=path[1]):
        load_translation_config(write_config(doc), repo_root=REPO_ROOT)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(manifest_path=3),
        lambda d: d.update(segmentation_method=[]),
        lambda d: d["inpainting_source"].update(run_id=4),
        lambda d: d["region_extraction"].update(min_area=True),
        lambda d: d["ocr"].update(preprocessing=[]),
        lambda d: d["ocr"]["preprocessing"].update(enabled="no"),
        lambda d: d["provider"].update(timeout_seconds="30"),
        lambda d: d["provider"]["retry"].update(retry_status_codes=[429, True]),
        lambda d: d["provider"]["cache"].update(enabled=1),
        lambda d: d["rendering"].update(color=[0, 256, 0]),
        lambda d: d["output_roots"].update(cache=None),
    ],
)
def test_wrong_nested_types_are_rejected(config_doc, write_config, mutate) -> None:
    doc = copy.deepcopy(config_doc)
    mutate(doc)
    with pytest.raises(ConfigError):
        load_translation_config(write_config(doc), repo_root=REPO_ROOT)


@pytest.mark.parametrize(
    "path,value",
    [
        (("segmentation_method",), "not-a-method"),
        (("target_language",), "ja"),
        (("inpainting_source", "algorithm"), "opencv"),
        (("region_extraction", "reading_order"), "diagonal"),
        (("provider", "failure_policy"), "silent"),
        (("rendering", "text_direction"), "auto-magically"),
        (("rendering", "align_h"), "justify"),
    ],
)
def test_invalid_enums_are_rejected(config_doc, write_config, path, value) -> None:
    doc = copy.deepcopy(config_doc)
    target = doc
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ConfigError):
        load_translation_config(write_config(doc), repo_root=REPO_ROOT)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["region_extraction"].update(min_area=0),
        lambda d: d["region_extraction"].update(max_area=0),
        lambda d: d["region_extraction"].update(max_area=0, min_area=1),
        lambda d: d["region_extraction"].update(merge_distance=-1),
        lambda d: d["region_extraction"].update(polygon_epsilon=0),
        lambda d: d["provider"].update(timeout_seconds=0),
        lambda d: d["provider"]["retry"].update(max_attempts=0),
        lambda d: d["provider"]["retry"].update(initial_backoff_seconds=4, max_backoff_seconds=2),
        lambda d: d["provider"]["retry"].update(retry_status_codes=[99]),
        lambda d: d["rendering"].update(font_size=0),
        lambda d: d["rendering"].update(min_font_size=40),
        lambda d: d["rendering"].update(stroke_width=-1),
        lambda d: d["rendering"].update(line_spacing=0),
    ],
)
def test_invalid_numeric_values_are_rejected(config_doc, write_config, mutate) -> None:
    doc = copy.deepcopy(config_doc)
    mutate(doc)
    with pytest.raises(ConfigError):
        load_translation_config(write_config(doc), repo_root=REPO_ROOT)


def test_ocr_settings_require_supported_engine_and_valid_preprocessing(config_doc, write_config) -> None:
    doc = copy.deepcopy(config_doc)
    doc["ocr"]["engine"] = ""
    with pytest.raises(ConfigError, match="engine"):
        load_translation_config(write_config(doc), repo_root=REPO_ROOT)
    doc = copy.deepcopy(config_doc)
    doc["ocr"]["preprocessing"]["upscale_factor"] = 0.5
    with pytest.raises(ConfigError, match="upscale_factor"):
        load_translation_config(write_config(doc), repo_root=REPO_ROOT)


def test_invalid_failure_policy_is_rejected(config_doc, write_config) -> None:
    config_doc["provider"]["failure_policy"] = "continue_silently"
    with pytest.raises(ConfigError, match="failure_policy"):
        load_translation_config(write_config(config_doc), repo_root=REPO_ROOT)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["provider"].update(name=""),
        lambda d: d["provider"].update(model=""),
        lambda d: d["provider"].update(base_url="file:///tmp/secret"),
        lambda d: d["provider"].update(base_url="https://user:password@example.test/v1"),
        lambda d: d["provider"].update(base_url="https://[invalid/v1"),
        lambda d: d["provider"].update(base_url="https://example.test:bad/v1"),
        lambda d: d["provider"].update(secret_env_var="literal-secret-value"),
        lambda d: d["provider"]["retry"].update(retry_status_codes=[]),
        lambda d: d["provider"]["retry"].update(retry_status_codes=[429, 429]),
        lambda d: d["provider"]["cache"].update(bypass="false"),
    ],
)
def test_invalid_provider_and_cache_settings_are_rejected(config_doc, write_config, mutate) -> None:
    doc = copy.deepcopy(config_doc)
    mutate(doc)
    with pytest.raises(ConfigError):
        load_translation_config(write_config(doc), repo_root=REPO_ROOT)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["rendering"].update(font_family=""),
        lambda d: d["rendering"].update(font_path=""),
        lambda d: d["rendering"].update(fallback_font_path=42),
        lambda d: d["rendering"].update(stroke_fill=[0, 0]),
        lambda d: d["rendering"].update(background=[0, 0, 0, 256]),
        lambda d: d["rendering"].update(wrap="true"),
    ],
)
def test_invalid_rendering_and_font_settings_are_rejected(config_doc, write_config, mutate) -> None:
    doc = copy.deepcopy(config_doc)
    mutate(doc)
    with pytest.raises(ConfigError):
        load_translation_config(write_config(doc), repo_root=REPO_ROOT)


def test_unknown_keys_follow_existing_loader_convention(config_doc, write_config) -> None:
    doc = copy.deepcopy(config_doc)
    doc["future_extension"] = {"enabled": True}
    doc["rendering"]["font_famly"] = "typo-field-is-ignored-like-existing-loaders"
    assert load_translation_config(write_config(doc), repo_root=REPO_ROOT).segmentation_method == "classical_baseline"


def test_credential_fields_are_rejected_even_when_unknown(config_doc, write_config) -> None:
    doc = copy.deepcopy(config_doc)
    doc["provider"]["api_key"] = "not-allowed"
    with pytest.raises(ConfigError, match="credential"):
        load_translation_config(write_config(doc), repo_root=REPO_ROOT)


def test_repository_config_contains_no_literal_credential(config_doc) -> None:
    provider = config_doc["provider"]
    assert provider["secret_env_var"] == "MANGA_TRANSLATE_API_KEY"
    assert not {"api_key", "token", "credential", "secret"} & provider.keys()
    assert "sk-" not in json.dumps(config_doc).lower()
