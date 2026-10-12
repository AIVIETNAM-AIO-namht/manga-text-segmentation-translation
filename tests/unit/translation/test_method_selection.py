"""T019 method selection refuses to guess when configuration is absent."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from manga_text_seg.config import load_translation_config
from manga_text_seg.translation.intake import _resolve_method

ROOT = Path(__file__).resolve().parents[3]


def test_explicit_configured_method_has_precedence():
    config = load_translation_config(ROOT / "configs" / "translation.json")
    assert config.segmentation_method == "unetpp_efficientnetv2"
    assert _resolve_method(config) == "unetpp_efficientnetv2"


def test_missing_explicit_method_without_recorded_selection_refuses():
    config = load_translation_config(ROOT / "configs" / "translation.json")
    config_without_method = replace(config, segmentation_method=None)

    with pytest.raises(ValueError, match="no Spec 2.*selection record"):
        _resolve_method(config_without_method)
