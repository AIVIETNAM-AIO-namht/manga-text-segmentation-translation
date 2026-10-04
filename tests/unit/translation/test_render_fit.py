"""T049 renderer fit-ladder tests (Pillow only, no OCR/provider runtime)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PIL import Image

from manga_text_seg.config import load_translation_config
from manga_text_seg.translation.regions import TextRegion
from manga_text_seg.translation.render import render_page
from manga_text_seg.translation.runs import ErrorReport
from manga_text_seg.translation.translate import TranslationResult

ROOT = Path(__file__).resolve().parents[3]


def _config(**changes):
    base = load_translation_config(ROOT / "configs" / "translation.json").rendering
    return replace(base, **changes)


def _region(box=(500, 400, 300, 160), orientation="horizontal"):
    return TextRegion(
        1,
        box,
        (
            (box[0], box[1]),
            (box[0] + box[2] - 1, box[1]),
            (box[0] + box[2] - 1, box[1] + box[3] - 1),
        ),
        orientation,
        0,
        1,
        None,
        box,
        False,
        box[2] * box[3],
    )


def _translation(text: str) -> TranslationResult:
    return TranslationResult(
        1,
        "source",
        text,
        "vi",
        "fixture",
        "fixture",
        "1",
        "miss",
        0.0,
        "ok",
        None,
        "keep_ocr_text",
    )


def _render(tmp_path, text, *, box=(500, 400, 300, 160), **config_changes):
    source = Image.new("RGB", (1654, 1170), "white")
    target = tmp_path / "rendered.png"
    result = render_page(
        source, [_region(box)], [_translation(text)], _config(**config_changes), target
    )
    return result, target


def test_short_text_fits_configured_size(tmp_path):
    result, target = _render(tmp_path, "Short")
    region = result.regions[0]
    assert region.render_status == "rendered"
    assert region.font_size_used == 32
    assert region.line_count == 1
    assert Image.open(target).size == (1654, 1170)


def test_wrapping_and_multiple_lines_are_recorded(tmp_path):
    result, _ = _render(
        tmp_path,
        "A fairly long translated phrase for wrapping",
        box=(500, 400, 120, 180),
    )
    assert result.regions[0].line_count > 1
    assert result.regions[0].render_status == "rendered"


def test_long_text_shrinks_font_before_overflow(tmp_path):
    result, _ = _render(
        tmp_path,
        "Long translated phrase",
        box=(500, 400, 190, 100),
        font_size=36,
        min_font_size=12,
    )
    assert result.regions[0].font_size_used < 36


def test_expansion_allowance_is_used_before_overflow(tmp_path):
    result, _ = _render(
        tmp_path,
        "A phrase that needs more room",
        box=(500, 400, 90, 10),
        font_size=20,
        min_font_size=12,
        expansion_allowance=40,
    )
    region = result.regions[0]
    assert region.display_box != (500, 400, 90, 10)
    assert region.render_status in {"rendered", "overflow_warning"}


def test_unfittable_text_is_explicit_overflow_not_silent_clip(tmp_path):
    errors = ErrorReport("render-test")
    source = Image.new("RGB", (1654, 1170), "white")
    region = _region((500, 400, 12, 10))
    result = render_page(
        source,
        [region],
        [_translation("An extremely long phrase that cannot fit")],
        _config(font_size=20, min_font_size=12),
        tmp_path / "rendered.png",
        error_report=errors,
        image_id="manga/001",
    )
    assert result.regions[0].render_status == "overflow_warning"
    assert result.regions[0].warning
    assert errors.counts["text_overflow"] == 1


def test_unreadable_font_uses_only_configured_fallback(tmp_path):
    fallback = ROOT / "assets" / "fonts" / "DejaVuSans.ttf"
    result, _ = _render(
        tmp_path,
        "fallback",
        font_path=tmp_path / "missing.ttf",
        fallback_font_path=fallback,
    )
    assert result.regions[0].render_status == "rendered"


def test_no_font_fallback_records_font_missing_and_continues(tmp_path):
    errors = ErrorReport("render-test")
    source = Image.new("RGB", (1654, 1170), "white")
    regions = [
        _region(),
        replace(
            _region(),
            region_id=2,
            bbox=(900, 400, 300, 160),
            crop_box=(900, 400, 300, 160),
        ),
    ]
    translations = [
        replace(_translation("one"), region_id=1),
        replace(_translation("two"), region_id=2),
    ]
    # T054 contract: a missing font is an isolated rendering error, not a page crash.
    result = render_page(
        source,
        regions,
        translations,
        _config(font_path=tmp_path / "missing.ttf", fallback_font_path=None),
        tmp_path / "rendered.png",
        error_report=errors,
        image_id="manga/001",
    )
    assert len(result.regions) == 2
    assert errors.counts["font_missing"] == 2
