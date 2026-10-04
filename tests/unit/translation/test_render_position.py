"""T050 placement, direction and deterministic-rendering tests."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from PIL import Image

from manga_text_seg.config import load_translation_config
from manga_text_seg.translation.regions import TextRegion
from manga_text_seg.translation.render import render_page
from manga_text_seg.translation.translate import TranslationResult

ROOT = Path(__file__).resolve().parents[3]


def _config(**changes):
    base = load_translation_config(ROOT / "configs" / "translation.json").rendering
    return replace(base, **changes)


def _region(box, orientation):
    x, y, width, height = box
    polygon = (
        (x, y),
        (x + width - 1, y),
        (x + width - 1, y + height - 1),
        (x, y + height - 1),
    )
    return TextRegion(
        1, box, polygon, orientation, 0, 1, None, box, False, width * height
    )


def _translation():
    return TranslationResult(
        1,
        "source",
        "Rendered text",
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


def test_text_pixels_are_placed_inside_bbox(tmp_path):
    source = Image.new("RGB", (1654, 1170), "white")
    region = _region((500, 400, 320, 120), "horizontal")
    result = render_page(
        source, [region], [_translation()], _config(), tmp_path / "rendered.png"
    )
    image = Image.open(result.image_path).convert("RGB")
    pixels = image.load()
    changed_points = [
        (x, y)
        for y in range(image.height)
        for x in range(image.width)
        if pixels[x, y] != (255, 255, 255)
    ]
    assert changed_points
    assert min(x for x, _ in changed_points) >= 500
    assert max(x for x, _ in changed_points) < 820
    assert min(y for _, y in changed_points) >= 400
    assert max(y for _, y in changed_points) < 520
    assert result.regions[0].display_box == region.bbox


def test_auto_direction_follows_horizontal_and_vertical_orientation(tmp_path):
    source = Image.new("RGB", (1654, 1170), "white")
    vertical_region = _region((700, 300, 90, 300), "vertical")
    result = render_page(
        source,
        [vertical_region],
        [_translation()],
        _config(text_direction="auto"),
        tmp_path / "vertical.png",
    )
    assert result.regions[0].direction_used == "vertical"
    explicit = render_page(
        source,
        [vertical_region],
        [_translation()],
        _config(text_direction="horizontal"),
        tmp_path / "horizontal.png",
    )
    assert explicit.regions[0].direction_used == "horizontal"


def test_placement_is_deterministic_and_does_not_mutate_source(tmp_path):
    source = Image.new("RGB", (1654, 1170), "white")
    source.putpixel((10, 10), (1, 2, 3))
    before = sha256(source.tobytes()).hexdigest()
    region = _region((500, 400, 320, 120), "horizontal")
    first = render_page(
        source, [region], [_translation()], _config(), tmp_path / "one.png"
    )
    render_page(source, [region], [_translation()], _config(), tmp_path / "two.png")
    assert (tmp_path / "one.png").read_bytes() == (tmp_path / "two.png").read_bytes()
    assert sha256(source.tobytes()).hexdigest() == before
    output = Image.open(first.image_path)
    assert output.format == "PNG"
    assert output.size == (1654, 1170)


def test_source_file_bytes_are_unchanged(tmp_path):
    source_path = tmp_path / "inpainted.png"
    Image.new("RGB", (1654, 1170), (20, 30, 40)).save(source_path, format="PNG")
    before = sha256(source_path.read_bytes()).hexdigest()
    render_page(
        source_path,
        [_region((500, 400, 320, 120), "horizontal")],
        [_translation()],
        _config(),
        tmp_path / "rendered.png",
    )
    assert sha256(source_path.read_bytes()).hexdigest() == before


def test_vertical_orientation_is_recorded_even_when_empty_text_is_skipped(tmp_path):
    source = Image.new("RGB", (1654, 1170), "white")
    empty = replace(_translation(), translated_text="")
    result = render_page(
        source,
        [_region((700, 300, 80, 260), "vertical")],
        [empty],
        _config(text_direction="auto"),
        tmp_path / "skipped.png",
    )
    assert result.regions[0].direction_used == "vertical"
