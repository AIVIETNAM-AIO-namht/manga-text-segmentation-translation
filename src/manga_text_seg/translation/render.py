"""Deterministic Pillow rendering for Spec 004 translated page regions."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..config import RenderingConfig
from .regions import PAGE_HEIGHT, PAGE_WIDTH, TextRegion
from .runs import ErrorReport
from .translate import TranslationResult


@dataclass(frozen=True)
class RenderedRegion:
    """Validated per-region rendering outcome, ready for page metadata."""

    region_id: int
    render_status: str
    rendered_text: str
    font_size_used: int | None
    line_count: int
    display_box: tuple[int, int, int, int]
    direction_used: str
    warning: str | None = None

    def __post_init__(self) -> None:
        if self.region_id < 1:
            raise ValueError("region_id must be positive")
        if self.render_status not in {"rendered", "overflow_warning", "skipped"}:
            raise ValueError("invalid render_status")
        if self.direction_used not in {"horizontal", "vertical"}:
            raise ValueError("invalid direction_used")
        if self.render_status == "overflow_warning" and not self.warning:
            raise ValueError("overflow warnings require a reason")


@dataclass(frozen=True)
class RenderedPage:
    image_path: Path
    regions: tuple[RenderedRegion, ...]


def _font(config: RenderingConfig, size: int) -> ImageFont.FreeTypeFont:
    """Load only configured font files; never search system fonts."""
    errors: list[OSError] = []
    for path in (config.font_path, config.fallback_font_path):
        if path is None:
            continue
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError as exc:
            errors.append(exc)
    raise OSError("configured rendering fonts are unavailable") from (
        errors[-1] if errors else None
    )


def _measure(font: ImageFont.FreeTypeFont, text: str, stroke: int) -> tuple[int, int]:
    left, top, right, bottom = font.getbbox(text or " ", stroke_width=stroke)
    return right - left, bottom - top


def _wrap(
    text: str, font: ImageFont.FreeTypeFont, width: int, stroke: int
) -> list[str]:
    words = text.split()
    if not words:
        return [text]
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and _measure(font, candidate, stroke)[0] > width:
            lines.append(current)
            current = word
        else:
            current = candidate
        if _measure(font, current, stroke)[0] > width:
            # Break long unspaced words deterministically by Unicode code point.
            chunk = ""
            for char in current:
                if chunk and _measure(font, chunk + char, stroke)[0] > width:
                    lines.append(chunk)
                    chunk = char
                else:
                    chunk += char
            current = chunk
    if current or not lines:
        lines.append(current)
    return lines


def _lines_size(
    lines: Sequence[str], font: ImageFont.FreeTypeFont, config: RenderingConfig
) -> tuple[int, int]:
    heights = [_measure(font, line, config.stroke_width)[1] for line in lines]
    width = max(
        (_measure(font, line, config.stroke_width)[0] for line in lines), default=0
    )
    height = sum(heights) + max(0, len(lines) - 1) * max(
        0, round(font.size * (config.line_spacing - 1))
    )
    return width, height


def _vertical_size(
    characters: Sequence[str], font: ImageFont.FreeTypeFont, config: RenderingConfig
) -> tuple[int, int]:
    widths = [
        _measure(font, character, config.stroke_width)[0] for character in characters
    ]
    heights = [
        _measure(font, character, config.stroke_width)[1] for character in characters
    ]
    spacing = max(0, round(font.size * (config.line_spacing - 1)))
    return max(widths, default=0), sum(heights) + spacing * max(0, len(heights) - 1)


def _fits(
    text: str, box: tuple[int, int, int, int], config: RenderingConfig, direction: str
) -> tuple[int, ImageFont.FreeTypeFont, list[str], int, int]:
    x, y, width, height = box
    vertical = direction == "vertical"
    for size in range(config.font_size, config.min_font_size - 1, -1):
        font = _font(config, size)
        lines = list(text) if vertical else [text]
        text_width, text_height = _lines_size(lines, font, config)
        if vertical:
            text_width, text_height = _vertical_size(lines, font, config)
        if text_width <= width and text_height <= height:
            return size, font, lines, text_width, text_height

    # FR-041 rung two: after exhausting font shrink, add lines at the minimum size.
    size = config.min_font_size
    font = _font(config, size)
    if vertical:
        lines = list(text)
        glyph_width, glyph_height = _vertical_size(lines, font, config)
        return size, font, lines, glyph_width, glyph_height
    lines = _wrap(text, font, width, config.stroke_width) if config.wrap else [text]
    text_width, text_height = _lines_size(lines, font, config)
    return size, font, lines, text_width, text_height


def _expanded_box(
    box: tuple[int, int, int, int], amount: int
) -> tuple[int, int, int, int]:
    x, y, width, height = box
    left, top = max(0, x - amount), max(0, y - amount)
    right, bottom = min(PAGE_WIDTH, x + width + amount), min(
        PAGE_HEIGHT, y + height + amount
    )
    return left, top, right - left, bottom - top


def _draw_region(
    canvas: Image.Image,
    region: TextRegion,
    result: TranslationResult,
    config: RenderingConfig,
    error_report: ErrorReport | None,
    image_id: str | None,
) -> RenderedRegion:
    direction = config.text_direction
    if direction == "auto":
        direction = region.orientation
    if direction not in {"horizontal", "vertical"}:
        raise ValueError("text_direction must resolve to horizontal or vertical")
    if result.status == "skipped" or result.failure_policy == "exclude_from_rendering":
        return RenderedRegion(
            region.region_id,
            "skipped",
            result.translated_text,
            None,
            0,
            region.bbox,
            direction,
        )
    if not result.translated_text:
        return RenderedRegion(
            region.region_id, "skipped", "", None, 0, region.bbox, direction
        )
    original_box = tuple(int(v) for v in region.bbox)
    try:
        size, font, lines, text_width, text_height = _fits(
            result.translated_text, original_box, config, direction
        )
    except OSError:
        reason = "configured rendering fonts are unavailable"
        if error_report is not None:
            if image_id is None:
                raise ValueError("image_id is required when writing render errors")
            error_report.add(
                "font_missing",
                image_id=image_id,
                region_id=region.region_id,
                stage="render",
                reason=reason,
            )
        return RenderedRegion(
            region.region_id,
            "skipped",
            result.translated_text,
            None,
            0,
            original_box,
            direction,
            reason,
        )
    display_box = original_box
    if text_width > original_box[2] or text_height > original_box[3]:
        if config.expansion_allowance > 0:
            display_box = _expanded_box(original_box, config.expansion_allowance)
            try:
                size, font, lines, text_width, text_height = _fits(
                    result.translated_text, display_box, config, direction
                )
            except OSError:
                reason = "configured rendering fonts are unavailable"
                if error_report is not None:
                    if image_id is None:
                        raise ValueError(
                            "image_id is required when writing render errors"
                        )
                    error_report.add(
                        "font_missing",
                        image_id=image_id,
                        region_id=region.region_id,
                        stage="render",
                        reason=reason,
                    )
                return RenderedRegion(
                    region.region_id,
                    "skipped",
                    result.translated_text,
                    None,
                    0,
                    original_box,
                    direction,
                    reason,
                )
    overflow = text_width > display_box[2] or text_height > display_box[3]
    warning = None
    if overflow:
        warning = "translated text does not fit after font, line, and expansion fitting"
        if error_report is not None:
            if image_id is None:
                raise ValueError("image_id is required when writing render errors")
            error_report.add(
                "text_overflow",
                image_id=image_id,
                region_id=region.region_id,
                stage="render",
                reason=warning,
            )

    x, y, width, height = display_box
    if config.background is not None:
        canvas_draw = ImageDraw.Draw(canvas)
        canvas_draw.rectangle(
            (x, y, x + width - 1, y + height - 1), fill=config.background
        )
    # Draw to a transparent layer, then composite only the display area. Even an
    # explicit overflow warning therefore never paints outside its recorded box.
    text_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer)
    if direction == "vertical":
        glyph_heights = [_measure(font, char, config.stroke_width)[1] for char in lines]
        spacing = max(0, round(font.size * (config.line_spacing - 1)))
        total_height = sum(glyph_heights) + spacing * max(0, len(glyph_heights) - 1)
        cy = (
            y + max(0, (height - total_height) // 2)
            if config.align_v == "middle"
            else y
        )
        for char, glyph_height in zip(lines, glyph_heights):
            glyph_width = _measure(font, char, config.stroke_width)[0]
            cx = x + max(0, (width - glyph_width) // 2)
            draw.text(
                (cx, cy),
                char,
                font=font,
                fill=config.color,
                stroke_width=config.stroke_width,
                stroke_fill=config.stroke_fill,
            )
            cy += glyph_height + spacing
    else:
        box_width, box_height = _lines_size(lines, font, config)
        if config.align_v == "top":
            top = y
        elif config.align_v == "bottom":
            top = y + height - box_height
        else:
            top = y + (height - box_height) // 2
        spacing = max(0, round(font.size * (config.line_spacing - 1)))
        for line in lines:
            line_width, line_height = _measure(font, line, config.stroke_width)
            if config.align_h == "left":
                left = x
            elif config.align_h == "right":
                left = x + width - line_width
            else:
                left = x + (width - line_width) // 2
            draw.text(
                (left, top),
                line,
                font=font,
                fill=config.color,
                stroke_width=config.stroke_width,
                stroke_fill=config.stroke_fill,
            )
            top += line_height + spacing
    left, top, box_width, box_height = display_box
    clipped_layer = text_layer.crop((left, top, left + box_width, top + box_height))
    if canvas.mode == "RGBA":
        canvas.alpha_composite(clipped_layer, (left, top))
    else:
        canvas.paste(clipped_layer, (left, top), clipped_layer.getchannel("A"))
    return RenderedRegion(
        region.region_id,
        "overflow_warning" if overflow else "rendered",
        result.translated_text,
        size,
        len(lines),
        display_box,
        direction,
        warning,
    )


def render_page(
    inpainted_image: str | Path | Image.Image,
    regions: Sequence[TextRegion],
    translations: Sequence[TranslationResult],
    config: RenderingConfig,
    output_path: str | Path,
    *,
    error_report: ErrorReport | None = None,
    image_id: str | None = None,
) -> RenderedPage:
    """Render translated text onto a copy of the aligned inpainted page.

    Input files are opened read-only and never resized or written. The only output
    is the caller-selected PNG path.
    """
    if not isinstance(config, RenderingConfig):
        raise TypeError("config must be RenderingConfig")
    if isinstance(inpainted_image, Image.Image):
        source = inpainted_image.copy()
    else:
        with Image.open(Path(inpainted_image)) as opened:
            source = opened.copy()
    if source.size != (PAGE_WIDTH, PAGE_HEIGHT):
        raise ValueError(
            f"inpainted image must be {(PAGE_WIDTH, PAGE_HEIGHT)}; "
            "resizing is prohibited"
        )
    canvas = (
        source.convert("RGBA")
        if source.mode in {"RGBA", "LA"}
        else source.convert("RGB")
    )
    by_id = {item.region_id: item for item in translations}
    if len(by_id) != len(translations):
        raise ValueError("translations must have unique region_id values")
    outcomes = []
    for region in regions:
        result = by_id.get(region.region_id)
        if result is None:
            raise ValueError(f"missing translation for region_id {region.region_id}")
        outcomes.append(
            _draw_region(canvas, region, result, config, error_report, image_id)
        )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        isinstance(inpainted_image, (str, Path))
        and destination.resolve() == Path(inpainted_image).resolve()
    ):
        raise ValueError("rendered output must not overwrite its source image")
    canvas.save(destination, format="PNG", optimize=False, compress_level=9)
    return RenderedPage(destination, tuple(outcomes))


def write_rendered_metadata(
    path: str | Path, regions: Sequence[RenderedRegion]
) -> Path:
    """Write the renderer's validated per-region outcomes as deterministic JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps([asdict(region) for region in regions], indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return destination
