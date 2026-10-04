"""Deterministic text-region extraction for Spec 004 (FR-014–FR-021)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from ..config import RegionExtractionConfig

if TYPE_CHECKING:
    from .runs import ErrorReport

PAGE_WIDTH = 1654
PAGE_HEIGHT = 1170


@dataclass(frozen=True)
class TextRegion:
    """A single extracted region using the Spec 004 TextRegion fields."""

    region_id: int
    bbox: tuple[int, int, int, int]
    polygon: tuple[tuple[int, int], ...]
    orientation: str
    reading_order_index: int
    merged_from: int
    crop_path: str | None
    crop_box: tuple[int, int, int, int]
    clipped: bool
    area: int


@dataclass(frozen=True)
class RegionExtractionResult:
    """Regions and recorded exclusions/outcome for one page mask."""

    regions: tuple[TextRegion, ...]
    status: str
    excluded_components: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _Component:
    label: int
    area: int
    bbox: tuple[int, int, int, int]
    pixels: np.ndarray


def _validate_mask(mask: np.ndarray) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a numpy.ndarray")
    if mask.dtype != np.uint8 or mask.ndim != 2:
        raise ValueError("mask must be a single-channel uint8 binary array")
    if mask.shape != (PAGE_HEIGHT, PAGE_WIDTH):
        raise ValueError(f"mask must have aligned shape {(PAGE_HEIGHT, PAGE_WIDTH)}")
    if not np.isin(mask, (0, 255)).all():
        raise ValueError("mask values must follow the binary convention {0, 255}")
    return mask


def _boxes_within_distance(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
    distance: int,
) -> bool:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    horizontal_gap = max(lx - (rx + rw), rx - (lx + lw), 0)
    vertical_gap = max(ly - (ry + rh), ry - (ly + lh), 0)
    return horizontal_gap < distance and vertical_gap < distance


def _groups(components: list[_Component], distance: int) -> list[list[_Component]]:
    parent = list(range(len(components)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    if distance > 0:
        for left in range(len(components)):
            for right in range(left + 1, len(components)):
                if _boxes_within_distance(
                    components[left].bbox, components[right].bbox, distance
                ):
                    union(left, right)

    grouped: dict[int, list[_Component]] = {}
    for index, component in enumerate(components):
        grouped.setdefault(find(index), []).append(component)
    return [grouped[key] for key in sorted(grouped)]


def _polygon(mask: np.ndarray, epsilon: float) -> tuple[tuple[int, int], ...]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("component group has no contour")
    contour = (
        contours[0]
        if len(contours) == 1
        else cv2.convexHull(np.concatenate(contours, axis=0))
    )
    perimeter = cv2.arcLength(contour, True)
    approximation = cv2.approxPolyDP(contour, epsilon * perimeter, True).reshape(-1, 2)
    points = [(int(x), int(y)) for x, y in approximation]
    if len(points) < 3:
        points = [(int(x), int(y)) for x, y in contour.reshape(-1, 2)]
    # Canonical winding and starting point make serialization independent of
    # OpenCV's contour start vertex.
    signed_area = sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )
    if signed_area < 0:
        points.reverse()
    start = min(
        range(len(points)), key=lambda index: (points[index][1], points[index][0])
    )
    points = points[start:] + points[:start]
    return tuple(points)


def _crop_geometry(
    bbox: tuple[int, int, int, int], padding: int
) -> tuple[tuple[int, int, int, int], bool]:
    x, y, width, height = bbox
    left, top = x - padding, y - padding
    right, bottom = x + width + padding, y + height + padding
    clipped = left < 0 or top < 0 or right > PAGE_WIDTH or bottom > PAGE_HEIGHT
    return (
        max(0, left),
        max(0, top),
        min(PAGE_WIDTH, right),
        min(PAGE_HEIGHT, bottom),
    ), clipped


def extract_regions(
    mask: np.ndarray,
    config: RegionExtractionConfig,
    *,
    image_id: str | None = None,
    error_report: ErrorReport | None = None,
) -> RegionExtractionResult:
    """Extract connected components, merge nearby boxes and assign stable IDs.

    ``mask`` must already satisfy Spec 1's binary aligned-mask contract; this
    function rejects invalid masks instead of silently normalizing them.
    """
    binary = _validate_mask(mask)
    if not isinstance(config, RegionExtractionConfig):
        raise TypeError("config must be a RegionExtractionConfig")
    if config.reading_order not in {"manga", "western"}:
        raise ValueError("reading_order must be 'manga' or 'western'")
    for name in ("min_area", "max_area", "merge_distance", "crop_padding"):
        value = getattr(config, name)
        minimum = 1 if name in {"min_area", "max_area"} else 0
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} has an invalid value")
    if config.max_area < config.min_area:
        raise ValueError("max_area must be >= min_area")
    if (
        not isinstance(config.polygon_epsilon, (float, int))
        or isinstance(config.polygon_epsilon, bool)
        or config.polygon_epsilon <= 0
    ):
        raise ValueError("polygon_epsilon must be a positive number")

    def record(category: str, reason: str) -> None:
        if error_report is not None:
            if image_id is None:
                raise ValueError("image_id is required when writing to ErrorReport")
            error_report.add(
                category,
                image_id=image_id,
                region_id=None,
                stage="regions",
                reason=reason,
            )

    if not np.any(binary):
        record("empty_mask", "prediction mask contains no text pixels")
        return RegionExtractionResult((), "no_regions", ())

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components: list[_Component] = []
    excluded: list[dict[str, Any]] = []
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        if area < config.min_area or area > config.max_area:
            excluded.append(
                {
                    "label": label,
                    "bbox": (x, y, width, height),
                    "area": area,
                    "reason": (
                        "below_min_area" if area < config.min_area else "above_max_area"
                    ),
                }
            )
            continue
        components.append(
            _Component(label, area, (x, y, width, height), labels == label)
        )

    if not components:
        record(
            "no_text_region_detected",
            "all mask components were excluded by area filters",
        )
        return RegionExtractionResult((), "no_regions", tuple(excluded))

    candidates: list[dict[str, Any]] = []
    for group in _groups(components, config.merge_distance):
        group_mask = np.zeros(binary.shape, dtype=np.uint8)
        for component in group:
            group_mask[component.pixels] = 255
        ys, xs = np.nonzero(group_mask)
        left, top = int(xs.min()), int(ys.min())
        right, bottom = int(xs.max()) + 1, int(ys.max()) + 1
        bbox = (left, top, right - left, bottom - top)
        crop_box, clipped = _crop_geometry(bbox, config.crop_padding)
        candidates.append(
            {
                "bbox": bbox,
                "polygon": _polygon(group_mask, float(config.polygon_epsilon)),
                "area": sum(component.area for component in group),
                "merged_from": len(group),
                "crop_box": crop_box,
                "clipped": clipped,
            }
        )

    def reading_key(item: dict[str, Any]) -> tuple[Any, ...]:
        x, y, width, height = item["bbox"]
        center_x_twice = 2 * x + width
        center_y_twice = 2 * y + height
        horizontal_key = (
            -center_x_twice if config.reading_order == "manga" else center_x_twice
        )
        return (
            horizontal_key,
            center_y_twice,
            -(width * height),
            y,
            x,
            item["polygon"],
        )

    candidates.sort(key=reading_key)
    regions = tuple(
        TextRegion(
            region_id=index,
            bbox=item["bbox"],
            polygon=item["polygon"],
            orientation=(
                "horizontal" if item["bbox"][2] >= item["bbox"][3] else "vertical"
            ),
            reading_order_index=index,
            merged_from=item["merged_from"],
            crop_path=None,
            crop_box=item["crop_box"],
            clipped=item["clipped"],
            area=item["area"],
        )
        for index, item in enumerate(candidates, start=1)
    )
    return RegionExtractionResult(regions, "ok", tuple(excluded))


def crop_region(image: np.ndarray, region: TextRegion) -> np.ndarray:
    """Return the region's padded crop from a page-sized image array."""
    if not isinstance(image, np.ndarray) or image.ndim not in {2, 3}:
        raise ValueError("image must be a two- or three-dimensional numpy array")
    if image.shape[:2] != (PAGE_HEIGHT, PAGE_WIDTH):
        raise ValueError(f"image must have aligned shape {(PAGE_HEIGHT, PAGE_WIDTH)}")
    left, top, right, bottom = region.crop_box
    if not (0 <= left < right <= PAGE_WIDTH and 0 <= top < bottom <= PAGE_HEIGHT):
        raise ValueError("region crop_box is outside the aligned page")
    return image[top:bottom, left:right].copy()
