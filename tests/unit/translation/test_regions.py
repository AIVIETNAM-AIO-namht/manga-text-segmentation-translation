"""T020 coverage for deterministic Spec 004 region extraction."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from manga_text_seg.config import load_translation_config
from manga_text_seg.translation.regions import crop_region, extract_regions
from manga_text_seg.translation.runs import ErrorReport


@pytest.fixture()
def region_config():
    config_path = Path(__file__).resolve().parents[3] / "configs" / "translation.json"
    return load_translation_config(config_path).region_extraction


@pytest.fixture()
def image_id() -> str:
    """Use a real identity from the checked-in Spec 2 page list."""
    page_list = Path(__file__).resolve().parents[3] / "benchmark" / "page-list.json"
    return json.loads(page_list.read_text(encoding="utf-8"))["pages"][0]["image_id"]


def test_aligned_fixture_yields_two_regions_with_bbox_and_polygon(
    aligned_mask, region_config
):
    config = replace(
        region_config, min_area=1, max_area=400_000, merge_distance=0
    )
    result = extract_regions(aligned_mask, config)

    assert len(result.regions) == 2
    assert {region.bbox for region in result.regions} == {
        (300, 200, 600, 200),
        (1200, 800, 300, 100),
    }
    assert all(
        len(region.bbox) == 4 and len(region.polygon) >= 3
        for region in result.regions
    )


def test_single_and_multiple_components(aligned_mask, region_config):
    config = replace(
        region_config, min_area=1, max_area=400_000, merge_distance=0
    )
    one = np.zeros_like(aligned_mask)
    one[100:130, 110:170] = 255

    assert len(extract_regions(one, config).regions) == 1
    assert len(extract_regions(aligned_mask, config).regions) == 2


def test_min_area_filtering_is_recorded(aligned_mask, region_config):
    config = replace(region_config, min_area=30_001, max_area=400_000, merge_distance=0)

    result = extract_regions(aligned_mask, config)

    assert len(result.regions) == 1
    assert result.regions[0].area == 120_000
    assert len(result.excluded_components) == 1
    assert result.excluded_components[0]["reason"] == "below_min_area"


def test_max_area_filtering_is_recorded(aligned_mask, region_config):
    config = replace(region_config, min_area=1, max_area=50_000, merge_distance=0)

    result = extract_regions(aligned_mask, config)

    assert len(result.regions) == 1
    assert result.regions[0].area == 30_000
    assert len(result.excluded_components) == 1
    assert result.excluded_components[0]["reason"] == "above_max_area"


def test_oversized_component_is_excluded_and_never_split(
    aligned_mask, region_config, image_id
):
    mask = np.zeros_like(aligned_mask)
    mask[20:200, 20:220] = 255
    config = replace(region_config, min_area=1, max_area=10_000, merge_distance=0)

    report = ErrorReport(run_id="oversized-test")
    result = extract_regions(mask, config, image_id=image_id, error_report=report)

    assert result.regions == ()
    assert len(result.excluded_components) == 1
    assert result.excluded_components[0]["reason"] == "above_max_area"
    assert result.excluded_components[0]["area"] == 36_000


def test_nearby_components_merge_using_configured_distance(aligned_mask, region_config):
    mask = np.zeros_like(aligned_mask)
    mask[300:330, 300:330] = 255
    mask[300:330, 335:365] = 255
    base = replace(region_config, min_area=1, max_area=10_000, merge_distance=0)

    separate = extract_regions(mask, base)
    merged = extract_regions(mask, replace(base, merge_distance=6))

    assert len(separate.regions) == 2
    assert len(merged.regions) == 1
    assert merged.regions[0].merged_from == 2
    assert merged.regions[0].bbox == (300, 300, 65, 30)


def test_border_crop_is_clipped_and_preserves_all_mask_pixels(
    aligned_mask, region_config
):
    mask = np.zeros_like(aligned_mask)
    mask[0:8, 0:9] = 255
    config = replace(
        region_config,
        min_area=1,
        max_area=100,
        merge_distance=0,
        crop_padding=5,
    )
    result = extract_regions(mask, config)
    region = result.regions[0]

    cropped = crop_region(mask, region)
    left, top, right, bottom = region.crop_box
    assert region.clipped is True
    assert region.crop_box == (0, 0, 14, 13)
    assert cropped.shape == (bottom - top, right - left)
    assert np.array_equal(cropped[:8, :9], mask[:8, :9])
    assert np.count_nonzero(cropped) == np.count_nonzero(mask)


def test_empty_mask_is_no_regions_and_records_empty_mask(
    aligned_mask, region_config, image_id
):
    report = ErrorReport(run_id="empty-mask-test")

    result = extract_regions(
        np.zeros_like(aligned_mask),
        region_config,
        image_id=image_id,
        error_report=report,
    )

    assert result.regions == ()
    assert result.status == "no_regions"
    assert report.counts["empty_mask"] == 1
    assert report.counts["no_text_region_detected"] == 0


def test_nonempty_mask_filtered_to_zero_is_recorded_without_crash(
    aligned_mask, region_config, image_id
):
    mask = np.zeros_like(aligned_mask)
    mask[400:402, 400:402] = 255
    report = ErrorReport(run_id="zero-regions-test")
    config = replace(region_config, min_area=5, max_area=10, merge_distance=0)

    result = extract_regions(
        mask, config, image_id=image_id, error_report=report
    )

    assert result.regions == ()
    assert result.status == "no_regions"
    assert report.counts["no_text_region_detected"] == 1
    assert report.counts["empty_mask"] == 0


def test_invalid_mask_is_rejected_without_silent_normalization(
    aligned_mask, region_config
):
    mask = aligned_mask.copy()
    mask[0, 0] = 1

    with pytest.raises(ValueError, match="binary convention"):
        extract_regions(mask, region_config)
