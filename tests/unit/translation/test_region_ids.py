"""T021 determinism, reading order, merge, and orientation checks."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from manga_text_seg.config import load_translation_config
from manga_text_seg.translation.regions import extract_regions


def _config():
    path = Path(__file__).resolve().parents[3] / "configs" / "translation.json"
    return load_translation_config(path).region_extraction


def test_region_ids_and_geometry_are_identical_across_runs(aligned_mask):
    config = replace(_config(), min_area=1, max_area=400_000, merge_distance=0)

    first = extract_regions(aligned_mask, config)
    second = extract_regions(aligned_mask.copy(), config)

    assert first.regions == second.regions
    assert [region.region_id for region in first.regions] == [1, 2]
    assert [region.reading_order_index for region in first.regions] == [1, 2]
    # Manga order is right-to-left; config controls western left-to-right.
    assert first.regions[0].bbox[0] > first.regions[1].bbox[0]
    western = extract_regions(aligned_mask, replace(config, reading_order="western"))
    assert western.regions[0].bbox[0] < western.regions[1].bbox[0]


def test_merge_distance_zero_disables_merge_and_nonzero_records_component_count(
    aligned_mask,
):
    mask = np.zeros_like(aligned_mask)
    mask[300:340, 500:540] = 255
    mask[300:340, 545:585] = 255
    config = replace(_config(), min_area=1, max_area=10_000, merge_distance=0)

    unmerged = extract_regions(mask, config)
    merged = extract_regions(mask, replace(config, merge_distance=6))

    assert [region.merged_from for region in unmerged.regions] == [1, 1]
    assert len(merged.regions) == 1
    assert merged.regions[0].merged_from == 2


def test_orientation_is_determined_from_region_geometry(aligned_mask):
    mask = np.zeros_like(aligned_mask)
    mask[100:120, 100:180] = 255
    mask[200:280, 200:220] = 255
    config = replace(_config(), min_area=1, max_area=10_000, merge_distance=0)

    result = extract_regions(mask, config)

    by_bbox = {region.bbox: region.orientation for region in result.regions}
    assert by_bbox[(100, 100, 80, 20)] == "horizontal"
    assert by_bbox[(200, 200, 20, 80)] == "vertical"
