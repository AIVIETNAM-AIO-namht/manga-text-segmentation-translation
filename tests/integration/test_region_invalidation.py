"""T036 integration coverage for region-set freshness and manual edit history."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from manga_text_seg.config import load_translation_config
from manga_text_seg.translation.ocr import apply_manual_edit, load_ocr_results
from manga_text_seg.translation.pipeline import rerun_stage, run_batch
from manga_text_seg.translation.translate import FixtureTranslationProvider

ROOT = Path(__file__).resolve().parents[2]


def _config(bundle, tmp_path):
    base = load_translation_config(ROOT / "configs" / "translation.json")
    cache = tmp_path / "cache"
    provider = replace(
        base.provider,
        name="fixture",
        model="fixture",
        version="1",
        secret_env_var=None,
        cache=replace(base.provider.cache, root=cache, enabled=False),
    )
    return replace(
        base,
        manifest_path=bundle["manifest"],
        output_roots=replace(
            base.output_roots, translation=tmp_path / "outputs", cache=cache
        ),
        provider=provider,
    )


def test_changed_region_invalidates_only_stale_ocr_and_refuses_downstream(
    tmp_path, translation_bundle
):
    bundle = translation_bundle()
    mask = np.zeros((1170, 1654), dtype=np.uint8)
    mask[100:130, 100:150] = 255
    mask[220:250, 200:260] = 255
    assert cv2.imwrite(str(bundle["prediction_mask"]), mask)
    calls = {"count": 0}

    def recognizer(_crop):
        calls["count"] += 1
        return "first OCR" if calls["count"] % 2 else "second OCR"

    config = _config(bundle, tmp_path)
    result = run_batch(
        config,
        "region-invalidation",
        repo_root=ROOT,
        provider=FixtureTranslationProvider(
            {"first OCR": "first translation", "second OCR": "second translation"}
        ),
        recognizer=recognizer,
        prediction_root=bundle["prediction_mask"].parents[2],
        inpainting_root=bundle["inpainted"].parents[3],
    )
    page = result.pages[0]
    metadata_path = page.page_dir / "metadata.json"
    before = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert len(before["regions"]) == 2
    changed_id = next(
        item["region_id"] for item in before["regions"] if item["bbox"][1] < 180
    )
    unchanged_id = next(
        item["region_id"] for item in before["regions"] if item["bbox"][1] >= 180
    )

    ocr_path = page.page_dir / "ocr.json"
    apply_manual_edit(
        ocr_path, changed_id, "human correction", edited_at="2026-01-01T00:00:00Z"
    )
    # An unchanged extraction preserves the active edit.
    rerun_stage("ocr", page.image_id, result.run_dir, recognizer=recognizer)
    same_set_results = load_ocr_results(ocr_path)
    same_set_by_id = {item.region_id: item for item in same_set_results}
    assert same_set_by_id[changed_id].text == "human correction"
    assert same_set_by_id[changed_id].manually_edited is True
    before_change = json.loads(metadata_path.read_text(encoding="utf-8"))

    # Move only region 1 without changing its reading-order identity. Region 2
    # keeps the same geometry and therefore remains current.
    changed = mask.copy()
    changed[100:130, 100:150] = 0
    changed[105:135, 100:150] = 255
    assert cv2.imwrite(str(bundle["prediction_mask"]), changed)
    shutil.copyfile(bundle["prediction_mask"], page.page_dir / "mask.png")
    rerun_stage("regions", page.image_id, result.run_dir)

    after_regions = json.loads(metadata_path.read_text(encoding="utf-8"))
    after_ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
    after_by_id = {item["region_id"]: item for item in after_regions["regions"]}
    before_by_id = {item["region_id"]: item for item in before_change["regions"]}
    assert after_by_id[changed_id]["translation_status"] == "skipped"
    assert after_by_id[changed_id]["render_status"] == "skipped"
    assert after_by_id[unchanged_id] == before_by_id[unchanged_id]
    assert after_ocr["stale_region_ids"] == [changed_id]
    edit = next(
        item for item in after_ocr["manual_edits"] if item["region_id"] == changed_id
    )
    assert edit["edited_text"] == "human correction"
    assert edit["invalidated_by_reextraction"] is True
    assert [item.region_id for item in load_ocr_results(ocr_path)] == [unchanged_id]

    translation_bytes = (page.page_dir / "translations.json").read_bytes()
    rendered_bytes = (page.page_dir / "rendered.png").read_bytes()
    provider = FixtureTranslationProvider({})
    with pytest.raises(ValueError, match="stale.*region set|stale regions"):
        rerun_stage("translate", page.image_id, result.run_dir, provider=provider)
    assert provider.calls == []
    assert (page.page_dir / "translations.json").read_bytes() == translation_bytes
    with pytest.raises(ValueError, match="stale.*region set|stale regions"):
        rerun_stage("render", page.image_id, result.run_dir)
    assert (page.page_dir / "rendered.png").read_bytes() == rendered_bytes

    # Full OCR re-extraction clears the stale marker, retains the audit history,
    # and allows downstream stages to proceed from current region geometry.
    rerun_stage("ocr", page.image_id, result.run_dir, recognizer=recognizer)
    refreshed = json.loads(ocr_path.read_text(encoding="utf-8"))
    assert refreshed["stale_region_ids"] == []
    edit = next(
        item for item in refreshed["manual_edits"] if item["region_id"] == changed_id
    )
    assert edit["invalidated_by_reextraction"] is True
    refreshed_by_id = {item.region_id: item for item in load_ocr_results(ocr_path)}
    assert refreshed_by_id[changed_id].text != "human correction"
