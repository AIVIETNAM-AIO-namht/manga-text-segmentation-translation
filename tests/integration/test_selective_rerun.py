"""T071–T074 selective rerun isolation checks (fixture-only providers)."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from manga_text_seg.config import load_translation_config
from manga_text_seg.manifest import Manifest, load_manifest, save_manifest
from manga_text_seg.translation.pipeline import (
    export_intermediates,
    rerun_region,
    rerun_stage,
    run_batch,
)
from manga_text_seg.translation.translate import FixtureTranslationProvider

ROOT = Path(__file__).resolve().parents[2]


def _digest_tree(path: Path) -> dict[str, str]:
    return {
        item.relative_to(path).as_posix(): hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _copy_bundle_artifacts(bundle, prediction_root: Path, inpainting_root: Path):
    page = bundle["image_id"].split("/")
    manga, stem = page
    prediction_page = prediction_root / "masks" / manga
    metadata_page = prediction_root / "metadata" / manga
    prediction_page.mkdir(parents=True, exist_ok=True)
    metadata_page.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bundle["prediction_mask"], prediction_page / f"{stem}.png")
    shutil.copyfile(bundle["mask_sidecar"], metadata_page / f"{stem}.json")

    method_root = inpainting_root / "classical_baseline" / "telea" / manga
    method_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bundle["inpainted"], method_root / f"{stem}.png")
    sidecar = json.loads(bundle["inpainted_sidecar"].read_text(encoding="utf-8"))
    sidecar["prediction_mask_path"] = str(prediction_page / f"{stem}.png")
    (method_root / f"{stem}.json").write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _setup_run(tmp_path: Path, translation_bundle):
    page_list = json.loads(
        (ROOT / "benchmark" / "page-list.json").read_text(encoding="utf-8")
    )
    image_ids = sorted(page["image_id"] for page in page_list["pages"])[:2]
    first, second = (translation_bundle(image_id=image_id) for image_id in image_ids)

    # Two components in the first page create two independently addressable regions.
    mask = np.zeros((1170, 1654), dtype=np.uint8)
    mask[100:130, 100:150] = 255
    mask[220:250, 200:260] = 255
    assert cv2.imwrite(str(first["prediction_mask"]), mask)

    prediction_root = tmp_path / "spec2"
    inpainting_root = tmp_path / "spec3"
    _copy_bundle_artifacts(first, prediction_root, inpainting_root)
    _copy_bundle_artifacts(second, prediction_root, inpainting_root)

    pairs = (
        load_manifest(first["manifest"]).pairs + load_manifest(second["manifest"]).pairs
    )
    manifest = Manifest(
        run_id="selective-rerun-fixture",
        gt_root=Path("unused-groundtruth-never-read"),
        raw_root=first["root"] / "raw",
        pairs=sorted(pairs, key=lambda pair: (pair.manga, pair.stem)),
    )
    manifest_path = tmp_path / "combined-manifest.json"
    save_manifest(manifest, manifest_path)

    base = load_translation_config(ROOT / "configs" / "translation.json")
    cache = tmp_path / "cache"
    provider_config = replace(
        base.provider,
        name="fixture",
        model="fixture",
        version="1",
        secret_env_var=None,
        cache=replace(base.provider.cache, root=cache, enabled=False),
    )
    config = replace(
        base,
        manifest_path=manifest_path,
        output_roots=replace(
            base.output_roots, translation=tmp_path / "outputs", cache=cache
        ),
        provider=provider_config,
    )
    calls = {"count": 0}

    def recognizer(_crop):
        calls["count"] += 1
        return f"source-{calls['count']}"

    provider = FixtureTranslationProvider(
        {
            "source-1": RuntimeError("controlled region failure"),
            "source-2": "translated-2",
            "source-3": RuntimeError("controlled page-stage failure"),
        }
    )
    result = run_batch(
        config,
        "selective-run",
        repo_root=ROOT,
        provider=provider,
        recognizer=recognizer,
        prediction_root=prediction_root,
        inpainting_root=inpainting_root,
    )
    return result, image_ids, calls, config, prediction_root, inpainting_root


def test_region_and_page_reruns_reuse_stage_outputs_and_stay_scoped(
    tmp_path, translation_bundle
):
    result, image_ids, recognizer_calls, config, prediction_root, inpainting_root = (
        _setup_run(tmp_path, translation_bundle)
    )
    assert all(
        page.status == "success" for page in result.pages
    ), result.error_report.to_dict()
    first_page = next(page for page in result.pages if page.image_id == image_ids[0])
    second_page = next(page for page in result.pages if page.image_id == image_ids[1])
    translation_path = first_page.page_dir / "translations.json"
    records_before = json.loads(translation_path.read_text(encoding="utf-8"))
    failed = next(item for item in records_before if item["status"] == "failed")
    sibling = next(
        item for item in records_before if item["region_id"] != failed["region_id"]
    )
    sibling_bytes = {
        name: (first_page.page_dir / name).read_bytes()
        for name in ("crops/2.png", "ocr.json")
    }
    sibling_record = dict(sibling)
    second_page_before = _digest_tree(second_page.page_dir)

    def other_recognizer(_crop):
        recognizer_calls["count"] += 1
        return f"source-{recognizer_calls['count']}"

    other_run = run_batch(
        config,
        "untouched-run",
        repo_root=ROOT,
        provider=FixtureTranslationProvider(
            {f"source-{index}": f"untouched-{index}" for index in range(4, 7)}
        ),
        recognizer=other_recognizer,
        prediction_root=prediction_root,
        inpainting_root=inpainting_root,
    )
    other_run_before = _digest_tree(other_run.run_dir)
    ocr_call_count = recognizer_calls["count"]

    repaired = FixtureTranslationProvider({failed["source_text"]: "repaired region"})
    rerun_region(
        failed["region_id"],
        first_page.image_id,
        result.run_dir,
        provider=repaired,
    )

    after_region = json.loads(translation_path.read_text(encoding="utf-8"))
    updated_failed = next(
        item for item in after_region if item["region_id"] == failed["region_id"]
    )
    updated_sibling = next(
        item for item in after_region if item["region_id"] == sibling["region_id"]
    )
    assert updated_failed["status"] == "ok"
    assert updated_sibling == sibling_record
    assert repaired.calls == [(failed["source_text"], "vi")]
    assert {
        name: (first_page.page_dir / name).read_bytes() for name in sibling_bytes
    } == sibling_bytes
    assert _digest_tree(second_page.page_dir) == second_page_before
    assert _digest_tree(other_run.run_dir) == other_run_before
    assert recognizer_calls["count"] == ocr_call_count

    source_page_before = _digest_tree(second_page.page_dir)
    source_run_before_export = _digest_tree(result.run_dir)
    export_dir = export_intermediates(
        result.run_dir,
        second_page.image_id,
        tmp_path / "exports" / "selected-page",
    )
    assert _digest_tree(export_dir) == source_page_before
    exported_render = export_dir / "rendered.png"
    exported_render.write_bytes(b"changed export copy")
    assert (
        second_page.page_dir / "rendered.png"
    ).read_bytes() != b"changed export copy"
    assert _digest_tree(second_page.page_dir) == source_page_before
    assert _digest_tree(result.run_dir) == source_run_before_export

    # Page-level translation rerun uses persisted OCR, and leaves the other
    # page's complete artifact tree byte-identical.
    other_page_before = _digest_tree(first_page.page_dir)
    page_provider = FixtureTranslationProvider({"source-3": "page rerun"})
    rerun_stage(
        "translate", second_page.image_id, result.run_dir, provider=page_provider
    )
    assert page_provider.calls == [("source-3", "vi")]
    assert _digest_tree(first_page.page_dir) == other_page_before
    assert _digest_tree(second_page.page_dir) != second_page_before
    assert _digest_tree(other_run.run_dir) == other_run_before
    assert recognizer_calls["count"] == ocr_call_count


def test_failed_page_reruns_only_render_and_recovers(
    tmp_path, translation_bundle, monkeypatch
):
    import manga_text_seg.translation.pipeline as pipeline_module

    original_render = pipeline_module.run_rendering
    calls = {"failed": False}
    page_list = json.loads(
        (ROOT / "benchmark" / "page-list.json").read_text(encoding="utf-8")
    )
    ids = sorted(page["image_id"] for page in page_list["pages"])[:2]
    target_id = ids[0]

    def fail_first_page_render(*args, **kwargs):
        if kwargs.get("image_id") == target_id and not calls["failed"]:
            calls["failed"] = True
            raise OSError("controlled page-level rendering failure")
        return original_render(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "run_rendering", fail_first_page_render)
    result, image_ids, _, _, _, _ = _setup_run(tmp_path, translation_bundle)
    assert image_ids == ids
    target = next(page for page in result.pages if page.image_id == target_id)
    sibling_page = next(page for page in result.pages if page.image_id != target_id)
    assert target.status == "failed"
    assert result.run_record.page_statuses[target_id] == "failed"
    assert result.run_record.page_statuses[sibling_page.image_id] == "success"
    assert any(
        item["stage"] == "render" and "controlled page-level" in item["reason"]
        for item in result.error_report.categories["output_write_failure"]
    )
    assert (target.page_dir / "translations.json").is_file()
    assert not (target.page_dir / "rendered.png").exists()
    metadata_path = target.page_dir / "metadata.json"
    assert metadata_path.is_file(), result.error_report.categories[
        "output_write_failure"
    ]
    failed_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert failed_metadata["status"] == "failed"
    preserved_artifacts = {
        name: (target.page_dir / name).read_bytes()
        for name in (
            "original.png",
            "inpainted.png",
            "mask.png",
            "regions.json",
            "ocr.json",
            "translations.json",
        )
    }
    sibling_before = _digest_tree(sibling_page.page_dir)

    monkeypatch.setattr(pipeline_module, "run_rendering", original_render)
    rerun_stage("render", target_id, result.run_dir)

    recovered_metadata = json.loads(
        (target.page_dir / "metadata.json").read_text(encoding="utf-8")
    )
    updated_run = json.loads((result.run_dir / "run.json").read_text(encoding="utf-8"))
    updated_errors = json.loads(
        (result.run_dir / "errors.json").read_text(encoding="utf-8")
    )
    assert recovered_metadata["status"] == "ok"
    assert updated_run["page_statuses"][target_id] == "success"
    assert (target.page_dir / "rendered.png").is_file()
    assert updated_errors["counts"]["output_write_failure"] == 0
    assert {
        name: (target.page_dir / name).read_bytes() for name in preserved_artifacts
    } == preserved_artifacts
    assert _digest_tree(sibling_page.page_dir) == sibling_before


def test_region_reruns_are_isolated_across_ocr_translate_and_render(
    tmp_path, translation_bundle
):
    from PIL import Image

    result, image_ids, _, _, _, _ = _setup_run(tmp_path, translation_bundle)
    page = next(item for item in result.pages if item.image_id == image_ids[0])
    sibling_page = next(item for item in result.pages if item.image_id == image_ids[1])
    metadata_path = page.page_dir / "metadata.json"
    metadata_before = json.loads(metadata_path.read_text(encoding="utf-8"))
    sibling_region = metadata_before["regions"][1]
    sibling_id = sibling_region["region_id"]
    sibling_box = sibling_region["display_box"]
    assert sibling_box is not None
    x, y, width, height = sibling_box
    with Image.open(page.page_dir / "rendered.png").convert("RGB") as rendered:
        before_pixels = rendered.crop((x, y, x + width, y + height)).tobytes()
    sibling_crop = (page.page_dir / f"crops/{sibling_id}.png").read_bytes()
    translations_before = json.loads(
        (page.page_dir / "translations.json").read_text(encoding="utf-8")
    )
    sibling_translation = next(
        item for item in translations_before if item["region_id"] == sibling_id
    )
    sibling_run_before = _digest_tree(sibling_page.page_dir)

    rerun_region(
        1,
        page.image_id,
        result.run_dir,
        stage_name="ocr",
        recognizer=lambda _crop: "corrected OCR source",
    )
    after_ocr = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert after_ocr["regions"][1] == metadata_before["regions"][1]
    assert (page.page_dir / f"crops/{sibling_id}.png").read_bytes() == sibling_crop

    provider = FixtureTranslationProvider(
        {"corrected OCR source": "corrected translation"}
    )
    rerun_region(
        1, page.image_id, result.run_dir, stage_name="translate", provider=provider
    )
    assert provider.calls == [("corrected OCR source", "vi")]
    after_translate = json.loads(
        (page.page_dir / "translations.json").read_text(encoding="utf-8")
    )
    assert (
        next(item for item in after_translate if item["region_id"] == sibling_id)
        == sibling_translation
    )
    rerun_region(1, page.image_id, result.run_dir, stage_name="render")
    with Image.open(page.page_dir / "rendered.png").convert("RGB") as rendered:
        after_pixels = rendered.crop((x, y, x + width, y + height)).tobytes()
    assert after_pixels == before_pixels
    after_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert after_metadata["regions"][1] == metadata_before["regions"][1]
    assert _digest_tree(sibling_page.page_dir) == sibling_run_before
