"""T058 end-to-end batch isolation and report-shape checks."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from manga_text_seg.config import load_translation_config
from manga_text_seg.manifest import Pair, load_manifest, save_manifest
from manga_text_seg.translation.pipeline import run_batch
from manga_text_seg.translation.runs import ERROR_CATEGORIES
from manga_text_seg.translation.translate import FixtureTranslationProvider

ROOT = Path(__file__).resolve().parents[2]


def _deny_file_writes(path: Path):
    """Deny writes to one file and prove the current process is denied too."""
    if os.name == "nt":
        identity = subprocess.run(
            ["whoami", "/user", "/fo", "csv", "/nh"],
            capture_output=True,
            text=True,
            check=False,
        )
        match = re.search(r"S-\d-\d+(?:-\d+)+", identity.stdout)
        if identity.returncode != 0 or match is None:
            return None
        sid = match.group(0)
        denied = subprocess.run(
            ["icacls", str(path), "/deny", f"*{sid}:(W)"],
            capture_output=True,
            text=True,
            check=False,
        )
        if denied.returncode != 0:
            return None

        def restore() -> None:
            subprocess.run(
                ["icacls", str(path), "/remove:d", f"*{sid}"],
                capture_output=True,
                text=True,
                check=False,
            )

    else:
        previous_mode = path.stat().st_mode
        path.chmod(previous_mode & ~0o222)

        def restore() -> None:
            path.chmod(previous_mode)

    try:
        with path.open("ab"):
            pass
    except PermissionError:
        return restore
    restore()
    return None


def _configured(bundle, tmp_path):
    base = load_translation_config(ROOT / "configs" / "translation.json")
    cache = tmp_path / "cache"
    provider = replace(
        base.provider,
        name="fixture",
        model="fixture",
        version="1",
        secret_env_var=None,
        cache=replace(base.provider.cache, root=cache),
    )
    return replace(
        base,
        manifest_path=bundle["manifest"],
        output_roots=replace(
            base.output_roots, translation=tmp_path / "outputs", cache=cache
        ),
        provider=provider,
    )


def test_page_and_region_failures_do_not_abort_manifest_batch(
    tmp_path, translation_bundle
):
    bundle = translation_bundle()
    config = _configured(bundle, tmp_path)

    # The second Spec 1 manifest entry is validly identified but its raw image is
    # missing. This induces a page rejection while retaining one accepted page.
    manifest = load_manifest(bundle["manifest"])
    first_id = manifest.pairs[0].image_id
    page_list = json.loads(
        (ROOT / "benchmark" / "page-list.json").read_text(encoding="utf-8")
    )
    second = next(page for page in page_list["pages"] if page["image_id"] != first_id)
    manifest.pairs.append(
        Pair(
            manga=second["manga"],
            stem=second["stem"],
            mask_path=bundle["root"] / "unused-prediction-mask.png",
            raw_path=bundle["root"] / "missing-raw.png",
            mask_encoding="magenta-black",
        )
    )
    save_manifest(manifest, bundle["manifest"])

    # Two components exercise per-region continuation: the recognizer fails only
    # the first call, while the next region still receives OCR/translation/render.
    mask = np.zeros((1170, 1654), dtype=np.uint8)
    mask[100:130, 100:150] = 255
    mask[220:250, 200:260] = 255
    assert cv2.imwrite(str(bundle["prediction_mask"]), mask)
    source_mask_bytes = bundle["prediction_mask"].read_bytes()
    calls = {"count": 0}

    def recognizer(_crop):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("fixture OCR failure")
        return "second region text"

    result = run_batch(
        config,
        "isolation-run",
        repo_root=ROOT,
        provider=FixtureTranslationProvider({"second region text": "translated"}),
        recognizer=recognizer,
        prediction_root=bundle["prediction_mask"].parents[2],
        inpainting_root=bundle["inpainted"].parents[3],
    )

    assert {page.image_id for page in result.pages} == {first_id, second["image_id"]}
    assert (
        result.run_record.page_statuses[first_id] == "success"
    ), result.error_report.to_dict()
    assert (
        result.run_record.page_statuses[second["image_id"]] == "missing_original_image"
    )
    assert result.run_record.page_list_identity == bundle["page_list_identity"]
    assert result.run_record.input_image_identity == {
        first_id: bundle["input_image_identity"],
        second["image_id"]: next(
            page["input_image_identity"]
            for page in page_list["pages"]
            if page["image_id"] == second["image_id"]
        ),
    }
    assert result.error_report.counts["ocr_failure"] == 1
    assert result.error_report.counts["missing_original_image"] == 1
    assert set(result.error_report.counts) == set(ERROR_CATEGORIES)
    assert all(
        category in result.error_report.categories for category in ERROR_CATEGORIES
    )
    page = next(page for page in result.pages if page.image_id == first_id)
    metadata = json.loads((page.page_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["regions"][0]["ocr_status"] == "failed"
    assert metadata["regions"][1]["ocr_status"] == "ok"
    assert (page.page_dir / "rendered.png").is_file()
    assert bundle["prediction_mask"].read_bytes() == source_mask_bytes
    errors = json.loads((result.run_dir / "errors.json").read_text(encoding="utf-8"))
    assert set(errors["counts"]) == set(ERROR_CATEGORIES)


@pytest.mark.parametrize("category", ERROR_CATEGORIES)
def test_each_error_category_is_induced_by_pipeline(
    category, tmp_path, translation_bundle, monkeypatch
):
    """Each report category is reached through its corresponding stage failure."""
    import manga_text_seg.translation.pipeline as pipeline_module
    import manga_text_seg.translation.render as render_module
    from manga_text_seg.manifest import Manifest, load_manifest, save_manifest

    page_list = json.loads(
        (ROOT / "benchmark" / "page-list.json").read_text(encoding="utf-8")
    )
    image_ids = sorted(page["image_id"] for page in page_list["pages"])[:2]
    if category == "output_write_failure":
        probe = tmp_path / "read-only-probe.txt"
        probe.write_text("probe", encoding="utf-8")
        restore_probe = _deny_file_writes(probe)
        if restore_probe is None:
            pytest.skip(
                "environment cannot enforce a genuine read-only file for this user"
            )
        restore_probe()

    options = {
        "missing_original_image": {"missing_original": True},
        "missing_prediction_mask": {"missing_prediction_mask": True},
        "missing_inpainted_image": {"missing_inpainted_image": True},
        "empty_mask": {"empty_mask": True},
    }.get(category, {})
    failed_bundle = translation_bundle(image_id=image_ids[0], **options)
    healthy_bundle = translation_bundle(image_id=image_ids[1])

    # Use two genuinely different connected components on the target page so
    # region-level isolation is observable independently of the healthy page.
    target_mask = np.zeros((1170, 1654), dtype=np.uint8)
    if category == "no_text_region_detected":
        target_mask[100:120, 100:120] = 255
    elif category != "empty_mask":
        target_mask[100:130, 100:150] = 255
        target_mask[220:250, 200:260] = 255
    if category not in {"missing_prediction_mask", "empty_mask"}:
        assert cv2.imwrite(str(failed_bundle["prediction_mask"]), target_mask)

    prediction_root = tmp_path / "spec2"
    inpainting_root = tmp_path / "spec3"
    pairs = []
    for bundle in (failed_bundle, healthy_bundle):
        manga, stem = bundle["image_id"].split("/")
        prediction_page = prediction_root / "masks" / manga
        metadata_page = prediction_root / "metadata" / manga
        prediction_page.mkdir(parents=True, exist_ok=True)
        metadata_page.mkdir(parents=True, exist_ok=True)
        if bundle["prediction_mask"].is_file():
            shutil.copyfile(bundle["prediction_mask"], prediction_page / f"{stem}.png")
        if bundle["mask_sidecar"].is_file():
            shutil.copyfile(bundle["mask_sidecar"], metadata_page / f"{stem}.json")
        method_root = inpainting_root / "classical_baseline" / "telea" / manga
        method_root.mkdir(parents=True, exist_ok=True)
        if bundle["inpainted"].is_file():
            shutil.copyfile(bundle["inpainted"], method_root / f"{stem}.png")
        if bundle["inpainted_sidecar"].is_file():
            sidecar = json.loads(
                bundle["inpainted_sidecar"].read_text(encoding="utf-8")
            )
            sidecar["prediction_mask_path"] = str(
                prediction_root / "masks" / manga / f"{stem}.png"
            )
            (method_root / f"{stem}.json").write_text(
                json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        pairs.extend(load_manifest(bundle["manifest"]).pairs)

    manifest = Manifest(
        run_id="error-matrix",
        gt_root=tmp_path / "groundtruth-never-read",
        raw_root=failed_bundle["root"] / "raw",
        pairs=sorted(pairs, key=lambda pair: (pair.manga, pair.stem)),
    )
    manifest_path = tmp_path / "error-matrix-manifest.json"
    save_manifest(manifest, manifest_path)
    config = _configured(failed_bundle, tmp_path)
    config = replace(config, manifest_path=manifest_path)
    if category == "no_text_region_detected":
        config = replace(
            config,
            region_extraction=replace(config.region_extraction, min_area=1000),
        )
    if category == "text_overflow":
        config = replace(
            config,
            rendering=replace(
                config.rendering,
                min_font_size=1,
                expansion_allowance=0,
            ),
        )
    if category == "api_timeout_or_rate_limit":
        config = replace(
            config,
            provider=replace(
                config.provider,
                retry=replace(config.provider.retry, max_attempts=3),
            ),
        )

    calls = {"ocr": 0, "provider": 0}
    provider_events = []

    def recognizer(_crop):
        calls["ocr"] += 1
        if category == "ocr_failure" and calls["ocr"] == 1:
            raise RuntimeError("controlled OCR backend failure")
        return f"source-{calls['ocr']}"

    class MatrixProvider:
        deterministic = True

        def translate(self, text, _language):
            calls["provider"] += 1
            if category == "translation_failure" and text == "source-1":
                raise RuntimeError("controlled provider failure")
            if category == "api_timeout_or_rate_limit" and text == "source-1":
                attempts = sum(event[0] == "source-1" for event in provider_events) + 1
                if attempts == 1:
                    provider_events.append((text, "timeout"))
                    raise TimeoutError("controlled provider timeout")
                if attempts == 2:
                    provider_events.append((text, "rate_limit"))
                    raise RateLimitedError("controlled provider rate limit")
                provider_events.append((text, "success"))
                return "recovered after timeout and rate limit"
            if category == "api_timeout_or_rate_limit":
                provider_events.append((text, "success"))
            if category == "text_overflow" and text == "source-1":
                return "translation " * 1000
            return "short translation"

    class RateLimitedError(RuntimeError):
        status_code = 429

    original_render = pipeline_module.run_rendering
    denied_write_errors = []

    def render_with_target_write_failure(*args, **kwargs):
        if (
            category == "output_write_failure"
            and kwargs.get("image_id") == image_ids[0]
        ):
            # Deny this file at the filesystem ACL/mode layer, then prove the
            # actual renderer receives an operating-system PermissionError.
            target = Path(args[3]) / "rendered.png"
            target.touch()
            restore_target = _deny_file_writes(target)
            if restore_target is None:
                pytest.skip(
                    "environment cannot enforce a genuine read-only output file"
                )
            try:
                return original_render(*args, **kwargs)
            except PermissionError as exc:
                denied_write_errors.append(exc)
                raise
            finally:
                restore_target()
        return original_render(*args, **kwargs)

    original_fits = render_module._fits
    font_failure = {"remaining": 1}

    def fits_with_one_font_open_failure(*args, **kwargs):
        if category == "font_missing" and font_failure["remaining"]:
            font_failure["remaining"] -= 1
            raise OSError("controlled font open failure")
        return original_fits(*args, **kwargs)

    monkeypatch.setattr(
        pipeline_module, "run_rendering", render_with_target_write_failure
    )
    monkeypatch.setattr(render_module, "_fits", fits_with_one_font_open_failure)
    provider = MatrixProvider()
    result = run_batch(
        config,
        f"error-{category}",
        repo_root=ROOT,
        provider=provider,
        recognizer=recognizer,
        prediction_root=prediction_root,
        inpainting_root=inpainting_root,
    )

    entries = result.error_report.categories[category]
    assert entries, result.error_report.to_dict()
    assert all(entry["reason"].strip() for entry in entries)
    assert all(entry["image_id"] == image_ids[0] for entry in entries)
    assert set(result.error_report.counts) == set(ERROR_CATEGORIES)
    assert result.run_record.page_statuses[image_ids[1]] == "success"
    target = next(page for page in result.pages if page.image_id == image_ids[0])
    if category in {
        "missing_original_image",
        "missing_prediction_mask",
        "missing_inpainted_image",
        "output_write_failure",
    }:
        assert target.status != "success"
    else:
        metadata = json.loads(
            (target.page_dir / "metadata.json").read_text(encoding="utf-8")
        )
        assert len(metadata["regions"]) <= 2
        if category == "ocr_failure":
            assert [item["ocr_status"] for item in metadata["regions"]] == [
                "failed",
                "ok",
            ]
        elif category == "translation_failure":
            assert [item["translation_status"] for item in metadata["regions"]] == [
                "failed",
                "ok",
            ]
        elif category == "font_missing":
            assert [item["render_status"] for item in metadata["regions"]] == [
                "skipped",
                "rendered",
            ]
        elif category == "text_overflow":
            assert [item["render_status"] for item in metadata["regions"]] == [
                "overflow_warning",
                "rendered",
            ]
        elif category == "api_timeout_or_rate_limit":
            assert any(
                region["ocr_status"] == "ok" and region["translation_status"] == "ok"
                for region in metadata["regions"]
            )
    healthy = next(page for page in result.pages if page.image_id == image_ids[1])
    assert (healthy.page_dir / "rendered.png").is_file()
    if category == "api_timeout_or_rate_limit":
        assert [event for event in provider_events if event[0] == "source-1"] == [
            ("source-1", "timeout"),
            ("source-1", "rate_limit"),
            ("source-1", "success"),
        ]
        assert [event for event in provider_events if event[0] == "source-2"] == [
            ("source-2", "success")
        ]
        retry_errors = result.error_report.categories["api_timeout_or_rate_limit"]
        assert len(retry_errors) == 2
        assert all(entry["region_id"] == 1 for entry in retry_errors)
        assert "controlled provider timeout" in retry_errors[0]["reason"]
        assert "controlled provider rate limit" in retry_errors[1]["reason"]
        assert not result.error_report.categories["translation_failure"]
        metadata = json.loads(
            (target.page_dir / "metadata.json").read_text(encoding="utf-8")
        )
        assert [item["translation_status"] for item in metadata["regions"]] == [
            "ok",
            "ok",
        ]
    elif category == "output_write_failure":
        assert len(denied_write_errors) == 1
        assert isinstance(denied_write_errors[0], PermissionError)
