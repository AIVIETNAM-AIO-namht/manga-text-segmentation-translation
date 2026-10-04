"""T082 real CLI coverage for fixture-backed quickstart intake and regions."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def _prepare_cli_workspace(tmp_path: Path, bundle: dict) -> tuple[Path, Path]:
    """Build Spec 004 fixture inputs under a disposable repo-shaped directory."""
    fixture_repo = tmp_path / "fixture-repo"
    (fixture_repo / "benchmark").mkdir(parents=True)
    (fixture_repo / "configs").mkdir()
    for name in ("page-list.json", "image-identity.json"):
        shutil.copyfile(ROOT / "benchmark" / name, fixture_repo / "benchmark" / name)

    manga, stem = bundle["image_id"].split("/")
    inpaint_dir = (
        fixture_repo
        / "outputs"
        / "inpainting"
        / "translation-cli-fixture"
        / "classical_baseline"
        / "telea"
        / manga
    )
    inpaint_dir.mkdir(parents=True)
    shutil.copyfile(bundle["inpainted"], inpaint_dir / f"{stem}.png")
    shutil.copyfile(bundle["inpainted_sidecar"], inpaint_dir / f"{stem}.json")

    config = json.loads((ROOT / "configs" / "translation.json").read_text())
    config.update(
        {
            "manifest_path": str(bundle["manifest"]),
            "segmentation_method": "classical_baseline",
            "inpainting_source": {
                "run_id": "translation-cli-fixture",
                "algorithm": "telea",
            },
            "provider": {
                **config["provider"],
                "name": "fixture",
                "model": "fixture-v1",
                "version": "1",
                "secret_env_var": None,
                "deterministic": True,
                "cache": {
                    **config["provider"]["cache"],
                    "enabled": False,
                    "root": str(fixture_repo / "cache" / "translation"),
                },
            },
            "rendering": {
                **config["rendering"],
                "font_path": str(ROOT / "assets" / "fonts" / "DejaVuSans.ttf"),
            },
            "output_roots": {
                "translation": str(fixture_repo / "outputs" / "translation"),
                "cache": str(fixture_repo / "cache" / "translation"),
            },
        }
    )
    config_path = fixture_repo / "configs" / "translation.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return fixture_repo, config_path


def _invoke_cli(config_path: Path, run_id: str, image_id: str, stage: str):
    return subprocess.run(
        [
            "uv",
            "run",
            "--locked",
            "--python",
            "3.12",
            "manga-text-seg",
            "translate",
            "--config",
            str(config_path),
            "--run",
            run_id,
            "--image-id",
            image_id,
            "--stage",
            stage,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_quickstart_scenarios_1_and_2_use_the_user_facing_cli(
    tmp_path: Path, translation_bundle, monkeypatch
) -> None:
    sentinel_secret = "round3-secret-sentinel-never-write-this-value"
    monkeypatch.setenv("ROUND3_TRANSLATION_SECRET_SENTINEL", sentinel_secret)
    bundle = translation_bundle()
    # Two disconnected prediction components make the selective region rerun
    # check meaningful; this is a synthetic prediction-mask perturbation.
    prediction = cv2.imread(str(bundle["prediction_mask"]), cv2.IMREAD_UNCHANGED)
    prediction[700:770, 1200:1280] = 255
    assert cv2.imwrite(str(bundle["prediction_mask"]), prediction)
    fixture_repo, config_path = _prepare_cli_workspace(tmp_path, bundle)
    run_id = "quickstart-cli"

    intake = _invoke_cli(config_path, run_id, bundle["image_id"], "intake")

    assert intake.returncode == 0, intake.stderr
    assert f"Intake '{run_id}' finished: 1 page(s)." in intake.stdout
    run_dir = fixture_repo / "outputs" / "translation" / run_id
    run_record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_record["run_id"] == run_id
    assert run_record["input_image_ids"] == [bundle["image_id"]]
    page_dir = (
        run_dir
        / run_record["segmentation_method"]
        / bundle["image_id"].split("/")[0]
        / bundle["image_id"].split("/")[1]
    )
    intake_metadata = json.loads(
        (page_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert intake_metadata["image_id"] == bundle["image_id"]
    assert intake_metadata["run_id"] == run_id
    assert (page_dir / "mask.png").read_bytes() == bundle[
        "prediction_mask"
    ].read_bytes()

    regions = _invoke_cli(config_path, run_id, bundle["image_id"], "regions")

    assert regions.returncode == 0, regions.stderr
    assert f"Re-ran regions for {bundle['image_id']}" in regions.stdout
    region_record = json.loads((page_dir / "regions.json").read_text(encoding="utf-8"))
    assert region_record["status"] == "ok"
    assert region_record["regions"]
    metadata = json.loads((page_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["image_id"] == bundle["image_id"]
    assert metadata["segmentation_method"] == "classical_baseline"
    assert (
        metadata["regions"][0]["region_id"] == region_record["regions"][0]["region_id"]
    )

    # Exercise the documented no-engine branch only when manga-ocr is absent;
    # this prevents an optional engine from initializing or downloading a model.
    if importlib.util.find_spec("manga_ocr") is None:
        ocr = _invoke_cli(config_path, run_id, bundle["image_id"], "ocr")
        assert ocr.returncode == 0, ocr.stderr
        ocr_record = json.loads((page_dir / "ocr.json").read_text(encoding="utf-8"))
        assert ocr_record["results"]
        assert all(item["status"] == "failed" for item in ocr_record["results"])
        assert all(item["text"] == "" for item in ocr_record["results"])
        assert all(
            "manga-ocr is not installed" in item["error"]
            for item in ocr_record["results"]
        )

        translated = _invoke_cli(config_path, run_id, bundle["image_id"], "translate")
        assert translated.returncode == 0, translated.stderr
        translations = json.loads(
            (page_dir / "translations.json").read_text(encoding="utf-8")
        )
        assert translations
        assert all(item["status"] in {"failed", "skipped"} for item in translations)
        assert all(item["translated_text"] == "" for item in translations)

        rendered = _invoke_cli(config_path, run_id, bundle["image_id"], "render")
        assert rendered.returncode == 0, rendered.stderr
        with Image.open(page_dir / "rendered.png") as image:
            assert image.size == (1654, 1170)
            assert image.format == "PNG"

        # The fixture has multiple components; re-running one OCR region must
        # preserve every sibling region's stored OCR record.
        sibling_before = ocr_record["results"][1]
        rerun = subprocess.run(
            [
                "uv",
                "run",
                "--locked",
                "--python",
                "3.12",
                "manga-text-seg",
                "translate",
                "--config",
                str(config_path),
                "--run",
                run_id,
                "--image-id",
                bundle["image_id"],
                "--stage",
                "ocr",
                "--region",
                str(ocr_record["results"][0]["region_id"]),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert rerun.returncode == 0, rerun.stderr
        after_rerun = json.loads((page_dir / "ocr.json").read_text(encoding="utf-8"))
        assert after_rerun["results"][1] == sibling_before

        # Quickstart scenario 9: two real CLI full runs use the deterministic
        # fixture provider and compare their complete artifact trees.
        from tests.integration.test_determinism import (
            _assert_tree_equal,
            _normalized_tree,
        )

        deterministic_a = _invoke_cli(
            config_path, "quickstart-det-a", bundle["image_id"], "full"
        )
        deterministic_b = _invoke_cli(
            config_path, "quickstart-det-b", bundle["image_id"], "full"
        )
        assert deterministic_a.returncode == 0, deterministic_a.stderr
        assert deterministic_b.returncode == 0, deterministic_b.stderr
        run_root = fixture_repo / "outputs" / "translation"
        _assert_tree_equal(
            _normalized_tree(run_root / "quickstart-det-a"),
            _normalized_tree(run_root / "quickstart-det-b"),
        )

        # Quickstart scenario 10's output-tree guard.
        for output_dir in (run_dir, run_root / "quickstart-det-a"):
            serialized = b"\n".join(
                path.read_bytes() for path in output_dir.rglob("*") if path.is_file()
            ).lower()
            assert sentinel_secret.encode("utf-8").lower() not in serialized
            assert b"groundtruth" not in serialized
            assert b"no-need-to-read" not in serialized
