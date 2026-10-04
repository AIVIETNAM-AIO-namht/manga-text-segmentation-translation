"""T015–T019 intake tests using the valid-ID handoff fixture."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from manga_text_seg.config import TranslationOutputRoots, load_translation_config
from manga_text_seg.translation.intake import IntakeRejectionError, assemble_bundle
from manga_text_seg.translation.runs import ErrorReport, page_directory

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = load_translation_config(
    REPO_ROOT / "configs" / "translation.json", REPO_ROOT
)


def _assemble(tmp_path: Path, bundle: dict, *, run_id: str = "intake-test"):
    config = replace(
        BASE_CONFIG,
        manifest_path=bundle["manifest"],
        output_roots=TranslationOutputRoots(
            translation=tmp_path / "translation-output", cache=tmp_path / "cache"
        ),
    )
    report = ErrorReport(run_id)
    result = assemble_bundle(
        config,
        bundle["image_id"],
        run_id,
        error_report=report,
        repo_root=REPO_ROOT,
        prediction_root=bundle["prediction_mask"].parents[2],
        inpainting_root=bundle["inpainted"].parents[3],
    )
    return result, report, bundle["prediction_mask"].read_bytes()


def test_valid_bundle_is_joined_by_image_id_and_mask_is_copied(
    translation_bundle, tmp_path
) -> None:
    bundle = translation_bundle()
    manifest_before = hashlib.sha256(bundle["manifest"].read_bytes()).hexdigest()
    result, report, upstream_bytes = _assemble(tmp_path, bundle)

    assert result.image_id == bundle["image_id"]
    assert result.original_image_path == bundle["original"]
    assert result.prediction_mask_path == bundle["prediction_mask"]
    assert result.inpainted_image_path == bundle["inpainted"]
    assert result.status == "accepted"
    assert (
        result.mask_copy_path
        == page_directory(
            tmp_path / "translation-output",
            "intake-test",
            "classical_baseline",
            result.image_id,
        )
        / "mask.png"
    )
    assert result.mask_copy_path.read_bytes() == upstream_bytes
    assert not result.mask_copy_path.is_symlink()
    assert bundle["prediction_mask"].read_bytes() == upstream_bytes
    assert report.counts["missing_prediction_mask"] == 0
    assert bundle["manifest"].is_file()
    assert bundle["manifest"].parent.name == "benchmark"
    assert (
        hashlib.sha256(bundle["manifest"].read_bytes()).hexdigest() == manifest_before
    )


@pytest.mark.parametrize(
    ("perturbation", "category", "reason_part"),
    [
        (
            {"missing_original": True},
            "missing_original_image",
            "original image is missing",
        ),
        (
            {"missing_prediction_mask": True},
            "missing_prediction_mask",
            "prediction mask is missing",
        ),
        (
            {"missing_mask_sidecar": True},
            "missing_prediction_mask",
            "sidecar is missing",
        ),
        ({"mask_size": (100, 100)}, "missing_inpainted_image", "alignment"),
        (
            {"missing_inpainted_image": True},
            "missing_inpainted_image",
            "inpainted image is missing",
        ),
        (
            {"missing_inpainted_sidecar": True},
            "missing_inpainted_image",
            "metadata sidecar is missing",
        ),
        ({"inpainted_size": (100, 100)}, "missing_inpainted_image", "never resized"),
    ],
)
def test_invalid_bundle_is_rejected_and_recorded(
    translation_bundle, tmp_path, perturbation, category, reason_part
):
    bundle = translation_bundle(**perturbation)
    run_id = "rejected-test"
    config = replace(
        BASE_CONFIG,
        manifest_path=bundle["manifest"],
        output_roots=TranslationOutputRoots(tmp_path / "out", tmp_path / "cache"),
    )
    report = ErrorReport(run_id)
    with pytest.raises(IntakeRejectionError, match=reason_part) as caught:
        assemble_bundle(
            config,
            bundle["image_id"],
            run_id,
            error_report=report,
            repo_root=REPO_ROOT,
            prediction_root=bundle["prediction_mask"].parents[2],
            inpainting_root=bundle["inpainted"].parents[3],
        )
    assert caught.value.category == category
    assert report.counts[category] == 1
    assert report.categories[category][0]["image_id"] == bundle["image_id"]
    assert report.categories[category][0]["stage"] == "intake"
    assert report.categories[category][0]["reason"]


def test_empty_mask_is_recorded_as_page_outcome_not_silently_changed(
    translation_bundle, tmp_path
) -> None:
    bundle = translation_bundle(empty_mask=True)
    result, report, source_bytes = _assemble(tmp_path, bundle)

    assert result.status == "empty_mask"
    assert report.counts["empty_mask"] == 1
    assert (
        result.mask_copy_path.read_bytes()
        == source_bytes
        == bundle["prediction_mask"].read_bytes()
    )


def test_identity_mismatch_is_rejected_and_upstream_mask_is_unchanged(
    translation_bundle, tmp_path
) -> None:
    bundle = translation_bundle()
    before = bundle["prediction_mask"].read_bytes()
    sidecar_path = bundle["mask_sidecar"]
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["input_image_identity"] = "mismatched-identity"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    config = replace(
        BASE_CONFIG,
        manifest_path=bundle["manifest"],
        output_roots=TranslationOutputRoots(tmp_path / "out", tmp_path / "cache"),
    )
    report = ErrorReport("identity-test")

    with pytest.raises(IntakeRejectionError, match="input_image_identity"):
        assemble_bundle(
            config,
            bundle["image_id"],
            "identity-test",
            error_report=report,
            repo_root=REPO_ROOT,
            prediction_root=bundle["prediction_mask"].parents[2],
            inpainting_root=bundle["inpainted"].parents[3],
        )

    assert report.counts["missing_prediction_mask"] == 1
    assert bundle["prediction_mask"].read_bytes() == before


def test_unconfigured_method_is_rejected_and_reported(
    translation_bundle, tmp_path
) -> None:
    bundle = translation_bundle()
    config = replace(
        BASE_CONFIG,
        manifest_path=bundle["manifest"],
        segmentation_method="",
        output_roots=TranslationOutputRoots(tmp_path / "out", tmp_path / "cache"),
    )
    report = ErrorReport("method-test")
    with pytest.raises(
        IntakeRejectionError, match="no Spec 2 benchmark method-selection record"
    ):
        assemble_bundle(
            config,
            bundle["image_id"],
            "method-test",
            error_report=report,
            repo_root=REPO_ROOT,
            prediction_root=bundle["prediction_mask"].parents[2],
            inpainting_root=bundle["inpainted"].parents[3],
        )
    assert report.counts["missing_prediction_mask"] == 1
