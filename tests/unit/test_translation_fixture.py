"""T007 checks that the shared translation bundle writes real, valid-ID artifacts."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest


def test_translation_bundle_writes_manifest_and_all_sidecars(translation_bundle) -> None:
    bundle = translation_bundle("classical_baseline", "ARMS/000")
    assert bundle["image_id"] == "ARMS/000"
    assert bundle["manifest"].is_file()
    manifest = json.loads(bundle["manifest"].read_text(encoding="utf-8"))
    assert [pair["manga"] + "/" + pair["stem"] for pair in manifest["pairs"]] == [
        bundle["image_id"]
    ]
    assert bundle["original"].is_file()
    assert bundle["prediction_mask"].is_file()
    assert bundle["mask_sidecar"].is_file()
    assert bundle["inpainted"].is_file()
    assert bundle["inpainted_sidecar"].is_file()
    mask_sidecar = json.loads(bundle["mask_sidecar"].read_text(encoding="utf-8"))
    assert mask_sidecar["page_list_identity"] == bundle["page_list_identity"]
    assert mask_sidecar["input_image_identity"] == bundle["input_image_identity"]
    inpaint_sidecar = json.loads(bundle["inpainted_sidecar"].read_text(encoding="utf-8"))
    assert inpaint_sidecar["status"] == "ok"
    assert "page_list_identity" not in inpaint_sidecar
    assert "input_image_identity" not in inpaint_sidecar


@pytest.mark.parametrize(
    "option,path_key",
    [
        ("missing_original", "original"),
        ("missing_prediction_mask", "prediction_mask"),
        ("missing_mask_sidecar", "mask_sidecar"),
        ("missing_inpainted_image", "inpainted"),
        ("missing_inpainted_sidecar", "inpainted_sidecar"),
    ],
)
def test_translation_bundle_missing_artifact_perturbations(translation_bundle, option, path_key) -> None:
    bundle = translation_bundle(**{option: True})
    assert not bundle[path_key].exists()


def test_translation_bundle_can_produce_wrong_size_and_empty_mask(translation_bundle) -> None:
    wrong = translation_bundle(
        inpainted_size=(1653, 1170),
        original_pixel_value=180,
        prediction_text_value=128,
        inpainted_pixel_value=200,
    )
    original = cv2.imread(str(wrong["original"]))
    inpainted = cv2.imread(str(wrong["inpainted"]))
    mask = cv2.imread(str(wrong["prediction_mask"]), cv2.IMREAD_UNCHANGED)
    assert np.all(original == 180)
    assert inpainted.shape[1::-1] == (1653, 1170)
    assert np.all(inpainted == 200)
    assert 128 in np.unique(mask)

    empty = translation_bundle(empty_mask=True)
    mask = cv2.imread(str(empty["prediction_mask"]), cv2.IMREAD_UNCHANGED)
    assert isinstance(mask, np.ndarray)
    assert not np.any(mask)


def test_translation_bundle_rejects_an_unlisted_image_id(translation_bundle) -> None:
    with pytest.raises(ValueError, match="Spec 2 page list"):
        translation_bundle(image_id="ARMS/999")


def test_translation_bundle_accepts_method_positionally_and_rejects_invalid_method(
    translation_bundle,
) -> None:
    bundle = translation_bundle("manga_text_segmentation", "ARMS/000")
    sidecar = json.loads(bundle["mask_sidecar"].read_text(encoding="utf-8"))
    assert sidecar["method"] == "manga_text_segmentation"
    assert bundle["inpainted"].parts[-4] == "manga_text_segmentation"
    with pytest.raises(ValueError, match="segmentation identity"):
        translation_bundle("unregistered", "ARMS/000")
