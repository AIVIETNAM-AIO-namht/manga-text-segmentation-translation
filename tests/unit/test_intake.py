"""T012 - unit tests for mask intake (Spec 003, US1).

Contract: contracts/intake-validation.md (FR-008 - FR-015). Written before
``intake.py`` exists, so every test here must FAIL until T015-T017 land.

Masks and sidecars are built by ``inpaint_fixture``; pages by
``synthetic_manifest``. Neither needs the real dataset, and the manifest's
ground-truth paths point at files that do not exist, so a passing test also
proves intake never reads ground truth (FR-033).

Intake is a pure reader: a rejection is raised as ``MaskRejection`` and the
caller records it. Nothing is written here.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from manga_text_seg.intake import AcceptedMask, intake_mask, source_available
from manga_text_seg.runs import (
    ERROR_CATEGORIES,
    AlignmentRecord,
    MaskRejection,
    RunError,
)

METHOD = "classical_baseline"
GT_CANVAS_SHAPE = (1176, 1656)  # numpy (height, width), the Spec 001 padded canvas


def _reject(root, manifest, manga, stem) -> MaskRejection:
    with pytest.raises(MaskRejection) as exc:
        intake_mask(root, manifest, manga, stem)
    return exc.value


# --------------------------------------------------------------------------- #
# MaskRejection
# --------------------------------------------------------------------------- #
def test_mask_rejection_carries_category_reason_and_image_id():
    rejection = MaskRejection("wrong_size", "1000x1000 is not a page size", "ARMS/000")

    assert rejection.category == "wrong_size"
    assert rejection.reason == "1000x1000 is not a page size"
    assert rejection.image_id == "ARMS/000"


def test_mask_rejection_refuses_a_category_outside_the_error_report():
    with pytest.raises(RunError):
        MaskRejection("made_up_category", "x", "ARMS/000")


# --------------------------------------------------------------------------- #
# Accepted input
# --------------------------------------------------------------------------- #
def test_accepts_a_valid_mask_at_the_aligned_size(inpaint_fixture, synthetic_manifest):
    manifest = synthetic_manifest()
    root = inpaint_fixture(METHOD, "ARMS/000")

    accepted = intake_mask(root, manifest, "ARMS", "000")

    assert isinstance(accepted, AcceptedMask)
    assert accepted.image_id == "ARMS/000"
    assert accepted.mask_path == root / "masks" / "ARMS" / "000.png"
    assert accepted.sidecar_path == root / "metadata" / "ARMS" / "000.json"
    assert accepted.page_path == manifest.pairs[0].raw_path
    assert accepted.raw_size == (1654, 1170)
    assert accepted.alignment == AlignmentRecord(
        rule="pass-through", pre_size=(1654, 1170), post_size=(1654, 1170)
    )
    on_disk = cv2.imread(str(accepted.mask_path), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(accepted.mask, on_disk)
    assert accepted.mask.dtype == np.uint8
    assert set(np.unique(accepted.mask)) <= {0, 255}


def test_accepts_the_gt_canvas_size_by_cropping_top_left(
    inpaint_fixture, synthetic_manifest
):
    manifest = synthetic_manifest()
    root = inpaint_fixture(METHOD, "ARMS/000", size=GT_CANVAS_SHAPE)

    accepted = intake_mask(root, manifest, "ARMS", "000")

    assert accepted.raw_size == (1656, 1176)
    assert accepted.mask.shape == (1170, 1654)
    assert accepted.alignment == AlignmentRecord(
        rule="crop-topleft", pre_size=(1656, 1176), post_size=(1654, 1170)
    )


# --------------------------------------------------------------------------- #
# Check 3: mapping by exact name only (FR-008, SC-001)
# --------------------------------------------------------------------------- #
def test_zero_match_is_an_image_id_mismatch(inpaint_fixture, synthetic_manifest):
    manifest = synthetic_manifest()
    root = inpaint_fixture(METHOD, "ARMS/999")

    rejection = _reject(root, manifest, "ARMS", "999")

    assert rejection.category == "image_id_mismatch"
    assert rejection.image_id == "ARMS/999"
    assert "ambiguous" not in rejection.reason.lower()


def test_multi_match_is_an_ambiguous_mapping(inpaint_fixture, synthetic_manifest):
    manifest = synthetic_manifest(duplicate=("ARMS/000",))
    root = inpaint_fixture(METHOD, "ARMS/000")

    rejection = _reject(root, manifest, "ARMS", "000")

    assert rejection.category == "image_id_mismatch"
    assert "ambiguous" in rejection.reason.lower()


@pytest.mark.parametrize(
    ("manga", "stem"),
    [("ARMS", "0"), ("ARMS", "00"), ("arms", "000"), ("ARM", "000")],
    ids=["unpadded", "half-padded", "wrong-case", "near-miss-name"],
)
def test_near_misses_are_never_matched_by_similarity(
    inpaint_fixture, synthetic_manifest, manga, stem
):
    """ARMS/000 is in the manifest. None of these names is it, and none may be
    renumbered, case-folded or fuzzy-matched onto it."""
    manifest = synthetic_manifest()
    root = inpaint_fixture(METHOD, f"{manga}/{stem}")

    rejection = _reject(root, manifest, manga, stem)

    assert rejection.category == "image_id_mismatch"
    assert rejection.image_id == f"{manga}/{stem}"


# --------------------------------------------------------------------------- #
# Checks 1 and 2: source page and mask exist and decode
# --------------------------------------------------------------------------- #
def test_missing_source_image(inpaint_fixture, synthetic_manifest):
    manifest = synthetic_manifest(missing_source=("ARMS/001",))
    root = inpaint_fixture(METHOD, "ARMS/001")

    assert _reject(root, manifest, "ARMS", "001").category == "missing_source_image"


def test_corrupt_source_image(inpaint_fixture, synthetic_manifest):
    manifest = synthetic_manifest(corrupt_source=("ARMS/001",))
    root = inpaint_fixture(METHOD, "ARMS/001")

    assert _reject(root, manifest, "ARMS", "001").category == "corrupt_image"


def test_missing_prediction_mask(inpaint_fixture, synthetic_manifest):
    manifest = synthetic_manifest()
    root = inpaint_fixture(METHOD, "ARMS/000")  # only ARMS/000 has a mask

    rejection = _reject(root, manifest, "ARMS", "001")

    assert rejection.category == "missing_prediction_mask"
    assert rejection.image_id == "ARMS/001"


def test_corrupt_prediction_mask(inpaint_fixture, synthetic_manifest):
    manifest = synthetic_manifest()
    root = inpaint_fixture(METHOD, "ARMS/000")
    (root / "masks" / "ARMS" / "000.png").write_bytes(b"this is not a png")

    assert _reject(root, manifest, "ARMS", "000").category == "corrupt_image"


# --------------------------------------------------------------------------- #
# Checks 4-6: size, values, emptiness
# --------------------------------------------------------------------------- #
def test_wrong_size(inpaint_fixture, synthetic_manifest):
    manifest = synthetic_manifest()
    root = inpaint_fixture(METHOD, "ARMS/000", size=(1000, 1000))

    assert _reject(root, manifest, "ARMS", "000").category == "wrong_size"


def test_non_binary_mask(inpaint_fixture, synthetic_manifest):
    manifest = synthetic_manifest()
    root = inpaint_fixture(METHOD, "ARMS/000", pixel_values=(0, 128, 255))

    rejection = _reject(root, manifest, "ARMS", "000")

    assert rejection.category == "non_binary_mask"
    assert "128" in rejection.reason


def test_empty_mask(inpaint_fixture, synthetic_manifest):
    manifest = synthetic_manifest()
    root = inpaint_fixture(METHOD, "ARMS/000", empty=True)

    assert _reject(root, manifest, "ARMS", "000").category == "empty_mask"


# --------------------------------------------------------------------------- #
# Check 7: the sidecar (FR-009)
# --------------------------------------------------------------------------- #
def test_missing_sidecar_is_missing_metadata(inpaint_fixture, synthetic_manifest):
    manifest = synthetic_manifest()
    root = inpaint_fixture(METHOD, "ARMS/000", with_sidecar=False)

    assert _reject(root, manifest, "ARMS", "000").category == "missing_metadata"


def test_a_sidecar_is_never_joined_from_another_root(
    inpaint_fixture, synthetic_manifest
):
    """A valid sidecar for the same image_id exists in another root. It must
    not be used to rescue a mask whose own root lacks one."""
    manifest = synthetic_manifest()
    without = inpaint_fixture(METHOD, "ARMS/000", with_sidecar=False)
    inpaint_fixture(METHOD, "ARMS/000")  # another root, with its sidecar

    assert _reject(without, manifest, "ARMS", "000").category == "missing_metadata"


def test_sidecar_that_does_not_parse_is_missing_metadata(
    inpaint_fixture, synthetic_manifest
):
    manifest = synthetic_manifest()
    root = inpaint_fixture(METHOD, "ARMS/000")
    (root / "metadata" / "ARMS" / "000.json").write_text("{not json", encoding="utf-8")

    assert _reject(root, manifest, "ARMS", "000").category == "missing_metadata"


def test_sidecar_naming_a_different_image_is_an_id_mismatch(
    inpaint_fixture, synthetic_manifest
):
    manifest = synthetic_manifest()
    root = inpaint_fixture(METHOD, "ARMS/000")
    sidecar_path = root / "metadata" / "ARMS" / "000.json"
    doc = json.loads(sidecar_path.read_text(encoding="utf-8"))
    doc["image_id"] = "ARMS/001"
    sidecar_path.write_text(json.dumps(doc), encoding="utf-8")

    rejection = _reject(root, manifest, "ARMS", "000")

    assert rejection.category == "image_id_mismatch"
    assert "ARMS/000" in rejection.reason and "ARMS/001" in rejection.reason


# --------------------------------------------------------------------------- #
# Every rejection is a category the ErrorReport accepts
# --------------------------------------------------------------------------- #
def test_every_rejection_category_is_reportable(inpaint_fixture, synthetic_manifest):
    manifest = synthetic_manifest(missing_source=("ARMS/001",))
    cases = [
        inpaint_fixture(METHOD, "ARMS/999"),
        inpaint_fixture(METHOD, "ARMS/000", size=(1000, 1000)),
        inpaint_fixture(METHOD, "ARMS/000", pixel_values=(0, 128, 255)),
        inpaint_fixture(METHOD, "ARMS/000", empty=True),
        inpaint_fixture(METHOD, "ARMS/000", with_sidecar=False),
    ]
    names = [("ARMS", "999")] + [("ARMS", "000")] * 4

    for root, (manga, stem) in zip(cases, names):
        assert _reject(root, manifest, manga, stem).category in ERROR_CATEGORIES


# --------------------------------------------------------------------------- #
# source_available: an identity with no admitted masks (Method A, Path 1)
# --------------------------------------------------------------------------- #
def test_source_available_when_masks_exist(inpaint_fixture):
    root = inpaint_fixture(METHOD, "ARMS/000")

    assert source_available(root) is True


def test_source_unavailable_when_the_root_does_not_exist(tmp_path):
    assert source_available(tmp_path / "<PENDING_REISSUE>") is False


def test_source_unavailable_when_masks_dir_is_empty(tmp_path):
    (tmp_path / "masks").mkdir()

    assert source_available(tmp_path) is False
