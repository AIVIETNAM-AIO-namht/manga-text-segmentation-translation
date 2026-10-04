"""T014 - five sample inputs through intake (Spec 003, US1, SC-001).

One valid input and four that each break exactly one rule. Asserts that
exactly the valid one passes, that each rejection carries the right category
and a specific reason, and that no file on disk was created, changed or
removed by intake (FR-013: a rejected input is never silently corrected).

Written before ``intake.py`` exists, so it must FAIL until T015-T017 land.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from manga_text_seg.intake import AcceptedMask, intake_mask
from manga_text_seg.runs import ErrorReport, MaskRejection

METHOD = "classical_baseline"


def _snapshot(root: Path) -> dict[str, str]:
    """Relative path -> sha256 for every file under ``root``."""
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_five_inputs_through_intake(tmp_path, inpaint_fixture, synthetic_manifest):
    manifest = synthetic_manifest(
        image_ids=("ARMS/000", "ARMS/001", "ARMS/002", "ARMS/003"),
        missing_source=("ARMS/003",),
    )

    # (manga, stem) -> root holding that one mask, each with one thing wrong.
    # The valid mask comes last, so a rejection has already happened before it
    # is processed: a failure must not stop the samples after it (FR-011).
    inputs = {
        ("ARMS", "001"): inpaint_fixture(METHOD, "ARMS/001", size=(1000, 1000)),
        ("ARMS", "002"): inpaint_fixture(
            METHOD, "ARMS/002", pixel_values=(0, 128, 255)
        ),
        ("ARMS", "003"): inpaint_fixture(METHOD, "ARMS/003"),
        ("ARMS", "999"): inpaint_fixture(METHOD, "ARMS/999"),
        ("ARMS", "000"): inpaint_fixture(METHOD, "ARMS/000"),
    }

    before = _snapshot(tmp_path)

    report = ErrorReport()
    accepted: dict[str, AcceptedMask] = {}
    rejected: dict[str, MaskRejection] = {}
    for (manga, stem), root in inputs.items():
        try:
            result = intake_mask(root, manifest, manga, stem)
        except MaskRejection as rejection:
            rejected[f"{manga}/{stem}"] = rejection
            report.add_rejection(rejection, method=METHOD)
        else:
            accepted[result.image_id] = result

    assert set(accepted) == {"ARMS/000"}
    assert {k: r.category for k, r in rejected.items()} == {
        "ARMS/001": "wrong_size",
        "ARMS/002": "non_binary_mask",
        "ARMS/003": "missing_source_image",
        "ARMS/999": "image_id_mismatch",
    }
    for image_id, rejection in rejected.items():
        assert rejection.image_id == image_id
        assert rejection.reason, f"{image_id} was rejected without a reason"
    assert "1000" in rejected["ARMS/001"].reason
    assert "128" in rejected["ARMS/002"].reason

    assert _snapshot(tmp_path) == before, (
        "intake must not create, modify or remove any file, accepted or not"
    )


# ---------------------------------------------------------------------------
# Spec 004 / Person C / T014
#
# This section intentionally extends the existing Spec 003 integration tests
# because Spec 004 T014 requires this exact assignment path.
#
# Existing Spec 003 tests above are preserved unchanged.
# ---------------------------------------------------------------------------


def test_spec004_six_bundles_record_rejections_and_continue(
    tmp_path, translation_bundle
):
    """Exercise the real Spec 004 intake path for all six contract bundles."""
    from dataclasses import replace

    from manga_text_seg.config import (
        TranslationOutputRoots,
        load_translation_config,
    )
    from manga_text_seg.translation.intake import (
        IntakeRejectionError,
        assemble_bundle,
    )
    from manga_text_seg.translation.runs import ERROR_CATEGORIES, ErrorReport

    repo_root = Path(__file__).resolve().parents[2]
    base_config = load_translation_config(
        repo_root / "configs" / "translation.json", repo_root=repo_root
    )
    run_id = "t014-spec004"
    cases = {
        "missing_original": {
            "fixture_options": {"missing_original": True},
            "category": "missing_original_image",
            "reason": "original image is missing",
        },
        "missing_prediction_mask": {
            "fixture_options": {"missing_prediction_mask": True},
            "category": "missing_prediction_mask",
            "reason": "prediction mask is missing",
        },
        "missing_prediction_sidecar": {
            "fixture_options": {"missing_mask_sidecar": True},
            "category": "missing_prediction_mask",
            "reason": "prediction mask sidecar is missing",
        },
        "missing_inpainted_image": {
            "fixture_options": {"missing_inpainted_image": True},
            "category": "missing_inpainted_image",
            "reason": "inpainted image is missing",
        },
        "wrong_inpainted_size": {
            "fixture_options": {"inpainted_size": (100, 100)},
            "category": "missing_inpainted_image",
            "reason": "images are never resized",
        },
        "valid": {"fixture_options": {}, "category": None, "reason": None},
    }

    bundles = {
        name: translation_bundle(**case["fixture_options"])
        for name, case in cases.items()
    }
    upstream_before = {
        name: _snapshot(bundle["root"]) for name, bundle in bundles.items()
    }
    report = ErrorReport(run_id)
    outcomes = {}

    # The valid bundle is intentionally last: every prior rejection must be
    # recorded without preventing the remaining intake work from proceeding.
    for name, case in cases.items():
        bundle = bundles[name]
        config = replace(
            base_config,
            manifest_path=bundle["manifest"],
            output_roots=TranslationOutputRoots(
                translation=tmp_path / "translation-output",
                cache=tmp_path / "translation-cache",
            ),
        )
        try:
            result = assemble_bundle(
                config,
                bundle["image_id"],
                run_id,
                error_report=report,
                repo_root=repo_root,
                prediction_root=bundle["prediction_mask"].parents[2],
                inpainting_root=bundle["inpainted"].parents[3],
            )
        except IntakeRejectionError as rejection:
            outcomes[name] = {
                "accepted": False,
                "category": rejection.category,
                "reason": rejection.reason,
            }
        else:
            outcomes[name] = {"accepted": True, "bundle": result}

    for name, case in cases.items():
        if case["category"] is None:
            continue
        outcome = outcomes[name]
        assert outcome["accepted"] is False, f"{name} was silently accepted"
        assert outcome["category"] == case["category"]
        assert case["reason"] in outcome["reason"]
        matching_entries = [
            entry
            for entry in report.categories[case["category"]]
            if entry["image_id"] == bundles[name]["image_id"]
            and case["reason"] in entry["reason"]
        ]
        assert len(matching_entries) == 1
        assert matching_entries[0]["region_id"] is None
        assert matching_entries[0]["stage"] == "intake"

    expected_counts = {category: 0 for category in ERROR_CATEGORIES}
    for case in cases.values():
        if case["category"] is not None:
            expected_counts[case["category"]] += 1
    assert report.counts == expected_counts
    assert set(report.categories) == set(ERROR_CATEGORIES)

    valid_bundle = bundles["valid"]
    valid_outcome = outcomes["valid"]
    assert valid_outcome["accepted"] is True
    accepted = valid_outcome["bundle"]
    assert accepted.status == "accepted"
    assert accepted.image_id == valid_bundle["image_id"]
    assert accepted.original_image_path == valid_bundle["original"]
    assert accepted.prediction_mask_path == valid_bundle["prediction_mask"]
    assert accepted.inpainted_image_path == valid_bundle["inpainted"]
    assert accepted.mask_copy_path.is_file()
    assert (
        accepted.mask_copy_path.read_bytes()
        == valid_bundle["prediction_mask"].read_bytes()
    )

    # Intake may write its translation-run copy, but never mutates any upstream
    # input or rewrites/rebuilds the fixture's existing manifest.
    assert {
        name: _snapshot(bundle["root"]) for name, bundle in bundles.items()
    } == upstream_before
