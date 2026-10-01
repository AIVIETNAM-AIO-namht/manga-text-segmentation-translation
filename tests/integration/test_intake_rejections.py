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
