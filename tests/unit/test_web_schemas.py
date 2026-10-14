"""Unit tests for the four models the demo's page route serializes.

These pin the parts a route could silently drift from: the closed FR-020
page-state vocabulary *and* the branch that picks from it, FR-024's per-region
fields, FR-018's provenance panel, FR-019's rule that an unrecorded value stays
null, and the fact that every model serializes to plain JSON — a filesystem
path in a response would break FR-013d and FR-028, which require run-relative
references.
"""

from __future__ import annotations

import json
from dataclasses import replace

from manga_text_seg.web import demo, errors, schemas


def test_fr020_page_states_are_exactly_the_eight_named_states() -> None:
    assert len(schemas.PAGE_STATES) == 8
    assert set(schemas.PAGE_STATES) == {
        "valid",
        "missing_prediction_mask",
        "missing_inpainting_output",
        "missing_ocr_translation_output",
        "no_text_regions",
        "region_ocr_failure",
        "region_translation_failure",
        "rendering_failure",
    }


def test_the_page_states_are_a_separate_vocabulary_from_the_fr079_categories() -> None:
    """FR-020's states describe a page; FR-079's categories describe a request.

    They share spellings (``missing_prediction_mask``) but not meanings: a page
    whose prediction mask is gone is a 200 with that state, never a 404
    envelope (FR-026). Keeping them as two tuples is what stops the two from
    being merged.
    """
    assert schemas.PAGE_STATES is not errors.ERROR_CATEGORIES
    assert set(schemas.PAGE_STATES) != set(errors.ERROR_CATEGORIES)


def test_every_state_the_route_can_return_is_in_the_fr020_vocabulary() -> None:
    """``_status`` is the only thing that picks a page's state.

    A literal it returns that PAGE_STATES does not carry would leave the
    frontend's state lookup empty and the banner blank, so the branch and the
    vocabulary are checked against each other rather than each on its own.
    """

    def region(**over: str) -> schemas.RegionView:
        return replace(
            schemas.RegionView(
                region_id=1,
                bbox=[0, 0, 1, 1],
                ocr_status="ok",
                translation_status="ok",
                render_status="ok",
            ),
            **over,
        )

    complete = schemas.PageArtifacts(
        original="o", prediction_mask="p", processed_mask="m", inpainted="i", rendered="r"
    )
    cases = [
        ("no_text_regions", schemas.PageArtifacts(), []),
        ("missing_prediction_mask", replace(complete, prediction_mask=None), [region()]),
        ("missing_inpainting_output", replace(complete, inpainted=None), [region()]),
        ("missing_ocr_translation_output", replace(complete, rendered=None), [region()]),
        ("region_ocr_failure", complete, [region(ocr_status="failed")]),
        ("region_translation_failure", complete, [region(translation_status="failed")]),
        ("rendering_failure", complete, [region(render_status="failed")]),
        ("valid", complete, [region()]),
    ]
    for expected, artifacts, regions in cases:
        assert demo._status(artifacts, regions) == expected
        assert expected in schemas.PAGE_STATES


def test_provenance_carries_every_fr018_field() -> None:
    assert set(schemas.Provenance(image_id="WebManga/001").to_dict()) == {
        "image_id",
        "segmentation_method",
        "spec2_run_id",
        "inpainting_run_id",
        "translation_run_id",
        "inpainting_algorithm",
        "target_language",
        "ocr_engine",
        "ocr_version",
        "translation_provider",
        "translation_model",
        "rendering_font",
        "rendering_config",
        "summary_provider",
        "summary_model",
        "summary_version",
    }


def test_unrecorded_provenance_values_stay_present_and_null() -> None:
    """FR-019: a value upstream never recorded is displayed as missing.

    The key must be present so the frontend can render "not recorded"; filling
    it in or omitting it would both read as a value the artifacts do not carry.
    """
    payload = schemas.Provenance(image_id="WebManga/001").to_dict()
    assert payload["spec2_run_id"] is None
    assert payload["summary_provider"] is None
    assert "spec2_run_id" in payload


def test_region_view_carries_the_fr024_fields() -> None:
    keys = set(schemas.RegionView(region_id=1, bbox=[0, 0, 1, 1]).to_dict())
    assert {
        "region_id",
        "bbox",
        "polygon",
        "reading_order_index",
        "ocr_text",
        "translation",
        "ocr_status",
        "translation_status",
        "render_status",
    } <= keys


def test_page_artifacts_cover_the_fr023_stages() -> None:
    assert set(schemas.PageArtifacts().to_dict()) == {
        "original",
        "prediction_mask",
        "processed_mask",
        "inpainted",
        "rendered",
    }


def test_a_page_bundle_serializes_to_plain_json() -> None:
    bundle = schemas.PageBundle(
        image_id="WebManga/001",
        status="valid",
        provenance=schemas.Provenance(
            image_id="WebManga/001", segmentation_method="unetpp_efficientnetv2"
        ),
        regions=[
            schemas.RegionView(
                region_id=1,
                bbox=[100, 100, 120, 60],
                reading_order_index=1,
                ocr_text="こんにちは",
                translation="Xin chào",
            )
        ],
        artifacts=schemas.PageArtifacts(
            original="raw/WebManga/001.jpg", rendered="rendered.png"
        ),
    )
    payload = bundle.to_dict()

    # Round-trips through JSON unchanged: no Path, no set, no dataclass left.
    assert json.loads(json.dumps(payload)) == payload
    assert payload["regions"][0]["region_id"] == 1
    assert payload["artifacts"]["prediction_mask"] is None