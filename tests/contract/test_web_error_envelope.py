"""Contract tests for the FR-079 category-to-status mapping and the error envelope (T007).

The mapping below is transcribed from ``plan.md``'s "Error envelope" table, which
in turn enumerates FR-079's categories. It is written out here rather than
imported so that the test is an independent record of the contract: a change to
either side has to be made deliberately in both places.
"""

import json

import pytest

# status -> the FR-079 categories plan.md assigns to it.
EXPECTED_STATUS_BY_CATEGORY = {
    400: {"invalid_request"},
    404: {
        "missing_prediction_mask",
        "missing_metadata_sidecar",
        "missing_inpainting_output",
        "missing_ocr_output",
        "missing_translation_output",
        "undeterminable_run",
    },
    409: {
        "duplicate_run_id",
        "cross_run_mismatch",
        "undeterminable_segmentation_method",
    },
    422: {"page_without_text_regions"},
    429: {"api_timeout_rate_limit"},
    500: {
        "ocr_failure",
        "translation_failure",
        "rendering_failure",
        "summary_failure",
        "output_write_failure",
        "internal",
    },
    503: {"not_ready"},
}

# FR-082: every candidate must appear in detail.candidates.
AMBIGUITY_CATEGORIES = {"undeterminable_run", "undeterminable_segmentation_method"}

# FR-083: the expected location must appear in detail.expected.
MISSING_ARTIFACT_CATEGORIES = {
    "missing_prediction_mask",
    "missing_metadata_sidecar",
    "missing_inpainting_output",
    "missing_ocr_output",
    "missing_translation_output",
}


def _errors():
    """Import the module under test inside the test, not at module scope."""
    from manga_text_seg.web import errors

    return errors


def test_every_fr079_category_is_mapped_exactly_once():
    errors = _errors()
    mapped = {c for categories in EXPECTED_STATUS_BY_CATEGORY.values() for c in categories}

    assert set(errors.STATUS_BY_CATEGORY) == mapped
    assert set(errors.ERROR_CATEGORIES) == mapped


@pytest.mark.parametrize(
    ("status", "categories"),
    [(status, sorted(cats)) for status, cats in sorted(EXPECTED_STATUS_BY_CATEGORY.items())],
)
def test_category_maps_to_its_documented_status(status, categories):
    errors = _errors()

    for category in categories:
        assert errors.STATUS_BY_CATEGORY[category] == status


def test_unknown_category_is_rejected_rather_than_defaulted():
    """An unmapped category is a programming error, not a 500 to guess at."""
    errors = _errors()

    with pytest.raises(errors.UnknownCategoryError):
        errors.error_response("not_a_real_category", "message")


def test_envelope_shape():
    errors = _errors()
    response = errors.error_response("invalid_request", "bad parameter")

    assert response.status_code == 400
    assert json.loads(response.body) == {
        "error": {
            "category": "invalid_request",
            "message": "bad parameter",
            "detail": {},
        }
    }


def test_envelope_detail_is_passed_through():
    errors = _errors()
    response = errors.error_response(
        "invalid_request", "bad parameter", detail={"field": "lang"}
    )

    assert json.loads(response.body)["error"]["detail"] == {"field": "lang"}


def test_ambiguity_error_requires_candidates():
    """FR-082: an ambiguity error without its candidates is unrepresentable."""
    errors = _errors()

    with pytest.raises(ValueError):
        errors.error_response("undeterminable_run", "ambiguous")

    response = errors.error_response(
        "undeterminable_run", "ambiguous", detail={"candidates": ["run-a", "run-b"]}
    )
    assert json.loads(response.body)["error"]["detail"]["candidates"] == [
        "run-a",
        "run-b",
    ]


def test_missing_artifact_error_requires_expected_location():
    """FR-083: a missing-artifact error without detail.expected is unrepresentable."""
    errors = _errors()

    with pytest.raises(ValueError):
        errors.error_response("missing_ocr_output", "absent")

    response = errors.error_response(
        "missing_ocr_output", "absent", detail={"expected": "pages/001/ocr.json"}
    )
    assert (
        json.loads(response.body)["error"]["detail"]["expected"]
        == "pages/001/ocr.json"
    )
