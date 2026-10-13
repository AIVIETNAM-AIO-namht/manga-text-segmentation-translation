"""FR-079 error categories, their HTTP statuses, and the uniform error envelope.

Every non-2xx JSON response the web service produces goes through
:func:`error_response`, so the envelope shape and the category-to-status
mapping live in exactly one place. A page-level failure does not come through
here: ``/api/page`` returns 200 with the page's FR-020 status and records the
failure in the run's error report (FR-026).
"""

from typing import Any

from fastapi.responses import JSONResponse

# FR-079's categories, in the order the requirement lists them.
ERROR_CATEGORIES: tuple[str, ...] = (
    "missing_prediction_mask",
    "missing_metadata_sidecar",
    "missing_inpainting_output",
    "missing_ocr_output",
    "missing_translation_output",
    "undeterminable_segmentation_method",
    "undeterminable_run",
    "ocr_failure",
    "translation_failure",
    "rendering_failure",
    "summary_failure",
    "api_timeout_rate_limit",
    "output_write_failure",
    "page_without_text_regions",
    "duplicate_run_id",
    "cross_run_mismatch",
    "invalid_request",
    "not_ready",
    "internal",
)

# The plan.md "Error envelope" table, as a category -> status lookup.
STATUS_BY_CATEGORY: dict[str, int] = {
    "invalid_request": 400,
    "missing_prediction_mask": 404,
    "missing_metadata_sidecar": 404,
    "missing_inpainting_output": 404,
    "missing_ocr_output": 404,
    "missing_translation_output": 404,
    "undeterminable_run": 404,
    "duplicate_run_id": 409,
    "cross_run_mismatch": 409,
    "undeterminable_segmentation_method": 409,
    "page_without_text_regions": 422,
    "api_timeout_rate_limit": 429,
    "ocr_failure": 500,
    "translation_failure": 500,
    "rendering_failure": 500,
    "summary_failure": 500,
    "output_write_failure": 500,
    "internal": 500,
    "not_ready": 503,
}

# FR-082: these must name every candidate they found, never pick one.
_AMBIGUITY_CATEGORIES = frozenset(
    {"undeterminable_run", "undeterminable_segmentation_method"}
)

# FR-083: these must name where the artifact was expected.
_MISSING_ARTIFACT_CATEGORIES = frozenset(
    {
        "missing_prediction_mask",
        "missing_metadata_sidecar",
        "missing_inpainting_output",
        "missing_ocr_output",
        "missing_translation_output",
    }
)


class UnknownCategoryError(ValueError):
    """A category outside FR-079 was requested.

    This is a defect in the service, not a user error, so it is raised rather
    than mapped to a guessed status — a silently-wrong status would hide the
    bug behind a plausible response.
    """


def _require_detail(category: str, detail: dict[str, Any], key: str, fr: str) -> None:
    """Refuse an error that omits the detail its category is defined to carry."""
    if not detail.get(key):
        raise ValueError(
            f"'{category}' must carry a non-empty detail.{key} ({fr})"
        )


def error_response(
    category: str, message: str, detail: dict[str, Any] | None = None
) -> JSONResponse:
    """Build the uniform error envelope for one FR-079 category.

    ``detail`` is passed through as-is; the two categories whose requirements
    mandate specific keys are checked here so an error that would hide the
    information the caller needs cannot be constructed.
    """
    if category not in STATUS_BY_CATEGORY:
        raise UnknownCategoryError(f"unknown FR-079 category: '{category}'")

    payload = dict(detail or {})
    if category in _AMBIGUITY_CATEGORIES:
        _require_detail(category, payload, "candidates", "FR-082")
    if category in _MISSING_ARTIFACT_CATEGORIES:
        _require_detail(category, payload, "expected", "FR-083")

    return JSONResponse(
        status_code=STATUS_BY_CATEGORY[category],
        content={"error": {"category": category, "message": message, "detail": payload}},
    )
