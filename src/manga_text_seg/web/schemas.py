"""The four models the demo's ``/api/demo/page`` route serializes.

These are the wire contract for :mod:`manga_text_seg.web.demo`, and nothing
else: the demo has one payload shape, so this module holds one payload shape.
Two rules run through it:

* **Run-relative references only.** Every artifact field is a ``str`` such as
  ``"rendered.png"`` or ``"raw/WebManga/001.jpg"``, never a ``Path`` and never
  an absolute filesystem path. A response carrying a real path would defeat
  FR-013d (``/api/artifact`` is the only byte-serving route). Keeping the
  models path-free is also what lets every one of them round-trip through
  ``json.dumps``.
* **A missing value is present and null, never absent.** FR-019 requires an
  unrecorded provenance value to be *displayed* as missing; the frontend can
  only do that if the key is there with a ``None``.

The FR-079 error categories are deliberately *not* re-spelled here: they live
in :mod:`manga_text_seg.web.errors`, where :func:`error_response` is the only
consumer. FR-020's page states look similar but mean something else — a page
state describes a successfully returned page (a 200), an FR-079 category
describes a refused request. ``missing_prediction_mask`` is a spelling both
vocabularies use; merging them would turn a reported page into a 404.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "PAGE_STATES",
    "PageArtifacts",
    "PageBundle",
    "Provenance",
    "RegionView",
]


class _Serializable:
    """Gives a frozen dataclass a JSON-ready ``to_dict``.

    Every model here is a pure value object — no ``Path`` to stringify, no key
    to rename — so ``asdict`` is the whole conversion, including for nested
    models. The house style elsewhere hand-writes ``to_dict`` where a path
    needs rewriting; nothing in this module does.
    """

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[arg-type]


#: FR-020's eight single-page states, as the API spells them. A page in any of
#: these states is a successful ``/api/demo/page`` response (FR-026); the demo
#: computes one with :func:`~manga_text_seg.web.demo._status`, and the frontend
#: switches on it to pick the banner. Only request-level failures produce an
#: error envelope, through ``errors.error_response``.
PAGE_STATES: tuple[str, ...] = (
    "valid",
    "missing_prediction_mask",
    "missing_inpainting_output",
    "missing_ocr_translation_output",
    "no_text_regions",
    "region_ocr_failure",
    "region_translation_failure",
    "rendering_failure",
)


@dataclass(frozen=True)
class Provenance(_Serializable):
    """FR-018's provenance panel.

    Every field but ``image_id`` is optional because FR-019 forbids filling in
    a value the upstream artifacts never recorded; ``None`` is how "not
    recorded" reaches the frontend. Three are always ``None`` in practice
    because the Spec 004 metadata schema is closed and carries no summary
    fields — the panel shows them as not recorded rather than the backend
    guessing.
    """

    image_id: str
    segmentation_method: str | None = None
    spec2_run_id: str | None = None
    inpainting_run_id: str | None = None
    translation_run_id: str | None = None
    inpainting_algorithm: str | None = None
    target_language: str | None = None
    ocr_engine: str | None = None
    ocr_version: str | None = None
    translation_provider: str | None = None
    translation_model: str | None = None
    rendering_font: str | None = None
    rendering_config: dict[str, Any] | None = None
    summary_provider: str | None = None
    summary_model: str | None = None
    summary_version: str | None = None


@dataclass(frozen=True)
class RegionView(_Serializable):
    """One region as FR-024 requires it: identifiers, text and statuses.

    ``bbox`` is ``[x, y, w, h]`` in aligned page space, copied verbatim from
    the metadata — a list, not a tuple, so the payload is already JSON.

    ``polygon`` rides along because FR-023's overlay cannot be drawn from
    ``bbox`` alone: on the recorded pages some regions have a degenerate box
    (a 1-pixel-tall line) whose polygon is the real shape. The frontend strokes
    the polygon and falls back to the box only when there is none.
    """

    region_id: int
    bbox: list[int]
    polygon: list[list[int]] = field(default_factory=list)
    reading_order_index: int | None = None
    ocr_text: str | None = None
    translation: str | None = None
    ocr_status: str | None = None
    translation_status: str | None = None
    render_status: str | None = None
    ocr_error: str | None = None
    translation_error: str | None = None
    render_warning: str | None = None


@dataclass(frozen=True)
class PageArtifacts(_Serializable):
    """FR-023's artifact references for one page.

    All optional: which of these exist is exactly what the page's FR-020 state
    reports, so a bundle for a page missing its prediction mask still
    serializes. The region overlay is *not* here — FR-023 has the frontend
    compose it from the region data, and it is never a stored artifact.
    """

    original: str | None = None
    prediction_mask: str | None = None
    processed_mask: str | None = None
    inpainted: str | None = None
    rendered: str | None = None


@dataclass(frozen=True)
class PageBundle(_Serializable):
    """``GET /api/demo/page`` — status, provenance, regions and artifact refs."""

    image_id: str
    status: str
    provenance: Provenance
    regions: list[RegionView] = field(default_factory=list)
    artifacts: PageArtifacts = field(default_factory=PageArtifacts)
    #: Populated only when ``status`` is a failure state (FR-026): the page's
    #: failure is reported in the bundle, not as an error envelope.
    error_message: str | None = None
