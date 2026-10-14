"""The demo routes: one page bundle, one artifact byte server, one catalog.

Deliberately not the 17 endpoints of plan.md. This is the subset that makes a
translated page already on disk visible in the Stitch UI, and nothing else:
no session, no jobs, no summaries, no reports, no batch.

Everything is read from the real runs through :attr:`WebConfig.artifact_roots`,
so the demo shows recorded results rather than fixtures. The three routes:

* ``GET /api/demo/catalog`` — the mangas/pages that have a translated page, and
  the Spec 2 benchmark rows.
* ``GET /api/demo/page?image_id=<manga>/<page>`` — the bundle for one page.
* ``GET /api/artifact?ref=<root>/<tail>`` — the bytes, through
  :func:`~manga_text_seg.web.security.resolve_artifact` so FR-013d holds even
  in a demo.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from ..config import WebConfig
from .errors import error_response
from .schemas import PageArtifacts, PageBundle, Provenance, RegionView
from .security import ArtifactRefused, resolve_artifact

#: Which artifact each chain card shows, as ``(root name, suffix under the
#: inpainting page directory)``. The five are FR-023's stages.
_INPAINT_SUFFIX = {
    "prediction_mask": ".raw.png",
    "processed_mask": ".mask.png",
    "inpainted": ".png",
}


@dataclass(frozen=True)
class _PageRef:
    """One translated page found on disk."""

    image_id: str
    run_id: str
    method: str
    manga: str
    page: str
    #: Path relative to the ``translation`` artifact root, POSIX separators.
    rel: str


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _translated_pages(config: WebConfig) -> list[_PageRef]:
    """Every page under the translation root that has a rendered output.

    The layout is ``<run>/<method>/<manga>/<page>/``; the glob fixes that depth
    rather than walking, so an unrelated nested directory cannot be mistaken
    for a page. ``rendered.png`` is the presence test — a page directory
    without it is a partial run, which the demo has nothing to show for.

    ponytail: only the run with the most pages is returned. The translation
    root also holds one-page smoke runs, and a page id would otherwise resolve
    to whichever of them sorts first, showing a stale run's page under the
    current run's provenance. Add a run selector when the demo has to compare
    runs rather than show one.
    """
    root = config.artifact_roots["translation"]
    refs: list[_PageRef] = []
    for meta in sorted(root.glob("*/*/*/*/metadata.json")):
        page_dir = meta.parent
        if not (page_dir / "rendered.png").is_file():
            continue
        run_id, method, manga, page = page_dir.relative_to(root).parts
        refs.append(
            _PageRef(
                image_id=f"{manga}/{page}",
                run_id=run_id,
                method=method,
                manga=manga,
                page=page,
                rel=page_dir.relative_to(root).as_posix(),
            )
        )
    if not refs:
        return []
    counts = Counter(ref.run_id for ref in refs)
    # Sorted by run id so two runs of equal size resolve the same way twice.
    main = min(counts, key=lambda run: (-counts[run], run))
    return [ref for ref in refs if ref.run_id == main]


def _find(config: WebConfig, image_id: str) -> _PageRef:
    for ref in _translated_pages(config):
        if ref.image_id == image_id:
            return ref
    raise LookupError(image_id)


def _ref_if_file(config: WebConfig, root_name: str, tail: str) -> str | None:
    """The run-relative reference for ``root_name/tail``, or ``None`` if absent.

    A missing artifact is not an error here: which of FR-023's stages exist is
    exactly what the page's status reports (FR-020), so the bundle carries a
    null and the frontend renders that stage as unavailable.
    """
    root = config.artifact_roots[root_name]
    if not (root / tail).is_file():
        return None
    return f"{root_name}/{tail}"


def _artifacts(config: WebConfig, ref: _PageRef, meta: dict[str, Any]) -> PageArtifacts:
    """FR-023's five stage references, each present only if its file is.

    Four come from the translation page directory. The middle three come from
    the inpainting run the metadata names — ``inpainting_method.run_id`` and
    ``.algorithm`` — because the translation run copies only the mask and the
    inpainted canvas forward, not the processed (dilated) mask.
    """
    inpaint = meta.get("inpainting_method") or {}
    run = inpaint.get("run_id")
    algorithm = inpaint.get("algorithm")
    prefix = f"{run}/{meta.get('segmentation_method')}/{algorithm}/{ref.image_id}"
    mid = {
        key: _ref_if_file(config, "inpainting", f"{prefix}{suffix}")
        for key, suffix in _INPAINT_SUFFIX.items()
    }
    return PageArtifacts(
        original=_ref_if_file(config, "translation", f"{ref.rel}/original.png"),
        inpainted=mid["inpainted"],
        prediction_mask=mid["prediction_mask"],
        processed_mask=mid["processed_mask"],
        rendered=_ref_if_file(config, "translation", f"{ref.rel}/rendered.png"),
    )


def _provenance(meta: dict[str, Any], image_id: str) -> Provenance:
    """FR-018's panel, filled from the metadata and null where it records none.

    ``spec2_run_id`` is always null: the Spec 004 metadata schema is closed and
    does not carry it, so FR-019 has the frontend show it as not recorded
    rather than the backend guessing from the mask's path.
    """
    inpaint = meta.get("inpainting_method") or {}
    ocr = meta.get("ocr_engine") or {}
    provider = meta.get("translation_provider") or {}
    rendering = meta.get("rendering_config") or {}
    font_path = rendering.get("font_path")
    return Provenance(
        image_id=image_id,
        segmentation_method=meta.get("segmentation_method"),
        inpainting_run_id=inpaint.get("run_id"),
        translation_run_id=meta.get("run_id"),
        inpainting_algorithm=inpaint.get("algorithm"),
        target_language=meta.get("target_language"),
        ocr_engine=ocr.get("name"),
        ocr_version=ocr.get("version"),
        translation_provider=provider.get("name"),
        translation_model=provider.get("model"),
        # The recorded value is an absolute Windows path. FR-013d keeps real
        # paths out of responses, so the panel gets the font's name; the whole
        # config still travels, because FR-018 asks for it.
        rendering_font=Path(font_path).name if font_path else None,
        rendering_config=rendering or None,
    )


def _regions(meta: dict[str, Any]) -> list[RegionView]:
    """FR-024's per-region view, plus the polygon the overlay needs.

    ``polygon`` is carried because ``bbox`` alone is not the geometry: on the
    recorded pages some regions have a degenerate box (a 1-pixel-tall line)
    whose polygon is the real shape. The frontend strokes the polygon and only
    falls back to the box when there is none.
    """
    views: list[RegionView] = []
    for region in meta.get("regions") or []:
        views.append(
            RegionView(
                region_id=region["region_id"],
                bbox=list(region.get("bbox") or []),
                polygon=[list(point) for point in region.get("polygon") or []],
                reading_order_index=region.get("reading_order_index"),
                ocr_text=region.get("ocr_text"),
                translation=region.get("translation"),
                ocr_status=region.get("ocr_status"),
                translation_status=region.get("translation_status"),
                render_status=region.get("render_status"),
                ocr_error=region.get("ocr_error"),
                translation_error=region.get("translation_error"),
                render_warning=region.get("render_warning"),
            )
        )
    return views


def _status(artifacts: PageArtifacts, regions: list[RegionView]) -> str:
    """FR-020's state for the page, most-fundamental condition first."""
    if not regions:
        return "no_text_regions"
    if artifacts.prediction_mask is None:
        return "missing_prediction_mask"
    if artifacts.inpainted is None:
        return "missing_inpainting_output"
    if artifacts.rendered is None:
        return "missing_ocr_translation_output"
    if any(r.ocr_status not in (None, "ok") for r in regions):
        return "region_ocr_failure"
    if any(r.translation_status not in (None, "ok") for r in regions):
        return "region_translation_failure"
    if any(r.render_status == "failed" for r in regions):
        return "rendering_failure"
    return "valid"


def build_bundle(config: WebConfig, image_id: str) -> PageBundle:
    """The whole ``/api/demo/page`` payload for one page."""
    ref = _find(config, image_id)
    meta = _load_json(config.artifact_roots["translation"] / ref.rel / "metadata.json")
    artifacts = _artifacts(config, ref, meta)
    regions = _regions(meta)
    return PageBundle(
        image_id=image_id,
        status=_status(artifacts, regions),
        provenance=_provenance(meta, image_id),
        regions=regions,
        artifacts=artifacts,
    )


def _benchmarks(config: WebConfig) -> list[dict[str, Any]]:
    """Spec 2's comparison rows, as recorded.

    Each metric is ``{"mean": ..., "std": ...}`` and two of the ten rows carry
    ``metrics: null`` (the methods that were never run). Both are read
    defensively: the row still appears, with a null mean, rather than the panel
    dropping a method or the request failing. The chart plots the mean; the std
    rides along so a later view can draw it without re-reading the file.
    """
    path = config.artifact_roots["deliverables"] / "spec002/comparison.json"
    if not path.is_file():
        return []
    rows = []
    for row in _load_json(path).get("rows") or []:
        metrics = row.get("metrics") or {}
        rows.append(
            {
                "method": row.get("method"),
                "status": row.get("status"),
                "reason": row.get("reason"),
                "iou": (metrics.get("iou") or {}).get("mean"),
                "iou_std": (metrics.get("iou") or {}).get("std"),
                "f1": (metrics.get("f1") or {}).get("mean"),
                "f1_std": (metrics.get("f1") or {}).get("std"),
            }
        )
    return rows


def _catalog(config: WebConfig) -> dict[str, Any]:
    pages = _translated_pages(config)
    mangas: dict[str, list[str]] = {}
    for ref in pages:
        mangas.setdefault(ref.manga, []).append(ref.page)
    return {
        "mangas": [
            {"manga": manga, "pages": sorted(set(pages_))}
            for manga, pages_ in sorted(mangas.items())
        ],
        "pages": [
            {"image_id": ref.image_id, "run_id": ref.run_id, "method": ref.method}
            for ref in pages
        ],
        "benchmarks": _benchmarks(config),
    }


def register_demo_routes(app: FastAPI, config: WebConfig) -> None:
    """Mount the demo routes. Called by :func:`create_app` before the bundle."""

    @app.get("/api/demo/catalog")
    async def catalog() -> dict:
        return _catalog(config)

    @app.get("/api/demo/page")
    async def page(image_id: str) -> JSONResponse:
        try:
            bundle = build_bundle(config, image_id)
        except LookupError:
            return error_response(
                "missing_translation_output",
                f"no translated page '{image_id}' is recorded under the "
                f"translation root",
                {"expected": f"{config.artifact_roots['translation']}/<run>/"
                             f"<method>/<manga>/<page>/metadata.json"},
            )
        return JSONResponse(bundle.to_dict())

    # response_model=None: this route returns bytes or an error envelope, and
    # FastAPI cannot build a response model from a Response union — it refuses
    # to start rather than guess. The two are already serialized.
    @app.get("/api/artifact", response_model=None)
    async def artifact(ref: str) -> FileResponse | JSONResponse:
        try:
            path = resolve_artifact(config, ref)
        except ArtifactRefused as exc:
            return error_response("invalid_request", str(exc))
        if not path.is_file():
            return error_response("invalid_request", f"'{ref}' names no artifact")
        # ponytail: no cache headers, no range support. One browser on the same
        # machine reads these once; add them when the demo is served remotely.
        return FileResponse(path)
