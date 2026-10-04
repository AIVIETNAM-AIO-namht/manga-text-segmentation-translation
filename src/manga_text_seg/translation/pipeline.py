"""Stage orchestration and failure isolation for Spec 004 translation runs."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import cv2

from ..config import (
    RegionExtractionConfig,
    RenderingConfig,
    TranslationCacheConfig,
    TranslationConfig,
    TranslationOcrConfig,
    TranslationOutputRoots,
    TranslationProviderConfig,
    TranslationRetryConfig,
)
from ..manifest import Manifest, load_manifest
from .intake import IntakeRejectionError, PageBundle, assemble_bundle
from .ocr import OcrAdapter, _record_direct_edits, _region_signature, load_ocr_results
from .regions import RegionExtractionResult, TextRegion, extract_regions
from .render import RenderedPage, render_page
from .runs import (
    ErrorReport,
    TranslationRun,
    create_run_directory,
    page_directory,
    write_metadata,
    write_run_json,
)
from .translate import (
    FixtureTranslationProvider,
    TranslationProviderAdapter,
    TranslationRequest,
    TranslationResult,
    TranslationService,
    write_translations_json,
)


@dataclass(frozen=True)
class PageResult:
    image_id: str
    status: str
    page_dir: Path
    regions: tuple[TextRegion, ...] = ()


@dataclass(frozen=True)
class BatchResult:
    run_dir: Path
    pages: tuple[PageResult, ...]
    error_report: ErrorReport
    run_record: TranslationRun


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def run_regions(
    mask_path: str | Path,
    page_dir: str | Path,
    config: TranslationConfig,
    *,
    image_id: str,
    error_report: ErrorReport,
) -> RegionExtractionResult:
    """Run only extraction from the stored prediction mask and save regions.json."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        error_report.add(
            "missing_prediction_mask",
            image_id=image_id,
            region_id=None,
            stage="regions",
            reason=f"prediction mask became unreadable: {mask_path}",
        )
        raise ValueError(f"prediction mask is unreadable: {mask_path}")
    result = extract_regions(
        mask, config.region_extraction, image_id=image_id, error_report=error_report
    )
    Path(page_dir).mkdir(parents=True, exist_ok=True)
    (Path(page_dir) / "regions.json").write_text(
        json.dumps(
            {
                "status": result.status,
                "regions": [_jsonable(region) for region in result.regions],
                "excluded_components": _jsonable(result.excluded_components),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def run_ocr(
    original_path: str | Path,
    regions: Sequence[TextRegion],
    page_dir: str | Path,
    config: TranslationConfig,
    *,
    image_id: str,
    error_report: ErrorReport,
    recognizer: Callable[..., Any] | None = None,
    signature_regions: Sequence[TextRegion] | None = None,
) -> Any:
    """Run OCR alone against stored regions and the original pre-inpaint page."""
    adapter = OcrAdapter(
        recognizer,
        engine=config.ocr.engine,
        error_report=error_report,
    )
    return adapter.run(
        original_path,
        regions,
        page_dir,
        config=config.ocr,
        image_id=image_id,
        signature_regions=signature_regions,
    )


def run_translation(
    ocr_results: Sequence[Any],
    page_dir: str | Path,
    config: TranslationConfig,
    provider: TranslationProviderAdapter,
    *,
    image_id: str,
    error_report: ErrorReport,
) -> tuple[Any, ...]:
    """Run translation alone from persisted OCR results."""
    requests = [
        TranslationRequest(result.region_id, result.text)
        for result in ocr_results
        if result.status == "ok" and result.text
    ]
    service = TranslationService(config, provider, error_report=error_report)
    translated = service.translate_regions(
        requests,
        image_id=image_id,
    )
    by_id = {result.region_id: result for result in translated}
    for ocr_result in ocr_results:
        if ocr_result.region_id not in by_id:
            by_id[ocr_result.region_id] = TranslationResult(
                region_id=ocr_result.region_id,
                source_text=ocr_result.text,
                translated_text="",
                target_language=config.target_language,
                provider=config.provider.name,
                model=config.provider.model,
                version=config.provider.version,
                cache_status="miss",
                processing_time_seconds=0.0,
                status="skipped",
                error=None,
                failure_policy="exclude_from_rendering",
            )
    ordered = tuple(by_id[key] for key in sorted(by_id))
    from .translate import write_translations_json

    write_translations_json(Path(page_dir) / "translations.json", ordered)
    return ordered


def run_rendering(
    inpainted_path: str | Path,
    regions: Sequence[TextRegion],
    translations: Sequence[Any],
    page_dir: str | Path,
    config: TranslationConfig,
    *,
    image_id: str,
    error_report: ErrorReport,
) -> RenderedPage:
    """Run rendering alone from the preceding stored stage outputs."""
    return render_page(
        inpainted_path,
        regions,
        translations,
        config.rendering,
        Path(page_dir) / "rendered.png",
        error_report=error_report,
        image_id=image_id,
    )


def _manifest_page_ids(config: TranslationConfig) -> tuple[str, ...]:
    manifest: Manifest = load_manifest(config.manifest_path)
    return tuple(pair.image_id for pair in manifest.pairs)


def _identity_records(
    repo_root: Path, image_ids: Sequence[str]
) -> tuple[str, dict[str, str]]:
    try:
        page_list = json.loads(
            (repo_root / "benchmark" / "page-list.json").read_text(encoding="utf-8")
        )
        image_identity = json.loads(
            (repo_root / "benchmark" / "image-identity.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Spec 2 identity records: {exc}") from exc
    if not isinstance(page_list, dict) or not isinstance(page_list.get("pages"), list):
        raise ValueError("Spec 2 page-list.json must contain a pages array")
    if not isinstance(image_identity, dict):
        raise ValueError("Spec 2 image-identity.json must contain an object")
    pages: dict[str, dict[str, Any]] = {}
    for page in page_list["pages"]:
        if (
            not isinstance(page, dict)
            or not isinstance(page.get("image_id"), str)
            or page["image_id"] in pages
        ):
            raise ValueError("Spec 2 page list contains a malformed or duplicate page")
        pages[page["image_id"]] = page
    identities: dict[str, str] = {}
    for image_id in image_ids:
        page = pages.get(image_id)
        if not isinstance(page, dict) or not isinstance(
            page.get("input_image_identity"), str
        ):
            raise ValueError(
                f"Spec 2 page list has no input identity for manifest page {image_id!r}"
            )
        record = image_identity.get(image_id)
        if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
            raise ValueError(
                f"Spec 2 image-identity record has no sha256 for {image_id!r}"
            )
        if page["input_image_identity"] != record["sha256"]:
            raise ValueError(
                "Spec 2 identity records disagree for input_image_identity of "
                f"{image_id!r}"
            )
        identities[image_id] = record["sha256"]
    identity = page_list.get("page_list_identity")
    if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{64}", identity):
        raise ValueError("Spec 2 page list has no valid sha256 page_list_identity")
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in identities.values()):
        raise ValueError("Spec 2 image identity must be a sha256 hex digest")
    return identity, identities


def _page_metadata(
    bundle: PageBundle,
    config: TranslationConfig,
    page_status: str,
    regions: Sequence[TextRegion],
    ocr_run: Any | None,
    translations: Sequence[Any],
    renders: Sequence[Any],
    timings: dict[str, float],
) -> dict[str, Any]:
    ocr_by_id = {item.region_id: item for item in (ocr_run.results if ocr_run else ())}
    translation_by_id = {item.region_id: item for item in translations}
    render_by_id = {item.region_id: item for item in renders}
    region_docs = []
    for region in regions:
        ocr = ocr_by_id.get(region.region_id)
        translated = translation_by_id.get(region.region_id)
        rendered = render_by_id.get(region.region_id)
        region_docs.append(
            {
                "region_id": region.region_id,
                "bbox": list(region.bbox),
                "polygon": [list(point) for point in region.polygon],
                "orientation": region.orientation,
                "reading_order_index": region.reading_order_index,
                "merged_from": region.merged_from,
                "crop_path": ocr.crop_path if ocr else None,
                "clipped": region.clipped,
                "ocr_text": ocr.text if ocr else "",
                "ocr_confidence": ocr.confidence if ocr else None,
                "ocr_status": ocr.status if ocr else "failed",
                "ocr_error": ocr.error if ocr else None,
                "manually_edited": ocr.manually_edited if ocr else False,
                "translation": translated.translated_text if translated else "",
                "translation_status": translated.status if translated else "skipped",
                "translation_error": translated.error if translated else None,
                "cache_status": translated.cache_status if translated else None,
                "failure_policy": translated.failure_policy if translated else None,
                "render_status": rendered.render_status if rendered else "skipped",
                "render_warning": rendered.warning if rendered else None,
                "font_size_used": rendered.font_size_used if rendered else None,
                "line_count": rendered.line_count if rendered else 0,
                "display_box": list(rendered.display_box) if rendered else None,
                "direction_used": rendered.direction_used if rendered else None,
                "ocr_time_seconds": ocr.processing_time_seconds if ocr else None,
                "translation_time_seconds": (
                    translated.processing_time_seconds if translated else None
                ),
                "render_time_seconds": timings.get(f"render:{region.region_id}"),
            }
        )
    return {
        "image_id": bundle.image_id,
        "run_id": "",
        "segmentation_method": config.segmentation_method,
        "inpainting_method": {
            "run_id": config.inpainting_run_id,
            "algorithm": bundle.inpainting_provenance["algorithm"],
            "inpaint_algorithm": bundle.inpainting_provenance["inpaint_algorithm"],
            "inpaint_radius": bundle.inpainting_provenance["inpaint_radius"],
            "dilation": bundle.inpainting_provenance["dilation"],
        },
        "target_language": config.target_language,
        "ocr_engine": {
            "name": ocr_run.engine if ocr_run else config.ocr.engine,
            "version": ocr_run.engine_version if ocr_run else None,
            "deterministic": True,
        },
        "translation_provider": {
            "name": config.provider.name,
            "model": config.provider.model,
            "version": config.provider.version,
            "deterministic": config.provider.deterministic,
            "secret_env_var": config.provider.secret_env_var,
        },
        "rendering_config": {
            "font_path": str(config.rendering.font_path),
            "font_size": config.rendering.font_size,
            "min_font_size": config.rendering.min_font_size,
            "text_direction": config.rendering.text_direction,
            "expansion_allowance": config.rendering.expansion_allowance,
            "color": list(config.rendering.color),
            "stroke_width": config.rendering.stroke_width,
            "background": (
                list(config.rendering.background)
                if config.rendering.background
                else None
            ),
            "wrap": config.rendering.wrap,
            "line_spacing": config.rendering.line_spacing,
            "align_h": config.rendering.align_h,
            "align_v": config.rendering.align_v,
        },
        "regions": region_docs,
        "stage_timings": timings,
        "status": "ok" if page_status == "success" else page_status,
    }


def _process_page(
    config: TranslationConfig,
    run_id: str,
    image_id: str,
    repo_root: Path,
    error_report: ErrorReport,
    provider: TranslationProviderAdapter,
    recognizer: Callable[..., Any] | None,
    prediction_root: str | Path | None,
    inpainting_root: str | Path | None,
) -> PageResult:
    page_dir = page_directory(
        config.output_roots.translation, run_id, config.segmentation_method, image_id
    )
    timings: dict[str, float] = {}
    regions: tuple[TextRegion, ...] = ()
    ocr_run = None
    translations: tuple[Any, ...] = ()
    renders: tuple[Any, ...] = ()
    bundle: PageBundle | None = None
    status = "failed"
    current_stage = "intake"
    try:
        page_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        bundle = assemble_bundle(
            config,
            image_id,
            run_id,
            error_report=error_report,
            repo_root=repo_root,
            prediction_root=prediction_root,
            inpainting_root=inpainting_root,
        )
        # Keep byte-for-byte local copies of the images needed by isolated OCR
        # and rendering reruns. Upstream artifacts remain read-only.
        shutil.copyfile(bundle.original_image_path, page_dir / "original.png")
        shutil.copyfile(bundle.inpainted_image_path, page_dir / "inpainted.png")
        timings["intake"] = time.perf_counter() - started
        current_stage = "regions"
        started = time.perf_counter()
        region_result = run_regions(
            bundle.mask_copy_path,
            page_dir,
            config,
            image_id=image_id,
            error_report=error_report,
        )
        regions = region_result.regions
        timings["regions"] = time.perf_counter() - started
        if region_result.status == "no_regions":
            status = "no_regions"
        else:
            current_stage = "ocr"
            started = time.perf_counter()
            ocr_run = run_ocr(
                bundle.original_image_path,
                regions,
                page_dir,
                config,
                image_id=image_id,
                error_report=error_report,
                recognizer=recognizer,
            )
            timings["ocr"] = time.perf_counter() - started
            current_stage = "translate"
            started = time.perf_counter()
            translations = run_translation(
                ocr_run.results,
                page_dir,
                config,
                provider,
                image_id=image_id,
                error_report=error_report,
            )
            timings["translate"] = time.perf_counter() - started
            current_stage = "render"
            started = time.perf_counter()
            rendered_page = run_rendering(
                bundle.inpainted_image_path,
                regions,
                translations,
                page_dir,
                config,
                image_id=image_id,
                error_report=error_report,
            )
            renders = rendered_page.regions
            timings["render"] = time.perf_counter() - started
            status = "success"
        metadata = _page_metadata(
            bundle, config, status, regions, ocr_run, translations, renders, timings
        )
        metadata["run_id"] = run_id
        write_metadata(page_dir, metadata)
        if status == "no_regions":
            (page_dir / "rendered.png").write_bytes(
                bundle.inpainted_image_path.read_bytes()
            )
    except IntakeRejectionError:
        category = error_report.categories
        status = next(
            (
                name
                for name, items in category.items()
                if items and items[-1]["image_id"] == image_id
            ),
            "failed",
        )
    except Exception as exc:
        status = "failed"
        category = "output_write_failure"
        stage = (
            current_stage
            if current_stage in {"regions", "ocr", "translate", "render"}
            else "write"
        )
        if not any(
            entry["image_id"] == image_id and entry["stage"] == stage
            for entry in error_report.categories[category]
        ):
            error_report.add(
                category,
                image_id=image_id,
                region_id=None,
                stage=stage,
                reason=f"{type(exc).__name__}: {exc}",
            )
        if bundle is not None:
            try:
                metadata = _page_metadata(
                    bundle,
                    config,
                    status,
                    regions,
                    ocr_run,
                    translations,
                    renders,
                    timings,
                )
                metadata["run_id"] = run_id
                metadata["error_message"] = f"{type(exc).__name__}: {exc}"
                write_metadata(page_dir, metadata)
            except Exception as metadata_exc:
                # Preserve the original stage failure; the error report remains
                # authoritative if a partial metadata write also fails.
                error_report.add(
                    "output_write_failure",
                    image_id=image_id,
                    region_id=None,
                    stage="write",
                    reason=f"could not persist failed-page metadata: {metadata_exc}",
                )
    return PageResult(image_id, status, page_dir, regions)


def run_batch(
    config: TranslationConfig,
    run_id: str,
    *,
    repo_root: str | Path,
    provider: TranslationProviderAdapter | None = None,
    recognizer: Callable[..., Any] | None = None,
    image_ids: Sequence[str] | None = None,
    prediction_root: str | Path | None = None,
    inpainting_root: str | Path | None = None,
    overwrite: bool = False,
) -> BatchResult:
    """Process Spec 1 manifest pages independently into one isolated run."""
    if not isinstance(config, TranslationConfig):
        raise TypeError("config must be TranslationConfig")
    if provider is None:
        if config.provider.secret_env_var:
            raise ValueError(
                "a provider adapter must be supplied; no live provider "
                "protocol is defined"
            )
        provider = FixtureTranslationProvider({})
    ids = tuple(image_ids) if image_ids is not None else _manifest_page_ids(config)
    manifest_ids = _manifest_page_ids(config)
    if any(image_id not in manifest_ids for image_id in ids):
        raise ValueError(
            "requested image_ids must be entries in the configured Spec 1 manifest"
        )
    repo = Path(repo_root).resolve()
    # Validate authoritative Spec 2 identity artifacts before creating or
    # mutating a translation run. A run without these values is not attributable.
    page_list_identity, identities = _identity_records(repo, ids)
    run_dir = create_run_directory(
        config.output_roots.translation, run_id, overwrite=overwrite
    )
    report = ErrorReport(run_id)
    results = tuple(
        _process_page(
            config,
            run_id,
            image_id,
            repo,
            report,
            provider,
            recognizer,
            prediction_root,
            inpainting_root,
        )
        for image_id in ids
    )
    try:
        run_record = TranslationRun(
            run_id=run_id,
            segmentation_method=config.segmentation_method,
            inpainting_source={
                "run_id": config.inpainting_run_id,
                "algorithm": config.inpainting_algorithm,
            },
            target_language=config.target_language,
            ocr_engine={
                "name": config.ocr.engine,
                "version": None,
                "deterministic": True,
            },
            provider={
                "name": config.provider.name,
                "model": config.provider.model,
                "version": config.provider.version,
                "deterministic": bool(
                    getattr(provider, "deterministic", config.provider.deterministic)
                ),
                "secret_env_var": config.provider.secret_env_var,
            },
            rendering={
                "font_family": config.rendering.font_family,
                "font_path": str(config.rendering.font_path),
                "fallback_font_path": (
                    str(config.rendering.fallback_font_path)
                    if config.rendering.fallback_font_path
                    else None
                ),
                "font_size": config.rendering.font_size,
                "min_font_size": config.rendering.min_font_size,
                "color": list(config.rendering.color),
                "stroke_width": config.rendering.stroke_width,
                "stroke_fill": list(config.rendering.stroke_fill),
                "background": (
                    list(config.rendering.background)
                    if config.rendering.background
                    else None
                ),
                "wrap": config.rendering.wrap,
                "line_spacing": config.rendering.line_spacing,
                "align_h": config.rendering.align_h,
                "align_v": config.rendering.align_v,
                "text_direction": config.rendering.text_direction,
                "expansion_allowance": config.rendering.expansion_allowance,
            },
            resolved_config={
                "manifest_path": str(config.manifest_path),
                "segmentation_method": config.segmentation_method,
                "inpainting_source": {
                    "run_id": config.inpainting_run_id,
                    "algorithm": config.inpainting_algorithm,
                },
                "target_language": config.target_language,
                "region_extraction": _jsonable(config.region_extraction),
                "ocr": _jsonable(config.ocr),
                "provider": _jsonable(config.provider),
                "rendering": _jsonable(config.rendering),
                "output_roots": _jsonable(config.output_roots),
            },
            input_image_ids=ids,
            page_statuses={page.image_id: page.status for page in results},
            page_list_identity=page_list_identity,
            input_image_identity=identities,
            region_extraction_config=_jsonable(config.region_extraction),
            counts={
                "pages_attempted": len(ids),
                "pages_succeeded": sum(
                    page.status in {"success", "no_regions"} for page in results
                ),
                "pages_failed": sum(
                    page.status not in {"success", "no_regions"} for page in results
                ),
            },
        )
        write_run_json(run_dir, run_record)
    except Exception as exc:
        if ids:
            report.add(
                "output_write_failure",
                image_id=ids[0],
                region_id=None,
                stage="write",
                reason=f"could not write run.json: {exc}",
            )
        raise
    report.write(run_dir)
    return BatchResult(run_dir, results, report, run_record)


def run_intake(
    config: TranslationConfig,
    run_id: str,
    *,
    repo_root: str | Path,
    image_ids: Sequence[str] | None = None,
    prediction_root: str | Path | None = None,
    inpainting_root: str | Path | None = None,
    overwrite: bool = False,
) -> BatchResult:
    """Create a run from admitted inputs only, ready for later isolated stages."""
    if not isinstance(config, TranslationConfig):
        raise TypeError("config must be TranslationConfig")
    manifest_ids = _manifest_page_ids(config)
    ids = tuple(image_ids) if image_ids is not None else manifest_ids
    if any(image_id not in manifest_ids for image_id in ids):
        raise ValueError(
            "requested image_ids must be entries in the configured Spec 1 manifest"
        )
    repo = Path(repo_root).resolve()
    page_list_identity, identities = _identity_records(repo, ids)
    run_dir = create_run_directory(
        config.output_roots.translation, run_id, overwrite=overwrite
    )
    report = ErrorReport(run_id)
    results: list[PageResult] = []
    for image_id in ids:
        target = page_directory(
            config.output_roots.translation,
            run_id,
            config.segmentation_method,
            image_id,
        )
        target.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        try:
            bundle = assemble_bundle(
                config,
                image_id,
                run_id,
                error_report=report,
                repo_root=repo,
                prediction_root=prediction_root,
                inpainting_root=inpainting_root,
            )
            shutil.copyfile(bundle.original_image_path, target / "original.png")
            shutil.copyfile(bundle.inpainted_image_path, target / "inpainted.png")
            page_status = "empty_mask" if bundle.status == "empty_mask" else "accepted"
            metadata = _page_metadata(
                bundle,
                config,
                "success",
                (),
                None,
                (),
                (),
                {"intake": time.perf_counter() - started},
            )
            metadata["run_id"] = run_id
            write_metadata(target, metadata)
            results.append(PageResult(image_id, page_status, target))
        except IntakeRejectionError as exc:
            results.append(PageResult(image_id, exc.category, target))
        except Exception as exc:
            report.add(
                "output_write_failure",
                image_id=image_id,
                region_id=None,
                stage="write",
                reason=f"{type(exc).__name__}: {exc}",
            )
            results.append(PageResult(image_id, "output_write_failure", target))

    statuses = {page.image_id: page.status for page in results}
    run_record = TranslationRun(
        run_id=run_id,
        segmentation_method=config.segmentation_method,
        inpainting_source={
            "run_id": config.inpainting_run_id,
            "algorithm": config.inpainting_algorithm,
        },
        target_language=config.target_language,
        ocr_engine={"name": config.ocr.engine, "version": None, "deterministic": True},
        provider={
            "name": config.provider.name,
            "model": config.provider.model,
            "version": config.provider.version,
            "deterministic": config.provider.deterministic,
            "secret_env_var": config.provider.secret_env_var,
        },
        rendering={
            "font_family": config.rendering.font_family,
            "font_path": str(config.rendering.font_path),
            "fallback_font_path": (
                str(config.rendering.fallback_font_path)
                if config.rendering.fallback_font_path
                else None
            ),
            "font_size": config.rendering.font_size,
            "min_font_size": config.rendering.min_font_size,
            "color": list(config.rendering.color),
            "stroke_width": config.rendering.stroke_width,
            "stroke_fill": list(config.rendering.stroke_fill),
            "background": (
                list(config.rendering.background)
                if config.rendering.background
                else None
            ),
            "wrap": config.rendering.wrap,
            "line_spacing": config.rendering.line_spacing,
            "align_h": config.rendering.align_h,
            "align_v": config.rendering.align_v,
            "text_direction": config.rendering.text_direction,
            "expansion_allowance": config.rendering.expansion_allowance,
        },
        resolved_config={
            "manifest_path": str(config.manifest_path),
            "segmentation_method": config.segmentation_method,
            "inpainting_source": {
                "run_id": config.inpainting_run_id,
                "algorithm": config.inpainting_algorithm,
            },
            "target_language": config.target_language,
            "region_extraction": _jsonable(config.region_extraction),
            "ocr": _jsonable(config.ocr),
            "provider": _jsonable(config.provider),
            "rendering": _jsonable(config.rendering),
            "output_roots": _jsonable(config.output_roots),
        },
        input_image_ids=ids,
        page_statuses=statuses,
        page_list_identity=page_list_identity,
        input_image_identity=identities,
        region_extraction_config=_jsonable(config.region_extraction),
        counts={
            "pages_attempted": len(ids),
            "pages_succeeded": sum(
                page.status in {"accepted", "empty_mask"} for page in results
            ),
            "pages_failed": sum(
                page.status not in {"accepted", "empty_mask"} for page in results
            ),
        },
    )
    write_run_json(run_dir, run_record)
    report.write(run_dir)
    return BatchResult(run_dir, tuple(results), report, run_record)


def _config_from_run(run_record: dict[str, Any]) -> TranslationConfig:
    """Rehydrate the exact validated settings persisted in run.json."""
    raw = run_record["resolved_config"]
    region = raw["region_extraction"]
    ocr = raw["ocr"]
    provider = raw["provider"]
    retry = provider["retry"]
    cache = provider["cache"]
    rendering = raw["rendering"]
    roots = raw["output_roots"]
    return TranslationConfig(
        manifest_path=Path(raw["manifest_path"]),
        segmentation_method=raw["segmentation_method"],
        inpainting_run_id=raw["inpainting_source"]["run_id"],
        inpainting_algorithm=raw["inpainting_source"]["algorithm"],
        target_language=raw["target_language"],
        region_extraction=RegionExtractionConfig(**region),
        ocr=TranslationOcrConfig(
            engine=ocr["engine"],
            preprocessing_enabled=ocr["preprocessing_enabled"],
            grayscale=ocr["grayscale"],
            upscale_factor=ocr["upscale_factor"],
        ),
        provider=TranslationProviderConfig(
            name=provider["name"],
            model=provider["model"],
            version=provider["version"],
            base_url=provider["base_url"],
            secret_env_var=provider["secret_env_var"],
            timeout_seconds=provider["timeout_seconds"],
            retry=TranslationRetryConfig(
                max_attempts=retry["max_attempts"],
                initial_backoff_seconds=retry["initial_backoff_seconds"],
                max_backoff_seconds=retry["max_backoff_seconds"],
                retry_status_codes=tuple(retry["retry_status_codes"]),
            ),
            failure_policy=provider["failure_policy"],
            deterministic=provider["deterministic"],
            cache=TranslationCacheConfig(
                enabled=cache["enabled"],
                bypass=cache["bypass"],
                root=Path(cache["root"]),
            ),
        ),
        rendering=RenderingConfig(
            font_family=rendering["font_family"],
            font_path=Path(rendering["font_path"]),
            fallback_font_path=(
                Path(rendering["fallback_font_path"])
                if rendering["fallback_font_path"]
                else None
            ),
            font_size=rendering["font_size"],
            min_font_size=rendering["min_font_size"],
            color=tuple(rendering["color"]),
            stroke_width=rendering["stroke_width"],
            stroke_fill=tuple(rendering["stroke_fill"]),
            background=(
                tuple(rendering["background"])
                if rendering["background"] is not None
                else None
            ),
            wrap=rendering["wrap"],
            line_spacing=rendering["line_spacing"],
            align_h=rendering["align_h"],
            align_v=rendering["align_v"],
            text_direction=rendering["text_direction"],
            expansion_allowance=rendering["expansion_allowance"],
        ),
        output_roots=TranslationOutputRoots(
            translation=Path(roots["translation"]), cache=Path(roots["cache"])
        ),
    )


def _load_regions(path: Path) -> tuple[TextRegion, ...]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        TextRegion(
            **{
                **item,
                "bbox": tuple(item["bbox"]),
                "polygon": tuple(tuple(point) for point in item["polygon"]),
                "crop_box": tuple(item["crop_box"]),
            }
        )
        for item in document["regions"]
    )


def _load_translations(path: Path) -> tuple[TranslationResult, ...]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return tuple(TranslationResult(**item) for item in document)


def _load_report(run_dir: Path, run_id: str) -> ErrorReport:
    path = run_dir / "errors.json"
    if not path.is_file():
        return ErrorReport(run_id)
    document = json.loads(path.read_text(encoding="utf-8"))
    return ErrorReport(run_id, document["categories"])


def _update_metadata_stage(
    page_dir: Path,
    image_id: str,
    stage_name: str,
    *,
    region_ids: set[int] | None = None,
) -> None:
    """Merge only the selected stage's persisted records into page metadata."""
    path = page_dir / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("status") == "success":
        # Earlier T060–T064 output used the BatchResult status vocabulary;
        # page metadata follows the official schema vocabulary.
        metadata["status"] = "ok"
    records = metadata["regions"]
    selected = [
        entry
        for entry in records
        if region_ids is None or entry["region_id"] in region_ids
    ]
    if stage_name == "ocr":
        ocr_by_id = {
            item.region_id: item for item in load_ocr_results(page_dir / "ocr.json")
        }
        for entry in selected:
            result = ocr_by_id[entry["region_id"]]
            changed = (
                entry.get("ocr_text") != result.text
                or entry.get("ocr_status") != result.status
            )
            entry.update(
                crop_path=result.crop_path,
                ocr_text=result.text,
                ocr_confidence=result.confidence,
                ocr_status=result.status,
                ocr_error=result.error,
                manually_edited=result.manually_edited,
                ocr_time_seconds=result.processing_time_seconds,
            )
            if changed:
                _clear_translation_and_render(entry)
    elif stage_name in {"translate", "translation"}:
        translations = {
            item.region_id: item
            for item in _load_translations(page_dir / "translations.json")
        }
        for entry in selected:
            result = translations[entry["region_id"]]
            changed = (
                entry.get("translation") != result.translated_text
                or entry.get("translation_status") != result.status
            )
            entry.update(
                translation=result.translated_text,
                translation_status=result.status,
                translation_error=result.error,
                cache_status=result.cache_status,
                failure_policy=result.failure_policy,
                translation_time_seconds=result.processing_time_seconds,
            )
            if changed:
                _clear_render(entry)
    elif stage_name in {"render", "rendering"}:
        # Rendering results are fully reflected in metadata by the renderer's
        # current metadata records; callers merge those records below.
        rendered = json.loads(
            (page_dir / "render-stage.json").read_text(encoding="utf-8")
        )
        render_by_id = {item["region_id"]: item for item in rendered}
        for entry in selected:
            result = render_by_id[entry["region_id"]]
            entry.update(result)
    write_metadata(page_dir, metadata)


def _clear_render(entry: dict[str, Any]) -> None:
    entry.update(
        render_status="skipped",
        render_warning=None,
        font_size_used=None,
        line_count=0,
        direction_used=None,
        render_time_seconds=None,
    )


def _clear_translation_and_render(entry: dict[str, Any]) -> None:
    entry.update(
        translation="",
        translation_status="skipped",
        translation_error=None,
        cache_status=None,
        failure_policy=None,
        translation_time_seconds=None,
    )
    _clear_render(entry)


def _assert_current_ocr(page_dir: Path, regions: Sequence[TextRegion]) -> None:
    path = page_dir / "ocr.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    expected = _region_signature(regions)
    if document.get("region_set") != expected:
        raise ValueError(
            "OCR artifact is stale for the current region set; rerun the OCR stage"
        )
    if document.get("stale_region_ids"):
        raise ValueError(
            "OCR artifact contains stale regions; rerun the OCR stage "
            "before consuming it"
        )


def _assert_current_translations(page_dir: Path, regions: Sequence[TextRegion]) -> None:
    _assert_current_ocr(page_dir, regions)
    ocr_by_id = {
        item.region_id: item for item in load_ocr_results(page_dir / "ocr.json")
    }
    translations = _load_translations(page_dir / "translations.json")
    expected_ids = {region.region_id for region in regions}
    by_id = {item.region_id: item for item in translations}
    if set(ocr_by_id) != expected_ids or set(by_id) != expected_ids:
        raise ValueError(
            "translation artifact is stale for the current region set; "
            "rerun translation"
        )
    if any(by_id[key].source_text != ocr_by_id[key].text for key in expected_ids):
        raise ValueError(
            "translation artifact is stale for current OCR text; rerun translation"
        )


def _update_metadata_regions(page_dir: Path, extracted: RegionExtractionResult) -> None:
    metadata_path = page_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    previous = {item["region_id"]: item for item in metadata["regions"]}
    updated = []
    for region in extracted.regions:
        geometry = {
            "region_id": region.region_id,
            "bbox": list(region.bbox),
            "polygon": [list(point) for point in region.polygon],
            "orientation": region.orientation,
            "reading_order_index": region.reading_order_index,
            "merged_from": region.merged_from,
            "clipped": region.clipped,
        }
        old = previous.get(region.region_id)
        if old is not None and all(
            old.get(key) == value for key, value in geometry.items()
        ):
            updated.append(old)
            continue
        updated.append(
            {
                **geometry,
                "crop_path": None,
                "ocr_text": "",
                "ocr_confidence": None,
                "ocr_status": "empty",
                "ocr_error": None,
                "manually_edited": False,
                "translation": "",
                "translation_status": "skipped",
                "translation_error": None,
                "cache_status": None,
                "failure_policy": None,
                "render_status": "skipped",
                "render_warning": None,
                "font_size_used": None,
                "line_count": 0,
                # Retain the previous painted bounds while marking rendering
                # skipped so a targeted rerun can clear those old pixels.
                "display_box": old.get("display_box") if old is not None else None,
                "direction_used": None,
                "ocr_time_seconds": None,
                "translation_time_seconds": None,
                "render_time_seconds": None,
            }
        )
    metadata["regions"] = updated
    metadata["status"] = "no_regions" if extracted.status == "no_regions" else "ok"
    ocr_path = page_dir / "ocr.json"
    if ocr_path.is_file():
        document = json.loads(ocr_path.read_text(encoding="utf-8"))
        _record_direct_edits(document)
        old_signature = document.get("region_set", [])
        new_signature = _region_signature(extracted.regions)
        old_by_id = {
            item.get("region_id"): item
            for item in old_signature
            if isinstance(item, dict)
        }
        new_by_id = {item["region_id"]: item for item in new_signature}
        stale_ids = set(document.get("stale_region_ids", []))
        stale_ids.update(
            region_id
            for region_id, old in old_by_id.items()
            if new_by_id.get(region_id) != old
        )
        if stale_ids:
            reports = document.setdefault("invalidation_reports", [])
            for edit in document.get("manual_edits", []):
                region_id = edit.get("region_id")
                if region_id in stale_ids and not edit.get(
                    "invalidated_by_reextraction", False
                ):
                    edit["invalidated_by_reextraction"] = True
                    report = {
                        "region_id": region_id,
                        "reason": "region set changed during re-extraction",
                    }
                    if report not in reports:
                        reports.append(report)
            document["stale_region_ids"] = sorted(stale_ids)
            ocr_path.write_text(
                json.dumps(document, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    write_metadata(page_dir, metadata)


def _rerun_stage(
    stage_name: str,
    image_id: str,
    run_dir: Path,
    *,
    provider: TranslationProviderAdapter | None,
    recognizer: Callable[..., Any] | None,
    region_ids: set[int] | None,
) -> Any:
    run_path = run_dir / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"run metadata not found: {run_path}")
    record = json.loads(run_path.read_text(encoding="utf-8"))
    run_id = record["run_id"]
    if run_dir.name != run_id:
        raise ValueError("run directory does not match its run.json run_id")
    if image_id not in record["input_image_ids"]:
        raise KeyError(f"image_id is not part of run {run_id!r}: {image_id!r}")
    config = _config_from_run(record)
    page_dir = page_directory(
        run_dir.parent, run_id, record["segmentation_method"], image_id
    )
    if not page_dir.is_dir():
        raise FileNotFoundError(f"page output directory not found: {page_dir}")
    report = _load_report(run_dir, run_id)
    aliases = {"translation": "translate", "rendering": "render"}
    stage = aliases.get(stage_name, stage_name)
    allowed = {"regions", "ocr", "translate", "render"}
    if stage not in allowed:
        raise ValueError(
            f"unsupported rerun stage {stage_name!r}; expected one of {sorted(allowed)}"
        )
    for category in report.categories:
        report.categories[category] = [
            entry
            for entry in report.categories[category]
            if not (
                entry["image_id"] == image_id
                and entry["stage"] == stage
                and (region_ids is None or entry["region_id"] in region_ids)
            )
        ]

    regions_path = page_dir / "regions.json"
    if stage == "regions" and not regions_path.exists():
        regions = ()
    else:
        regions = _load_regions(regions_path)
    selected_regions = tuple(
        item for item in regions if region_ids is None or item.region_id in region_ids
    )
    if (
        region_ids is not None
        and {item.region_id for item in selected_regions} != region_ids
    ):
        known_ids = {item.region_id for item in selected_regions}
        missing_ids = sorted(region_ids - known_ids)
        raise KeyError(f"unknown region_id(s): {missing_ids}")

    if stage == "regions":
        result = run_regions(
            page_dir / "mask.png",
            page_dir,
            config,
            image_id=image_id,
            error_report=report,
        )
        _update_metadata_regions(page_dir, result)
    elif stage == "ocr":
        if region_ids is not None:
            ocr_path = page_dir / "ocr.json"
            previous = json.loads(ocr_path.read_text(encoding="utf-8"))
            _record_direct_edits(previous)
            with tempfile.TemporaryDirectory(
                prefix="ocr-region-rerun-", dir=page_dir
            ) as temporary:
                temporary_dir = Path(temporary)
                run_ocr(
                    page_dir / "original.png",
                    selected_regions,
                    temporary_dir,
                    config,
                    image_id=image_id,
                    error_report=report,
                    recognizer=recognizer,
                    signature_regions=regions,
                )
                generated = json.loads(
                    (temporary_dir / "ocr.json").read_text(encoding="utf-8")
                )
                for record in generated["results"]:
                    relative = Path(record["crop_path"])
                    final_crop = page_dir / relative
                    final_crop.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(temporary_dir / relative, final_crop)
            generated_by_id = {item["region_id"]: item for item in generated["results"]}
            results_by_id = {item["region_id"]: item for item in previous["results"]}
            results_by_id.update(generated_by_id)
            previous["results"] = [results_by_id[key] for key in sorted(results_by_id)]
            recognized = dict(previous.get("recognized_text", {}))
            for key, value in generated.get("recognized_text", {}).items():
                if int(key) in region_ids:
                    recognized[key] = value
            previous["recognized_text"] = recognized
            previous["region_set"] = _region_signature(regions)
            previous["stale_region_ids"] = sorted(
                set(previous.get("stale_region_ids", [])) - region_ids
            )
            previous["manual_edits"] = previous.get("manual_edits", [])
            ocr_path.write_text(
                json.dumps(previous, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            result = load_ocr_results(ocr_path)
        else:
            result = run_ocr(
                page_dir / "original.png",
                selected_regions,
                page_dir,
                config,
                image_id=image_id,
                error_report=report,
                recognizer=recognizer,
            )
    elif stage == "translate":
        _assert_current_ocr(page_dir, regions)
        if provider is None and config.provider.secret_env_var is not None:
            raise ValueError(
                "the configured provider requires an injected adapter; this repository "
                "does not define a live provider wire protocol"
            )
        provider = provider or FixtureTranslationProvider({})
        translations_path = page_dir / "translations.json"
        old = (
            _load_translations(translations_path) if translations_path.is_file() else ()
        )
        ocr_results = [
            item
            for item in load_ocr_results(page_dir / "ocr.json")
            if region_ids is None or item.region_id in region_ids
        ]
        new = run_translation(
            ocr_results,
            page_dir,
            config,
            provider,
            image_id=image_id,
            error_report=report,
        )
        if region_ids is not None:
            merged = {item.region_id: item for item in old}
            merged.update({item.region_id: item for item in new})
            new = tuple(merged[key] for key in sorted(merged))
            write_translations_json(page_dir / "translations.json", new)
        result = new
    else:
        _assert_current_translations(page_dir, regions)
        translations = _load_translations(page_dir / "translations.json")
        selected_translations = tuple(
            item
            for item in translations
            if region_ids is None or item.region_id in region_ids
        )
        if region_ids is None:
            render_source = page_dir / "inpainted.png"
            render_regions = regions
            render_translations = translations
        else:
            # Restore only the target's previous display box from the stored
            # clean inpainted page, then paint that region over the existing
            # rendered page. Sibling regions and their pixels remain untouched.
            import numpy as np
            from PIL import Image

            prior_render_path = page_dir / "render-stage.json"
            if prior_render_path.is_file():
                prior_render = json.loads(prior_render_path.read_text(encoding="utf-8"))
                prior_ids = {item["region_id"] for item in prior_render}
                current_ids = {item.region_id for item in regions}
                if prior_ids != current_ids:
                    raise ValueError(
                        "rendered page is stale for the current region set; "
                        "rerun the full render stage"
                    )

            clean = np.asarray(Image.open(page_dir / "inpainted.png").convert("RGBA"))
            canvas = np.asarray(
                Image.open(page_dir / "rendered.png").convert("RGBA")
            ).copy()
            for entry in json.loads(
                (page_dir / "metadata.json").read_text(encoding="utf-8")
            )["regions"]:
                if entry["region_id"] not in region_ids or entry["display_box"] is None:
                    continue
                x, y, width, height = entry["display_box"]
                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(canvas.shape[1], x + width), min(
                    canvas.shape[0], y + height
                )
                if x0 < x1 and y0 < y1:
                    canvas[y0:y1, x0:x1] = clean[y0:y1, x0:x1]
            render_source = page_dir / "rerun-base.png"
            Image.fromarray(canvas).save(
                render_source, format="PNG", optimize=False, compress_level=9
            )
            render_regions = selected_regions
            render_translations = selected_translations
        try:
            rendered = run_rendering(
                render_source,
                render_regions,
                render_translations,
                page_dir,
                config,
                image_id=image_id,
                error_report=report,
            )
        finally:
            if region_ids is not None:
                render_source.unlink(missing_ok=True)
        stage_records = []
        for item in rendered.regions:
            stage_records.append(
                {
                    "region_id": item.region_id,
                    "render_status": item.render_status,
                    "render_warning": item.warning,
                    "font_size_used": item.font_size_used,
                    "line_count": item.line_count,
                    "display_box": list(item.display_box) if item.display_box else None,
                    "direction_used": item.direction_used,
                }
            )
        (page_dir / "render-stage.json").write_text(
            json.dumps(stage_records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        result = rendered

    if stage != "regions":
        _update_metadata_stage(page_dir, image_id, stage, region_ids=region_ids)
    if stage == "render":
        metadata_path = page_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") == "failed":
            metadata["status"] = "ok"
            metadata.pop("error_message", None)
            write_metadata(page_dir, metadata)
            run_record_path = run_dir / "run.json"
            run_record = json.loads(run_record_path.read_text(encoding="utf-8"))
            statuses = run_record["page_statuses"]
            statuses[image_id] = "success"
            run_record["counts"]["pages_succeeded"] = sum(
                value in {"success", "no_regions"} for value in statuses.values()
            )
            run_record["counts"]["pages_failed"] = sum(
                value not in {"success", "no_regions"} for value in statuses.values()
            )
            validated = TranslationRun(
                run_id=run_record["run_id"],
                segmentation_method=run_record["segmentation_method"],
                inpainting_source=run_record["inpainting_source"],
                target_language=run_record["target_language"],
                ocr_engine=run_record["ocr_engine"],
                provider=run_record["translation_provider"],
                rendering=run_record["rendering_config"],
                resolved_config=run_record["resolved_config"],
                input_image_ids=tuple(run_record["input_image_ids"]),
                page_statuses=statuses,
                page_list_identity=run_record["page_list_identity"],
                input_image_identity=run_record["input_image_identity"],
                region_extraction_config=run_record["region_extraction_config"],
                counts=run_record["counts"],
            )
            run_record_path.write_text(
                json.dumps(validated.to_dict(), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    (run_dir / "errors.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def rerun_stage(
    stage_name: str,
    image_id: str,
    run_dir: str | Path,
    *,
    provider: TranslationProviderAdapter | None = None,
    recognizer: Callable[..., Any] | None = None,
) -> Any:
    """Re-run one stage for one page, consuming this run's stored prior artifacts."""
    return _rerun_stage(
        stage_name,
        image_id,
        Path(run_dir),
        provider=provider,
        recognizer=recognizer,
        region_ids=None,
    )


def rerun_region(
    region_id: int,
    image_id: str,
    run_dir: str | Path,
    *,
    stage_name: str = "translate",
    provider: TranslationProviderAdapter | None = None,
    recognizer: Callable[..., Any] | None = None,
) -> Any:
    """Re-run a single region's failed stage without changing sibling records."""
    if not isinstance(region_id, int) or isinstance(region_id, bool) or region_id < 1:
        raise ValueError("region_id must be a positive integer")
    return _rerun_stage(
        stage_name,
        image_id,
        Path(run_dir),
        provider=provider,
        recognizer=recognizer,
        region_ids={region_id},
    )


def export_intermediates(
    run_dir: str | Path, image_id: str, destination: str | Path
) -> Path:
    """Copy a page's complete intermediate tree outside its immutable run tree."""
    run_root = Path(run_dir).resolve()
    record = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    if run_root.name != record.get("run_id"):
        raise ValueError("run directory does not match its run.json run_id")
    if image_id not in record.get("input_image_ids", []):
        raise KeyError(f"image_id is not part of run {run_root.name!r}: {image_id!r}")
    source = page_directory(
        run_root.parent,
        run_root.name,
        record["segmentation_method"],
        image_id,
    ).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"page output directory not found: {source}")
    target = Path(destination).resolve()
    if (
        target == run_root
        or target.is_relative_to(run_root)
        or run_root.is_relative_to(target)
    ):
        raise ValueError(
            "export destination must be outside and disjoint from the run tree"
        )
    if target.exists():
        raise FileExistsError(f"export destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    return target
