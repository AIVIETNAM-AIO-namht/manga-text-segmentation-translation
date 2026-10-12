"""Read-only assembly and validation of Spec 1–3 page inputs for Spec 004."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import SEGMENTATION_IDENTITIES, TranslationConfig
from ..manifest import Manifest, load_manifest
from .runs import ErrorReport, page_directory

ALIGNED_SIZE = (1654, 1170)  # width, height


class IntakeRejectionError(ValueError):
    """A page bundle that cannot be admitted, with its ErrorReport category."""

    def __init__(self, image_id: str, category: str, reason: str) -> None:
        super().__init__(reason)
        self.image_id = image_id
        self.category = category
        self.reason = reason


@dataclass(frozen=True)
class PageBundle:
    """Resolved paths and verbatim provenance for one manifest page."""

    image_id: str
    original_image_path: Path
    prediction_mask_path: Path
    prediction_mask_sidecar_path: Path
    inpainted_image_path: Path
    inpainting_metadata_path: Path
    inpainting_provenance: dict[str, Any]
    page_list_identity: str
    input_image_identity: str
    status: str
    reason: str | None = None
    mask_copy_path: Path | None = None


def _read_json(
    path: Path, *, image_id: str, category: str, label: str
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IntakeRejectionError(
            image_id, category, f"cannot read {label} JSON sidecar {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise IntakeRejectionError(
            image_id, category, f"{label} sidecar must contain a JSON object"
        )
    return value


def _load_manifest(value: Manifest | str | Path) -> Manifest:
    return value if isinstance(value, Manifest) else load_manifest(Path(value))


def _resolve_method(config: TranslationConfig) -> str:
    method = config.segmentation_method
    if not isinstance(method, str) or not method:
        raise ValueError(
            "segmentation method is not configured and this repository has no Spec 2 "
            "benchmark method-selection record to resolve it"
        )
    if method not in SEGMENTATION_IDENTITIES:
        raise ValueError(f"unsupported segmentation method: {method!r}")
    return method


def _record_rejection(report: ErrorReport, exc: IntakeRejectionError) -> None:
    report.add(
        exc.category,
        image_id=exc.image_id,
        region_id=None,
        stage="intake",
        reason=exc.reason,
    )


def _reject(
    image_id: str,
    category: str,
    reason: str,
    report: ErrorReport,
) -> None:
    rejection = IntakeRejectionError(image_id, category, reason)
    _record_rejection(report, rejection)
    raise rejection


def assemble_bundle(
    config: TranslationConfig,
    image_id: str,
    run_id: str,
    *,
    error_report: ErrorReport,
    repo_root: str | Path,
    prediction_root: str | Path | None = None,
    inpainting_root: str | Path | None = None,
) -> PageBundle:
    """Resolve a manifest page and copy its admitted Spec 2 mask into this run.

    ``prediction_root`` is an optional Spec 2 method root (the directory
    containing ``masks/`` and ``metadata/``). If omitted, the path quoted by
    the Spec 3 sidecar is authoritative. ``inpainting_root`` is the
    selected Spec 3 run directory containing ``run.json`` and method folders.
    These optional roots make fixture/upstream handoffs addressable without
    guessing their internal paths.
    """
    if error_report.run_id != run_id:
        raise ValueError("error_report.run_id must match the translation run_id")
    repo = Path(repo_root).resolve()
    try:
        manifest = _load_manifest(config.manifest_path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        _reject(
            image_id,
            "missing_original_image",
            f"cannot load configured Spec 1 manifest: {exc}",
            error_report,
        )
    pairs = [pair for pair in manifest.pairs if pair.image_id == image_id]
    if len(pairs) != 1:
        _reject(
            image_id,
            "missing_original_image",
            f"image_id {image_id!r} must identify exactly one entry in the "
            f"configured Spec 1 manifest; found {len(pairs)}",
            error_report,
        )
    pair = pairs[0]
    original = Path(pair.raw_path)
    if not original.is_file():
        _reject(
            image_id,
            "missing_original_image",
            f"manifest original image is missing: {original}",
            error_report,
        )
    original_pixels = cv2.imread(str(original), cv2.IMREAD_UNCHANGED)
    if original_pixels is None:
        _reject(
            image_id,
            "missing_original_image",
            f"manifest original image is unreadable: {original}",
            error_report,
        )
    if original_pixels.shape[:2] != (ALIGNED_SIZE[1], ALIGNED_SIZE[0]):
        _reject(
            image_id,
            "missing_original_image",
            f"original image size is {original_pixels.shape[1]}x"
            f"{original_pixels.shape[0]}, "
            f"expected {ALIGNED_SIZE[0]}x{ALIGNED_SIZE[1]}",
            error_report,
        )

    try:
        method = _resolve_method(config)
    except ValueError as exc:
        _reject(image_id, "missing_prediction_mask", str(exc), error_report)
    manga, stem = image_id.split("/", 1)
    selected_inpainting_run = (
        Path(inpainting_root).resolve()
        if inpainting_root
        else repo / "outputs" / "inpainting" / config.inpainting_run_id
    )
    inpainting_page_dir = (
        selected_inpainting_run / method / config.inpainting_algorithm / manga
    )
    inpainted_path = inpainting_page_dir / f"{stem}.png"
    inpaint_metadata_path = inpainting_page_dir / f"{stem}.json"
    if not inpainted_path.is_file():
        _reject(
            image_id,
            "missing_inpainted_image",
            f"configured Spec 3 inpainted image is missing: {inpainted_path}",
            error_report,
        )
    if not inpaint_metadata_path.is_file():
        _reject(
            image_id,
            "missing_inpainted_image",
            f"configured Spec 3 metadata sidecar is missing: {inpaint_metadata_path}",
            error_report,
        )
    inpaint_metadata = _read_json(
        inpaint_metadata_path,
        image_id=image_id,
        category="missing_inpainted_image",
        label="Spec 3 inpainting metadata",
    )
    required_inpaint_fields = {
        "image_id",
        "method",
        "algorithm",
        "model_repository",
        "experiment_id",
        "input_image_path",
        "prediction_mask_path",
        "mask_size",
        "alignment",
        "dilation",
        "inpaint_algorithm",
        "inpaint_radius",
        "output_format",
        "processing_time_seconds",
        "status",
    }
    missing = required_inpaint_fields - inpaint_metadata.keys()
    if missing:
        _reject(
            image_id,
            "missing_inpainted_image",
            f"Spec 3 sidecar is missing required fields: {sorted(missing)}",
            error_report,
        )
    if inpaint_metadata["image_id"] != image_id or inpaint_metadata["method"] != method:
        _reject(
            image_id,
            "missing_inpainted_image",
            "Spec 3 sidecar image_id or method disagrees with the configured page",
            error_report,
        )
    if (
        inpaint_metadata["algorithm"] != config.inpainting_algorithm
        or inpaint_metadata["status"] != "ok"
    ):
        _reject(
            image_id,
            "missing_inpainted_image",
            "Spec 3 sidecar algorithm or status disagrees with the configured source",
            error_report,
        )
    if (
        inpaint_metadata["inpaint_algorithm"]
        != f"INPAINT_{config.inpainting_algorithm.upper()}"
    ):
        _reject(
            image_id,
            "missing_inpainted_image",
            "Spec 3 sidecar inpaint_algorithm disagrees with the configured algorithm",
            error_report,
        )
    if (
        not isinstance(inpaint_metadata["prediction_mask_path"], str)
        or not inpaint_metadata["prediction_mask_path"]
    ):
        _reject(
            image_id,
            "missing_inpainted_image",
            "Spec 3 sidecar prediction_mask_path must be a non-empty path",
            error_report,
        )
    if (
        not isinstance(inpaint_metadata["input_image_path"], str)
        or not inpaint_metadata["input_image_path"]
    ):
        _reject(
            image_id,
            "missing_inpainted_image",
            "Spec 3 sidecar input_image_path must be a non-empty path",
            error_report,
        )
    sidecar_mask_size = inpaint_metadata["mask_size"]
    if (
        not isinstance(sidecar_mask_size, list)
        or len(sidecar_mask_size) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in sidecar_mask_size
        )
    ):
        _reject(
            image_id,
            "missing_inpainted_image",
            "Spec 3 sidecar mask_size must be a [width, height] pair",
            error_report,
        )
    alignment = inpaint_metadata["alignment"]
    if (
        not isinstance(alignment, dict)
        or alignment.get("rule") not in {"pass-through", "crop-topleft"}
        or alignment.get("post_size") != [ALIGNED_SIZE[0], ALIGNED_SIZE[1]]
    ):
        _reject(
            image_id,
            "missing_inpainted_image",
            "Spec 3 sidecar alignment must resolve to [1654, 1170]",
            error_report,
        )
    dilation = inpaint_metadata["dilation"]
    if not isinstance(dilation, dict) or not isinstance(dilation.get("enabled"), bool):
        _reject(
            image_id,
            "missing_inpainted_image",
            "Spec 3 sidecar dilation must declare enabled as a boolean",
            error_report,
        )
    radius = inpaint_metadata["inpaint_radius"]
    if not isinstance(radius, (int, float)) or isinstance(radius, bool) or radius <= 0:
        _reject(
            image_id,
            "missing_inpainted_image",
            "Spec 3 sidecar inpaint_radius must be positive",
            error_report,
        )
    experiment_id = inpaint_metadata["experiment_id"]
    if not isinstance(experiment_id, str) or not experiment_id:
        _reject(
            image_id,
            "missing_prediction_mask",
            "Spec 3 sidecar has no usable experiment_id for the prediction hand-off",
            error_report,
        )

    recorded_prediction_path = Path(inpaint_metadata["prediction_mask_path"])
    if not recorded_prediction_path.is_absolute():
        recorded_prediction_path = repo / recorded_prediction_path
    recorded_prediction_path = recorded_prediction_path.resolve()
    if (
        recorded_prediction_path.name != f"{stem}.png"
        or recorded_prediction_path.parent.name != manga
        or recorded_prediction_path.parent.parent.name != "masks"
    ):
        _reject(
            image_id,
            "missing_prediction_mask",
            "Spec 3 prediction_mask_path does not address this image_id in "
            "the Spec 2 masks layout",
            error_report,
        )
    method_root = (
        Path(prediction_root).resolve()
        if prediction_root is not None
        else recorded_prediction_path.parents[2]
    )
    prediction_path = method_root / "masks" / manga / f"{stem}.png"
    if prediction_path.resolve() != recorded_prediction_path:
        _reject(
            image_id,
            "missing_prediction_mask",
            "configured prediction root disagrees with Spec 3 prediction_mask_path",
            error_report,
        )
    mask_sidecar_path = method_root / "metadata" / manga / f"{stem}.json"
    if not prediction_path.is_file():
        _reject(
            image_id,
            "missing_prediction_mask",
            f"Spec 2 prediction mask is missing: {prediction_path}",
            error_report,
        )
    if not mask_sidecar_path.is_file():
        _reject(
            image_id,
            "missing_prediction_mask",
            f"Spec 2 prediction mask sidecar is missing: {mask_sidecar_path}",
            error_report,
        )
    mask_sidecar = _read_json(
        mask_sidecar_path,
        image_id=image_id,
        category="missing_prediction_mask",
        label="Spec 2 mask metadata",
    )

    page_list_path = repo / "benchmark" / "page-list.json"
    image_identity_path = repo / "benchmark" / "image-identity.json"
    try:
        page_list = json.loads(page_list_path.read_text(encoding="utf-8"))
        identity_record = json.loads(image_identity_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _reject(
            image_id,
            "missing_prediction_mask",
            f"cannot read committed Spec 2 identity records: {exc}",
            error_report,
        )
    pages = [
        page for page in page_list.get("pages", []) if page.get("image_id") == image_id
    ]
    if len(pages) != 1:
        _reject(
            image_id,
            "missing_original_image",
            f"image_id {image_id!r} is not uniquely present in the committed "
            "Spec 2 page list",
            error_report,
        )
    canonical_page = pages[0]
    expected_page_identity = page_list.get("page_list_identity")
    expected_image_identity = canonical_page.get("input_image_identity")
    identity_entry = identity_record.get(image_id)
    if not isinstance(expected_page_identity, str) or not isinstance(
        expected_image_identity, str
    ):
        _reject(
            image_id,
            "missing_prediction_mask",
            "committed Spec 2 page list is missing an identity value",
            error_report,
        )
    if (
        not isinstance(identity_entry, dict)
        or identity_entry.get("sha256") != expected_image_identity
    ):
        _reject(
            image_id,
            "missing_prediction_mask",
            "Spec 2 page-list and image-identity records disagree",
            error_report,
        )
    # ponytail: Spec 2's adapters register the hyphenated spelling
    # (``unetpp-efficientnetv2``) while Specs 3-5 fix the underscored identity.
    # Same method, two vocabularies; normalize rather than rewrite either.
    sidecar_method = mask_sidecar.get("method")
    if isinstance(sidecar_method, str):
        sidecar_method = sidecar_method.replace("-", "_")
    if mask_sidecar.get("image_id") != image_id or sidecar_method != method:
        _reject(
            image_id,
            "missing_prediction_mask",
            "Spec 2 sidecar image_id or method disagrees with the requested page",
            error_report,
        )
    if mask_sidecar.get("page_list_identity") != expected_page_identity:
        _reject(
            image_id,
            "missing_prediction_mask",
            "Spec 2 sidecar page_list_identity does not match the committed page list",
            error_report,
        )
    if mask_sidecar.get("input_image_identity") != expected_image_identity:
        _reject(
            image_id,
            "missing_prediction_mask",
            "Spec 2 sidecar input_image_identity does not match the committed "
            "image identity",
            error_report,
        )
    if mask_sidecar.get("status") != "success":
        _reject(
            image_id,
            "missing_prediction_mask",
            "Spec 2 sidecar status is not success",
            error_report,
        )
    if mask_sidecar.get("output_size") != [ALIGNED_SIZE[0], ALIGNED_SIZE[1]]:
        _reject(
            image_id,
            "missing_prediction_mask",
            "Spec 2 sidecar output_size is not [1654, 1170]",
            error_report,
        )
    if mask_sidecar.get("output_size") != [ALIGNED_SIZE[0], ALIGNED_SIZE[1]]:
        _reject(
            image_id,
            "missing_prediction_mask",
            "Spec 2 sidecar output_size is not [1654, 1170]",
            error_report,
        )

    mask = cv2.imread(str(prediction_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        _reject(
            image_id,
            "missing_prediction_mask",
            f"Spec 2 prediction mask is unreadable: {prediction_path}",
            error_report,
        )
    if mask.dtype != np.uint8 or mask.ndim != 2:
        _reject(
            image_id,
            "missing_prediction_mask",
            "prediction mask must be single-channel uint8",
            error_report,
        )
    if mask.shape != (ALIGNED_SIZE[1], ALIGNED_SIZE[0]):
        _reject(
            image_id,
            "missing_prediction_mask",
            f"prediction mask shape is {mask.shape}, expected "
            f"{(ALIGNED_SIZE[1], ALIGNED_SIZE[0])}",
            error_report,
        )
    if not np.isin(mask, (0, 255)).all():
        _reject(
            image_id,
            "missing_prediction_mask",
            "prediction mask contains values outside {0, 255}",
            error_report,
        )

    inpainted = cv2.imread(str(inpainted_path), cv2.IMREAD_UNCHANGED)
    if inpainted is None:
        _reject(
            image_id,
            "missing_inpainted_image",
            f"Spec 3 inpainted image is unreadable: {inpainted_path}",
            error_report,
        )
    if inpainted.shape[:2] != (ALIGNED_SIZE[1], ALIGNED_SIZE[0]):
        _reject(
            image_id,
            "missing_inpainted_image",
            f"inpainted image size is {inpainted.shape[1]}x{inpainted.shape[0]}, "
            f"expected {ALIGNED_SIZE[0]}x{ALIGNED_SIZE[1]}; images are never resized",
            error_report,
        )

    target_page_dir = page_directory(
        config.output_roots.translation, run_id, method, image_id
    )
    target_page_dir.mkdir(parents=True, exist_ok=True)
    mask_copy_path = target_page_dir / "mask.png"
    if mask_copy_path.exists():
        _reject(
            image_id,
            "output_write_failure",
            f"refusing to overwrite existing translation mask copy: {mask_copy_path}",
            error_report,
        )
    try:
        shutil.copyfile(prediction_path, mask_copy_path)
    except OSError as exc:
        _reject(
            image_id,
            "output_write_failure",
            f"could not copy prediction mask into translation run: {exc}",
            error_report,
        )
    if mask_copy_path.read_bytes() != prediction_path.read_bytes():
        _reject(
            image_id,
            "output_write_failure",
            "copied prediction mask is not byte-identical to its upstream source",
            error_report,
        )

    if not np.any(mask):
        error_report.add(
            "empty_mask",
            image_id=image_id,
            region_id=None,
            stage="intake",
            reason="prediction mask is valid but contains no text pixels",
        )
        status, reason = "empty_mask", "prediction mask contains no text pixels"
    else:
        status, reason = "accepted", None
    return PageBundle(
        image_id=image_id,
        original_image_path=original,
        prediction_mask_path=prediction_path,
        prediction_mask_sidecar_path=mask_sidecar_path,
        inpainted_image_path=inpainted_path,
        inpainting_metadata_path=inpaint_metadata_path,
        inpainting_provenance={
            key: inpaint_metadata[key]
            for key in ("algorithm", "inpaint_algorithm", "inpaint_radius", "dilation")
        },
        page_list_identity=expected_page_identity,
        input_image_identity=expected_image_identity,
        status=status,
        reason=reason,
        mask_copy_path=mask_copy_path,
    )
