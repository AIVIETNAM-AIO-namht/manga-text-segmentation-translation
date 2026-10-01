"""Receive and validate one prediction mask for Spec 003 (US1).

* T015 - input location: one code path for all four identities; only the root
  differs, and the caller takes it from ``configs/inpainting.json``.
* T016 - mapping: a mask is tied to a manifest page by exact name only (FR-008).
* T017 - the receipt checks of ``contracts/intake-validation.md`` (FR-010).

Intake is a pure reader. It writes nothing, so a rejected input produces no
artifact and is never altered (FR-013). A refusal is raised as
``MaskRejection`` and the caller records it and carries on (FR-011).

Ground truth is not read: ``Pair.mask_path`` is the GT path and is never opened
here. Nor does this module look at ``notebooks/deliverables/`` or at
``benchmark/availability.json`` - see the contract's "Input roots" for why.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from manga_text_seg.imaging import ImageError, load_raw_page
from manga_text_seg.manifest import Manifest, Pair
from manga_text_seg.maskproc import align_mask, check_not_empty, normalize_mask
from manga_text_seg.runs import AlignmentRecord, MaskRejection

#: Layout under a source root (contract "Input roots"). Fixed by the contract.
MASKS_DIRNAME = "masks"
METADATA_DIRNAME = "metadata"
MASK_SUFFIX = ".png"
SIDECAR_SUFFIX = ".json"


@dataclass(frozen=True)
class AcceptedMask:
    """A mask that passed every check, ready for dilation and inpainting.

    ``mask`` is normalized and aligned. ``raw_size`` and ``alignment`` record
    what it looked like on arrival and what was done to it (FR-018).
    """

    image_id: str
    page_path: Path
    mask_path: Path
    sidecar_path: Path
    mask: np.ndarray
    raw_size: tuple[int, int]
    alignment: AlignmentRecord


def source_available(source_root: Path) -> bool:
    """Whether ``source_root`` holds any admitted mask at all.

    An identity with nothing under ``masks/`` (Method A left unavailable, or a
    root still pointing at a placeholder) is skipped by the caller as a whole,
    rather than reported as one missing mask per page.
    """
    masks_dir = Path(source_root) / MASKS_DIRNAME
    return masks_dir.is_dir() and any(masks_dir.rglob(f"*{MASK_SUFFIX}"))


def _map_to_page(manifest: Manifest, manga: str, stem: str, image_id: str) -> Pair:
    """Check 3: exactly one manifest page, by exact name (FR-008, SC-001).

    Zero matches and several matches are both refused. There is no fallback to
    similarity, case folding or renumbering.
    """
    matches = [p for p in manifest.pairs if p.manga == manga and p.stem == stem]
    if not matches:
        raise MaskRejection(
            "image_id_mismatch", f"no manifest page is named {image_id!r}", image_id
        )
    if len(matches) > 1:
        raise MaskRejection(
            "image_id_mismatch",
            f"ambiguous mapping: {len(matches)} manifest pages are named {image_id!r}",
            image_id,
        )
    return matches[0]


def _check_source_page(page_path: Path, image_id: str) -> None:
    """Check 1: the source page exists and decodes."""
    if not page_path.is_file():
        raise MaskRejection(
            "missing_source_image", f"source page not found: {page_path}", image_id
        )
    try:
        load_raw_page(page_path)
    except ImageError as error:
        raise MaskRejection("corrupt_image", str(error), image_id) from error


def _read_mask(mask_path: Path, image_id: str) -> np.ndarray:
    """Check 2: the mask exists and decodes, read as stored.

    Not ``imaging.load_mask``: that one folds "cannot decode" together with
    "not binary" and refuses multi-channel files, and this contract needs the
    two told apart (FR-012) and channel information kept (FR-014).
    """
    if not mask_path.is_file():
        raise MaskRejection(
            "missing_prediction_mask",
            f"prediction mask not found: {mask_path}",
            image_id,
        )
    raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise MaskRejection(
            "corrupt_image", f"prediction mask cannot be decoded: {mask_path}", image_id
        )
    return raw


def _check_sidecar(sidecar_path: Path, image_id: str) -> None:
    """Check 7: the sidecar is present and names this ``image_id`` (FR-009).

    Only ``sidecar_path`` is looked at; a sidecar is never joined in from
    another root.
    """
    if not sidecar_path.is_file():
        raise MaskRejection(
            "missing_metadata", f"sidecar not found: {sidecar_path}", image_id
        )
    try:
        document = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise MaskRejection(
            "missing_metadata", f"sidecar cannot be read: {sidecar_path}", image_id
        ) from error
    if not isinstance(document, dict):
        raise MaskRejection(
            "missing_metadata",
            f"sidecar is not a JSON object: {sidecar_path}",
            image_id,
        )
    named = document.get("image_id")
    if named != image_id:
        raise MaskRejection(
            "image_id_mismatch",
            f"sidecar names {named!r} but the mask is {image_id!r}",
            image_id,
        )


def intake_mask(
    source_root: Path, manifest: Manifest, manga: str, stem: str
) -> AcceptedMask:
    """Receive one mask from ``source_root`` and run every check on it.

    ``manga`` and ``stem`` are the names the mask carries on disk. Raises
    ``MaskRejection`` (with ``image_id`` set) on the first check that fails, in
    contract order; returns an ``AcceptedMask`` otherwise.
    """
    root = Path(source_root)
    image_id = f"{manga}/{stem}"

    page = _map_to_page(manifest, manga, stem, image_id)
    _check_source_page(page.raw_path, image_id)

    mask_path = root / MASKS_DIRNAME / manga / f"{stem}{MASK_SUFFIX}"
    sidecar_path = root / METADATA_DIRNAME / manga / f"{stem}{SIDECAR_SUFFIX}"
    raw = _read_mask(mask_path, image_id)
    raw_size = (int(raw.shape[1]), int(raw.shape[0]))

    # Values are checked on the full canvas *before* alignment, so a stray
    # value in the padding that a crop would discard is still caught.
    try:
        normalized = normalize_mask(raw)
        aligned, alignment = align_mask(normalized)
        check_not_empty(aligned)
    except MaskRejection as rejection:
        rejection.image_id = image_id
        raise

    _check_sidecar(sidecar_path, image_id)

    return AcceptedMask(
        image_id=image_id,
        page_path=page.raw_path,
        mask_path=mask_path,
        sidecar_path=sidecar_path,
        mask=aligned,
        raw_size=raw_size,
        alignment=alignment,
    )
