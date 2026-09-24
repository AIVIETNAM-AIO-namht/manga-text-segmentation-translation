"""The distributable page list and its image bundle (FR-054, FR-054a).

Two artefacts leave this project: the page list, which says *which* pages are
being benchmarked and under *which* conventions, and the bundle of input images
those pages denote.  Neither carries a pixel of ground truth — FR-054a's "the
images are the question and the masks are the answer".

Both are relative to each other (research.md R2), so a runner can relocate the
pair without rewriting anything, and both are covered by the page list's own
identity, so a receipt can tell whether the list it was handed is the list this
project published.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .imaging import load_raw_page
from .manifest import Manifest

#: The page list's filename, and the identity record that travels beside it.
PAGE_LIST_NAME = "page-list.json"
IMAGE_IDENTITY_NAME = "image-identity.json"

#: Where the input images live, relative to the page list.
BUNDLE_ROOT = "dist/images"

#: The one field the identity covers but cannot contain.
IDENTITY_FIELD = "page_list_identity"

#: The shared alignment convention (FR-053c) — see ``align.py``.
ALIGNMENT = "crop-topleft"

#: The shared mask convention (FR-004), restated so a runner needs no access here.
MASK_CONVENTION = {
    "dtype": "uint8",
    "channels": 1,
    "values": [0, 255],
    "text": 255,
    "background": 0,
}

_CHUNK = 1 << 20


class PageListError(RuntimeError):
    """Raised when a manifest cannot be turned into a distributable page list."""


def sha256_file(path: Path) -> str:
    """The sha256 of a file's exact bytes, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(doc: dict) -> str:
    """The serialisation the identity is computed over, and what is written."""
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def page_list_identity(doc: dict) -> str:
    """The identity of a page list: sha256 over its canonical form, field omitted.

    Recomputable, so a receipt verifies it without trusting the file it arrived
    in; covering the whole document, so an edit that drops a page changes it.
    """
    body = {key: value for key, value in doc.items() if key != IDENTITY_FIELD}
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


def build_page_list(manifest: Manifest) -> dict[str, Any]:
    """Derive the page list from a manifest (FR-054).

    Every page is decoded to read its size, and the sizes must agree: the list
    records one ``aligned_size``, and publishing the first page's size would
    silently misdescribe every page after it (FR-024a).
    """
    pairs = sorted(manifest.pairs, key=lambda pair: (pair.manga, pair.stem))
    if not pairs:
        raise PageListError(
            f"Manifest {manifest.run_id!r} holds no pairs; nothing to export"
        )

    pages = []
    aligned: tuple[int, int] | None = None

    for pair in pairs:
        image = load_raw_page(pair.raw_path)
        if image.ndim != 2:
            raise PageListError(
                f"{pair.image_id}: expected a single-channel page, got shape "
                f"{image.shape}"
            )
        size = (image.shape[1], image.shape[0])
        if aligned is None:
            aligned = size
        elif size != aligned:
            raise PageListError(
                f"{pair.image_id}: aligned size {size} differs from {aligned} "
                f"recorded for the pages before it; one aligned size is recorded "
                f"for the whole list (FR-024a)"
            )

        pages.append(
            {
                "manga": pair.manga,
                "stem": pair.stem,
                "image_id": pair.image_id,
                "image_ref": f"{pair.manga}/{pair.raw_path.name}",
                "input_image_identity": sha256_file(pair.raw_path),
            }
        )

    doc: dict[str, Any] = {
        "generated_at_run": manifest.run_id,
        "aligned_size": list(aligned),
        "alignment": ALIGNMENT,
        "mask_convention": MASK_CONVENTION,
        "image_bundle": {"root": BUNDLE_ROOT, "identity_record": IMAGE_IDENTITY_NAME},
        "pages": pages,
    }
    doc[IDENTITY_FIELD] = page_list_identity(doc)
    return doc


def write_image_bundle(manifest: Manifest, benchmark_dir: Path, doc: dict) -> dict:
    """Copy the paired raw images into the bundle; return the identity record.

    The record is built from the *source* file's bytes — the bytes the classical
    baseline ran on (FR-054a) — and every file is copied unconditionally.  A
    "same size, skip it" shortcut is exactly the silent corruption this project
    refuses.
    """
    by_id = {page["image_id"]: page for page in doc["pages"]}
    root = Path(benchmark_dir) / BUNDLE_ROOT
    record: dict[str, dict] = {}

    for pair in manifest.pairs:
        page = by_id.get(pair.image_id)
        if page is None:
            raise PageListError(f"{pair.image_id}: in the manifest but not in the list")

        digest = sha256_file(pair.raw_path)
        if digest != page["input_image_identity"]:
            raise PageListError(
                f"{pair.image_id}: source bytes changed between listing and copying"
            )

        destination = root / page["image_ref"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(pair.raw_path.read_bytes())

        record[pair.image_id] = {
            "sha256": digest,
            "size_bytes": pair.raw_path.stat().st_size,
        }

    return record


def write_page_list(doc: dict, benchmark_dir: Path) -> Path:
    """Write ``page-list.json``; return the path it landed at."""
    path = Path(benchmark_dir) / PAGE_LIST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical(doc), encoding="utf-8")
    return path


def export(manifest: Manifest, benchmark_dir: Path) -> Path:
    """Write the page list and the image bundle (FR-054, FR-054a)."""
    benchmark_dir = Path(benchmark_dir)
    doc = build_page_list(manifest)
    record = write_image_bundle(manifest, benchmark_dir, doc)

    for page in doc["pages"]:
        entry = record.get(page["image_id"])
        if entry is None or entry["sha256"] != page["input_image_identity"]:
            raise PageListError(
                f"{page['image_id']}: identity record disagrees with the page list"
            )

    (benchmark_dir / IMAGE_IDENTITY_NAME).write_text(
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return write_page_list(doc, benchmark_dir)


def load_page_list(benchmark_dir: Path) -> dict:
    """Read ``page-list.json`` back."""
    return json.loads(
        (Path(benchmark_dir) / PAGE_LIST_NAME).read_text(encoding="utf-8")
    )


def load_identity_record(benchmark_dir: Path) -> dict:
    """Read ``image-identity.json`` back, keyed by ``image_id``."""
    return json.loads(
        (Path(benchmark_dir) / IMAGE_IDENTITY_NAME).read_text(encoding="utf-8")
    )
