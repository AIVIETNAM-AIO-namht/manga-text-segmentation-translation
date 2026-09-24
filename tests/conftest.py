"""Pytest bootstrap and the fixtures spec 002's tests share (T006).

The bootstrap half makes ``manga_text_seg`` importable from the src layout: the
package uses a ``src/`` layout (pyproject.toml), so ``src/`` is inserted at the
front of ``sys.path`` and the suite runs without a prior
``pip install -e ".[dev]"``.  Harmless when the package is already installed —
the repo's own source is then preferred, which is exactly what the tests assert
against.

The fixtures half supplies the three things every story's tests need
(FR-050, FR-050a): a two-page manifest drawn from the real one, a synthetic
binary mask at the aligned page size, and a hand-off builder that writes the
``returned-result.md`` layout.  None of them loads a checkpoint or touches the
network (FR-051).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_MANIFEST = REPO_ROOT / "outputs" / "segmentation" / "default" / "manifest.json"

#: Aligned page size as the page-list schema records it: ``[width, height]``.
ALIGNED_SIZE = [1654, 1170]
#: The same size as an ``np.ndarray`` shape is ``(height, width)``.
ALIGNED_SHAPE = (1170, 1654)
#: How many pages the fixtures draw from the real manifest.
FIXTURE_PAGES = 2


@pytest.fixture(scope="session")
def real_manifest():
    """The real 390-pair manifest, or a skip when the dataset is absent."""
    from manga_text_seg.manifest import load_manifest

    if not REAL_MANIFEST.is_file():
        pytest.skip(f"Real manifest not present: {REAL_MANIFEST}")
    return load_manifest(REAL_MANIFEST)


@pytest.fixture()
def two_page_manifest(tmp_path: Path, real_manifest) -> Path:
    """A manifest file holding the first two real pairs (ARMS/000, ARMS/001).

    Written through the real ``save_manifest`` so its shape is the shape
    ``load_manifest`` expects, and pointing at the real raw images so the
    fixtures hash real bytes.
    """
    from manga_text_seg.manifest import Manifest, save_manifest

    subset = Manifest(
        run_id="fixture",
        gt_root=real_manifest.gt_root,
        raw_root=real_manifest.raw_root,
        pairs=real_manifest.pairs[:FIXTURE_PAGES],
    )
    path = tmp_path / "manifest.json"
    save_manifest(subset, path)
    return path


@pytest.fixture()
def aligned_mask() -> np.ndarray:
    """A deterministic synthetic prediction mask at the aligned page size.

    Single-channel uint8, values ⊆ {0, 255}, shape ``(1170, 1654)`` — the
    convention every admitted mask must satisfy (FR-004, FR-043).
    """
    mask = np.zeros(ALIGNED_SHAPE, dtype=np.uint8)
    mask[200:400, 300:900] = 255
    mask[800:900, 1200:1500] = 255
    return mask


@pytest.fixture()
def make_handoff(tmp_path: Path):
    """Return a builder that writes a hand-off directory.

    ``build(page_list, masks, method=..., provenance=..., sidecar=...)`` writes
    ``masks/<manga>/<stem>.png``, ``metadata/<manga>/<stem>.json``,
    ``provenance.json`` and ``errors.json`` — the layout of
    ``contracts/returned-result.md`` — and returns the directory.

    ``masks`` maps ``image_id`` to an ``np.ndarray``; a sidecar is derived per
    page from the page list so the identities are quoted verbatim, which is
    what the receipt checks.  ``provenance`` and ``sidecar`` are update hooks
    for the refusal tests: ``sidecar(image_id, doc) -> doc`` and
    ``provenance(doc) -> doc`` each receive the generated document and return
    the perturbed one.
    """
    from manga_text_seg.imaging import save_mask

    counter = {"n": 0}

    def _build(
        page_list: dict,
        masks: dict[str, np.ndarray],
        *,
        method: str = "standin",
        provenance: dict | None = None,
        sidecar=None,
    ) -> Path:
        counter["n"] += 1
        root = tmp_path / f"handoff-{counter['n']}"
        (root / "masks").mkdir(parents=True)
        (root / "metadata").mkdir(parents=True)

        by_id = {p["image_id"]: p for p in page_list["pages"]}
        for image_id, mask in masks.items():
            page = by_id[image_id]
            manga, stem = page["manga"], page["stem"]
            save_mask(root / "masks" / manga / f"{stem}.png", mask)

            doc = {
                "image_id": image_id,
                "page_list_identity": page_list["page_list_identity"],
                "input_image_identity": page["input_image_identity"],
                "method": method,
                # [width, height], matching aligned_size's order.
                "output_size": [mask.shape[1], mask.shape[0]],
                "status": "ok",
                "fold_attribution": None,
            }
            if sidecar is not None:
                doc = sidecar(image_id, doc)
            meta_dir = root / "metadata" / manga
            meta_dir.mkdir(parents=True, exist_ok=True)
            (meta_dir / f"{stem}.json").write_text(
                json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )

        record = provenance if provenance is not None else _provenance_record(method)
        (root / "provenance.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (root / "errors.json").write_text("[]\n", encoding="utf-8")
        return root

    return _build


@pytest.fixture()
def default_provenance() -> dict:
    """A complete provenance record, freshly built per test.

    A fixture rather than a bare helper so each test gets its own dict — the
    refusal tests mutate what they are handed.
    """
    return _provenance_record()


@pytest.fixture(scope="session")
def real_page_list(real_manifest) -> dict:
    """The page list built from the real 390-pair manifest (FR-054).

    Session-scoped: building it decodes and hashes all 390 raw images, and both
    the export tests and the schema tests need the same instance.
    """
    from manga_text_seg.pagelist import build_page_list

    return build_page_list(real_manifest)


@pytest.fixture()
def exported(tmp_path: Path, two_page_manifest: Path) -> dict:
    """A two-page export read back from disk: page list, identity record, bundle.

    Goes through the real ``export``, so the page list the receipt tests hand
    out is one this project actually produced rather than one a test assembled.
    """
    from manga_text_seg.manifest import load_manifest
    from manga_text_seg.pagelist import export, load_identity_record, load_page_list

    benchmark_dir = tmp_path / "benchmark"
    export(load_manifest(two_page_manifest), benchmark_dir)
    return {
        "dir": benchmark_dir,
        "manifest": load_manifest(two_page_manifest),
        "page_list": load_page_list(benchmark_dir),
        "identity_record": load_identity_record(benchmark_dir),
    }


def _provenance_record(method: str = "standin") -> dict:
    """A provenance record with every field ``provenance.schema.json`` requires.

    Deliberately a *complete* record: the refusal tests perturb one field each,
    so anything missing here would be refused for the wrong reason.
    """
    return {
        "method": method,
        "repository": {
            "url": "https://example.invalid/standin",
            "revision": "0" * 40,
        },
        "checkpoint": {
            "identity": "standin.bin",
            "size_bytes": 1,
            "sha256": "0" * 64,
        },
        "code_license": "MIT",
        "weight_license": "MIT",
        "checkpoint_load_evidence": {
            "evidenced": True,
            "method": "state_dict key set plus checksum comparison",
            "detail": "fixture: 1 key, sha256 0000...",
        },
        "device": {"type": "cpu", "name": "fixture-cpu"},
        "interpreter": "3.11.0",
        "packages": {"numpy": "2.2.6"},
        "seed": 42,
        "run_timestamp": "2026-09-19T00:00:00+07:00",
    }
