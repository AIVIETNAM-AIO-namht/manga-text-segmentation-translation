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


# --------------------------------------------------------------------------- #
# Spec 3 fixtures — intake + inpainting (T003)
# --------------------------------------------------------------------------- #
@pytest.fixture()
def inpaint_fixture(tmp_path: Path):
    """Return a builder for one admitted-run-shaped mask + sidecar (Spec 3 intake).

    ``build(method, image_id, *, size=ALIGNED_SHAPE, pixel_values=(0, 255),
    empty=False, with_sidecar=True)`` writes exactly one mask and, unless
    ``with_sidecar`` is False, its metadata sidecar, at
    ``<root>/masks/<manga>/<stem>.png`` and ``<root>/metadata/<manga>/<stem>.json``
    (contracts/intake-validation.md "Input roots") under a fresh tmp root that
    stands in for an admitted ``deliverables/<run_id>/<method>/`` directory.

    Returns the root ``Path`` so a test points ``load_inpaint_config``'s
    resolved ``source_root`` at it directly.

    Parameters let a caller perturb exactly one thing at a time:
    - ``size``: an ``(height, width)`` shape different from ``ALIGNED_SHAPE``
      to trigger the "wrong size" rejection.
    - ``pixel_values``: an iterable of values written into the mask; include
      something outside ``{0, 255}`` (e.g. ``(0, 128, 255)``) to trigger the
      "non-binary mask" rejection.
    - ``empty``: ``True`` writes an all-zero mask (the "empty mask" case).
    - ``with_sidecar``: ``False`` skips writing the metadata sidecar (the
      "missing metadata" case).
    """
    from manga_text_seg.imaging import save_mask

    counter = {"n": 0}

    def _build(
        method: str,
        image_id: str,
        *,
        size: tuple[int, int] = ALIGNED_SHAPE,
        pixel_values: tuple[int, ...] = (0, 255),
        empty: bool = False,
        with_sidecar: bool = True,
    ) -> Path:
        counter["n"] += 1
        root = tmp_path / f"inpaint-fixture-{counter['n']}"
        manga, stem = image_id.split("/")

        mask = np.zeros(size, dtype=np.uint8)
        if not empty:
            # Deterministic small text-like blob so "at least one text pixel"
            # holds whenever the mask isn't explicitly requested empty.
            h, w = size
            fill_value = pixel_values[-1] if pixel_values else 255
            mask[h // 4 : h // 4 + 20, w // 4 : w // 4 + 60] = fill_value
            if len(pixel_values) > 1:
                # Stamp every requested value somewhere, for the non-binary case.
                for i, value in enumerate(pixel_values):
                    mask[i, i] = value

        mask_path = root / "masks" / manga / f"{stem}.png"
        save_mask(mask_path, mask)

        if with_sidecar:
            sidecar = {
                "image_id": image_id,
                "page_list_identity": "b52aa60d-fixture",
                "input_image_identity": f"fixture-sha256-{image_id}",
                "method": method,
                "output_size": [mask.shape[1], mask.shape[0]],
                "status": "ok",
                "fold_attribution": None,
            }
            meta_path = root / "metadata" / manga / f"{stem}.json"
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(
                json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        return root

    return _build


@pytest.fixture()
def synthetic_manifest(tmp_path: Path):
    """Return a builder for a small manifest that needs no real dataset (T012, T014).

    ``build(image_ids=("ARMS/000", "ARMS/001"), *, missing_source=(),
    corrupt_source=(), duplicate=())`` returns a ``Manifest`` whose source pages
    are written under ``tmp_path`` at the aligned size.

    - ``missing_source``: listed in the manifest, but its page file is never
      written (the "missing source image" case).
    - ``corrupt_source``: its page file holds bytes that do not decode (the
      "corrupt or undecodable image" case).
    - ``duplicate``: listed twice in ``pairs`` (the "ambiguous mapping" case).

    Every pair's ``mask_path`` (the ground-truth path) points at a file that is
    never created. Intake must not read ground truth (FR-033), so a test that
    passes here also proves it did not try.
    """
    import cv2

    from manga_text_seg.manifest import Manifest, Pair

    counter = {"n": 0}

    def _build(
        image_ids: tuple[str, ...] = ("ARMS/000", "ARMS/001"),
        *,
        missing_source: tuple[str, ...] = (),
        corrupt_source: tuple[str, ...] = (),
        duplicate: tuple[str, ...] = (),
    ) -> Manifest:
        counter["n"] += 1
        raw_root = tmp_path / f"synthetic-raw-{counter['n']}"
        gt_root = tmp_path / f"synthetic-gt-never-created-{counter['n']}"

        pairs: list[Pair] = []
        for image_id in image_ids:
            manga, stem = image_id.split("/")
            raw_path = raw_root / manga / f"{stem}.png"
            if image_id not in missing_source:
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                if image_id in corrupt_source:
                    raw_path.write_bytes(b"this is not an image")
                else:
                    page = np.full(ALIGNED_SHAPE, 200, dtype=np.uint8)
                    cv2.imwrite(str(raw_path), page)
            pair = Pair(
                manga=manga,
                stem=stem,
                mask_path=gt_root / manga / f"{stem}.png",
                raw_path=raw_path,
                mask_encoding="magenta-black",
            )
            pairs.append(pair)
            if image_id in duplicate:
                pairs.append(pair)

        pairs.sort(key=lambda p: (p.manga, p.stem))
        return Manifest(
            run_id="synthetic", gt_root=gt_root, raw_root=raw_root, pairs=pairs
        )

    return _build


@pytest.fixture()
def inpaint_config(tmp_path: Path):
    """Return a builder for an ``InpaintConfig`` that lives entirely under tmp (T022-T025).

    ``build(source_roots=None, *, dilation=None, radius=3,
    algorithms=("telea", "ns"), output_format="png")`` returns a config in
    which every FR-026 identity has a source root (``source_roots`` for the
    ones a test cares about, an unused tmp directory for the rest) and
    ``output_root`` is a fresh tmp directory, so nothing touches the real
    ``outputs/inpainting/``. ``dilation`` defaults to the shared setting of
    ``configs/inpainting.json`` (ellipse, 3x3, one iteration).
    """
    from manga_text_seg.config import (
        DilationConfig,
        InpaintConfig,
        InpaintingConfig,
        InpaintMethodConfig,
        MaskProcessingConfig,
    )
    from manga_text_seg.runs import INPAINT_METHOD_ORDER

    def _build(
        source_roots: dict[str, Path] | None = None,
        *,
        dilation=None,
        radius: float = 3,
        algorithms: tuple[str, ...] = ("telea", "ns"),
        output_format: str = "png",
    ) -> InpaintConfig:
        roots = dict(source_roots or {})
        if dilation is None:
            dilation = DilationConfig(
                enabled=True, kernel_shape="ellipse", kernel_size=(3, 3), iterations=1
            )
        methods = {
            name: InpaintMethodConfig(
                source_root=roots.get(name, tmp_path / f"unused-source-{name}")
            )
            for name in INPAINT_METHOD_ORDER
        }
        return InpaintConfig(
            methods=methods,
            mask_processing=MaskProcessingConfig(dilation=dilation),
            inpainting=InpaintingConfig(
                radius=radius, algorithms=algorithms, output_format=output_format
            ),
            selection_n=5,
            output_root=tmp_path / "inpainting-out",
        )

    return _build
