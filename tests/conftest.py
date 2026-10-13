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
import shutil
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
def translation_bundle(tmp_path: Path):
    """Build a valid-ID Spec 004 input bundle with independently perturbable files.

    The selected ID and both Spec 2 identity strings come from the checked-in
    Spec 2 page-list metadata. Source images/masks are synthetic and no GT file
    is created or read. ``benchmark/manifest.json`` is fixture-local and is
    written with the repository's Spec 1 Manifest serializer.
    """
    import cv2

    from manga_text_seg.imaging import save_mask
    from manga_text_seg.manifest import Manifest, Pair, save_manifest

    page_list = json.loads(
        (REPO_ROOT / "benchmark" / "page-list.json").read_text(encoding="utf-8")
    )
    valid_pages = {page["image_id"]: page for page in page_list["pages"]}
    default_id = sorted(valid_pages)[0]

    def build(
        # ponytail: default matches the shipped config's resolved method, so a
        # default bundle is admissible against configs/translation.json.
        method: str = "unetpp_efficientnetv2",
        image_id: str = default_id,
        *,
        missing_original: bool = False,
        missing_prediction_mask: bool = False,
        missing_mask_sidecar: bool = False,
        missing_inpainted_image: bool = False,
        missing_inpainted_sidecar: bool = False,
        original_size: tuple[int, int] = (1654, 1170),
        mask_size: tuple[int, int] = (1654, 1170),
        inpainted_size: tuple[int, int] = (1654, 1170),
        original_pixel_value: int = 210,
        prediction_text_value: int = 255,
        inpainted_pixel_value: int = 230,
        empty_mask: bool = False,
    ) -> dict:
        if method not in {
            "classical_baseline",
            "manga_text_segmentation",
            "comic_text_detector",
            "unetpp_efficientnetv2",
        }:
            raise ValueError(
                f"method is not a Spec 2 segmentation identity: {method!r}"
            )
        if image_id not in valid_pages:
            raise ValueError(
                f"image_id is not present in the Spec 2 page list: {image_id!r}"
            )
        for label, value in (
            ("original_pixel_value", original_pixel_value),
            ("prediction_text_value", prediction_text_value),
            ("inpainted_pixel_value", inpainted_pixel_value),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 0 <= value <= 255
            ):
                raise ValueError(f"{label} must be an integer from 0 to 255")
        page = valid_pages[image_id]
        manga, stem = page["manga"], page["stem"]
        root = (
            tmp_path
            / f"translation-bundle-{image_id.replace('/', '-')}-{len(list(tmp_path.iterdir()))}"
        )
        raw_root = root / "raw"
        prediction_root = root / "prediction"
        inpaint_root = root / "inpainted"
        original_path = raw_root / manga / f"{stem}.png"
        prediction_path = prediction_root / "masks" / manga / f"{stem}.png"
        mask_sidecar_path = prediction_root / "metadata" / manga / f"{stem}.json"
        inpainted_method_root = inpaint_root / method / "telea"
        inpainted_path = inpainted_method_root / manga / f"{stem}.png"
        inpainted_sidecar_path = inpainted_method_root / manga / f"{stem}.json"
        height, width = original_size[1], original_size[0]
        if not missing_original:
            original_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(original_path),
                np.full((height, width, 3), original_pixel_value, dtype=np.uint8),
            )

        mask = np.zeros((mask_size[1], mask_size[0]), dtype=np.uint8)
        if not empty_mask:
            mask[100:160, 100:220] = prediction_text_value
        if not missing_prediction_mask:
            save_mask(prediction_path, mask)

        sidecar = {
            "image_id": image_id,
            "page_list_identity": page_list["page_list_identity"],
            "input_image_identity": page["input_image_identity"],
            "method": method,
            "output_size": list(mask_size),
            "status": "success",
            "fold_attribution": None,
        }
        if not missing_mask_sidecar:
            mask_sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            mask_sidecar_path.write_text(
                json.dumps(sidecar, indent=2) + "\n", encoding="utf-8"
            )

        ih, iw = inpainted_size[1], inpainted_size[0]
        if not missing_inpainted_image:
            inpainted_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(inpainted_path),
                np.full((ih, iw, 3), inpainted_pixel_value, dtype=np.uint8),
            )
        inpaint_meta = {
            "image_id": image_id,
            "method": method,
            "algorithm": "telea",
            "model_repository": None,
            "experiment_id": "translation-fixture",
            "input_image_path": str(original_path),
            "prediction_mask_path": str(prediction_path),
            "mask_size": list(mask_size),
            "alignment": {
                "rule": "pass-through",
                "pre_size": list(mask_size),
                "post_size": list(mask_size),
            },
            "dilation": {"enabled": False},
            "inpaint_algorithm": "INPAINT_TELEA",
            "inpaint_radius": 3.0,
            "output_format": "png",
            "processing_time_seconds": 0.0,
            "status": "ok",
        }
        if not missing_inpainted_sidecar:
            inpainted_sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            inpainted_sidecar_path.write_text(
                json.dumps(inpaint_meta, indent=2) + "\n", encoding="utf-8"
            )

        pair = Pair(
            manga=manga,
            stem=stem,
            mask_path=root / "gt-never-created" / manga / f"{stem}.png",
            raw_path=original_path,
            mask_encoding="magenta-black",
        )
        manifest = Manifest(
            run_id="translation-fixture",
            gt_root=root / "gt-never-created",
            raw_root=raw_root,
            pairs=[pair],
        )
        manifest_path = root / "benchmark" / "manifest.json"
        save_manifest(manifest, manifest_path)
        return {
            "root": root,
            "image_id": image_id,
            "method": method,
            "page_list_identity": page_list["page_list_identity"],
            "input_image_identity": page["input_image_identity"],
            "original": original_path,
            "prediction_mask": prediction_path,
            "mask_sidecar": mask_sidecar_path,
            "inpainted": inpainted_path,
            "inpainted_sidecar": inpainted_sidecar_path,
            "manifest": manifest_path,
            "manifest_entry": manifest.to_dict()["pairs"][0],
        }

    return build


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


# --------------------------------------------------------------------------- #
# Spec 5 fixtures — the web demo (T005)
# --------------------------------------------------------------------------- #

#: The one synthetic page every web fixture builds.  ``<manga>/<NNN>`` is the
#: Spec 4 addressing rule and the same pattern the pipeline metadata schema
#: enforces on ``image_id``.
WEB_PAGE_MANGA = "WebManga"
WEB_PAGE_STEM = "001"
WEB_PAGE_ID = f"{WEB_PAGE_MANGA}/{WEB_PAGE_STEM}"
#: Spec 3's inpainting run and Spec 4's translation run.  They must differ:
#: ``assemble_bundle`` refuses a page whose run already holds a ``mask.png``.
WEB_INPAINTING_RUN = "spec003-fixture"
WEB_TRANSLATION_RUN = "spec004-fixture"
#: The segmentation identity both configs select.
WEB_METHOD = "unetpp_efficientnetv2"


@pytest.fixture()
def web_bundle(tmp_path: Path):
    """Build a throwaway Spec 1-4 tree and the configs that address it (T005).

    ``build(**flags)`` lays the whole chain out under ``tmp_path`` at the paths
    the two config files name, loads both configs through their real loaders,
    and returns a dict of every path it wrote.  Each flag omits exactly one
    artifact, so a test can ask for the incomplete bundle its case needs
    (SC-002) without editing files afterwards.

    Unlike ``translation_bundle`` — which builds the shapes Spec 4's own tests
    hand to ``assemble_bundle`` — this tree is *run-rooted*, because the web
    config's ``artifact_roots`` are enumerated per run id (FR-014).  The two
    fixtures cover different contracts and neither is a superset of the other.
    """
    import cv2

    from manga_text_seg.config import load_translation_config, load_web_config
    from manga_text_seg.imaging import save_mask
    from manga_text_seg.manifest import Manifest, Pair, save_manifest
    from manga_text_seg.translation import runs as translation_runs

    def _region(
        *,
        ocr_failed: bool,
        translation_failed: bool,
        render_failed: bool,
    ) -> dict:
        """One schema-valid metadata region exercising whichever stage fails."""
        region = {
            "region_id": 1,
            "bbox": [100, 100, 220, 160],
            "polygon": [[100, 100], [220, 100], [220, 160], [100, 160]],
            "orientation": "horizontal",
            "reading_order_index": 1,
            "merged_from": 1,
            "crop_path": "crops/1.png",
            "clipped": False,
            "ocr_text": "こんにちは",
            "ocr_confidence": 0.91,
            "ocr_status": "ok",
            "ocr_error": None,
            "manually_edited": False,
            "translation": "Xin chào",
            "translation_status": "ok",
            "translation_error": None,
            "cache_status": "miss",
            "failure_policy": "keep_ocr_text",
            "render_status": "rendered",
            "render_warning": None,
            "font_size_used": 20,
            "line_count": 1,
            "display_box": [100, 100, 220, 160],
            "direction_used": "horizontal",
            "ocr_time_seconds": 0.1,
            "translation_time_seconds": 0.2,
            "render_time_seconds": 0.3,
        }
        if ocr_failed:
            region.update(
                ocr_text="", ocr_confidence=None, ocr_status="failed",
                ocr_error="fixture: OCR engine failed",
            )
        if translation_failed:
            region.update(
                translation="", translation_status="failed",
                translation_error="fixture: provider returned HTTP 401",
            )
        if render_failed:
            region.update(
                render_status="skipped", font_size_used=None,
                direction_used=None, render_warning="fixture: no renderable text",
            )
        return region

    def _build(
        *,
        # Spec 1-3 omissions.
        missing_original: bool = False,
        missing_prediction_mask: bool = False,
        missing_mask_sidecar: bool = False,
        missing_inpainted_image: bool = False,
        missing_inpainted_sidecar: bool = False,
        empty_mask: bool = False,
        # Spec 4 omissions.
        missing_ocr: bool = False,
        missing_translations: bool = False,
        missing_regions: bool = False,
        missing_render_stage: bool = False,
        missing_rendered: bool = False,
        missing_page_metadata: bool = False,
        # FR-020 page states.
        no_regions: bool = False,
        ocr_failed: bool = False,
        translation_failed: bool = False,
        render_failed: bool = False,
        # Config-level switches.
        missing_static_root: bool = False,
    ) -> dict:
        root = tmp_path
        method = WEB_METHOD
        run3, run4 = WEB_INPAINTING_RUN, WEB_TRANSLATION_RUN
        language = "vi"

        segmentation_root = root / "outputs" / "segmentation"
        inpainting_root = root / "outputs" / "inpainting"
        translation_root = root / "outputs" / "translation"
        raw_root = root / "outputs" / "raw"
        deliverables_root = root / "deliverables"
        groundtruth_root = root / "groundtruth"
        run_root = root / "outputs" / "end-to-end"
        forbidden_root = root / "no-need-to-read"
        static_root = root / "static"
        alt_static_root = root / "static-alt"

        manifest_path = segmentation_root / "default" / "manifest.json"
        original_path = raw_root / WEB_PAGE_MANGA / f"{WEB_PAGE_STEM}.png"
        prediction_path = segmentation_root / "masks" / WEB_PAGE_MANGA / f"{WEB_PAGE_STEM}.png"
        mask_sidecar_path = (
            segmentation_root / "metadata" / WEB_PAGE_MANGA / f"{WEB_PAGE_STEM}.json"
        )
        inpainted_dir = inpainting_root / run3 / method / "telea" / WEB_PAGE_MANGA
        inpainted_path = inpainted_dir / f"{WEB_PAGE_STEM}.png"
        inpainted_sidecar_path = inpainted_dir / f"{WEB_PAGE_STEM}.json"

        height, width = ALIGNED_SHAPE

        # ---- Spec 1: the raw page and the manifest -------------------------
        if not missing_original:
            original_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(original_path),
                np.full((height, width, 3), 210, dtype=np.uint8),
            )

        pair = Pair(
            manga=WEB_PAGE_MANGA,
            stem=WEB_PAGE_STEM,
            mask_path=groundtruth_root / WEB_PAGE_MANGA / f"{WEB_PAGE_STEM}.png",
            raw_path=original_path,
            mask_encoding="magenta-black",
        )
        manifest = Manifest(
            run_id="web-fixture", gt_root=groundtruth_root, raw_root=raw_root, pairs=[pair]
        )
        save_manifest(manifest, manifest_path)

        # ---- Spec 2: the prediction mask and its sidecar -------------------
        mask = np.zeros(ALIGNED_SHAPE, dtype=np.uint8)
        if not empty_mask:
            mask[100:160, 100:220] = 255
        if not missing_prediction_mask:
            save_mask(prediction_path, mask)

        page_list_identity = "web-fixture-page-list-identity"
        input_image_identity = "web-fixture-input-image-identity"
        if not missing_mask_sidecar:
            mask_sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            mask_sidecar_path.write_text(
                json.dumps(
                    {
                        "image_id": WEB_PAGE_ID,
                        "page_list_identity": page_list_identity,
                        "input_image_identity": input_image_identity,
                        "method": method,
                        "output_size": list(ALIGNED_SIZE),
                        "status": "success",
                        "fold_attribution": None,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        # ---- Spec 3: the inpainted page and its 15-field sidecar -----------
        if not missing_inpainted_image:
            inpainted_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(inpainted_path),
                np.full((height, width, 3), 230, dtype=np.uint8),
            )
        if not missing_inpainted_sidecar:
            inpainted_sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            inpainted_sidecar_path.write_text(
                json.dumps(
                    {
                        "image_id": WEB_PAGE_ID,
                        "method": method,
                        "algorithm": "telea",
                        "model_repository": None,
                        "experiment_id": "web-fixture",
                        "input_image_path": str(original_path),
                        "prediction_mask_path": str(prediction_path),
                        "mask_size": list(ALIGNED_SIZE),
                        "alignment": {
                            "rule": "pass-through",
                            "pre_size": list(ALIGNED_SIZE),
                            "post_size": list(ALIGNED_SIZE),
                        },
                        "dilation": {"enabled": False},
                        "inpaint_algorithm": "INPAINT_TELEA",
                        "inpaint_radius": 3.0,
                        "output_format": "png",
                        "processing_time_seconds": 0.0,
                        "status": "ok",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        # ---- Spec 4: the translation run's page directory ------------------
        page_dir = translation_runs.page_directory(
            translation_root, run4, method, WEB_PAGE_ID
        )
        page_dir.mkdir(parents=True, exist_ok=True)
        regions = [] if no_regions else [
            _region(
                ocr_failed=ocr_failed,
                translation_failed=translation_failed,
                render_failed=render_failed,
            )
        ]
        metadata = {
            "image_id": WEB_PAGE_ID,
            "run_id": run4,
            "segmentation_method": method,
            "inpainting_method": {
                "run_id": run3,
                "algorithm": "telea",
                "inpaint_algorithm": "INPAINT_TELEA",
                "inpaint_radius": 3.0,
                "dilation": {"enabled": False},
            },
            "target_language": language,
            "ocr_engine": {"name": "manga-ocr", "version": "0.1.16", "deterministic": False},
            "translation_provider": {
                "name": "fixture",
                "model": "fixture-model",
                "version": None,
                "deterministic": False,
                "secret_env_var": None,
            },
            "rendering_config": {
                "font_path": "assets/fonts/DejaVuSans.ttf",
                "font_size": 32,
                "min_font_size": 12,
                "text_direction": "horizontal",
                "align_h": "center",
                "align_v": "middle",
                "wrap": True,
                "line_spacing": 1.0,
                "expansion_allowance": 0,
            },
            "regions": regions,
            "stage_timings": {"ocr": 0.1, "translation": 0.2, "rendering": 0.3},
            "status": "no_regions" if no_regions else "ok",
        }
        if not missing_page_metadata:
            translation_runs.write_metadata(page_dir, metadata)

        if not missing_ocr:
            translation_runs.write_ocr_json(
                page_dir,
                {
                    "engine": "manga-ocr",
                    "engine_version": "0.1.16",
                    "preprocessing": None,
                    "region_set": [
                        {"region_id": 1, "bbox": [100, 100, 220, 160], "polygon": None}
                    ]
                    if regions
                    else [],
                    "recognized_text": {"1": "こんにちは"} if regions else {},
                    "results": [
                        {
                            "region_id": 1,
                            "crop_path": "crops/1.png",
                            "text": "" if ocr_failed else "こんにちは",
                            "confidence": None if ocr_failed else 0.91,
                            "engine": "manga-ocr",
                            "engine_version": "0.1.16",
                            "preprocessing": None,
                            "processing_time_seconds": 0.1,
                            "status": "failed" if ocr_failed else "ok",
                            "error": "fixture: OCR engine failed" if ocr_failed else None,
                            "manually_edited": False,
                        }
                    ]
                    if regions
                    else [],
                    "manual_edits": [],
                    "invalidation_reports": [],
                    "stale_region_ids": [],
                },
            )

        if not missing_translations:
            translation_runs.write_translations_json(
                page_dir,
                [
                    {
                        "region_id": 1,
                        "source_text": "こんにちは",
                        "translated_text": "" if translation_failed else "Xin chào",
                        "target_language": language,
                        "provider": "fixture",
                        "model": "fixture-model",
                        "version": None,
                        "cache_status": "miss",
                        "processing_time_seconds": 0.2,
                        "status": "failed" if translation_failed else "ok",
                        "error": (
                            "fixture: provider returned HTTP 401"
                            if translation_failed
                            else None
                        ),
                        "failure_policy": "keep_ocr_text",
                    }
                ]
                if regions
                else [],
            )

        if not missing_regions:
            (page_dir / "regions.json").write_text(
                json.dumps(
                    {
                        "status": "no_regions" if no_regions else "ok",
                        "regions": [
                            {
                                "region_id": 1,
                                "bbox": [100, 100, 220, 160],
                                "polygon": [[100, 100], [220, 100], [220, 160], [100, 160]],
                                "orientation": "horizontal",
                                "reading_order_index": 1,
                                "merged_from": 1,
                                "crop_path": "crops/1.png",
                                "crop_box": [100, 100, 220, 160],
                                "clipped": False,
                                "area": 7200,
                            }
                        ]
                        if regions
                        else [],
                        "excluded_components": [],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        if not missing_render_stage:
            (page_dir / "render-stage.json").write_text(
                json.dumps(
                    [
                        {
                            "region_id": 1,
                            "render_status": "skipped" if render_failed else "rendered",
                            "render_warning": (
                                "fixture: no renderable text" if render_failed else None
                            ),
                            "font_size_used": None if render_failed else 20,
                            "line_count": 1,
                            "display_box": [100, 100, 220, 160],
                            "direction_used": None if render_failed else "horizontal",
                        }
                    ]
                    if regions
                    else [],
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        crop_path = None
        if regions:
            # ponytail: write_crop copies bytes or a path, never an ndarray —
            # encode in memory rather than round-tripping through a temp file.
            encoded_ok, encoded = cv2.imencode(
                ".png", np.full((60, 120, 3), 255, dtype=np.uint8)
            )
            assert encoded_ok
            crop_path = translation_runs.write_crop(page_dir, 1, encoded.tobytes())

        if not missing_rendered:
            translation_runs.write_rendered_png(page_dir, inpainted_path)

        if not missing_prediction_mask:
            translation_runs.write_mask_copy(page_dir, prediction_path)
        if not missing_original:
            shutil.copyfile(original_path, page_dir / "original.png")
        if not missing_inpainted_image:
            shutil.copyfile(inpainted_path, page_dir / "inpainted.png")

        # ---- The Spec 2 identity records the intake contract reads ---------
        benchmark_root = root / "benchmark"
        benchmark_root.mkdir(parents=True, exist_ok=True)
        page_list_path = benchmark_root / "page-list.json"
        page_list_path.write_text(
            json.dumps(
                {
                    "generated_at_run": "web-fixture",
                    "aligned_size": list(ALIGNED_SIZE),
                    "alignment": "crop-topleft",
                    "mask_convention": {
                        "dtype": "uint8",
                        "channels": 1,
                        "values": [0, 255],
                        "text": 255,
                        "background": 0,
                    },
                    "image_bundle": {
                        "root": "dist/images",
                        "identity_record": "image-identity.json",
                    },
                    "pages": [
                        {
                            "manga": WEB_PAGE_MANGA,
                            "stem": WEB_PAGE_STEM,
                            "image_id": WEB_PAGE_ID,
                            "image_ref": f"{WEB_PAGE_ID}.jpg",
                            "input_image_identity": input_image_identity,
                        }
                    ],
                    "page_list_identity": page_list_identity,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        image_identity_path = benchmark_root / "image-identity.json"
        image_identity_path.write_text(
            json.dumps(
                {WEB_PAGE_ID: {"sha256": input_image_identity, "size_bytes": 4096}},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        # ---- The static bundle the backend serves verbatim (FR-013a) -------
        if not missing_static_root:
            static_root.mkdir(parents=True, exist_ok=True)
            (static_root / "index.html").write_text(
                "<!doctype html><title>fixture-bundle</title>\n", encoding="utf-8"
            )
        alt_static_root.mkdir(parents=True, exist_ok=True)
        (alt_static_root / "index.html").write_text(
            "<!doctype html><title>fixture-bundle-alt</title>\n", encoding="utf-8"
        )

        deliverables_root.mkdir(parents=True, exist_ok=True)
        groundtruth_root.mkdir(parents=True, exist_ok=True)
        run_root.mkdir(parents=True, exist_ok=True)
        forbidden_root.mkdir(parents=True, exist_ok=True)
        (forbidden_root / "secret.txt").write_text("must never be served\n", encoding="utf-8")

        # ---- The two configs, written where their loaders resolve root -----
        cache_root = root / "cache" / "translation"
        configs_root = root / "configs"
        configs_root.mkdir(parents=True, exist_ok=True)

        translation_config_path = configs_root / "translation.json"
        translation_config_path.write_text(
            json.dumps(
                {
                    "manifest_path": str(manifest_path),
                    "segmentation_method": method,
                    "inpainting_source": {"run_id": run3, "algorithm": "telea"},
                    "target_language": language,
                    "region_extraction": {
                        "min_area": 50,
                        "max_area": 12000,
                        "merge_distance": 8,
                        "crop_padding": 0,
                        "reading_order": "manga",
                        "polygon_epsilon": 1.0,
                    },
                    "ocr": {
                        "engine": "manga-ocr",
                        "preprocessing": {
                            "enabled": False,
                            "grayscale": False,
                            "upscale_factor": 1.0,
                        },
                    },
                    "provider": {
                        "name": "fixture",
                        "model": "fixture-model",
                        "version": None,
                        "base_url": "https://fixture.invalid/v1",
                        "secret_env_var": "MANGA_TEXT_SEG_FIXTURE_TOKEN",
                        "timeout_seconds": 30.0,
                        "retry": {
                            "max_attempts": 3,
                            "initial_backoff_seconds": 1.0,
                            "max_backoff_seconds": 30.0,
                            "retry_status_codes": [429, 500, 502, 503, 504],
                        },
                        "failure_policy": "keep_ocr_text",
                        "deterministic": False,
                        # ponytail: must resolve to output_roots.cache, or the
                        # loader refuses the file outright.
                        "cache": {"enabled": True, "bypass": False, "root": str(cache_root)},
                    },
                    "rendering": {
                        "font_family": "DejaVu Sans",
                        "font_path": "assets/fonts/DejaVuSans.ttf",
                        "fallback_font_path": None,
                        "font_size": 32,
                        "min_font_size": 12,
                        "color": [0, 0, 0, 255],
                        "stroke_width": 0,
                        "stroke_fill": [255, 255, 255, 255],
                        "background": None,
                        "wrap": True,
                        "line_spacing": 1.0,
                        "align_h": "center",
                        "align_v": "middle",
                        "text_direction": "horizontal",
                        "expansion_allowance": 0,
                    },
                    "output_roots": {
                        "translation": str(translation_root),
                        "cache": str(cache_root),
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        web_config_path = configs_root / "web.json"
        web_config_path.write_text(
            json.dumps(
                {
                    "server": {"host": "127.0.0.1", "port": 8765},
                    "static_root": str(static_root),
                    "artifact_roots": {
                        "segmentation": str(segmentation_root),
                        "inpainting": str(inpainting_root),
                        "translation": str(translation_root),
                        "deliverables": str(deliverables_root),
                        "raw_images": str(raw_root),
                        "groundtruth": str(groundtruth_root),
                    },
                    "forbidden_roots": [str(forbidden_root)],
                    "run_root": str(run_root),
                    "session": {
                        "header_name": "X-Manga-Session",
                        "cookie_name": "manga_session",
                    },
                    "provider": {
                        "name": "fixture",
                        "model": "fixture-model",
                        "version": None,
                        "base_url": "https://fixture.invalid/v1",
                        "secret_env_var": "MANGA_TEXT_SEG_FIXTURE_TOKEN",
                        "timeout_seconds": 30.0,
                        "retry": {
                            "max_attempts": 3,
                            "initial_backoff_seconds": 1.0,
                            "max_backoff_seconds": 30.0,
                            "retry_status_codes": [429, 500, 502, 503, 504],
                        },
                        "cache": {"enabled": True, "bypass": False, "root": str(cache_root)},
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        return {
            "root": root,
            "web_config": load_web_config(web_config_path, repo_root=root),
            "translation_config": load_translation_config(
                translation_config_path, repo_root=root
            ),
            "web_config_path": web_config_path,
            "translation_config_path": translation_config_path,
            "static_root": static_root,
            "alt_static_root": alt_static_root,
            "forbidden_root": forbidden_root,
            "run_root": run_root,
            "image_id": WEB_PAGE_ID,
            "method": method,
            "inpainting_run": run3,
            "translation_run": run4,
            "language": language,
            "manifest": manifest_path,
            "original": original_path,
            "prediction_mask": prediction_path,
            "mask_sidecar": mask_sidecar_path,
            "inpainted": inpainted_path,
            "inpainted_sidecar": inpainted_sidecar_path,
            "page_list": page_list_path,
            "image_identity": image_identity_path,
            "page_dir": page_dir,
            "metadata": page_dir / "metadata.json",
            "ocr": page_dir / "ocr.json",
            "translations": page_dir / "translations.json",
            "regions": page_dir / "regions.json",
            "render_stage": page_dir / "render-stage.json",
            "rendered": page_dir / "rendered.png",
            "mask_copy": page_dir / "mask.png",
            "crop": crop_path,
            "manifest_entry": manifest.to_dict()["pairs"][0],
        }

    return _build


@pytest.fixture()
def web_app(web_bundle):
    """Build the Spec 005 app against the fixture bundle's own configuration.

    The built bundle is attached at ``app.state.fixture`` so a test can reach
    the paths it addresses without rebuilding the tree; ``web_bundle`` is
    function-scoped, so requesting it separately would re-run the whole
    builder against the same ``tmp_path``.

    For a different static root (SC-014), rebuild from the same config rather
    than rewriting the config file — ``WebConfig`` is a frozen dataclass:

        create_app(dataclasses.replace(config, static_root=other))
    """
    from manga_text_seg.web.app import create_app

    built = web_bundle()
    app = create_app(built["web_config"])
    app.state.fixture = built
    return app


@pytest.fixture()
def web_client(web_app):
    """A ``TestClient`` whose ``Host`` header is the configured loopback bind.

    ``TestClient`` defaults to ``base_url="http://testserver"``, which the
    FR-013c ``Host`` guard is required to refuse — every request would come
    back a refusal that looks like a guard bug. Binding the base URL to the
    configured host and port is what makes the client speak as a real
    loopback caller.
    """
    from fastapi.testclient import TestClient

    server = web_app.state.fixture["web_config"].server
    with TestClient(web_app, base_url=f"http://{server.host}:{server.port}") as client:
        yield client
