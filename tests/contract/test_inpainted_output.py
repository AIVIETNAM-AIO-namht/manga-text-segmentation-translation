"""T022 - contract test for the inpainted output tree (Spec 003, US2).

Contract: contracts/inpainted-output.md (FR-025 - FR-029). Written before
``inpaint.py`` exists, so every test here must FAIL until T026-T028 land.

Every sample gets, per algorithm, five files beside each other:

    <method>/<algorithm>/<manga>/<page_id>.png       the inpainted page
                                 <page_id>.json      SampleMetadata sidecar
                                 <page_id>.raw.png   raw prediction mask copy
                                 <page_id>.mask.png  post-dilation mask
                                 <page_id>.page.png  the original page

Across the two algorithms these are the six FR-025 artifacts: original page,
raw mask copy, processed mask, TELEA result, NS result, metadata.

``run.json``, ``errors.json`` and ``performance.json`` sit at the run root.
"""

from __future__ import annotations

import cv2
import numpy as np

from manga_text_seg.inpaint import process_sample
from manga_text_seg.runs import (
    ErrorReport,
    InpaintRun,
    MethodSource,
    SampleCounts,
    create_run,
    sample_paths,
    write_performance_summary,
    write_run_record,
)

METHOD = "classical_baseline"
RUN_ID = "run-t022"
IMAGE_ID = "ARMS/000"
ALGORITHMS = ("telea", "ns")
GT_CANVAS_SHAPE = (1176, 1656)  # numpy (height, width), the Spec 001 padded canvas
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _produce(inpaint_fixture, synthetic_manifest, inpaint_config, **fixture_kwargs):
    """Create a run and process ARMS/000 into it; return (config, root, manifest)."""
    manifest = synthetic_manifest()
    root = inpaint_fixture(METHOD, IMAGE_ID, **fixture_kwargs)
    config = inpaint_config({METHOD: root})
    create_run(config.output_root, RUN_ID)
    process_sample(
        manifest,
        "ARMS",
        "000",
        method=METHOD,
        run_id=RUN_ID,
        config=config,
        experiment_id="fixture-experiment",
        model_repository=None,
    )
    return config, root, manifest


def _artifacts(config, algorithm):
    paths = sample_paths(config.output_root, RUN_ID, METHOD, algorithm, IMAGE_ID)
    directory = paths.image_path.parent
    stem = paths.image_path.stem
    return {
        "inpainted": paths.image_path,
        "metadata": paths.metadata_path,
        "raw_mask": directory / f"{stem}.raw.png",
        "processed_mask": directory / f"{stem}.mask.png",
        "page": directory / f"{stem}.page.png",
    }


def _read(path):
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


# --------------------------------------------------------------------------- #
# The per-sample tree
# --------------------------------------------------------------------------- #
def test_every_artifact_exists_at_its_contract_address(
    inpaint_fixture, synthetic_manifest, inpaint_config
):
    config, _, _ = _produce(inpaint_fixture, synthetic_manifest, inpaint_config)

    for algorithm in ALGORITHMS:
        for kind, path in _artifacts(config, algorithm).items():
            assert path.is_file(), f"{algorithm}: missing {kind} at {path}"


def test_the_six_fr025_artifacts_are_all_present(
    inpaint_fixture, synthetic_manifest, inpaint_config
):
    config, _, _ = _produce(inpaint_fixture, synthetic_manifest, inpaint_config)
    telea, ns = (_artifacts(config, a) for a in ALGORITHMS)

    six = {
        "original page": telea["page"],
        "raw mask copy": telea["raw_mask"],
        "processed mask": telea["processed_mask"],
        "TELEA result": telea["inpainted"],
        "NS result": ns["inpainted"],
        "metadata": telea["metadata"],
    }
    for name, path in six.items():
        assert path.is_file(), f"FR-025 artifact missing: {name}"


def test_the_run_tree_holds_exactly_the_contract_files(
    inpaint_fixture, synthetic_manifest, inpaint_config
):
    """Nothing else is written: no ground truth, no stray copies (FR-033)."""
    config, _, _ = _produce(inpaint_fixture, synthetic_manifest, inpaint_config)
    run_dir = config.output_root / RUN_ID

    expected = {
        path.relative_to(run_dir)
        for algorithm in ALGORITHMS
        for path in _artifacts(config, algorithm).values()
    }
    found = {p.relative_to(run_dir) for p in run_dir.rglob("*") if p.is_file()}

    assert found == expected


def test_inpainted_pages_are_the_aligned_size_in_the_configured_format(
    inpaint_fixture, synthetic_manifest, inpaint_config, aligned_mask
):
    config, _, _ = _produce(inpaint_fixture, synthetic_manifest, inpaint_config)

    for algorithm in ALGORITHMS:
        path = _artifacts(config, algorithm)["inpainted"]
        assert _read(path).shape[:2] == aligned_mask.shape
        assert path.read_bytes()[:8] == PNG_SIGNATURE


def test_the_raw_mask_copy_is_byte_identical_to_the_admitted_mask(
    inpaint_fixture, synthetic_manifest, inpaint_config
):
    config, root, _ = _produce(inpaint_fixture, synthetic_manifest, inpaint_config)
    admitted = (root / "masks" / "ARMS" / "000.png").read_bytes()

    for algorithm in ALGORITHMS:
        assert _artifacts(config, algorithm)["raw_mask"].read_bytes() == admitted


def test_the_processed_mask_is_the_dilated_aligned_mask(
    inpaint_fixture, synthetic_manifest, inpaint_config, aligned_mask
):
    config, root, _ = _produce(inpaint_fixture, synthetic_manifest, inpaint_config)
    raw = _read(root / "masks" / "ARMS" / "000.png")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    expected = cv2.dilate(raw, kernel, iterations=1)

    for algorithm in ALGORITHMS:
        processed = _read(_artifacts(config, algorithm)["processed_mask"])
        assert processed.shape == aligned_mask.shape
        assert set(np.unique(processed)) <= {0, 255}
        assert np.array_equal(processed, expected)


def test_the_page_copy_has_the_pixels_of_the_original_page(
    inpaint_fixture, synthetic_manifest, inpaint_config
):
    config, _, manifest = _produce(inpaint_fixture, synthetic_manifest, inpaint_config)
    original = _read(manifest.pairs[0].raw_path)

    for algorithm in ALGORITHMS:
        copy = _read(_artifacts(config, algorithm)["page"])
        assert np.array_equal(copy, original)


def test_a_gt_canvas_mask_is_copied_raw_and_processed_at_the_aligned_size(
    inpaint_fixture, synthetic_manifest, inpaint_config, aligned_mask
):
    """The raw copy is the mask exactly as admitted (1656x1176); the processed
    mask is what cv2.inpaint got (cropped to 1654x1170, then dilated)."""
    config, root, _ = _produce(
        inpaint_fixture, synthetic_manifest, inpaint_config, size=GT_CANVAS_SHAPE
    )
    source = root / "masks" / "ARMS" / "000.png"
    canvas = _read(source)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    expected = cv2.dilate(canvas[:1170, :1654], kernel, iterations=1)

    for algorithm in ALGORITHMS:
        artifacts = _artifacts(config, algorithm)
        assert artifacts["raw_mask"].read_bytes() == source.read_bytes()
        assert _read(artifacts["raw_mask"]).shape == GT_CANVAS_SHAPE
        assert _read(artifacts["processed_mask"]).shape == aligned_mask.shape
        assert np.array_equal(_read(artifacts["processed_mask"]), expected)
        assert _read(artifacts["inpainted"]).shape[:2] == aligned_mask.shape


# --------------------------------------------------------------------------- #
# The run root
# --------------------------------------------------------------------------- #
def _complete_run(inpaint_fixture, synthetic_manifest, inpaint_config):
    """Process a sample, then write the run-level files with the existing writers."""
    config, root, _ = _produce(inpaint_fixture, synthetic_manifest, inpaint_config)
    run_dir = config.output_root / RUN_ID
    run = InpaintRun(
        run_id=RUN_ID,
        method_sources={
            METHOD: MethodSource(str(root), "adaptive", "research R1 (fixture)")
        },
        mask_processing_config=config.mask_processing,
        inpainting_config=config.inpainting,
        page_list_identity="fixture-page-list-identity",
        input_image_identity={"ARMS/000": "fixture-image-identity"},
        counts={METHOD: {a: SampleCounts(1, 1, 0) for a in ALGORITHMS}},
    )
    write_run_record(run_dir, run)
    ErrorReport().write(run_dir / "errors.json")
    write_performance_summary(
        run_dir,
        {METHOD: {a: [0.01] for a in ALGORITHMS}},
        {METHOD: {a: SampleCounts(1, 1, 0) for a in ALGORITHMS}},
    )
    return run_dir


def test_run_root_holds_run_json_and_errors_json(
    inpaint_fixture, synthetic_manifest, inpaint_config
):
    run_dir = _complete_run(inpaint_fixture, synthetic_manifest, inpaint_config)

    assert (run_dir / "run.json").is_file()
    assert (run_dir / "errors.json").is_file()


def test_run_root_holds_performance_json(
    inpaint_fixture, synthetic_manifest, inpaint_config
):
    run_dir = _complete_run(inpaint_fixture, synthetic_manifest, inpaint_config)

    assert (run_dir / "performance.json").is_file()
