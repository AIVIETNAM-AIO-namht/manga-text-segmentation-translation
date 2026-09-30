"""T024 - unit tests for the inpainting wrapper and ``process_sample`` (Spec 003, US2).

Contract: contracts/inpainted-output.md; FR-019 - FR-022, FR-025. Written before
``inpaint.py`` exists, so every test here must FAIL until T026 and T028 land.

``cv2.inpaint`` is spied on (it still runs for real) or, where only timing
matters, replaced by a stub. Both work because ``inpaint.py`` must call it as
``cv2.inpaint(...)`` through the module.

``process_sample(manifest, manga, stem, *, method, run_id, config,
experiment_id, model_repository, algorithms=None, ablation=False)`` reads the
mask from ``config.methods[method].source_root``, runs intake, dilates, inpaints
with each algorithm and writes the artifacts. It returns ``{algorithm:
SampleMetadata}``. ``algorithms=None`` means the configured set, which in the
main benchmark is always both (FR-019).
"""

from __future__ import annotations

import json
import time

import cv2
import numpy as np
import pytest

from manga_text_seg.config import DilationConfig, InpaintingConfig
from manga_text_seg.inpaint import inpaint_page, process_sample
from manga_text_seg.runs import MaskRejection, RunError, create_run, sample_paths

METHOD = "classical_baseline"
RUN_ID = "run-t024"
IMAGE_ID = "ARMS/000"
FLAGS = {"telea": cv2.INPAINT_TELEA, "ns": cv2.INPAINT_NS}


@pytest.fixture()
def cv2_inpaint_spy(monkeypatch):
    """Record every ``cv2.inpaint`` call (arguments copied) and run the real one."""
    calls: list[dict] = []
    real = cv2.inpaint

    def spy(src, inpaintMask, inpaintRadius, flags, dst=None):
        calls.append(
            {
                "src": src.copy(),
                "mask": inpaintMask.copy(),
                "radius": inpaintRadius,
                "flags": flags,
            }
        )
        return real(src, inpaintMask, inpaintRadius, flags)

    monkeypatch.setattr(cv2, "inpaint", spy)
    return calls


def _sample(inpaint_fixture, synthetic_manifest, inpaint_config, **config_kwargs):
    """Run one valid sample through ``process_sample``; return what a test needs."""
    manifest = synthetic_manifest()
    root = inpaint_fixture(METHOD, IMAGE_ID)
    config = inpaint_config({METHOD: root}, **config_kwargs)
    create_run(config.output_root, RUN_ID)

    def run(**kwargs):
        return process_sample(
            manifest,
            "ARMS",
            "000",
            method=METHOD,
            run_id=RUN_ID,
            config=config,
            experiment_id="fixture-experiment",
            model_repository=None,
            **kwargs,
        )

    return run, root, config


def _address(config, algorithm):
    return sample_paths(config.output_root, RUN_ID, METHOD, algorithm, IMAGE_ID)


def _dilated(mask, shape="ellipse", size=(3, 3), iterations=1):
    kernel_shape = {"ellipse": cv2.MORPH_ELLIPSE, "rect": cv2.MORPH_RECT}[shape]
    kernel = cv2.getStructuringElement(kernel_shape, size)
    return cv2.dilate(mask, kernel, iterations=iterations)


# --------------------------------------------------------------------------- #
# inpaint_page: the wrapper (T026)
# --------------------------------------------------------------------------- #
def _page_with_text(shape):
    """A flat grey page with a block of black text, and the mask covering it."""
    page = np.full(shape, 200, dtype=np.uint8)
    mask = np.zeros(shape, dtype=np.uint8)
    page[300:330, 400:520] = 0
    mask[300:330, 400:520] = 255
    return page, mask


@pytest.mark.parametrize("algorithm", ["telea", "ns"])
def test_inpaint_page_fills_the_masked_pixels_and_leaves_the_rest(
    aligned_mask, algorithm
):
    page, mask = _page_with_text(aligned_mask.shape)
    page_before, mask_before = page.copy(), mask.copy()
    settings = InpaintingConfig(radius=3, algorithms=("telea", "ns"))

    result = inpaint_page(page, mask, algorithm, settings)

    assert result.shape == aligned_mask.shape
    assert result.dtype == np.uint8
    assert np.array_equal(result[mask == 0], page[mask == 0]), "outside the mask"
    assert (result[mask == 255] > 100).all(), "the text was painted over"
    assert np.array_equal(page, page_before), "input page must not be modified"
    assert np.array_equal(mask, mask_before), "input mask must not be modified"


@pytest.mark.parametrize("algorithm", ["telea", "ns"])
def test_inpaint_page_uses_the_configured_radius_and_the_right_flag(
    aligned_mask, cv2_inpaint_spy, algorithm
):
    page, mask = _page_with_text(aligned_mask.shape)
    settings = InpaintingConfig(radius=5, algorithms=("telea", "ns"))

    inpaint_page(page, mask, algorithm, settings)

    assert len(cv2_inpaint_spy) == 1
    assert cv2_inpaint_spy[0]["radius"] == 5
    assert cv2_inpaint_spy[0]["flags"] == FLAGS[algorithm]


def test_inpaint_page_refuses_an_unknown_algorithm(aligned_mask):
    page, mask = _page_with_text(aligned_mask.shape)

    with pytest.raises(RunError):
        inpaint_page(page, mask, "bogus", InpaintingConfig(radius=3, algorithms=()))


# --------------------------------------------------------------------------- #
# process_sample: both algorithms, on the same input (FR-019)
# --------------------------------------------------------------------------- #
def test_both_algorithms_run_and_write_aligned_size_output(
    inpaint_fixture, synthetic_manifest, inpaint_config, aligned_mask
):
    run, _, config = _sample(inpaint_fixture, synthetic_manifest, inpaint_config)

    results = run()

    assert set(results) == {"telea", "ns"}
    for algorithm in ("telea", "ns"):
        image = cv2.imread(str(_address(config, algorithm).image_path), -1)
        assert image.shape[:2] == aligned_mask.shape


def test_both_algorithms_receive_the_same_input(
    inpaint_fixture, synthetic_manifest, inpaint_config, cv2_inpaint_spy
):
    run, _, _ = _sample(inpaint_fixture, synthetic_manifest, inpaint_config)

    run()

    assert {c["flags"] for c in cv2_inpaint_spy} == {cv2.INPAINT_TELEA, cv2.INPAINT_NS}
    first, second = cv2_inpaint_spy
    assert np.array_equal(first["src"], second["src"])
    assert np.array_equal(first["mask"], second["mask"])


@pytest.mark.parametrize(
    ("chosen", "other"), [("telea", "ns"), ("ns", "telea")], ids=["telea", "ns"]
)
def test_a_single_algorithm_writes_only_its_own_output(
    inpaint_fixture, synthetic_manifest, inpaint_config, cv2_inpaint_spy, chosen, other
):
    run, _, config = _sample(inpaint_fixture, synthetic_manifest, inpaint_config)

    results = run(algorithms=(chosen,))

    assert set(results) == {chosen}
    assert _address(config, chosen).image_path.is_file()
    assert _address(config, chosen).metadata_path.is_file()
    assert not _address(config, other).image_path.parent.exists()
    assert [c["flags"] for c in cv2_inpaint_spy] == [FLAGS[chosen]]


def test_an_unknown_algorithm_is_refused_before_anything_is_written(
    inpaint_fixture, synthetic_manifest, inpaint_config
):
    run, _, config = _sample(inpaint_fixture, synthetic_manifest, inpaint_config)

    with pytest.raises(RunError):
        run(algorithms=("bogus",))

    assert not any(p.is_file() for p in config.output_root.rglob("*"))


# --------------------------------------------------------------------------- #
# The mask cv2.inpaint receives, and the raw copy (FR-016, FR-025)
# --------------------------------------------------------------------------- #
def test_cv2_inpaint_receives_the_mask_after_dilation(
    inpaint_fixture, synthetic_manifest, inpaint_config, cv2_inpaint_spy
):
    run, root, _ = _sample(inpaint_fixture, synthetic_manifest, inpaint_config)
    raw = cv2.imread(str(root / "masks" / "ARMS" / "000.png"), -1)

    run()

    expected = _dilated(raw)
    assert not np.array_equal(expected, raw), "the fixture mask must actually grow"
    assert len(cv2_inpaint_spy) == 2
    for call in cv2_inpaint_spy:
        assert np.array_equal(call["mask"], expected)


def test_dilation_settings_come_from_the_config(
    inpaint_fixture, synthetic_manifest, inpaint_config, cv2_inpaint_spy
):
    dilation = DilationConfig(
        enabled=True, kernel_shape="rect", kernel_size=(5, 5), iterations=2
    )
    run, root, _ = _sample(
        inpaint_fixture, synthetic_manifest, inpaint_config, dilation=dilation
    )
    raw = cv2.imread(str(root / "masks" / "ARMS" / "000.png"), -1)

    run()

    for call in cv2_inpaint_spy:
        assert np.array_equal(call["mask"], _dilated(raw, "rect", (5, 5), 2))


def test_disabled_dilation_hands_the_aligned_mask_over_unchanged(
    inpaint_fixture, synthetic_manifest, inpaint_config, cv2_inpaint_spy
):
    run, root, _ = _sample(
        inpaint_fixture,
        synthetic_manifest,
        inpaint_config,
        dilation=DilationConfig(enabled=False),
    )
    raw = cv2.imread(str(root / "masks" / "ARMS" / "000.png"), -1)

    run()

    for call in cv2_inpaint_spy:
        assert np.array_equal(call["mask"], raw)


def test_the_raw_mask_copy_is_byte_identical_to_the_admitted_mask(
    inpaint_fixture, synthetic_manifest, inpaint_config
):
    run, root, config = _sample(inpaint_fixture, synthetic_manifest, inpaint_config)
    source = root / "masks" / "ARMS" / "000.png"
    before = source.read_bytes()

    run()

    for algorithm in ("telea", "ns"):
        image_path = _address(config, algorithm).image_path
        copy = image_path.with_name(f"{image_path.stem}.raw.png")
        assert copy.read_bytes() == before
    assert source.read_bytes() == before, "the source is copied, never moved or changed"


# --------------------------------------------------------------------------- #
# Configured parameters land in the metadata (FR-021, FR-025)
# --------------------------------------------------------------------------- #
def test_radius_and_format_come_from_the_config_and_reach_the_metadata(
    inpaint_fixture, synthetic_manifest, inpaint_config, cv2_inpaint_spy
):
    run, _, config = _sample(
        inpaint_fixture, synthetic_manifest, inpaint_config, radius=5
    )

    results = run()

    assert {c["radius"] for c in cv2_inpaint_spy} == {5}
    for algorithm, metadata in results.items():
        on_disk = json.loads(
            _address(config, algorithm).metadata_path.read_text(encoding="utf-8")
        )
        assert on_disk == json.loads(json.dumps(metadata.to_dict()))
        assert on_disk["inpaint_radius"] == 5
        assert on_disk["output_format"] == "png"
        assert on_disk["inpaint_algorithm"] == f"INPAINT_{algorithm.upper()}"
        assert on_disk["algorithm"] == algorithm
        assert on_disk["method"] == METHOD
        assert on_disk["image_id"] == IMAGE_ID
        assert on_disk["experiment_id"] == "fixture-experiment"
        assert on_disk["model_repository"] is None
        assert on_disk["status"] == "ok"
        assert on_disk["dilation"] == {
            "enabled": True,
            "kernel_shape": "ellipse",
            "kernel_size": [3, 3],
            "iterations": 1,
        }
        assert on_disk["alignment"] == {
            "rule": "pass-through",
            "pre_size": [1654, 1170],
            "post_size": [1654, 1170],
        }
        assert on_disk["mask_size"] == [1654, 1170]


def test_inpainted_file_is_in_the_configured_lossless_format(
    inpaint_fixture, synthetic_manifest, inpaint_config
):
    run, _, config = _sample(inpaint_fixture, synthetic_manifest, inpaint_config)

    run()

    for algorithm in ("telea", "ns"):
        head = _address(config, algorithm).image_path.read_bytes()[:8]
        assert head == b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------- #
# Timing, recorded per algorithm (FR-022)
# --------------------------------------------------------------------------- #
def test_timing_is_recorded_separately_for_each_algorithm(
    monkeypatch, inpaint_fixture, synthetic_manifest, inpaint_config
):
    def stub(src, inpaintMask, inpaintRadius, flags, dst=None):
        if flags == cv2.INPAINT_NS:
            time.sleep(0.2)
        return src.copy()

    monkeypatch.setattr(cv2, "inpaint", stub)
    run, _, _ = _sample(inpaint_fixture, synthetic_manifest, inpaint_config)

    results = run()

    assert results["ns"].processing_time_seconds >= 0.2
    assert results["telea"].processing_time_seconds < 0.2, (
        "TELEA's time must not include NS's"
    )


# --------------------------------------------------------------------------- #
# Determinism of order, and failure without leftovers (T029, T028)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "requested",
    [("telea", "ns"), ("ns", "telea"), ("telea", "ns", "telea")],
    ids=["config-order", "reversed", "duplicated"],
)
def test_algorithms_always_run_in_sorted_order(
    inpaint_fixture, synthetic_manifest, inpaint_config, cv2_inpaint_spy, requested
):
    run, _, _ = _sample(inpaint_fixture, synthetic_manifest, inpaint_config)

    results = run(algorithms=requested)

    assert list(results) == ["ns", "telea"]
    assert [c["flags"] for c in cv2_inpaint_spy] == [cv2.INPAINT_NS, cv2.INPAINT_TELEA]


def test_an_inpainting_failure_is_a_rejection_and_leaves_nothing_behind(
    monkeypatch, inpaint_fixture, synthetic_manifest, inpaint_config
):
    real = cv2.inpaint

    def fails_for_ns(src, inpaintMask, inpaintRadius, flags, dst=None):
        if flags == cv2.INPAINT_NS:
            raise cv2.error("simulated failure")
        return real(src, inpaintMask, inpaintRadius, flags)

    monkeypatch.setattr(cv2, "inpaint", fails_for_ns)
    run, _, config = _sample(inpaint_fixture, synthetic_manifest, inpaint_config)

    with pytest.raises(MaskRejection) as exc:
        run()

    assert exc.value.category == "inpainting_failure"
    assert exc.value.image_id == IMAGE_ID
    assert "ns" in exc.value.reason
    assert not any(p.is_file() for p in config.output_root.rglob("*")), (
        "TELEA succeeded, but a sample with any failure must write no artifact"
    )
