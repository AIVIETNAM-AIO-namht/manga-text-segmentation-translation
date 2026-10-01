"""T025 - the full inpainting pipeline on a fixture (Spec 003, US2).

Independent test of US2: one valid mask goes in; each algorithm produces a
page at the aligned size; the processed mask differs from the raw mask by the
configured dilation and nothing else; every parameter appears in the metadata;
and the same input and config give byte-identical output.

Written before ``inpaint.py``'s ``process_sample`` exists, so it must FAIL
until T028 lands, and its determinism test until T029 as well.
"""

from __future__ import annotations

import json
import re
from dataclasses import fields

import cv2
import numpy as np

from manga_text_seg.config import DilationConfig
from manga_text_seg.inpaint import process_sample
from manga_text_seg.runs import SampleMetadata, create_run, sample_paths

METHOD = "classical_baseline"
IMAGE_ID = "ARMS/001"
ALGORITHMS = ("telea", "ns")
_TIME_FIELD = re.compile(r'("processing_time_seconds": )[-+0-9.eE]+')


def _setup(inpaint_fixture, synthetic_manifest, inpaint_config, **config_kwargs):
    manifest = synthetic_manifest()  # ARMS/000 and ARMS/001
    root = inpaint_fixture(METHOD, IMAGE_ID)
    config = inpaint_config({METHOD: root}, **config_kwargs)

    def run(run_id: str):
        create_run(config.output_root, run_id)
        return process_sample(
            manifest,
            "ARMS",
            "001",
            method=METHOD,
            run_id=run_id,
            config=config,
            experiment_id="fixture-experiment",
            model_repository=None,
        )

    return run, root, config


def _files(config, run_id, algorithm):
    paths = sample_paths(config.output_root, run_id, METHOD, algorithm, IMAGE_ID)
    directory, stem = paths.image_path.parent, paths.image_path.stem
    return {
        "inpainted": paths.image_path,
        "metadata": paths.metadata_path,
        "raw": directory / f"{stem}.raw.png",
        "mask": directory / f"{stem}.mask.png",
        "page": directory / f"{stem}.page.png",
    }


def _read(path):
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def test_a_valid_mask_becomes_two_aligned_inpainted_pages(
    inpaint_fixture, synthetic_manifest, inpaint_config, aligned_mask
):
    run, _, config = _setup(inpaint_fixture, synthetic_manifest, inpaint_config)

    results = run("run-a")

    assert set(results) == set(ALGORITHMS)
    for algorithm in ALGORITHMS:
        page = _read(_files(config, "run-a", algorithm)["inpainted"])
        assert page.shape[:2] == aligned_mask.shape


def test_the_processed_mask_differs_from_the_raw_mask_only_by_the_dilation(
    inpaint_fixture, synthetic_manifest, inpaint_config
):
    run, _, config = _setup(inpaint_fixture, synthetic_manifest, inpaint_config)
    run("run-a")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for algorithm in ALGORITHMS:
        files = _files(config, "run-a", algorithm)
        raw, processed = _read(files["raw"]), _read(files["mask"])

        assert not np.array_equal(processed, raw), "the mask must have been dilated"
        assert np.array_equal(processed, cv2.dilate(raw, kernel, iterations=1))
        assert (processed >= raw).all(), "dilation only adds text pixels"


def test_with_dilation_off_the_processed_mask_equals_the_raw_mask(
    inpaint_fixture, synthetic_manifest, inpaint_config
):
    run, _, config = _setup(
        inpaint_fixture,
        synthetic_manifest,
        inpaint_config,
        dilation=DilationConfig(enabled=False),
    )
    run("run-a")

    for algorithm in ALGORITHMS:
        files = _files(config, "run-a", algorithm)
        assert np.array_equal(_read(files["mask"]), _read(files["raw"]))


def test_every_parameter_appears_in_the_metadata(
    inpaint_fixture, synthetic_manifest, inpaint_config
):
    run, _, config = _setup(
        inpaint_fixture, synthetic_manifest, inpaint_config, radius=4
    )
    run("run-a")
    expected_fields = {f.name for f in fields(SampleMetadata)}

    for algorithm in ALGORITHMS:
        doc = json.loads(
            _files(config, "run-a", algorithm)["metadata"].read_text(encoding="utf-8")
        )
        assert set(doc) == expected_fields
        assert doc["inpaint_radius"] == 4
        assert doc["dilation"]["enabled"] is True
        assert doc["dilation"]["kernel_shape"] == "ellipse"
        assert doc["dilation"]["kernel_size"] == [3, 3]
        assert doc["dilation"]["iterations"] == 1
        assert doc["alignment"]["rule"] == "pass-through"
        assert doc["output_format"] == "png"
        assert doc["processing_time_seconds"] >= 0
        assert doc["status"] == "ok"
        assert doc["error_message"] is None
        for name in ("image_id", "method", "experiment_id", "input_image_path"):
            assert doc[name] not in (None, ""), f"{name} is empty"


def _normalised(path) -> bytes:
    """A file's bytes, with the one non-deterministic field blanked out."""
    data = path.read_bytes()
    if path.suffix == ".json":
        text = data.decode("utf-8")
        return _TIME_FIELD.sub(r"\g<1>0", text).encode("utf-8")
    return data


def test_two_runs_of_the_same_input_are_byte_identical(
    inpaint_fixture, synthetic_manifest, inpaint_config
):
    """Everything but ``processing_time_seconds`` (and the run ID, which is the
    directory name) must match byte for byte (SC-012)."""
    run, _, config = _setup(inpaint_fixture, synthetic_manifest, inpaint_config)
    run("run-a")
    run("run-b")
    dir_a, dir_b = config.output_root / "run-a", config.output_root / "run-b"

    files_a = sorted(p.relative_to(dir_a) for p in dir_a.rglob("*") if p.is_file())
    files_b = sorted(p.relative_to(dir_b) for p in dir_b.rglob("*") if p.is_file())

    assert files_a == files_b
    assert files_a, "the runs wrote nothing"
    for relative in files_a:
        assert _normalised(dir_a / relative) == _normalised(dir_b / relative), (
            f"{relative} differs between two runs of the same input"
        )
