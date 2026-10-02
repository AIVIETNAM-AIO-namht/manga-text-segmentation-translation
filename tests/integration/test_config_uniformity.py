"""T053: main-benchmark methods share one mask and inpainting configuration."""

import json

from manga_text_seg.inpaint import process_batch
from manga_text_seg.runs import INPAINT_METHOD_ORDER, sample_paths


def test_all_methods_record_identical_mask_processing_and_radius(
    inpaint_fixture, synthetic_manifest, inpaint_config
):
    methods = INPAINT_METHOD_ORDER
    roots = {
        method: inpaint_fixture(method, "ARMS/001") for method in methods
    }
    config = inpaint_config(roots, algorithms=("telea",), radius=4)
    manifest = synthetic_manifest()
    origins = {method: ("fixture-run", None) for method in methods}

    process_batch(
        manifest, [("ARMS", "001")], methods=methods, run_id="uniform",
        config=config, origins=origins, algorithms=("telea",),
    )

    recorded = []
    for method in methods:
        sidecar = sample_paths(
            config.output_root, "uniform", method, "telea", "ARMS/001"
        ).metadata_path
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        recorded.append((metadata["dilation"], metadata["inpaint_radius"]))
    assert len(recorded) == len(methods)
    assert all(item == recorded[0] for item in recorded)
    assert recorded[0][1] == 4
