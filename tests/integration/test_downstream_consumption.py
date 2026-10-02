"""T056: downstream access needs only run, method and image identity."""

import json

from manga_text_seg.inpaint import process_sample
from manga_text_seg.runs import artifact_paths, sample_paths


def test_consumer_resolves_page_and_metadata_from_public_address(
    inpaint_fixture, synthetic_manifest, inpaint_config
):
    manifest = synthetic_manifest()
    source = inpaint_fixture("classical_baseline", "ARMS/001")
    config = inpaint_config({"classical_baseline": source}, algorithms=("telea",))
    process_sample(
        manifest, "ARMS", "001", method="classical_baseline", run_id="consumer",
        config=config, experiment_id="fixture", model_repository=None,
        algorithms=("telea",),
    )

    # This is the entire consumer contract; no manifest or run index is read.
    paths = artifact_paths(sample_paths(
        config.output_root, "consumer", "classical_baseline", "telea", "ARMS/001"
    ))
    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    assert paths.inpainted.is_file()
    assert metadata["image_id"] == "ARMS/001"
    assert metadata["method"] == "classical_baseline"
    assert metadata["algorithm"] == "telea"
    assert paths.raw_mask.is_file() and paths.processed_mask.is_file()
