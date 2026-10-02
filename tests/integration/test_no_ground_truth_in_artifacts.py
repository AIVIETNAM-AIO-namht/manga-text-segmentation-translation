"""T054: generated inpainting artifacts and boards contain no GT input."""

import cv2

from manga_text_seg.boards import PANEL_LABELS, render_board, write_board
from manga_text_seg.inpaint import process_sample
from manga_text_seg.runs import artifact_paths, sample_paths


def test_inpaint_outputs_and_board_do_not_reference_ground_truth(
    tmp_path, inpaint_fixture, synthetic_manifest, inpaint_config
):
    manifest = synthetic_manifest()
    source = inpaint_fixture("classical_baseline", "ARMS/001")
    config = inpaint_config({"classical_baseline": source})
    process_sample(
        manifest, "ARMS", "001", method="classical_baseline", run_id="no-gt",
        config=config, experiment_id="fixture", model_repository=None,
    )
    telea = artifact_paths(sample_paths(
        config.output_root, "no-gt", "classical_baseline", "telea", "ARMS/001"
    ))
    ns = artifact_paths(sample_paths(
        config.output_root, "no-gt", "classical_baseline", "ns", "ARMS/001"
    ))
    load = lambda path: cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    board_sample = {
        "image_id": "ARMS/001",
        "method": "classical_baseline",
        "algorithm": "telea+ns",
        "mask_proc_config": {"dilation": {"enabled": True}, "inpaint_radius": 3},
        "panels": {
            "original_page": load(telea.page),
            "raw_prediction_mask": load(telea.raw_mask),
            "dilated_mask": load(telea.processed_mask),
            "telea_result": load(telea.inpainted),
            "ns_result": load(ns.inpainted),
        },
    }
    assert set(render_board(board_sample)["panels"]) == set(PANEL_LABELS)
    board_path = write_board(board_sample, tmp_path / "board.png")

    text_files = [
        path for path in config.output_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".txt", ".md", ".csv"}
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in text_files)
    assert str(manifest.gt_root) not in combined
    assert "groundtruth" not in combined.lower()
    assert board_path.is_file()
