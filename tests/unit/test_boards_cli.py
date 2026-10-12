"""``cmd_boards`` reads each method's metrics from the right place (Spec 003 US4).

Two independent regressions, both silent-or-crash:

- ``classical_baseline``'s ``resolved_identity`` must be the pinned Spec 1
  *method* (``adaptive``), not the Spec 1 run id (``default``). The boards
  filter ``outputs/segmentation/default/metrics.csv`` on it, so the run id
  matched no row and the command wrote zero boards while still exiting 0.
- A deep-learning method's per-page metrics live at ``<source_root>/metrics/
  per-page.csv`` (the admitted layout, deliverables/README.md), not one
  directory above it. The old path raised ``FileNotFoundError``.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from manga_text_seg.cli import _resolved_identity, cmd_boards

ROOT = Path(__file__).resolve().parents[2]
IMAGE_ID = "ARMS/001"
METHOD = "comic_text_detector"
RUN_ID = "board-run"
SIZE = (64, 96)  # (height, width); boards only require equal panel sizes

DILATION = {"enabled": True, "kernel_shape": "ellipse", "kernel_size": [3, 3], "iterations": 1}


def test_classical_baseline_resolves_to_the_spec1_method_not_the_run():
    """The pinned Spec 1 method (research R1), read from the real config."""
    config = json.loads((ROOT / "configs" / "inpainting.json").read_text("utf-8"))
    classical = config["methods"]["classical_baseline"]["source_root"]

    assert _resolved_identity("classical_baseline", ROOT / classical) == "adaptive"
    # A DL identity still resolves to its admitted run id (the parent segment).
    assert _resolved_identity(METHOD, Path("deliverables/spec002") / METHOD) == "spec002"


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full(SIZE, 255, np.uint8))


def _write_config(repo: Path) -> Path:
    config = {
        "methods": {
            "classical_baseline": {"source_root": "outputs/segmentation/default/adaptive"},
            "manga_text_segmentation": {"source_root": "deliverables/spec002/manga-text-segmentation"},
            "comic_text_detector": {"source_root": f"deliverables/spec002/{METHOD}"},
            "unetpp_efficientnetv2": {"source_root": "deliverables/spec002/unetpp-efficientnetv2"},
        },
        "mask_processing": {"dilation": DILATION},
        "inpaint": {"radius": 3, "algorithms": ["telea", "ns"], "output_format": "png"},
        "selection": {"n": 5, "manual_artifact_flags": []},
        "output_root": "outputs/inpainting/",
    }
    path = repo / "configs" / "inpainting.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def test_boards_read_per_page_metrics_from_the_method_root(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    manga, stem = IMAGE_ID.split("/")
    admitted = repo / "deliverables" / "spec002" / METHOD
    config_path = _write_config(repo)

    # The admitted metrics file sits *inside* the method root.
    metrics = admitted / "metrics" / "per-page.csv"
    metrics.parent.mkdir(parents=True)
    metrics.write_text(
        "manga,stem,method,iou,precision,recall,f1,tp,fp,fn,inference_time_seconds\n"
        f"{manga},{stem},{METHOD},0.5,0.5,0.5,0.5,10,5,5,0.01\n",
        encoding="utf-8",
    )

    run_dir = repo / "outputs" / "inpainting" / RUN_ID
    for algorithm in ("telea", "ns"):
        sample_dir = run_dir / METHOD / algorithm / manga
        _write_mask(sample_dir / f"{stem}.png")  # inpainted
        _write_mask(sample_dir / f"{stem}.page.png")
        _write_mask(sample_dir / f"{stem}.raw.png")
        _write_mask(sample_dir / f"{stem}.mask.png")

    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "methods": [METHOD],
                "method_sources": {
                    METHOD: {
                        "source_root": str(admitted),
                        "resolved_identity": "spec002",
                        "reason": None,
                    }
                },
                "mask_processing_config": {"dilation": DILATION},
                "inpainting_config": {"radius": 3, "algorithms": ["telea", "ns"], "output_format": "png"},
                "page_list_identity": "fixture",
                "input_image_identity": {},
                "counts": {},
                "ablation": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    assert cmd_boards(str(config_path), RUN_ID, METHOD) == 0

    boards = sorted((run_dir / "boards" / METHOD).rglob("*.png"))
    assert boards, "cmd_boards wrote no boards — it read the wrong metrics file"
    assert (run_dir / "selection.json").is_file()
