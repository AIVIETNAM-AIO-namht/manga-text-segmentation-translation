"""Regression tests for visualization panel alignment (FR-032/FR-033).

Covers the broadcast-crash fix in `visualize._panels`: the GT mask is
aligned to the prediction's geometry with the same
``align(gt, pred, cfg.alignment_strategy)`` call the sweep uses (run.py),
instead of loading the larger unaligned GT (1176x1656 in production)
next to the aligned prediction (1170x1654) — which crashed
``_build_overlay`` with a shape-mismatch ValueError.

Fixture geometry mirrors production: GT mask 120x110, raw page +
prediction 110x100 (6px x 2px larger GT, R2 of align.py).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from manga_text_seg.config import Config
from manga_text_seg.manifest import Manifest, Pair, save_manifest
from manga_text_seg.report import write_metrics_csv
from manga_text_seg.visualize import _panels, render_composites

# Production sizes: GT mask 1176x1656, raw page + aligned prediction
# 1170x1654. Scaled down 10x for test speed while keeping the mismatch.
GT_H, GT_W = 120, 110
PAGE_H, PAGE_W = 110, 100
MANGA, STEM = "TestTitle", "001"


def _write_pair_files(tmp_path: Path) -> tuple[Config, Manifest]:
    """Create raw page, GT mask, aligned prediction, manifest and metrics."""
    raw_root = tmp_path / "raw"
    gt_root = tmp_path / "gt"
    out_root = tmp_path / "out"

    raw_dir = raw_root / MANGA
    gt_dir = gt_root / MANGA
    raw_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    # Raw page at page geometry (110x100).
    raw_path = raw_dir / f"{STEM}.png"
    page = np.random.randint(0, 256, (PAGE_H, PAGE_W), dtype=np.uint8)
    cv2.imwrite(str(raw_path), page)

    # Color GT mask at the LARGER gt geometry (120x110) — the R2 mismatch.
    mask_path = gt_dir / f"{STEM}.png"
    gt = np.zeros((GT_H, GT_W, 3), dtype=np.uint8)
    gt[: GT_H // 2, :, :] = (255, 0, 255)  # magenta BGR
    gt[GT_H // 2:, :, :] = (0, 0, 0)       # black
    cv2.imwrite(str(mask_path), gt)

    cfg = Config(
        raw_root=raw_root,
        gt_root=gt_root,
        output_root=out_root,
        run_id="regression",
        methods=[{"name": "otsu", "params": {}}],
        viz_mode="first-N",
        viz_n=20,
    )

    # Prediction PNG at page geometry — the sweep saves predictions ALIGNED
    # to the raw page (run.py), so this is the shape _build_overlay expects.
    pred_path = cfg.output_dir / "otsu" / MANGA / f"{STEM}.png"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred = np.zeros((PAGE_H, PAGE_W), dtype=np.uint8)
    pred[: PAGE_H // 3, :] = 255
    cv2.imwrite(str(pred_path), pred)

    manifest = Manifest(
        run_id=cfg.run_id,
        gt_root=cfg.gt_root,
        raw_root=cfg.raw_root,
        pairs=[
            Pair(
                manga=MANGA, stem=STEM,
                mask_path=mask_path,
                raw_path=raw_path,
                mask_encoding="magenta-black",
            )
        ],
    )
    save_manifest(manifest, cfg.output_dir / "manifest.json")
    write_metrics_csv(
        [{
            "manga": MANGA, "stem": STEM, "method": "otsu",
            "iou": 0.5, "precision": 0.6, "recall": 0.7, "f1": 0.65,
            "tp": 10, "fp": 3, "fn": 4, "inference_time_seconds": 0.12,
        }],
        cfg.output_dir / "metrics.csv",
    )
    return cfg, manifest


class TestPanelsAlignment:
    """_panels must return panels sharing one geometry when pred exists."""

    def test_panels_share_prediction_geometry(self, tmp_path: Path) -> None:
        """Raw, GT and pred panels all 110x100 — no 120x110 GT panel."""
        cfg, manifest = _write_pair_files(tmp_path)
        panels = _panels(cfg, manifest, "otsu", MANGA, STEM)
        assert len(panels) == 3
        assert {p.shape for p in panels} == {(PAGE_H, PAGE_W)}

    def test_gt_cropped_to_prediction_geometry(self, tmp_path: Path) -> None:
        """GT panel is crop-topleft aligned to pred (was 120x110 pre-fix)."""
        cfg, manifest = _write_pair_files(tmp_path)
        panels = _panels(cfg, manifest, "otsu", MANGA, STEM)
        assert panels[1].shape == (PAGE_H, PAGE_W)
        # Top-left (0,0) region preserved by crop-topleft alignment.
        assert panels[1][0, 0] == 255  # magenta → text (255)

    def test_unaligned_gt_returned_without_pred(self, tmp_path: Path) -> None:
        """No prediction file → raw GT panel returned at its own geometry."""
        cfg, manifest = _write_pair_files(tmp_path)
        pred_path = cfg.output_dir / "otsu" / MANGA / f"{STEM}.png"
        pred_path.unlink()
        panels = _panels(cfg, manifest, "otsu", MANGA, STEM)
        assert len(panels) == 2
        assert panels[1].shape == (GT_H, GT_W)


class TestRenderCompositesAlignment:
    """End-to-end: render_composites must not broadcast-crash on R2 mismatch."""

    def test_mismatched_gt_pred_no_broadcast_crash(self, tmp_path: Path) -> None:
        """Regression for ValueError: operands could not be broadcast together."""
        cfg, _ = _write_pair_files(tmp_path)
        rendered = render_composites(cfg)
        assert rendered["otsu"] == 1

        out_png = cfg.output_dir / "viz" / "otsu" / "first-20" / f"{MANGA}_{STEM}.png"
        assert out_png.is_file()
        composite = cv2.imread(str(out_png))
        assert composite is not None
        assert composite.ndim == 3
        assert composite.shape[0] > 0 and composite.shape[1] > 0