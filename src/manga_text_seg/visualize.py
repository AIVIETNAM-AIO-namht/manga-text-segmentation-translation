"""Visualization: 4-panel comparison composites (FR-032, FR-033).

Panels: raw page | ground-truth mask | prediction mask | overlay.
Selection modes (FR-033):
  - ``first-N`` : first N pairs in deterministic manifest order (default)
  - ``best-N``  : N pairs ranked by F1, best first
  - ``worst-N`` : N pairs ranked by F1, worst first

Runs from persisted metrics.csv WITHOUT re-running inference
(contract output-layout.md rule 4).  Output:
  outputs/segmentation/<run_id>/viz/<method>/<mode>-<n>/<manga>_<stem>.png
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .align import align
from .config import Config
from .imaging import load_gt_mask, load_mask, load_raw_page
from .manifest import Manifest, load_manifest
from .normalize import normalize_mask
from .report import load_metrics_csv

OVERLAY_GT_COLOR = (0, 255, 0)      # green: GT only
OVERLAY_PRED_COLOR = (255, 0, 0)    # blue: prediction only
OVERLAY_BOTH_COLOR = (0, 255, 255)  # yellow: overlap
OVERLAY_BG_COLOR = (40, 40, 40)

# Directory labels for selection modes (FR-033). quickstart.md documents the
# composite path as viz/<method>/<mode>-<n>/ with the SHORT label
# ("first-20", "best-10", ...), NOT the raw config value "first-N" — so the
# config mode string is mapped to its label here.
MODE_LABELS = {
    "first-N": "first",
    "best-N": "best",
    "worst-N": "worst",
}


def _selection_image_ids(
    cfg: Config,
    manifest: Manifest,
    metrics_by_method: dict[str, dict[str, float]],
    method: str,
) -> list[tuple[str, str]]:
    """Select (manga, stem) pairs for one method per FR-033."""
    n = cfg.viz_n
    if cfg.viz_mode == "first-N":
        selected = [(p.manga, p.stem) for p in manifest.pairs[:n]]
    else:
        f1_by_id = metrics_by_method.get(method, {})
        ranked = sorted(
            f1_by_id.items(),
            key=lambda kv: (kv[1], kv[0]),
            reverse=(cfg.viz_mode == "best-N"),
        )
        selected = []
        for image_id, _ in ranked[:n]:
            manga, stem = image_id.split("/", 1)
            selected.append((manga, stem))
    return selected


def _build_overlay(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """GT-green / pred-blue / overlap-yellow overlay (FR-032 panel 4)."""
    h, w = gt.shape
    overlay = np.full((h, w, 3), OVERLAY_BG_COLOR, dtype=np.uint8)
    gt_bool = gt == 255
    pred_bool = pred == 255
    overlay[gt_bool & ~pred_bool] = OVERLAY_GT_COLOR
    overlay[pred_bool & ~gt_bool] = OVERLAY_PRED_COLOR
    overlay[gt_bool & pred_bool] = OVERLAY_BOTH_COLOR
    return overlay


def _render_composite(row_img: np.ndarray, gt: np.ndarray,
                      pred: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Assemble a 2x2 grid of the four panels, resized to a common short side."""
    panels = [row_img, gt, pred, overlay]
    short_side = 500
    scaled = []
    for panel in panels:
        if panel.ndim == 2:
            panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        h, w = panel.shape[:2]
        scale = short_side / max(h, w)
        scaled.append(cv2.resize(panel, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA))
    ph = max(p.shape[0] for p in scaled)
    pw = max(p.shape[1] for p in scaled)
    padded = [cv2.copyMakeBorder(p, 0, ph - p.shape[0], 0, pw - p.shape[1],
                                 cv2.BORDER_CONSTANT, value=OVERLAY_BG_COLOR)
              for p in scaled]
    top = np.hstack([padded[0], padded[1]])
    bottom = np.hstack([padded[2], padded[3]])
    return np.vstack([top, bottom])


def _panels(
    cfg: Config,
    manifest: Manifest,
    method: str,
    manga: str,
    stem: str,
) -> list[np.ndarray]:
    """Load the four panels for one pair (or fewer if a file is missing)."""
    pair = next((p for p in manifest.pairs
                 if p.manga == manga and p.stem == stem), None)
    panels = []
    if pair is not None:
        panels.append(load_raw_page(pair.raw_path))
        gt = normalize_mask(load_gt_mask(pair.mask_path))
        pred_path = cfg.output_dir / method / manga / f"{stem}.png"
        if pred_path.is_file():
            pred = load_mask(pred_path)
            # GT masks are 6x2 px larger than raw pages in this dataset (R2);
            # align GT to the prediction's geometry (crop-topleft, FR-015) so
            # the overlay and panels share the exact dimensions the sweep's
            # metrics were computed on. Without this, _build_overlay would
            # broadcast-crash on mismatched GT/pred shapes.
            aligned = align(gt, pred, cfg.alignment_strategy)
            panels.append(aligned.mask)
            panels.append(aligned.page)
        else:
            panels.append(gt)
    return panels


def render_composites(cfg: Config) -> dict:
    """Render composites for every configured method (FR-032, FR-033)."""
    manifest_path = cfg.output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json not found at {manifest_path}; run 'discover' first")
    manifest = load_manifest(manifest_path)

    metrics_path = cfg.output_dir / "metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics.csv not found at {metrics_path}; run 'sweep' first")
    rows = load_metrics_csv(metrics_path)

    metrics_by_method: dict[str, dict[str, float]] = {}
    for row in rows:
        metrics_by_method.setdefault(row["method"], {})[
            f"{row['manga']}/{row['stem']}"] = float(row["f1"])

    rendered: dict[str, int] = {}
    out_root = cfg.output_dir / "viz"
    for entry in cfg.methods:
        method = entry["name"]
        selected = _selection_image_ids(cfg, manifest, metrics_by_method, method)
        count = 0
        for manga, stem in selected:
            panels = _panels(cfg, manifest, method, manga, stem)
            if len(panels) == 3:
                overlay = _build_overlay(panels[1], panels[2])
                composite = _render_composite(panels[0], panels[1], panels[2], overlay)
                out_dir = out_root / method / f"{MODE_LABELS[cfg.viz_mode]}-{cfg.viz_n}"
                out_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_dir / f"{manga}_{stem}.png"), composite)
                count += 1
        rendered[method] = count
    return rendered