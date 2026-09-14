"""Single-method inference loop + mask sidecar writing (US1, FR-021-FR-024, FR-035).

Pipeline per pair (contract output-layout.md / method-interface.md):
  1. load raw page (grayscale)
  2. preprocess (grayscale/denoise) — BEFORE segment
  3. method.segment(pre) -> binary {0, 255} at raw-page size
  4. normalize GT mask (KMeans-2) and align (crop-topleft) with the prediction
  5. save prediction PNG at aligned size + sidecar JSON
  6. compute per-image metrics (FR-025)

Failures never abort the run: a ``status: "failed"`` sidecar is written and
the pair is recorded for FR-031 reporting.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .align import align
from .config import Config
from .imaging import load_gt_mask, load_raw_page, save_mask
from .manifest import Manifest, Pair
from .methods import BaseSegmentationMethod, create
from .metrics import compute_metrics
from .normalize import normalize_mask
from .preprocess import preprocess


class RunError(RuntimeError):
    """Raised when a method cannot be created or the run setup is invalid."""


def _find_params(cfg: Config, method_name: str) -> dict:
    for entry in cfg.methods:
        if entry.get("name") == method_name:
            return dict(entry.get("params") or {})
    raise RunError(f"Method '{method_name}' is not configured in cfg.methods")


def _sidecar(
    image_id: str,
    method: BaseSegmentationMethod,
    cfg: Config,
    pre_size: tuple[int, int],
    post_size: tuple[int, int],
    inference_time_seconds: float,
    threshold: str,
    status: str = "ok",
    error: Optional[str] = None,
) -> dict:
    """Build the metadata sidecar document (output-layout.md)."""
    doc = {
        "image_id": image_id,
        "method": method.name,
        "preprocessing": {"grayscale": True, "denoise": cfg.denoise},
        "postprocessing": {"fill_holes": False},
        "threshold": threshold,
        "alignment": {
            "strategy": cfg.alignment_strategy,
            "pre_size": [pre_size[0], pre_size[1]],
            "post_size": [post_size[0], post_size[1]],
        },
        "inference_time_seconds": round(inference_time_seconds, 4),
        "status": status,
    }
    if status == "failed":
        doc["error"] = error or "unknown error"
    return doc


def _process_pair(
    cfg: Config,
    method: BaseSegmentationMethod,
    pair: Pair,
    collect_metrics: bool = True,
) -> dict:
    """Process one pair, persist prediction + sidecar, and return a metrics row."""
    page = load_raw_page(pair.raw_path)
    pre = preprocess(page, cfg.denoise)

    t0 = time.perf_counter()
    raw_pred = method.segment(pre)
    elapsed = time.perf_counter() - t0

    gt_raw = load_gt_mask(pair.mask_path)
    gt = normalize_mask(gt_raw)
    aligned = align(gt, raw_pred, cfg.alignment_strategy)
    if aligned.mask.shape != aligned.page.shape:
        raise RunError(f"Alignment produced mismatched shapes for {pair.image_id}")

    out_dir = cfg.output_dir / method.name / pair.manga
    pred_path = out_dir / f"{pair.stem}.png"
    sidecar_path = out_dir / f"{pair.stem}.json"

    save_mask(pred_path, aligned.page)
    with open(sidecar_path, "w", encoding="utf-8") as fh:
        json.dump(
            _sidecar(
                image_id=pair.image_id,
                method=method,
                cfg=cfg,
                pre_size=gt_raw.shape,
                post_size=aligned.page.shape,
                inference_time_seconds=elapsed,
                threshold=getattr(method, "threshold", "auto"),
            ),
            fh,
            indent=2,
            ensure_ascii=False,
        )
        fh.write("\n")

    row = {"manga": pair.manga, "stem": pair.stem, "method": method.name,
           "inference_time_seconds": elapsed}
    if collect_metrics:
        row.update(compute_metrics(aligned.mask, aligned.page))
    else:
        row.update({"iou": None, "precision": None, "recall": None,
                    "f1": None, "tp": None, "fp": None, "fn": None})
    return row


def run_method_on_manifest(
    cfg: Config,
    manifest: Manifest,
    method_name: str,
) -> dict:
    """Run one method across every manifest pair; returns a summary dict.

    Outputs (output-layout.md):
      outputs/segmentation/<run_id>/<method>/<manga>/<stem>.png (+ .json)
    """
    method = create(method_name, _find_params(cfg, method_name))
    failures: list[dict] = []
    ok = 0

    for pair in manifest.pairs:
        try:
            _process_pair(cfg, method, pair)
            ok += 1
        except Exception as exc:  # noqa: BLE001 - pair isolation, FR-031
            failures.append({"image_id": pair.image_id, "error": str(exc)})
            # Persist a failure sidecar so the pair is traceable.
            out_dir = cfg.output_dir / method.name / pair.manga
            out_dir.mkdir(parents=True, exist_ok=True)
            sidecar_path = out_dir / f"{pair.stem}.json"
            with open(sidecar_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "image_id": pair.image_id,
                        "method": method.name,
                        "status": "failed",
                        "error": str(exc),
                    },
                    fh,
                    indent=2,
                )

    summary = {"method": method_name, "ok": ok, "failed": len(failures), "failures": failures}
    return summary


def sweep_all_methods(
    cfg: Config,
    manifest: Manifest,
) -> dict:
    """Run every configured method across the dataset (FR-003, US2).

    Persists per-image metrics (metrics.csv), per-method summaries
    (summaries.json) and per-method failure lists (failures.json),
    per output-layout.md.  Returns an aggregate summary.
    """
    from .report import write_metrics_csv, write_summaries_json, write_failures_json

    rows: list[dict] = []
    failures_by_method: dict[str, list[dict]] = {}
    method_totals: dict[str, dict] = {}

    for entry in cfg.methods:
        method_name = entry["name"]
        method = create(method_name, dict(entry.get("params") or {}))
        ok = 0
        failed = 0
        for pair in manifest.pairs:
            try:
                row = _process_pair(cfg, method, pair)
                rows.append(row)
                ok += 1
            except Exception as exc:  # noqa: BLE001 - pair isolation, FR-031
                failed += 1
                failures_by_method.setdefault(method_name, []).append(
                    {"image_id": pair.image_id, "error": str(exc)}
                )
        method_totals[method_name] = {"ok": ok, "failed": failed}

    write_metrics_csv(rows, cfg.output_dir / "metrics.csv")
    write_summaries_json(rows, cfg.output_dir / "summaries.json", cfg.run_id)
    write_failures_json(failures_by_method, cfg.output_dir / "failures.json", cfg.run_id)

    return {
        "methods": method_totals,
        "total_pairs": manifest.total,
        "total_rows": len(rows),
        "failures": failures_by_method,
    }