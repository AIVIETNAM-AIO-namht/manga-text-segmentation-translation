"""Report generation: metrics.csv, summaries.json, failures.json (FR-030, FR-031).

Schema follows contracts/metrics.schema.json — no Pixel Accuracy field.

Contract rule (output-layout.md): ``report`` runs from persisted ``metrics.csv``
WITHOUT re-running inference.  ``sweep`` writes metrics.csv; ``report`` reads it
back and (re)builds the per-method summaries.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config

METRIC_COLUMNS = [
    "manga", "stem", "method",
    "iou", "precision", "recall", "f1",
    "tp", "fp", "fn", "inference_time_seconds",
]


def write_metrics_csv(rows: list[dict], path: Path) -> None:
    """Write per-image metric rows to CSV (metrics.schema.json row fields)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    df.to_csv(path, index=False)


def summarize(rows: list[dict]) -> list[dict]:
    """Aggregate per-method means across metric rows (metrics.schema.json)."""
    if not rows:
        return []
    df = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    for method_name, group in df.groupby("method", sort=True):
        summaries.append({
            "method": method_name,
            "n_images": int(len(group)),
            "mean_iou": float(group["iou"].mean()),
            "std_iou": float(group["iou"].std(ddof=0)) if len(group) > 1 else 0.0,
            "mean_precision": float(group["precision"].mean()),
            "mean_recall": float(group["recall"].mean()),
            "mean_f1": float(group["f1"].mean()),
            "mean_time_seconds": float(group["inference_time_seconds"].mean()),
        })
    return summaries


def write_summaries_json(rows: list[dict], path: Path, run_id: str) -> None:
    """Write per-method summaries as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"run_id": run_id, "summaries": summarize(rows)}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def write_failures_json(
    failures_by_method: dict[str, list[dict]],
    path: Path,
    run_id: str,
) -> None:
    """Write per-method failure lists as JSON (FR-031)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"run_id": run_id, "failures": failures_by_method}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


SCORECARD_COLUMNS = [
    "method", "n_images",
    "mean_iou", "std_iou",
    "mean_precision", "mean_recall", "mean_f1",
    "mean_time_seconds",
]


def write_scorecard_csv(summaries: list[dict], path: Path) -> None:
    """Write the per-method scorecard (US5)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summaries, columns=SCORECARD_COLUMNS)
    df.to_csv(path, index=False)


def load_metrics_csv(path: Path) -> list[dict]:
    """Read persisted metrics.csv back into row dicts."""
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def build_report(cfg: Config) -> Path:
    """Recompute summaries.json from persisted metrics.csv (no inference).

    Returns the written summaries path.
    """
    metrics_path = cfg.output_dir / "metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics.csv not found at {metrics_path}; run 'sweep' first")
    rows = load_metrics_csv(metrics_path)
    summaries_path = cfg.output_dir / "summaries.json"
    write_summaries_json(rows, summaries_path, cfg.run_id)
    return summaries_path


def build_scorecard(cfg: Config, filename: str = "metrics_summary.csv") -> Path:
    """Export the per-method scorecard CSV from persisted metrics.csv (US5).

    Reads metrics.csv (no inference), summarizes per method, writes the
    scorecard. Returns the written CSV path.
    """
    metrics_path = cfg.output_dir / "metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics.csv not found at {metrics_path}; run 'sweep' first")
    rows = load_metrics_csv(metrics_path)
    summaries = summarize(rows)
    scorecard_path = cfg.output_dir / filename
    write_scorecard_csv(summaries, scorecard_path)
    return scorecard_path