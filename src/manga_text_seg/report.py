"""Report generation: metrics.csv, summaries.json, failures.json (FR-030, FR-031).

Schema follows contracts/metrics.schema.json — no Pixel Accuracy field.

Contract rule (output-layout.md): ``report`` runs from persisted ``metrics.csv``
WITHOUT re-running inference.  ``sweep`` writes metrics.csv; ``report`` reads it
back and (re)builds the per-method summaries.

The cross-method comparison (T038) is additive: ``write_comparison_csv`` and
``write_comparison_charts`` consume the comparison ``benchmark.py`` assembles and
render whatever it hands over.  Nothing here branches on a method's name — the
device label and the contamination disclosure are already resolved into the row
(FR-051, SC-011).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: no display in a benchmark run

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import pandas as pd  # noqa: E402

from .config import Config  # noqa: E402

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
    """Read persisted metrics.csv back into row dicts.

    ``stem`` is read as text because it is a page identifier, not a quantity:
    it is zero-padded (``"000"``), and letting pandas infer it as an integer
    silently rewrites it to ``0``, so every caller that joins a row back to a
    page by ``manga/stem`` — the failure counts, the visualisation selection —
    would quietly match nothing.
    """
    df = pd.read_csv(path, dtype={"stem": str})
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


# --- The cross-method comparison (T038; FR-037, FR-038, FR-060, SC-011) -------

#: The four accuracy metrics, and only those — never Pixel Accuracy (FR-035).
COMPARISON_METRICS = ("iou", "precision", "recall", "f1")

#: The one summary table's columns: every metric as mean and std, both timing
#: figures, the producing device by name, the contamination disclosure beside
#: the scores, and the reason an unrun method has no numbers (FR-037, FR-060).
COMPARISON_COLUMNS = [
    "method",
    "status",
    *[f"{metric}_{stat}" for metric in COMPARISON_METRICS for stat in ("mean", "std")],
    "timing_mean_seconds",
    "timing_total_seconds",
    "device_name",
    "disclosure",
    "reason",
]


def _comparison_row(row: dict) -> dict:
    """Flatten one comparison row into the table's columns.

    An unrun method carries no number at all: its metric and timing cells are
    left absent so pandas renders them as NaN, which is what FR-036 requires —
    a zero would read as a score (US3 scenario 5).
    """
    flat: dict[str, Any] = {
        "method": row["method"],
        "status": row["status"],
        "device_name": (row.get("device") or {}).get("name"),
        "disclosure": row.get("disclosure"),
        "reason": row.get("reason"),
    }
    metrics = row.get("metrics")
    if metrics:
        for metric in COMPARISON_METRICS:
            flat[f"{metric}_mean"] = metrics[metric]["mean"]
            flat[f"{metric}_std"] = metrics[metric]["std"]
    timing = row.get("timing")
    if timing:
        flat["timing_mean_seconds"] = timing["mean_seconds"]
        flat["timing_total_seconds"] = timing["total_seconds"]
    return flat


def write_comparison_csv(comparison: dict, path: Path) -> Path:
    """Write the one summary table, sorted by method (FR-037).

    Sorted by name and never by time: FR-060 forbids ranking across devices, and
    a table ordered by a timing figure is exactly that ranking (US3 scenario 4).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        (_comparison_row(row) for row in comparison["rows"]),
        key=lambda row: row["method"],
    )
    pd.DataFrame(rows, columns=COMPARISON_COLUMNS).to_csv(path, index=False)
    return path


def write_comparison_charts(comparison: dict, out_dir: Path) -> list[Path]:
    """Write the accuracy chart set: the four metrics, every method (FR-038).

    One grouped bar chart, because FR-038 asks for the metrics compared across
    methods and one chart shows all four against each other.  Only scored rows
    appear — an unrun method has no bar to draw, and drawing a zero-length one
    would be the placeholder FR-036 forbids.

    FR-038 also allows a table in place of a timing chart, and the comparison
    CSV is that table, so timing is not charted here.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scored = [row for row in comparison["rows"] if row["metrics"] is not None]
    if not scored:
        return []

    methods = [row["method"] for row in scored]
    positions = range(len(COMPARISON_METRICS))
    width = 0.8 / max(len(methods), 1)

    figure, axes = plt.subplots(figsize=(10, 6))
    for offset, row in enumerate(scored):
        axes.bar(
            [position + offset * width for position in positions],
            [row["metrics"][metric]["mean"] for metric in COMPARISON_METRICS],
            width,
            yerr=[row["metrics"][metric]["std"] for metric in COMPARISON_METRICS],
            capsize=3,
            label=row["method"],
        )
    axes.set_xticks([position + 0.4 - width / 2 for position in positions])
    axes.set_xticklabels([metric.upper() for metric in COMPARISON_METRICS])
    axes.set_ylabel("score")
    axes.set_ylim(0, 1)
    axes.set_title(f"Segmentation accuracy by method — run {comparison['run_id']}")
    axes.legend(loc="lower right", fontsize="small")
    figure.tight_layout()

    path = out_dir / "accuracy-by-method.png"
    figure.savefig(path, dpi=100)
    plt.close(figure)
    return [path]