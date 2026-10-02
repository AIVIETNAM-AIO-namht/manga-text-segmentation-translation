"""Metric-based sample selection for Spec 003 comparative boards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from manga_text_seg.runs import INPAINT_METHOD_ORDER

SELECTION_RULES = (
    "highest_iou_f1",
    "lowest_iou_f1",
    "most_false_positives",
    "most_false_negatives",
    "manual_artifact_flags",
)


def require_main_run(run_record: dict[str, Any], run_id: str) -> None:
    """Reject ablation records before a main comparison selects samples."""
    if run_record.get("ablation") is True:
        raise ValueError(
            f"Run {run_id!r} is an ablation run and is excluded from main comparisons"
        )


def select_samples(
    metrics: dict[str, list[dict[str, Any]]],
    rule: str,
    n: int,
    manual_flags: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Select up to ``n`` page rows independently for each segmentation method.

    Metric rows use the shared ``image_id``, ``iou``, ``f1``, ``fp`` and ``fn``
    schema. Ties break on image ID, making the selection repeatable. A short
    pool is reported as a shortfall and is never padded with unrelated pages.
    """
    if rule not in SELECTION_RULES:
        raise ValueError(f"Unknown selection rule {rule!r}; expected one of {SELECTION_RULES}")
    if n < 0:
        raise ValueError("n must be non-negative")

    results: dict[str, dict[str, Any]] = {}
    for method in INPAINT_METHOD_ORDER:
        if method not in metrics:
            continue
        rows = list(metrics[method])
        if rule == "highest_iou_f1":
            ordered = sorted(rows, key=lambda row: (-((float(row["iou"]) + float(row["f1"])) / 2), str(row["image_id"])))
        elif rule == "lowest_iou_f1":
            ordered = sorted(rows, key=lambda row: (((float(row["iou"]) + float(row["f1"])) / 2), str(row["image_id"])))
        elif rule == "most_false_positives":
            ordered = sorted(rows, key=lambda row: (-int(row["fp"]), str(row["image_id"])))
        elif rule == "most_false_negatives":
            ordered = sorted(rows, key=lambda row: (-int(row["fn"]), str(row["image_id"])))
        else:
            by_id = {str(row["image_id"]): row for row in rows}
            ordered = [by_id.get(str(image_id), {"image_id": str(image_id)}) for image_id in (manual_flags or [])]
            # Duplicate manual flags do not consume multiple places.
            ordered = list({str(row["image_id"]): row for row in ordered}.values())
        selected = ordered[:n]
        results[method] = {
            "method": method,
            "rule": rule,
            "selected": selected,
            "shortfall": max(0, n - len(selected)),
        }
    return results


def write_selection(path: Path, selections: dict[str, dict[str, dict[str, Any]]]) -> Path:
    """Persist rule → method selections as stable UTF-8 JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selections, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
