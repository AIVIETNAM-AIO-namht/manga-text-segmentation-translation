"""Per-image metrics: IoU, Precision, Recall, F1 (FR-025, FR-027, Research R3).

No Pixel Accuracy — the metric set is frozen.  Empty-mask edge cases
(Research R3):
  - GT and prediction both empty            -> all metrics 1.0
  - GT empty, prediction non-empty          -> P=0, R=1, F1=0, IoU=0
  - GT non-empty, prediction empty          -> P=1, R=0, F1=0, IoU=0

All inputs are binary uint8 masks with values in {0, 255}.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

METRIC_KEYS = ("iou", "precision", "recall", "f1")


class MetricError(ValueError):
    """Raised when metric inputs violate the binary-mask invariant."""


def _as_binary(mask: np.ndarray, label: str) -> np.ndarray:
    if mask.dtype != np.uint8:
        raise MetricError(f"{label} must be uint8, got {mask.dtype}")
    if mask.ndim != 2:
        raise MetricError(f"{label} must be 2D, got shape {mask.shape}")
    values = set(np.unique(mask))
    if not values.issubset({0, 255}):
        raise MetricError(f"{label} must contain only {{0, 255}}, got {sorted(values)}")
    return mask


def compute_metrics(
    gt: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    """Compute IoU / Precision / Recall / F1 between ground-truth and prediction.

    Shapes must match; the caller aligns beforehand (crop-topleft).
    """
    gt = _as_binary(gt, "gt")
    prediction = _as_binary(prediction, "prediction")
    if gt.shape != prediction.shape:
        raise MetricError(
            f"Shape mismatch: gt {gt.shape} vs prediction {prediction.shape}; align first"
        )

    gt_bool = gt == 255
    pred_bool = prediction == 255
    gt_count = int(gt_bool.sum())
    pred_count = int(pred_bool.sum())

    # Both empty -> perfect agreement by convention (Research R3).
    if gt_count == 0 and pred_count == 0:
        return {"iou": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0,
                "tp": 0, "fp": 0, "fn": 0}

    tp = int((gt_bool & pred_bool).sum())
    fp = pred_count - tp
    fn = gt_count - tp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    iou = tp / (gt_count + pred_count - tp) if (gt_count + pred_count - tp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }