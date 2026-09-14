"""Configuration loader and schema validation.

Single configuration file (FR-036): all dataset paths, alignment, methods,
metrics and visualization settings come from one JSON document. No logic-level
hardcoding. Relative paths are resolved against the repository root
(the config file's parent parent).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REQUIRED_KEYS = {"dataset", "output_root", "run_id"}
ALIGNMENT_STRATEGIES = {"crop-topleft", "resize"}
VISUALIZATION_MODES = {"first-N", "best-N", "worst-N"}
DEFAULT_METHODS = [
    {"name": "otsu", "params": {"blur_ksize": 5}},
    {"name": "adaptive", "params": {"block_size": 35, "c": 5}},
    {"name": "mser", "params": {}},
    {"name": "edges", "params": {"low": 50, "high": 150}},
    {"name": "components", "params": {"min_area": 50, "max_aspect_ratio": 20.0}},
    {"name": "morphology", "params": {"kernel": 3, "open_iter": 1, "close_iter": 1}},
]


class ConfigError(ValueError):
    """Raised when the configuration file is missing keys or has invalid values."""


@dataclass(frozen=True)
class Config:
    """Validated runtime configuration."""

    raw_root: Path
    gt_root: Path
    output_root: Path
    run_id: str
    alignment_strategy: str = "crop-topleft"
    denoise: str = "none"
    methods: list[dict[str, Any]] = field(default_factory=lambda: list(DEFAULT_METHODS))
    metrics: list[str] = field(default_factory=lambda: ["iou", "precision", "recall", "f1"])
    viz_mode: str = "first-N"
    viz_n: int = 20

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.run_id


def _require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    """Validate that ``key`` exists and is a mapping. Fail fast with a clear error."""
    if key not in data:
        raise ConfigError(f"Missing required configuration key: '{key}'")
    value = data[key]
    if not isinstance(value, dict):
        raise ConfigError(f"Configuration key '{key}' must be an object, got {type(value).__name__}")
    return value


def _resolve(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def load_config(path: str | Path, repo_root: Path | None = None) -> Config:
    """Load and validate a JSON configuration file (FR-036)."""
    config_path = Path(path)
    with open(config_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ConfigError("Configuration root must be a JSON object")

    missing = REQUIRED_KEYS - set(data)
    if missing:
        raise ConfigError(f"Missing required configuration key(s): {sorted(missing)}")

    root = (repo_root or config_path.resolve().parent.parent).resolve()

    dataset = _require_mapping(data, "dataset")
    if "raw_root" not in dataset or "gt_root" not in dataset:
        raise ConfigError("'dataset' must define both 'raw_root' and 'gt_root'")

    alignment = _require_mapping(data, "alignment") if "alignment" in data else {}
    strategy = alignment.get("strategy", "crop-topleft")
    if strategy not in ALIGNMENT_STRATEGIES:
        raise ConfigError(f"Unsupported alignment strategy '{strategy}'; "
                          f"expected one of {sorted(ALIGNMENT_STRATEGIES)}")

    preprocessing = _require_mapping(data, "preprocessing") if "preprocessing" in data else {}
    denoise = preprocessing.get("denoise", "none")

    methods = data.get("methods", DEFAULT_METHODS)
    if not isinstance(methods, list) or any(not isinstance(m, dict) for m in methods):
        raise ConfigError("'methods' must be a list of {name, params} objects")

    metrics = data.get("metrics", ["iou", "precision", "recall", "f1"])
    if not isinstance(metrics, list) or not metrics:
        raise ConfigError("'metrics' must be a non-empty list")

    viz = _require_mapping(data, "visualization") if "visualization" in data else {}
    viz_mode = viz.get("mode", "first-N")
    if viz_mode not in VISUALIZATION_MODES:
        raise ConfigError(f"Unsupported visualization mode '{viz_mode}'")
    viz_n = int(viz.get("n", 20))
    if viz_n < 1:
        raise ConfigError("'visualization.n' must be >= 1")

    return Config(
        raw_root=_resolve(Path(dataset["raw_root"]), root),
        gt_root=_resolve(Path(dataset["gt_root"]), root),
        output_root=_resolve(Path(data["output_root"]), root),
        run_id=str(data["run_id"]),
        alignment_strategy=strategy,
        denoise=str(denoise),
        methods=list(methods),
        metrics=list(metrics),
        viz_mode=viz_mode,
        viz_n=viz_n,
    )