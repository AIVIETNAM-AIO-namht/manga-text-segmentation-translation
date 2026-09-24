"""Configuration loader and schema validation.

Two configuration documents, two loaders:

* :func:`load_config` reads the classical benchmark configuration
  (``configs/default.json``) — dataset paths, alignment, methods, metrics and
  visualization settings (FR-036). No logic-level hardcoding.
* :func:`load_dl_config` reads the deep-learning method configuration
  (``configs/dl.json``) — the three external methods' repositories, checkpoints
  and licences. It is a separate document with a separate loader because it
  answers a different question and is consumed by a different part of the
  pipeline; folding it into ``Config`` would give every classical caller fields
  it has no use for (research.md R11).

Relative paths are resolved against the repository root (the config file's
parent parent).
"""
from __future__ import annotations

import json
import re
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


# --------------------------------------------------------------------------- #
# Deep-learning method configuration (configs/dl.json)
# --------------------------------------------------------------------------- #

REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
DEVICE_TYPES = {"cpu", "cuda"}
PADDING_MODES = {"pad-to-multiple", "letterbox"}

DL_METHOD_KEYS = ("repository", "checkpoints", "code_license", "weight_license")


@dataclass(frozen=True)
class Checkpoint:
    """One weight file: identity, where it is configured, and how to verify it."""

    identity: str
    path: Path
    size_bytes: int
    #: ``None`` where no digest is published. A null here is the recorded
    #: absence of a published value, not a gap — the runner hashes the file it
    #: obtains and records the observed value in provenance (FR-044).
    sha256: str | None
    fold: int | None = None


@dataclass(frozen=True)
class DLMethod:
    """One external deep-learning method, as configured."""

    name: str
    repository: dict[str, Any]
    checkpoints: list[Checkpoint]
    code_license: str
    weight_license: str
    device: dict[str, Any]
    threshold: float
    padding: dict[str, Any]
    channel_order: str
    notes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DLConfig:
    """Validated deep-learning method configuration."""

    methods: dict[str, DLMethod]


def _require_non_empty_string(mapping: dict[str, Any], key: str, where: str) -> str:
    if key not in mapping:
        raise ConfigError(f"{where}: missing required key '{key}'")
    value = mapping[key]
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where}: '{key}' must be a non-empty string")
    return value


def _parse_checkpoint(raw: Any, where: str, repo_root: Path) -> Checkpoint:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: each checkpoint must be an object")
    for key in ("identity", "path", "size_bytes", "sha256"):
        if key not in raw:
            # Present-but-null is allowed for sha256 only; absent never is.
            raise ConfigError(f"{where}: checkpoint is missing required key '{key}'")
    identity = _require_non_empty_string(raw, "identity", where)

    size = raw["size_bytes"]
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ConfigError(f"{where}/{identity}: 'size_bytes' must be a positive integer")

    sha = raw["sha256"]
    if sha is not None and not (isinstance(sha, str) and len(sha) == 64):
        raise ConfigError(
            f"{where}/{identity}: 'sha256' must be a 64-character hex digest or null"
        )

    fold = raw.get("fold")
    if fold is not None and not isinstance(fold, int):
        raise ConfigError(f"{where}/{identity}: 'fold' must be an integer when present")

    return Checkpoint(
        identity=identity,
        path=_resolve(Path(str(raw["path"])), repo_root),
        size_bytes=size,
        sha256=sha,
        fold=fold,
    )


def _parse_dl_method(name: str, raw: Any, repo_root: Path) -> DLMethod:
    where = f"DL method '{name}'"
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: definition must be an object")

    for key in DL_METHOD_KEYS:
        if key not in raw:
            raise ConfigError(f"{where}: missing required key '{key}'")

    repository = _require_mapping(raw, "repository")
    _require_non_empty_string(repository, "url", where)
    revision = _require_non_empty_string(repository, "revision", where)
    if not REVISION_PATTERN.match(revision):
        raise ConfigError(
            f"{where}: 'revision' must be a full 40-character commit SHA; "
            f"a branch name, tag or short SHA is not a revision (got {revision!r})"
        )

    checkpoints_raw = raw["checkpoints"]
    if not isinstance(checkpoints_raw, list) or not checkpoints_raw:
        raise ConfigError(f"{where}: 'checkpoints' must be a non-empty list")
    checkpoints = [_parse_checkpoint(c, where, repo_root) for c in checkpoints_raw]

    device = _require_mapping(raw, "device")
    device_type = device.get("type")
    if device_type not in DEVICE_TYPES:
        raise ConfigError(
            f"{where}: 'device.type' must be one of {sorted(DEVICE_TYPES)}, got {device_type!r}"
        )
    # FR-060: 'cuda' alone is not enough; the specific accelerator is required.
    _require_non_empty_string(device, "name", where)

    padding = _require_mapping(raw, "padding") if "padding" in raw else {}
    if padding:
        mode = padding.get("mode")
        if mode not in PADDING_MODES:
            raise ConfigError(
                f"{where}: 'padding.mode' must be one of {sorted(PADDING_MODES)}, got {mode!r}"
            )
        multiple = padding.get("multiple")
        if not isinstance(multiple, int) or isinstance(multiple, bool) or multiple < 1:
            raise ConfigError(f"{where}: 'padding.multiple' must be a positive integer")

    notes = {
        key: str(raw[key])
        for key in ("sha256_note", "channel_order_note")
        if isinstance(raw.get(key), str)
    }

    return DLMethod(
        name=name,
        repository=dict(repository),
        checkpoints=checkpoints,
        code_license=_require_non_empty_string(raw, "code_license", where),
        weight_license=_require_non_empty_string(raw, "weight_license", where),
        device=dict(device),
        threshold=float(raw.get("threshold", 0.5)),
        padding=dict(padding),
        channel_order=str(raw.get("channel_order", "rgb")),
        notes=notes,
    )


def load_dl_config(path: str | Path, repo_root: Path | None = None) -> DLConfig:
    """Load and validate the deep-learning method configuration (FR-036, FR-044).

    Every method's revision, checkpoint identity, and both licences are required
    here rather than discovered at run time: FR-056 refuses a result whose
    provenance has gaps, and a gap the configuration already knew about is not
    something to discover after 390 pages have been inferred.
    """
    config_path = Path(path)
    with open(config_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ConfigError("Configuration root must be a JSON object")

    methods_raw = data.get("methods")
    if not isinstance(methods_raw, dict) or not methods_raw:
        raise ConfigError("'methods' must be a non-empty object keyed by method name")

    root = (repo_root or config_path.resolve().parent.parent).resolve()
    methods = {
        name: _parse_dl_method(name, raw, root)
        for name, raw in methods_raw.items()
    }
    return DLConfig(methods=methods)