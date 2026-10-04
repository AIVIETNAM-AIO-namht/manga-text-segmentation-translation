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
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

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
    metrics: list[str] = field(
        default_factory=lambda: ["iou", "precision", "recall", "f1"]
    )
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
        raise ConfigError(
            f"Configuration key '{key}' must be an object, got {type(value).__name__}"
        )
    return value


def _resolve(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


SEGMENTATION_IDENTITIES = {
    "classical_baseline",
    "manga_text_segmentation",
    "comic_text_detector",
    "unetpp_efficientnetv2",
}
INPAINTING_ALGORITHMS = {"telea", "ns"}
TRANSLATION_LANGUAGES = {"vi", "en"}
TRANSLATION_FAILURE_POLICIES = {"keep_ocr_text", "exclude_from_rendering"}
TRANSLATION_READING_ORDERS = {"manga", "western"}
TRANSLATION_TEXT_DIRECTIONS = {"auto", "horizontal", "vertical"}


@dataclass(frozen=True)
class RegionExtractionConfig:
    min_area: int
    max_area: int
    merge_distance: int
    crop_padding: int
    reading_order: str
    polygon_epsilon: float


@dataclass(frozen=True)
class TranslationOcrConfig:
    engine: str
    preprocessing_enabled: bool
    grayscale: bool
    upscale_factor: float


@dataclass(frozen=True)
class TranslationRetryConfig:
    max_attempts: int
    initial_backoff_seconds: float
    max_backoff_seconds: float
    retry_status_codes: tuple[int, ...]


@dataclass(frozen=True)
class TranslationCacheConfig:
    enabled: bool
    bypass: bool
    root: Path


@dataclass(frozen=True)
class TranslationProviderConfig:
    name: str
    model: str
    version: str | None
    base_url: str
    secret_env_var: str | None
    timeout_seconds: float
    retry: TranslationRetryConfig
    failure_policy: str
    deterministic: bool
    cache: TranslationCacheConfig


@dataclass(frozen=True)
class RenderingConfig:
    font_family: str
    font_path: Path
    fallback_font_path: Path | None
    font_size: int
    min_font_size: int
    color: tuple[int, ...]
    stroke_width: int
    stroke_fill: tuple[int, ...]
    background: tuple[int, ...] | None
    wrap: bool
    line_spacing: float
    align_h: str
    align_v: str
    text_direction: str
    expansion_allowance: int


@dataclass(frozen=True)
class TranslationOutputRoots:
    translation: Path
    cache: Path


@dataclass(frozen=True)
class TranslationConfig:
    manifest_path: Path
    segmentation_method: str
    inpainting_run_id: str
    inpainting_algorithm: str
    target_language: str
    region_extraction: RegionExtractionConfig
    ocr: TranslationOcrConfig
    provider: TranslationProviderConfig
    rendering: RenderingConfig
    output_roots: TranslationOutputRoots


def _translation_object(
    raw: Any, where: str, required: set[str]
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ConfigError(f"'{where}' must be an object, got {type(raw).__name__}")
    missing = required - raw.keys()
    if missing:
        raise ConfigError(f"'{where}' is missing required key(s): {sorted(missing)}")
    sensitive = {
        key
        for key in raw
        if key != "secret_env_var"
        and any(
            marker in str(key).lower().replace("-", "_")
            for marker in ("api_key", "access_token", "token", "credential", "secret")
        )
    }
    if sensitive:
        raise ConfigError(f"'{where}' must not contain literal credential field(s): {sorted(sensitive)}")
    # Existing config loaders ignore unrelated keys. Preserve that convention while
    # explicitly refusing credential-bearing keys instead of silently discarding them.
    return raw


def _translation_mapping(
    container: dict[str, Any], key: str, required: set[str]
) -> dict[str, Any]:
    """Use the established mapping guard, then check required/credential fields."""
    return _translation_object(_require_mapping(container, key), key, required)


def _translation_int(raw: Any, where: str, minimum: int = 0) -> int:
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < minimum:
        raise ConfigError(f"'{where}' must be an integer >= {minimum}")
    return raw


def _translation_number(raw: Any, where: str, minimum: float = 0.0) -> float:
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise ConfigError(f"'{where}' must be a number >= {minimum}")
    try:
        number = float(raw)
    except (OverflowError, ValueError) as exc:
        raise ConfigError(f"'{where}' must be a finite number >= {minimum}") from exc
    if not math.isfinite(number) or number < minimum:
        raise ConfigError(f"'{where}' must be a finite number >= {minimum}")
    return number


def _translation_bool(raw: Any, where: str) -> bool:
    if not isinstance(raw, bool):
        raise ConfigError(f"'{where}' must be a boolean")
    return raw


def _translation_text(raw: Any, where: str, allow_null: bool = False) -> str | None:
    if raw is None and allow_null:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError(f"'{where}' must be a non-empty string")
    return raw


def _translation_color(raw: Any, where: str, allow_null: bool = False) -> tuple[int, ...] | None:
    if raw is None and allow_null:
        return None
    if not isinstance(raw, list) or len(raw) not in (3, 4):
        raise ConfigError(f"'{where}' must be an RGB or RGBA array")
    values = tuple(_translation_int(value, where, 0) for value in raw)
    if any(value > 255 for value in values):
        raise ConfigError(f"'{where}' values must be <= 255")
    return values


def load_translation_config(
    path: str | Path, repo_root: Path | None = None
) -> TranslationConfig:
    """Load the Spec 004 translation settings with strict nested validation."""
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Could not load translation config '{config_path}': {exc}") from exc

    top_keys = {
        "manifest_path", "segmentation_method", "inpainting_source", "target_language",
        "region_extraction", "ocr", "provider", "rendering", "output_roots",
    }
    data = _translation_object(data, "translation config", top_keys)
    root = (repo_root or config_path.resolve().parent.parent).resolve()

    manifest_path_raw = _translation_text(data["manifest_path"], "manifest_path")
    method = _translation_text(data["segmentation_method"], "segmentation_method")
    if method not in SEGMENTATION_IDENTITIES:
        raise ConfigError(f"'segmentation_method' must be one of {sorted(SEGMENTATION_IDENTITIES)}")
    language = _translation_text(data["target_language"], "target_language")
    if language not in TRANSLATION_LANGUAGES:
        raise ConfigError(f"'target_language' must be one of {sorted(TRANSLATION_LANGUAGES)}")

    source = _translation_mapping(data, "inpainting_source", {"run_id", "algorithm"})
    source_run = _translation_text(source["run_id"], "inpainting_source.run_id")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", source_run):
        raise ConfigError("'inpainting_source.run_id' must be path-safe")
    algorithm = _translation_text(source["algorithm"], "inpainting_source.algorithm")
    if algorithm not in INPAINTING_ALGORITHMS:
        raise ConfigError(f"'inpainting_source.algorithm' must be one of {sorted(INPAINTING_ALGORITHMS)}")

    region_keys = {"min_area", "max_area", "merge_distance", "crop_padding", "reading_order", "polygon_epsilon"}
    region = _translation_mapping(data, "region_extraction", region_keys)
    min_area = _translation_int(region["min_area"], "region_extraction.min_area", 1)
    max_area = _translation_int(region["max_area"], "region_extraction.max_area", 1)
    if max_area < min_area:
        raise ConfigError("'region_extraction.max_area' must be >= min_area")
    reading_order = _translation_text(region["reading_order"], "region_extraction.reading_order")
    if reading_order not in TRANSLATION_READING_ORDERS:
        raise ConfigError(f"'region_extraction.reading_order' must be one of {sorted(TRANSLATION_READING_ORDERS)}")
    region_config = RegionExtractionConfig(
        min_area=min_area,
        max_area=max_area,
        merge_distance=_translation_int(region["merge_distance"], "region_extraction.merge_distance"),
        crop_padding=_translation_int(region["crop_padding"], "region_extraction.crop_padding"),
        reading_order=reading_order,
        polygon_epsilon=_translation_number(region["polygon_epsilon"], "region_extraction.polygon_epsilon", 0.000001),
    )

    ocr_raw = _translation_mapping(data, "ocr", {"engine", "preprocessing"})
    ocr_engine = _translation_text(ocr_raw["engine"], "ocr.engine")
    prep = _translation_mapping(ocr_raw, "preprocessing", {"enabled", "grayscale", "upscale_factor"})
    ocr_config = TranslationOcrConfig(
        engine=ocr_engine,
        preprocessing_enabled=_translation_bool(prep["enabled"], "ocr.preprocessing.enabled"),
        grayscale=_translation_bool(prep["grayscale"], "ocr.preprocessing.grayscale"),
        upscale_factor=_translation_number(prep["upscale_factor"], "ocr.preprocessing.upscale_factor", 1.0),
    )

    provider_keys = {"name", "model", "version", "base_url", "secret_env_var", "timeout_seconds", "retry", "failure_policy", "deterministic", "cache"}
    provider = _translation_mapping(data, "provider", provider_keys)
    provider_name = _translation_text(provider["name"], "provider.name")
    model = _translation_text(provider["model"], "provider.model")
    version = _translation_text(provider["version"], "provider.version", True)
    base_url = _translation_text(provider["base_url"], "provider.base_url")
    try:
        parsed_base_url = urlsplit(base_url)
        parsed_base_url.port
    except ValueError as exc:
        raise ConfigError("'provider.base_url' is malformed") from exc
    if (
        parsed_base_url.scheme not in {"http", "https"}
        or not parsed_base_url.hostname
        or parsed_base_url.username is not None
        or parsed_base_url.password is not None
        or parsed_base_url.query
        or parsed_base_url.fragment
    ):
        raise ConfigError(
            "'provider.base_url' must be an HTTP(S) base URL without credentials or query data"
        )
    secret_env = _translation_text(provider["secret_env_var"], "provider.secret_env_var", True)
    if secret_env is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", secret_env):
        raise ConfigError("'provider.secret_env_var' must be an environment variable name, not a secret value")
    timeout = _translation_number(provider["timeout_seconds"], "provider.timeout_seconds", 0.000001)
    retry = _translation_mapping(
        provider,
        "retry",
        {"max_attempts", "initial_backoff_seconds", "max_backoff_seconds", "retry_status_codes"},
    )
    attempts = _translation_int(retry["max_attempts"], "provider.retry.max_attempts", 1)
    initial_backoff = _translation_number(retry["initial_backoff_seconds"], "provider.retry.initial_backoff_seconds")
    max_backoff = _translation_number(retry["max_backoff_seconds"], "provider.retry.max_backoff_seconds")
    if max_backoff < initial_backoff:
        raise ConfigError("'provider.retry.max_backoff_seconds' must be >= initial_backoff_seconds")
    statuses_raw = retry["retry_status_codes"]
    if not isinstance(statuses_raw, list) or not statuses_raw:
        raise ConfigError("'provider.retry.retry_status_codes' must be a non-empty list")
    statuses = tuple(_translation_int(code, "provider.retry.retry_status_codes", 100) for code in statuses_raw)
    if any(code > 599 for code in statuses) or len(set(statuses)) != len(statuses):
        raise ConfigError("'provider.retry.retry_status_codes' must be unique HTTP status codes from 100 to 599")
    failure_policy = _translation_text(provider["failure_policy"], "provider.failure_policy")
    if failure_policy not in TRANSLATION_FAILURE_POLICIES:
        raise ConfigError(f"'provider.failure_policy' must be one of {sorted(TRANSLATION_FAILURE_POLICIES)}")
    deterministic = _translation_bool(provider["deterministic"], "provider.deterministic")
    cache = _translation_mapping(provider, "cache", {"enabled", "bypass", "root"})
    cache_config = TranslationCacheConfig(
        enabled=_translation_bool(cache["enabled"], "provider.cache.enabled"),
        bypass=_translation_bool(cache["bypass"], "provider.cache.bypass"),
        root=_resolve(Path(_translation_text(cache["root"], "provider.cache.root")), root),
    )
    provider_config = TranslationProviderConfig(
        name=provider_name, model=model, version=version, base_url=base_url,
        secret_env_var=secret_env,
        timeout_seconds=timeout,
        retry=TranslationRetryConfig(attempts, initial_backoff, max_backoff, statuses),
        failure_policy=failure_policy, deterministic=deterministic, cache=cache_config,
    )

    render_keys = {"font_family", "font_path", "fallback_font_path", "font_size", "min_font_size", "color", "stroke_width", "stroke_fill", "background", "wrap", "line_spacing", "align_h", "align_v", "text_direction", "expansion_allowance"}
    rendering = _translation_mapping(data, "rendering", render_keys)
    font_size = _translation_int(rendering["font_size"], "rendering.font_size", 1)
    min_font_size = _translation_int(rendering["min_font_size"], "rendering.min_font_size", 1)
    if min_font_size > font_size:
        raise ConfigError("'rendering.min_font_size' must be <= font_size")
    align_h = _translation_text(rendering["align_h"], "rendering.align_h")
    align_v = _translation_text(rendering["align_v"], "rendering.align_v")
    if align_h not in {"left", "center", "right"}:
        raise ConfigError("'rendering.align_h' must be left, center, or right")
    if align_v not in {"top", "middle", "bottom"}:
        raise ConfigError("'rendering.align_v' must be top, middle, or bottom")
    direction = _translation_text(rendering["text_direction"], "rendering.text_direction")
    if direction not in TRANSLATION_TEXT_DIRECTIONS:
        raise ConfigError(f"'rendering.text_direction' must be one of {sorted(TRANSLATION_TEXT_DIRECTIONS)}")
    fallback_raw = _translation_text(rendering["fallback_font_path"], "rendering.fallback_font_path", True)
    rendering_config = RenderingConfig(
        font_family=_translation_text(rendering["font_family"], "rendering.font_family"),
        font_path=_resolve(Path(_translation_text(rendering["font_path"], "rendering.font_path")), root),
        fallback_font_path=_resolve(Path(fallback_raw), root) if fallback_raw else None,
        font_size=font_size, min_font_size=min_font_size,
        color=_translation_color(rendering["color"], "rendering.color"),
        stroke_width=_translation_int(rendering["stroke_width"], "rendering.stroke_width"),
        stroke_fill=_translation_color(rendering["stroke_fill"], "rendering.stroke_fill"),
        background=_translation_color(rendering["background"], "rendering.background", True),
        wrap=_translation_bool(rendering["wrap"], "rendering.wrap"),
        line_spacing=_translation_number(rendering["line_spacing"], "rendering.line_spacing", 0.000001),
        align_h=align_h, align_v=align_v, text_direction=direction,
        expansion_allowance=_translation_int(rendering["expansion_allowance"], "rendering.expansion_allowance"),
    )

    output_raw = _translation_mapping(data, "output_roots", {"translation", "cache"})
    output_roots = TranslationOutputRoots(
        translation=_resolve(Path(_translation_text(output_raw["translation"], "output_roots.translation")), root),
        cache=_resolve(Path(_translation_text(output_raw["cache"], "output_roots.cache")), root),
    )
    if cache_config.root != output_roots.cache:
        raise ConfigError("'provider.cache.root' and 'output_roots.cache' must resolve to the same path")

    return TranslationConfig(
        manifest_path=_resolve(Path(manifest_path_raw), root), segmentation_method=method,
        inpainting_run_id=source_run, inpainting_algorithm=algorithm,
        target_language=language, region_extraction=region_config, ocr=ocr_config,
        provider=provider_config, rendering=rendering_config, output_roots=output_roots,
    )


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
        raise ConfigError(
            f"Unsupported alignment strategy '{strategy}'; "
            f"expected one of {sorted(ALIGNMENT_STRATEGIES)}"
        )

    preprocessing = (
        _require_mapping(data, "preprocessing") if "preprocessing" in data else {}
    )
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
        raise ConfigError(
            f"{where}/{identity}: 'size_bytes' must be a positive integer"
        )

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
        name: _parse_dl_method(name, raw, root) for name, raw in methods_raw.items()
    }
    return DLConfig(methods=methods)


# --------------------------------------------------------------------------- #
# Inpainting configuration (configs/inpainting.json) — Spec 003 (FR-038)
# --------------------------------------------------------------------------- #

#: The four FR-026 identities. Fixed names — never derived, never guessed.
INPAINT_METHOD_IDENTITIES = {
    "classical_baseline",
    "manga_text_segmentation",
    "comic_text_detector",
    "unetpp_efficientnetv2",
}
DILATION_KERNEL_SHAPES = {"rect", "ellipse", "cross"}
INPAINT_ALGORITHMS = {"telea", "ns"}
INPAINT_OUTPUT_FORMATS = {"png", "webp", "bmp", "tiff"}


@dataclass(frozen=True)
class DilationConfig:
    """One dilation setting. Required fields are only meaningful when enabled."""

    enabled: bool
    kernel_shape: str | None = None
    kernel_size: tuple[int, int] | None = None
    iterations: int | None = None


@dataclass(frozen=True)
class MaskProcessingConfig:
    """The shared preprocessing configuration (FR-014-FR-018), identical for
    every page and every method in the main benchmark (FR-017)."""

    dilation: DilationConfig


@dataclass(frozen=True)
class InpaintingConfig:
    """FR-019-FR-021. No per-method override exists — the main benchmark uses
    one radius, one algorithm set and one output format for all four methods."""

    radius: float
    algorithms: tuple[str, ...]
    output_format: str = "png"


@dataclass(frozen=True)
class InpaintMethodConfig:
    """One FR-026 identity's resolved mask source (research R1, R5)."""

    source_root: Path


@dataclass(frozen=True)
class InpaintConfig:
    """Validated runtime configuration for Spec 003."""

    methods: dict[str, InpaintMethodConfig]
    mask_processing: MaskProcessingConfig
    inpainting: InpaintingConfig
    selection_n: int
    output_root: Path
    manual_artifact_flags: list[str] = field(default_factory=list)


def _parse_inpaint_method(name: str, raw: Any, repo_root: Path) -> InpaintMethodConfig:
    where = f"inpaint method '{name}'"
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: definition must be an object")
    if "source_root" not in raw:
        raise ConfigError(f"{where}: missing required key 'source_root'")
    return InpaintMethodConfig(
        source_root=_resolve(Path(str(raw["source_root"])), repo_root)
    )


def _parse_mask_processing(raw: dict[str, Any]) -> MaskProcessingConfig:
    dilation_raw = _require_mapping(raw, "dilation")
    if "enabled" not in dilation_raw:
        raise ConfigError("mask_processing.dilation: missing required key 'enabled'")
    enabled = bool(dilation_raw["enabled"])

    kernel_shape: str | None = None
    kernel_size: tuple[int, int] | None = None
    iterations: int | None = None

    if enabled:
        kernel_shape = dilation_raw.get("kernel_shape")
        if kernel_shape not in DILATION_KERNEL_SHAPES:
            raise ConfigError(
                f"mask_processing.dilation: 'kernel_shape' must be one of "
                f"{sorted(DILATION_KERNEL_SHAPES)} when enabled, got {kernel_shape!r}"
            )
        raw_size = dilation_raw.get("kernel_size")
        if not (isinstance(raw_size, list) and len(raw_size) == 2):
            raise ConfigError(
                "mask_processing.dilation: 'kernel_size' must be a [width, height] "
                "pair when enabled"
            )
        kernel_size = (int(raw_size[0]), int(raw_size[1]))
        iterations = dilation_raw.get("iterations")
        if (
            not isinstance(iterations, int)
            or isinstance(iterations, bool)
            or iterations < 1
        ):
            raise ConfigError(
                "mask_processing.dilation: 'iterations' must be a positive integer "
                "when enabled"
            )

    return MaskProcessingConfig(
        dilation=DilationConfig(
            enabled=enabled,
            kernel_shape=kernel_shape,
            kernel_size=kernel_size,
            iterations=iterations,
        )
    )


def _parse_inpainting_config(raw: dict[str, Any]) -> InpaintingConfig:
    if "radius" not in raw:
        raise ConfigError("'inpaint': missing required key 'radius'")
    radius = raw["radius"]
    if not isinstance(radius, (int, float)) or isinstance(radius, bool) or radius <= 0:
        raise ConfigError("'inpaint.radius' must be a positive number")

    algorithms = raw.get("algorithms")
    if not isinstance(algorithms, list) or not algorithms:
        raise ConfigError("'inpaint.algorithms' must be a non-empty list")
    for algo in algorithms:
        if algo not in INPAINT_ALGORITHMS:
            raise ConfigError(
                f"'inpaint.algorithms' contains {algo!r}; expected one of "
                f"{sorted(INPAINT_ALGORITHMS)}"
            )

    output_format = raw.get("output_format", "png")
    if output_format not in INPAINT_OUTPUT_FORMATS:
        raise ConfigError(
            f"'inpaint.output_format' must be one of {sorted(INPAINT_OUTPUT_FORMATS)}, "
            f"got {output_format!r}"
        )

    return InpaintingConfig(
        radius=float(radius),
        algorithms=tuple(algorithms),
        output_format=str(output_format),
    )


def load_inpaint_config(
    path: str | Path, repo_root: Path | None = None
) -> InpaintConfig:
    """Load and validate ``configs/inpainting.json`` (FR-038).

    Every path and parameter for Spec 003 lives here; nothing may be
    hard-coded in the processing logic that consumes this config.
    """
    config_path = Path(path)
    with open(config_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ConfigError("Configuration root must be a JSON object")

    required_top = {"methods", "mask_processing", "inpaint", "selection", "output_root"}
    missing = required_top - set(data)
    if missing:
        raise ConfigError(f"Missing required configuration key(s): {sorted(missing)}")

    root = (repo_root or config_path.resolve().parent.parent).resolve()

    methods_raw = _require_mapping(data, "methods")
    missing_identities = INPAINT_METHOD_IDENTITIES - set(methods_raw)
    if missing_identities:
        raise ConfigError(
            f"'methods' is missing required identity(ies): {sorted(missing_identities)}"
        )
    methods = {
        name: _parse_inpaint_method(name, methods_raw[name], root)
        for name in INPAINT_METHOD_IDENTITIES
    }

    mask_processing = _parse_mask_processing(_require_mapping(data, "mask_processing"))
    inpainting = _parse_inpainting_config(_require_mapping(data, "inpaint"))

    selection_raw = _require_mapping(data, "selection")
    if "n" not in selection_raw:
        raise ConfigError("'selection' must define 'n'")
    selection_n = int(selection_raw["n"])
    manual_artifact_flags = list(selection_raw.get("manual_artifact_flags", []))

    output_root = _resolve(Path(str(data["output_root"])), root)

    return InpaintConfig(
        methods=methods,
        mask_processing=mask_processing,
        inpainting=inpainting,
        selection_n=selection_n,
        output_root=output_root,
        manual_artifact_flags=manual_artifact_flags,
    )
