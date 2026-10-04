"""Run addressing, error reporting, and run-level metadata for Spec 004."""

from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..config import (
    INPAINTING_ALGORITHMS,
    SEGMENTATION_IDENTITIES,
    TRANSLATION_LANGUAGES,
)

ERROR_CATEGORIES = (
    "missing_original_image",
    "missing_prediction_mask",
    "missing_inpainted_image",
    "empty_mask",
    "no_text_region_detected",
    "ocr_failure",
    "translation_failure",
    "font_missing",
    "text_overflow",
    "api_timeout_or_rate_limit",
    "output_write_failure",
)
PAGE_STATUSES = {
    "accepted",
    "success",
    "failed",
    "no_regions",
    "empty_mask",
    "no_text_region_detected",
    *ERROR_CATEGORIES,
}


def _validate_identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ValueError(f"{label} must be a non-empty path-safe string")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"{label} contains unsupported path characters: {value!r}")
    return value


def _validate_image_id(image_id: str) -> tuple[str, str]:
    if not isinstance(image_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*/[0-9]{3}", image_id
    ):
        raise ValueError("image_id must be a manifest identity in <manga>/<NNN> form")
    return tuple(image_id.split("/", 1))  # type: ignore[return-value]


def run_root(output_root: str | Path, run_id: str) -> Path:
    return Path(output_root) / _validate_identifier(run_id, "run_id")


def page_directory(
    output_root: str | Path,
    run_id: str,
    segmentation_method: str,
    image_id: str,
) -> Path:
    if segmentation_method not in SEGMENTATION_IDENTITIES:
        raise ValueError(f"unsupported segmentation method: {segmentation_method!r}")
    manga, page = _validate_image_id(image_id)
    return run_root(output_root, run_id) / segmentation_method / manga / page


@dataclass
class ErrorReport:
    """Contract-shaped errors.json with every category represented, including zeroes."""

    run_id: str
    categories: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: {category: [] for category in ERROR_CATEGORIES}
    )

    def __post_init__(self) -> None:
        _validate_identifier(self.run_id, "run_id")
        if set(self.categories) != set(ERROR_CATEGORIES):
            raise ValueError("categories must contain exactly the eleven Spec 004 keys")
        for category, entries in self.categories.items():
            if not isinstance(entries, list):
                raise ValueError(f"categories[{category!r}] must be a list")
            for entry in entries:
                self._validate_entry(category, entry)

    @staticmethod
    def _validate_entry(category: str, entry: Any) -> None:
        if not isinstance(entry, dict):
            raise ValueError("error entries must be objects")
        if set(entry) != {"image_id", "region_id", "stage", "reason"}:
            raise ValueError(
                "error entries must have image_id, region_id, stage, and reason only"
            )
        _validate_image_id(entry["image_id"])
        region_id = entry["region_id"]
        if region_id is not None and (
            not isinstance(region_id, int)
            or isinstance(region_id, bool)
            or region_id < 1
        ):
            raise ValueError("region_id must be a positive integer or null")
        stages = {"intake", "regions", "ocr", "translate", "render", "write"}
        if entry["stage"] not in stages:
            raise ValueError(f"stage must be one of {sorted(stages)}")
        if not isinstance(entry["reason"], str) or not entry["reason"].strip():
            raise ValueError("reason must be a specific non-empty string")

    @property
    def counts(self) -> dict[str, int]:
        return {
            category: len(self.categories[category]) for category in ERROR_CATEGORIES
        }

    def add(
        self,
        category: str,
        *,
        image_id: str,
        region_id: int | None,
        stage: str,
        reason: str,
    ) -> None:
        if category not in ERROR_CATEGORIES:
            raise ValueError(f"unknown error category: {category!r}")
        entry = {
            "image_id": image_id,
            "region_id": region_id,
            "stage": stage,
            "reason": reason,
        }
        self._validate_entry(category, entry)
        self.categories[category].append(entry)

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            "run_id": self.run_id,
            "counts": self.counts,
            "categories": self.categories,
        }

    def write(self, run_dir: str | Path) -> Path:
        directory = Path(run_dir)
        if directory.name != self.run_id:
            raise ValueError("run directory name must match ErrorReport.run_id")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "errors.json"
        if path.exists():
            raise FileExistsError(f"error report already exists: {path}")
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path


@dataclass(frozen=True)
class TranslationRun:
    """Validated run record persisted at run.json (FR-046, FR-056, FR-059)."""

    run_id: str
    segmentation_method: str
    inpainting_source: dict[str, Any]
    target_language: str
    ocr_engine: dict[str, Any]
    provider: dict[str, Any]
    rendering: dict[str, Any]
    resolved_config: dict[str, Any]
    input_image_ids: tuple[str, ...]
    page_statuses: dict[str, str]
    page_list_identity: str
    input_image_identity: dict[str, str]
    region_extraction_config: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.run_id, "run_id")
        if self.segmentation_method not in SEGMENTATION_IDENTITIES:
            raise ValueError(
                f"unsupported segmentation method: {self.segmentation_method!r}"
            )
        if self.target_language not in TRANSLATION_LANGUAGES:
            raise ValueError(f"unsupported target language: {self.target_language!r}")
        if not isinstance(self.inpainting_source, dict) or set(
            self.inpainting_source
        ) != {"run_id", "algorithm"}:
            raise ValueError("inpainting_source must contain run_id and algorithm")
        _validate_identifier(
            self.inpainting_source["run_id"], "inpainting_source.run_id"
        )
        if self.inpainting_source["algorithm"] not in INPAINTING_ALGORITHMS:
            raise ValueError("inpainting_source.algorithm must be telea or ns")
        self._identity_string(self.page_list_identity, "page_list_identity")
        if not isinstance(self.input_image_ids, tuple) or len(
            set(self.input_image_ids)
        ) != len(self.input_image_ids):
            raise ValueError(
                "input_image_ids must be a tuple of unique manifest image IDs"
            )
        for image_id in self.input_image_ids:
            _validate_image_id(image_id)
        if not isinstance(self.input_image_identity, dict) or set(
            self.input_image_identity
        ) != set(self.input_image_ids):
            raise ValueError(
                "input_image_identity must map every input image_id exactly once"
            )
        for identity in self.input_image_identity.values():
            self._identity_string(identity, "input_image_identity value")
        if not isinstance(self.page_statuses, dict) or set(self.page_statuses) != set(
            self.input_image_ids
        ):
            raise ValueError("page_statuses must map every input image_id exactly once")
        for image_id, status in self.page_statuses.items():
            _validate_image_id(image_id)
            if status not in PAGE_STATUSES:
                raise ValueError(f"unsupported page status: {status!r}")
        self._require_object(self.ocr_engine, "ocr_engine")
        if (
            not isinstance(self.ocr_engine.get("name"), str)
            or not self.ocr_engine["name"].strip()
        ):
            raise ValueError("ocr_engine.name must be non-empty")
        if "version" not in self.ocr_engine or (
            self.ocr_engine["version"] is not None
            and not isinstance(self.ocr_engine["version"], str)
        ):
            raise ValueError("ocr_engine.version must be a string or null")
        if not isinstance(self.ocr_engine.get("deterministic", True), bool):
            raise ValueError("ocr_engine.deterministic must be a boolean when present")
        if set(self.ocr_engine) - {"name", "version", "deterministic"}:
            raise ValueError("ocr_engine has unknown fields")
        self._require_object(self.provider, "provider")
        for key in ("name", "model"):
            if (
                not isinstance(self.provider.get(key), str)
                or not self.provider[key].strip()
            ):
                raise ValueError(f"provider.{key} must be non-empty")
        if "version" not in self.provider or not isinstance(
            self.provider.get("deterministic"), bool
        ):
            raise ValueError("provider.version and provider.deterministic are required")
        if self.provider["version"] is not None and not isinstance(
            self.provider["version"], str
        ):
            raise ValueError("provider.version must be a string or null")
        if set(self.provider) - {
            "name",
            "model",
            "version",
            "deterministic",
            "secret_env_var",
        }:
            raise ValueError("provider has unknown fields")
        secret_env = self.provider.get("secret_env_var")
        if secret_env is not None and (
            not isinstance(secret_env, str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", secret_env)
        ):
            raise ValueError(
                "provider.secret_env_var must be an environment variable name"
            )
        for label, value in (
            ("provider", self.provider),
            ("rendering", self.rendering),
            ("resolved_config", self.resolved_config),
        ):
            self._reject_secret_fields(value, label)
        self._require_object(self.rendering, "rendering")
        render_required = {
            "font_family",
            "font_path",
            "fallback_font_path",
            "font_size",
            "min_font_size",
            "color",
            "stroke_width",
            "stroke_fill",
            "background",
            "wrap",
            "line_spacing",
            "align_h",
            "align_v",
            "text_direction",
            "expansion_allowance",
        }
        if set(self.rendering) != render_required:
            raise ValueError(
                "rendering must contain the complete Spec 004 rendering configuration"
            )
        if (
            not isinstance(self.rendering["font_path"], str)
            or not self.rendering["font_path"].strip()
        ):
            raise ValueError("rendering.font_path must be a resolved file path")
        font_size = self.rendering["font_size"]
        min_font_size = self.rendering["min_font_size"]
        if (
            not isinstance(font_size, int)
            or isinstance(font_size, bool)
            or font_size < 1
            or not isinstance(min_font_size, int)
            or isinstance(min_font_size, bool)
            or min_font_size < 1
            or min_font_size > font_size
        ):
            raise ValueError(
                "rendering font sizes must be positive and min_font_size <= font_size"
            )
        if self.rendering["text_direction"] not in {"auto", "horizontal", "vertical"}:
            raise ValueError("rendering.text_direction is invalid")
        if self.rendering["align_h"] not in {"left", "center", "right"}:
            raise ValueError("rendering.align_h is invalid")
        if self.rendering["align_v"] not in {"top", "middle", "bottom"}:
            raise ValueError("rendering.align_v is invalid")
        self._require_object(self.resolved_config, "resolved_config")
        required_config = {
            "manifest_path",
            "segmentation_method",
            "inpainting_source",
            "target_language",
            "region_extraction",
            "ocr",
            "provider",
            "rendering",
            "output_roots",
        }
        if set(self.resolved_config) != required_config:
            raise ValueError(
                "resolved_config must contain all nine validated config sections"
            )
        self._require_object(self.region_extraction_config, "region_extraction_config")
        self._require_object(self.counts, "counts")
        required_counts = {"pages_attempted", "pages_succeeded", "pages_failed"}
        if not required_counts <= self.counts.keys():
            raise ValueError(f"counts must include {sorted(required_counts)}")
        for key in required_counts:
            value = self.counts[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"counts.{key} must be a non-negative integer")
        try:
            json.dumps(self.to_dict_without_validation(), ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("TranslationRun fields must be JSON-serializable") from exc

    @staticmethod
    def _identity_string(value: Any, label: str) -> None:
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"{label} must be a Spec 2 sha256 identity string")

    @staticmethod
    def _require_object(value: Any, label: str) -> None:
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be an object")

    @classmethod
    def _reject_secret_fields(cls, value: Any, label: str) -> None:
        sensitive = {"secret", "api_key", "token", "credential", "authorization"}
        if isinstance(value, dict):
            for key, nested in value.items():
                lowered = str(key).lower()
                if lowered in sensitive:
                    raise ValueError(f"{label} must not contain secret field {key!r}")
                if lowered == "secret_env_var" and nested is not None:
                    if not isinstance(nested, str) or not re.fullmatch(
                        r"[A-Za-z_][A-Za-z0-9_]*", nested
                    ):
                        raise ValueError(
                            f"{label}.secret_env_var must be an environment variable "
                            "name"
                        )
                cls._reject_secret_fields(nested, f"{label}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                cls._reject_secret_fields(nested, f"{label}[{index}]")

    def to_dict_without_validation(self) -> dict[str, Any]:
        """Build a serializable projection without recursively revalidating."""
        return {
            "run_id": self.run_id,
            "segmentation_method": self.segmentation_method,
            "inpainting_source": self.inpainting_source,
            "target_language": self.target_language,
            "ocr_engine": self.ocr_engine,
            "translation_provider": self.provider,
            "region_extraction_config": self.region_extraction_config,
            "rendering_config": self.rendering,
            "resolved_config": self.resolved_config,
            "input_image_ids": list(self.input_image_ids),
            "counts": self.counts,
            "page_statuses": self.page_statuses,
            "page_list_identity": self.page_list_identity,
            "input_image_identity": self.input_image_identity,
        }

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return self.to_dict_without_validation()


def write_run_json(run_dir: str | Path, run_record: TranslationRun) -> Path:
    """Validate a TranslationRun and persist its deterministic run metadata."""
    if not isinstance(run_record, TranslationRun):
        raise TypeError("run_record must be a validated TranslationRun")
    directory = Path(run_dir)
    if directory.name != run_record.run_id:
        raise ValueError("run directory name must match TranslationRun.run_id")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "run.json"
    if path.exists():
        raise FileExistsError(f"run metadata already exists: {path}")
    path.write_text(
        json.dumps(run_record.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def create_run_directory(
    output_root: str | Path,
    run_id: str,
    *,
    overwrite: bool = False,
) -> Path:
    """Create only this run's directory; overwrite never targets sibling runs."""
    root = Path(output_root).resolve()
    candidate = root / _validate_identifier(run_id, "run_id")
    if candidate.is_symlink():
        raise ValueError("refusing to follow a symlink as the target run directory")
    target = candidate.resolve()
    if target.parent != root:
        raise ValueError("resolved run directory must be a direct child of output_root")
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"translation run already exists: {target}")
        if target.is_symlink() or not target.is_dir():
            raise ValueError(
                "refusing to overwrite a non-directory or symlink run target"
            )
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=False)
    return target


def _metadata_integer(value: Any, label: str, minimum: int = 0) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


@lru_cache(maxsize=1)
def _official_pipeline_metadata_schema() -> dict[str, Any]:
    schema_path = (
        Path(__file__).resolve().parents[3]
        / "specs"
        / "004-ocr-translation-rendering"
        / "contracts"
        / "pipeline-metadata.schema.json"
    )
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot load official Spec 004 metadata schema at {schema_path}: {exc}"
        ) from exc
    if not isinstance(schema, dict):
        raise RuntimeError("official Spec 004 metadata schema must be a JSON object")
    _assert_supported_schema_keywords(schema)
    return schema


_SCHEMA_ANNOTATIONS = {"$schema", "$id", "title", "description"}
_SCHEMA_VALIDATION_KEYWORDS = {
    "type",
    "enum",
    "const",
    "pattern",
    "minLength",
    "minimum",
    "exclusiveMinimum",
    "minItems",
    "maxItems",
    "items",
    "required",
    "properties",
    "additionalProperties",
    "allOf",
    "if",
    "then",
}


def _assert_supported_schema_keywords(schema: dict[str, Any], path: str = "$") -> None:
    """Fail closed when the official schema adds an unenforced keyword."""
    unsupported = set(schema) - _SCHEMA_ANNOTATIONS - _SCHEMA_VALIDATION_KEYWORDS
    if unsupported:
        raise RuntimeError(
            f"unsupported keyword(s) in official metadata schema at {path}: "
            f"{sorted(unsupported)}"
        )

    for key, value in schema.get("properties", {}).items():
        _assert_supported_schema_keywords(value, f"{path}/properties/{key}")
    items = schema.get("items")
    if isinstance(items, dict):
        _assert_supported_schema_keywords(items, f"{path}/items")
    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        _assert_supported_schema_keywords(additional, f"{path}/additionalProperties")
    for keyword in ("allOf",):
        for index, child in enumerate(schema.get(keyword, [])):
            _assert_supported_schema_keywords(child, f"{path}/{keyword}/{index}")
    for keyword in ("if", "then"):
        child = schema.get(keyword)
        if isinstance(child, dict):
            _assert_supported_schema_keywords(child, f"{path}/{keyword}")


def _schema_type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    raise RuntimeError(f"unsupported JSON Schema type in official schema: {expected}")


def _schema_errors(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type is not None:
        allowed = expected_type if isinstance(expected_type, list) else [expected_type]
        if not any(_schema_type_matches(value, name) for name in allowed):
            return [f"{path}: expected type {allowed}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value is outside enum")
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: value differs from const")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path}: string is shorter than minLength")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            errors.append(f"{path}: string does not match pattern")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            errors.append(f"{path}: number must be finite")
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: number is below minimum")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: number is below exclusiveMinimum")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: array has too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: array has too many items")
        if "items" in schema:
            for index, item in enumerate(value):
                errors.extend(_schema_errors(item, schema["items"], f"{path}[{index}]"))
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required property {key!r}")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                errors.extend(_schema_errors(item, properties[key], f"{path}.{key}"))
            elif additional is False:
                errors.append(f"{path}: unexpected property {key!r}")
            elif isinstance(additional, dict):
                errors.extend(_schema_errors(item, additional, f"{path}.{key}"))
    for subschema in schema.get("allOf", []):
        errors.extend(_schema_errors(value, subschema, path))
    condition = schema.get("if")
    if condition is not None and not _schema_errors(value, condition, path):
        errors.extend(_schema_errors(value, schema.get("then", {}), path))
    return errors


def _validate_pipeline_metadata(metadata: Any) -> dict[str, Any]:
    """Validate metadata against the official Spec 004 JSON Schema."""
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    errors = _schema_errors(metadata, _official_pipeline_metadata_schema())
    if errors:
        raise ValueError(
            "metadata violates official Spec 004 schema: " + "; ".join(errors)
        )
    return metadata


def _write_json_file(path: Path, value: Any) -> Path:
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")
    return path


def write_metadata(page_dir: str | Path, metadata: dict[str, Any]) -> Path:
    """Validate and write FR-047 metadata without normalizing upstream provenance."""
    directory = Path(page_dir)
    _validate_pipeline_metadata(metadata)
    return _write_json_file(directory / "metadata.json", metadata)


def resolve_page_directory(
    output_root: str | Path,
    run_id: str,
    segmentation_method: str,
    image_id: str,
) -> Path:
    """Resolve the existing run/method/image_id addressing contract (FR-052)."""
    return page_directory(output_root, run_id, segmentation_method, image_id)


def write_crop(
    page_dir: str | Path, region_id: int, source: bytes | str | Path
) -> Path:
    """Write one independently addressable PNG crop, copying path inputs."""
    _metadata_integer(region_id, "region_id", 1)
    destination = Path(page_dir) / "crops" / f"{region_id}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(source, bytes):
        destination.write_bytes(source)
    else:
        shutil.copyfile(Path(source), destination)
    return destination


def write_mask_copy(page_dir: str | Path, source: str | Path) -> Path:
    """Physically copy the selected prediction mask to the page artifact tree."""
    destination = Path(page_dir) / "mask.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(source), destination)
    return destination


def write_ocr_json(page_dir: str | Path, document: Any) -> Path:
    """Persist the OCR stage's standalone JSON artifact."""
    return _write_json_file(Path(page_dir) / "ocr.json", document)


def write_translations_json(page_dir: str | Path, document: Any) -> Path:
    """Persist the translation stage's standalone JSON artifact."""
    return _write_json_file(Path(page_dir) / "translations.json", document)


def write_rendered_png(page_dir: str | Path, source: str | Path) -> Path:
    """Copy a rendered PNG into its contract location without altering it."""
    destination = Path(page_dir) / "rendered.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(source), destination)
    return destination
