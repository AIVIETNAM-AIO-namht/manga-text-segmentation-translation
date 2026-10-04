"""T065–T070: validate page metadata and stage artifacts against Spec 004."""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any

import pytest

from manga_text_seg.translation.runs import (
    page_directory,
    resolve_page_directory,
    write_crop,
    write_mask_copy,
    write_metadata,
    write_ocr_json,
    write_rendered_png,
    write_translations_json,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = (
    ROOT
    / "specs"
    / "004-ocr-translation-rendering"
    / "contracts"
    / "pipeline-metadata.schema.json"
)

SCHEMA_ANNOTATION_KEYWORDS = {"$schema", "$id", "title", "description"}
SCHEMA_VALIDATION_KEYWORDS = {
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


def _schema_keywords(schema: dict) -> set[str]:
    """Collect keywords from schema nodes, not arbitrary example values."""
    found = set(schema)
    for child in schema.get("properties", {}).values():
        found.update(_schema_keywords(child))
    items = schema.get("items")
    if isinstance(items, dict):
        found.update(_schema_keywords(items))
    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        found.update(_schema_keywords(additional))
    for child in schema.get("allOf", []):
        found.update(_schema_keywords(child))
    for keyword in ("if", "then"):
        child = schema.get(keyword)
        if isinstance(child, dict):
            found.update(_schema_keywords(child))
    return found


def _is_type(value: Any, name: str) -> bool:
    return {
        "string": lambda: isinstance(value, str),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda: isinstance(value, (int, float))
        and not isinstance(value, bool),
        "boolean": lambda: isinstance(value, bool),
        "array": lambda: isinstance(value, list),
        "object": lambda: isinstance(value, dict),
        "null": lambda: value is None,
    }[name]()


def _validate(
    instance: Any, schema: dict, path: str = "$", root: bool = True
) -> list[str]:
    """Small draft-07 validator for the keywords used by the official schema."""
    errors: list[str] = []
    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_is_type(instance, name) for name in names):
            return [f"{path}: expected {names}, got {type(instance).__name__}"]
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: value is outside enum")
    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: value differs from const")
    if isinstance(instance, str):
        if len(instance) < schema.get("minLength", 0):
            errors.append(f"{path}: string is shorter than minLength")
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            errors.append(f"{path}: string does not match pattern")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if not math.isfinite(float(instance)):
            errors.append(f"{path}: number is not finite")
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: below minimum")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: not above exclusiveMinimum")
    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: too few items")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: too many items")
        if "items" in schema:
            for index, item in enumerate(instance):
                errors.extend(
                    _validate(item, schema["items"], f"{path}[{index}]", False)
                )
    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}: missing required property {key!r}")
        props = schema.get("properties", {})
        for key, value in instance.items():
            if key in props:
                errors.extend(_validate(value, props[key], f"{path}.{key}", False))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected property {key!r}")
            elif isinstance(schema.get("additionalProperties"), dict):
                errors.extend(
                    _validate(
                        value, schema["additionalProperties"], f"{path}.{key}", False
                    )
                )
    for subschema in schema.get("allOf", []):
        errors.extend(_validate(instance, subschema, path, False))
    if "if" in schema and not _validate(instance, schema["if"], path, False):
        errors.extend(_validate(instance, schema.get("then", {}), path, False))
    return errors


@pytest.fixture(scope="module")
def schema() -> dict:
    assert SCHEMA_PATH.is_file(), f"official schema missing: {SCHEMA_PATH}"
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _region(**updates: Any) -> dict:
    record = {
        "region_id": 1,
        "bbox": [10, 20, 80, 40],
        "polygon": [[10, 20], [89, 20], [89, 59]],
        "orientation": "horizontal",
        "reading_order_index": 0,
        "merged_from": 1,
        "crop_path": "crops/1.png",
        "clipped": False,
        "ocr_text": "recognized",
        "ocr_confidence": None,
        "ocr_status": "ok",
        "ocr_error": None,
        "manually_edited": False,
        "translation": "translated",
        "translation_status": "ok",
        "translation_error": None,
        "cache_status": "miss",
        "failure_policy": "keep_ocr_text",
        "render_status": "rendered",
        "render_warning": None,
        "font_size_used": 24,
        "line_count": 1,
        "display_box": [10, 20, 80, 40],
        "direction_used": "horizontal",
        "ocr_time_seconds": 0.0,
        "translation_time_seconds": 0.0,
        "render_time_seconds": 0.0,
    }
    record.update(updates)
    return record


def _metadata(**updates: Any) -> dict:
    record = {
        "image_id": "ARMS/000",
        "run_id": "schema-run",
        "segmentation_method": "classical_baseline",
        "inpainting_method": {
            "run_id": "inpaint-run",
            "algorithm": "telea",
            "inpaint_algorithm": "INPAINT_TELEA",
            "inpaint_radius": 3.0,
            "dilation": {"enabled": False},
        },
        "target_language": "vi",
        "ocr_engine": {"name": "fake-ocr", "version": "1.0", "deterministic": True},
        "translation_provider": {
            "name": "fixture",
            "model": "fixture-model",
            "version": "1",
            "deterministic": True,
            "secret_env_var": None,
        },
        "rendering_config": {
            "font_path": str(ROOT / "assets" / "fonts" / "DejaVuSans.ttf"),
            "font_size": 32,
            "min_font_size": 12,
            "text_direction": "auto",
            "expansion_allowance": 0,
            "color": [0, 0, 0, 255],
            "stroke_width": 0,
            "background": None,
            "wrap": True,
            "line_spacing": 1.0,
            "align_h": "center",
            "align_v": "middle",
        },
        "regions": [_region()],
        "stage_timings": {"intake": 0.0, "regions": 0.0},
        "status": "ok",
    }
    record.update(updates)
    return record


def test_metadata_writer_output_validates_official_schema_and_preserves_provenance(
    schema, tmp_path
):
    record = _metadata()
    source_inpainting = copy.deepcopy(record["inpainting_method"])
    path = write_metadata(tmp_path, record)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert _validate(persisted, schema) == []
    assert persisted["inpainting_method"] == source_inpainting


def test_validator_covers_every_keyword_in_the_official_schema(schema):
    """A schema extension cannot silently bypass the metadata validator."""
    assert _schema_keywords(schema) <= (
        SCHEMA_ANNOTATION_KEYWORDS | SCHEMA_VALIDATION_KEYWORDS
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc.update(run_id=""),
        lambda doc: doc["inpainting_method"].update(run_id=""),
        lambda doc: doc["ocr_engine"].update(name=""),
        lambda doc: doc["translation_provider"].update(name=""),
        lambda doc: doc["translation_provider"].update(model=""),
    ],
    ids=[
        "run-id-min-length",
        "inpainting-run-id-min-length",
        "ocr-name-min-length",
        "provider-name-min-length",
        "provider-model-min-length",
    ],
)
def test_min_length_constraints_are_enforced_by_official_schema_and_writer(
    schema, tmp_path, mutate
):
    document = _metadata()
    mutate(document)

    assert _validate(document, schema)
    with pytest.raises(ValueError, match="official Spec 004 schema"):
        write_metadata(tmp_path, document)


def test_failed_region_is_retained_and_validates(schema, tmp_path):
    failed = _region(
        ocr_text="",
        ocr_status="failed",
        ocr_error="recognizer failed",
        translation="",
        translation_status="skipped",
        failure_policy=None,
        render_status="skipped",
        font_size_used=None,
        line_count=0,
        display_box=None,
        direction_used=None,
    )
    doc = _metadata(regions=[_region(), failed])
    persisted = json.loads(write_metadata(tmp_path, doc).read_text(encoding="utf-8"))
    assert _validate(persisted, schema) == []
    assert len(persisted["regions"]) == 2
    assert persisted["regions"][1]["ocr_status"] == "failed"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda doc: doc.pop("run_id"),
        lambda doc: doc.update(unlisted_field=True),
        lambda doc: doc.update(status="complete"),
        lambda doc: doc.update(status="failed"),
    ],
    ids=[
        "missing-required",
        "additional-properties-false",
        "bad-status",
        "failed-requires-message",
    ],
)
def test_schema_and_writer_reject_contract_violations(schema, tmp_path, mutation):
    doc = _metadata()
    mutation(doc)
    assert _validate(doc, schema)
    with pytest.raises(ValueError):
        write_metadata(tmp_path, doc)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda doc: doc["regions"][0].update(orientation="diagonal"),
        lambda doc: doc["regions"][0].update(polygon=[[1, 2], [3]]),
        lambda doc: doc["regions"][0].update(polygon=[[1, "2"], [3, 4]]),
        lambda doc: doc["regions"][0].update(ocr_status="complete"),
        lambda doc: doc.update(stage_timings={"ocr": -0.01}),
        lambda doc: doc["regions"][0].pop("bbox"),
        lambda doc: doc.update(ocr_engine=[]),
        lambda doc: doc["inpainting_method"].update(dilation={"enabled": True}),
        lambda doc: doc["inpainting_method"].update(inpaint_radius=0),
        lambda doc: doc["inpainting_method"].update(
            dilation={
                "enabled": True,
                "kernel_shape": "ellipse",
                "kernel_size": [3, "3"],
                "iterations": 1,
            }
        ),
    ],
    ids=[
        "invalid-orientation",
        "malformed-polygon-point",
        "polygon-point-wrong-type",
        "invalid-region-status",
        "invalid-timing",
        "missing-required-region-field",
        "wrong-nested-object-type",
        "enabled-dilation-missing-provenance",
        "non-positive-inpaint-radius",
        "malformed-dilation-kernel-size",
    ],
)
def test_writer_rejects_nested_official_schema_violations(schema, tmp_path, mutation):
    doc = _metadata()
    mutation(doc)
    assert _validate(doc, schema)
    with pytest.raises(ValueError, match="official Spec 004 schema"):
        write_metadata(tmp_path, doc)


def test_addressing_and_intermediate_writers_use_contract_paths(tmp_path):
    page = resolve_page_directory(tmp_path, "r1", "classical_baseline", "ARMS/000")
    assert page == page_directory(tmp_path, "r1", "classical_baseline", "ARMS/000")
    crop = write_crop(page, 1, b"crop")
    mask_source = tmp_path / "upstream-mask.png"
    mask_source.write_bytes(b"mask")
    mask_copy = write_mask_copy(page, mask_source)
    ocr = write_ocr_json(page, {"results": []})
    translations = write_translations_json(page, [])
    rendered_source = tmp_path / "source-rendered.png"
    rendered_source.write_bytes(b"png fixture bytes")
    rendered = write_rendered_png(page, rendered_source)
    assert crop.relative_to(page).as_posix() == "crops/1.png"
    assert mask_copy.name == "mask.png" and mask_copy.read_bytes() == b"mask"
    assert ocr.name == "ocr.json" and json.loads(ocr.read_text()) == {"results": []}
    assert (
        translations.name == "translations.json"
        and json.loads(translations.read_text()) == []
    )
    assert rendered.name == "rendered.png"
    assert rendered.read_bytes() == rendered_source.read_bytes()
    assert mask_source.read_bytes() == b"mask"


def test_metadata_has_contract_stage_status_and_finite_timings(tmp_path):
    for status in ("ok", "no_regions", "failed"):
        doc = _metadata(status=status)
        if status == "failed":
            doc["error_message"] = "page failed"
        path = write_metadata(tmp_path / status, doc)
        assert json.loads(path.read_text())["status"] == status
    bad = _metadata(stage_timings={"ocr": float("nan")})
    with pytest.raises((ValueError, TypeError)):
        write_metadata(tmp_path / "non-finite", bad)
