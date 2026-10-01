"""T023 - contract test for the per-sample metadata sidecar (Spec 003, US2).

Contract: contracts/sample-metadata.schema.json (FR-025, FR-018).

The repo has no ``jsonschema`` and does not add one (Constitution Principle
III), so this file carries a small validator for exactly the draft-07 keywords
the schema uses. Because a lenient validator would let anything pass, the
validator is itself tested against documents that must be refused.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from manga_text_seg.config import DilationConfig
from manga_text_seg.inpaint import process_sample
from manga_text_seg.runs import (
    AlignmentRecord,
    SampleMetadata,
    create_run,
    sample_paths,
    write_sample_metadata,
)

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "specs"
    / "003-text-removal-inpainting"
    / "contracts"
    / "sample-metadata.schema.json"
)
METHOD = "classical_baseline"
RUN_ID = "run-t023"
IMAGE_ID = "ARMS/000"


# --------------------------------------------------------------------------- #
# A validator for the keywords this schema uses
# --------------------------------------------------------------------------- #
def _is_type(value: Any, name: str) -> bool:
    if name == "string":
        return isinstance(value, str)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "array":
        return isinstance(value, list)
    if name == "object":
        return isinstance(value, dict)
    if name == "null":
        return value is None
    raise AssertionError(f"validator does not know type {name!r}")


def validate(instance: Any, schema: dict, path: str = "$") -> list[str]:
    """Return every violation of ``schema`` found in ``instance`` (empty = valid)."""
    errors: list[str] = []

    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_is_type(instance, n) for n in names):
            return [f"{path}: expected {names}, got {type(instance).__name__}"]
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} is not one of {schema['enum']}")
    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: {instance!r} is not {schema['const']!r}")

    if (
        isinstance(instance, str)
        and "pattern" in schema
        and not re.search(schema["pattern"], instance)
    ):
        errors.append(f"{path}: {instance!r} does not match {schema['pattern']}")

    if _is_type(instance, "number"):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} < minimum {schema['minimum']}")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: {instance} <= {schema['exclusiveMinimum']}")

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        if "items" in schema:
            for i, item in enumerate(instance):
                errors += validate(item, schema["items"], f"{path}[{i}]")

    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}: missing required property {key!r}")
        properties = schema.get("properties", {})
        for key, value in instance.items():
            if key in properties:
                errors += validate(value, properties[key], f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected property {key!r}")

    for sub in schema.get("allOf", []):
        errors += validate(instance, sub, path)
    if "if" in schema and not validate(instance, schema["if"], path):
        errors += validate(instance, schema.get("then", {}), path)

    return errors


@pytest.fixture(scope="module")
def schema() -> dict:
    assert SCHEMA_PATH.is_file(), f"contract file not found: {SCHEMA_PATH}"
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _metadata(**overrides: Any) -> SampleMetadata:
    values: dict[str, Any] = {
        "image_id": IMAGE_ID,
        "method": METHOD,
        "algorithm": "telea",
        "model_repository": None,
        "experiment_id": "fixture-experiment",
        "input_image_path": "pages/ARMS/000.jpg",
        "prediction_mask_path": "masks/ARMS/000.png",
        "mask_size": (1654, 1170),
        "alignment": AlignmentRecord("pass-through", (1654, 1170), (1654, 1170)),
        "dilation": DilationConfig(True, "ellipse", (3, 3), 1),
        "inpaint_algorithm": "INPAINT_TELEA",
        "inpaint_radius": 3,
        "output_format": "png",
        "processing_time_seconds": 0.25,
        "status": "ok",
        "error_message": None,
    }
    values.update(overrides)
    return SampleMetadata(**values)


def _as_json(metadata: SampleMetadata) -> dict:
    return json.loads(json.dumps(metadata.to_dict()))


# --------------------------------------------------------------------------- #
# The schema and the writer agree on the field list
# --------------------------------------------------------------------------- #
def test_schema_requires_exactly_the_fifteen_fields(schema):
    assert len(schema["required"]) == 15
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == set(schema["required"]) | {"error_message"}


def test_writer_and_schema_name_the_same_fields(schema):
    written = {f.name for f in fields(SampleMetadata)}

    assert written == set(schema["properties"])
    assert set(schema["required"]) == written - {"error_message"}


# --------------------------------------------------------------------------- #
# The validator refuses what the schema forbids
# --------------------------------------------------------------------------- #
def _without(key: str):
    def mutate(doc: dict) -> None:
        del doc[key]

    return mutate


def _set(path: tuple, value: Any):
    def mutate(doc: dict) -> None:
        target = doc
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


def _fail_without_message(doc: dict) -> None:
    doc["status"] = "failed"
    del doc["error_message"]


def _delete(path: tuple):
    def mutate(doc: dict) -> None:
        target = doc
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]

    return mutate


@pytest.mark.parametrize(
    "mutate",
    [
        _without("processing_time_seconds"),
        _without("dilation"),
        _set(("surprise",), 1),
        _set(("alignment", "surprise"), 1),
        _set(("dilation", "surprise"), 1),
        _set(("method",), "some_other_method"),
        _set(("algorithm",), "bogus"),
        _set(("inpaint_algorithm",), "INPAINT_OTHER"),
        _set(("output_format",), "jpeg"),
        _set(("image_id",), "ARMS/0"),
        _set(("image_id",), "no-separator"),
        _set(("mask_size",), [1654]),
        _set(("mask_size",), [1654, 0]),
        _set(("inpaint_radius",), 0),
        _set(("processing_time_seconds",), -1),
        _set(("status",), "done"),
        _set(("alignment", "rule"), "resize"),
        _fail_without_message,
        _delete(("dilation", "kernel_shape")),
        _delete(("dilation", "kernel_size")),
        _delete(("dilation", "iterations")),
    ],
    ids=[
        "missing-time",
        "missing-dilation",
        "extra-top-level",
        "extra-in-alignment",
        "extra-in-dilation",
        "unknown-method",
        "unknown-algorithm",
        "unknown-inpaint-constant",
        "unknown-format",
        "image-id-unpadded",
        "image-id-no-separator",
        "mask-size-one-item",
        "mask-size-zero",
        "radius-zero",
        "negative-time",
        "unknown-status",
        "unknown-rule",
        "failed-without-message",
        "enabled-without-shape",
        "enabled-without-size",
        "enabled-without-iterations",
    ],
)
def test_validator_refuses_a_document_that_breaks_the_schema(schema, mutate):
    doc = _as_json(_metadata())
    assert validate(doc, schema) == [], "the starting document must be valid"

    broken = copy.deepcopy(doc)
    mutate(broken)

    assert broken != doc, "the mutation changed nothing"
    assert validate(broken, schema), f"expected a violation, got none: {broken}"


# --------------------------------------------------------------------------- #
# Sidecars the writer produces validate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "metadata",
    [
        _metadata(),
        _metadata(
            algorithm="ns",
            inpaint_algorithm="INPAINT_NS",
            model_repository="https://example.invalid/repo",
        ),
        _metadata(dilation=DilationConfig(enabled=False)),
        _metadata(
            alignment=AlignmentRecord("crop-topleft", (1656, 1176), (1654, 1170)),
            mask_size=(1656, 1176),
        ),
        _metadata(status="failed", error_message="inpainting_failure: cv2.error"),
    ],
    ids=["ok-telea", "ok-ns-with-repo", "dilation-off", "cropped", "failed"],
)
def test_written_sidecar_validates(schema, tmp_path, metadata):
    path = tmp_path / "sidecar.json"
    write_sample_metadata(path, metadata)

    doc = json.loads(path.read_text(encoding="utf-8"))

    assert validate(doc, schema) == []
    assert set(schema["required"]) <= set(doc)


# --------------------------------------------------------------------------- #
# Sidecars the pipeline produces validate
# --------------------------------------------------------------------------- #
def test_every_sidecar_from_process_sample_validates(
    schema, inpaint_fixture, synthetic_manifest, inpaint_config
):
    manifest = synthetic_manifest()
    root = inpaint_fixture(METHOD, IMAGE_ID)
    config = inpaint_config({METHOD: root})
    create_run(config.output_root, RUN_ID)

    process_sample(
        manifest,
        "ARMS",
        "000",
        method=METHOD,
        run_id=RUN_ID,
        config=config,
        experiment_id="fixture-experiment",
        model_repository=None,
    )

    for algorithm in ("telea", "ns"):
        path = sample_paths(
            config.output_root, RUN_ID, METHOD, algorithm, IMAGE_ID
        ).metadata_path
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert validate(doc, schema) == [], f"{algorithm}: {validate(doc, schema)}"
        assert doc["alignment"]["post_size"] == [1654, 1170], "SC-002"
        assert doc["status"] == "ok"
