"""Contract tests for the distributable page list (T015).

Validates the exported ``page-list.json`` against
``contracts/page-list.schema.json`` — the schema itself, loaded from disk, not a
restatement of it. The validator below walks the subset of draft-07 the schema
actually uses: ``type``, ``enum``, ``pattern``, ``minItems``, ``maxItems``,
``items``, ``required``, ``properties`` and ``additionalProperties: false``.

Hand-rolled, no jsonschema dependency (Constitution Principle III) — the same
choice ``tests/contract/test_manifest.py`` made.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from ..conftest import ALIGNED_SIZE, REPO_ROOT

SCHEMA_PATH = (
    REPO_ROOT
    / "specs"
    / "002-deep-learning-segmentation-benchmark"
    / "contracts"
    / "page-list.schema.json"
)
EXPECTED_PAGES = 390


# -- A draft-07 subset validator ---------------------------------------------


def _type_name(value) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "null"


def _matches(value, expected: str) -> bool:
    actual = _type_name(value)
    # bool is not an integer, and an integer is a valid number.
    if expected == "number":
        return actual in ("integer", "number")
    return actual == expected


def _validate(instance, schema: dict, path: str = "$") -> list[str]:
    """Return a list of schema violations; empty means valid."""
    if "type" in schema and not _matches(instance, schema["type"]):
        return [f"{path}: expected {schema['type']}, got {_type_name(instance)}"]

    errors: list[str] = []

    if "enum" in schema and not any(instance == option for option in schema["enum"]):
        errors.append(f"{path}: {instance!r} not one of {schema['enum']!r}")

    if "pattern" in schema and not (
        isinstance(instance, str) and re.search(schema["pattern"], instance)
    ):
        errors.append(f"{path}: {instance!r} does not match {schema['pattern']!r}")

    if isinstance(instance, list):
        if len(instance) < schema.get("minItems", 0):
            errors.append(f"{path}: {len(instance)} items, minItems is {schema['minItems']}")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: {len(instance)} items, maxItems is {schema['maxItems']}")
        if "items" in schema:
            for index, item in enumerate(instance):
                errors += _validate(item, schema["items"], f"{path}[{index}]")

    if isinstance(instance, dict):
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}: missing required property {key!r}")
        if schema.get("additionalProperties") is False:
            for key in instance:
                if key not in properties:
                    errors.append(f"{path}: unexpected property {key!r}")
        for key, sub_schema in properties.items():
            if key in instance:
                errors += _validate(instance[key], sub_schema, f"{path}.{key}")

    return errors


@pytest.fixture(scope="session")
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@pytest.fixture()
def corrupt(real_page_list):
    """Return a mutator that applies ``change`` to a deep copy of the page list."""
    def _corrupt(change):
        doc = copy.deepcopy(real_page_list)
        change(doc)
        return doc

    return _corrupt


# -- The export satisfies the contract ---------------------------------------


def test_exported_page_list_validates(real_page_list, schema):
    assert _validate(real_page_list, schema) == []


def test_page_list_records_the_shared_conventions(real_page_list):
    """FR-053c/FR-004 restated in the list, so a runner needs no access here."""
    assert real_page_list["aligned_size"] == ALIGNED_SIZE == [1654, 1170]
    assert real_page_list["alignment"] == "crop-topleft"
    assert real_page_list["mask_convention"] == {
        "dtype": "uint8",
        "channels": 1,
        "values": [0, 255],
        "text": 255,
        "background": 0,
    }


def test_page_list_holds_every_page(real_page_list):
    assert len(real_page_list["pages"]) == EXPECTED_PAGES
    assert all(page["image_id"] == f"{page['manga']}/{page['stem']}"
               for page in real_page_list["pages"])


def test_bundle_is_relocatable(real_page_list):
    """Both paths are relative, so a moved bundle still resolves (research.md R2)."""
    bundle = real_page_list["image_bundle"]
    assert not Path(bundle["root"]).is_absolute()
    assert not Path(bundle["identity_record"]).is_absolute()
    assert all(not Path(p["image_ref"]).is_absolute() for p in real_page_list["pages"])


# -- The validator is not vacuous --------------------------------------------


@pytest.mark.parametrize(
    "label, change",
    [
        ("missing pages", lambda doc: doc.pop("pages")),
        ("root extra property", lambda doc: doc.update(gt_root="data/groundtruth")),
        (
            "identity is not a sha256",
            lambda doc: doc.update(page_list_identity="not-a-hash"),
        ),
        (
            "aligned_size has three elements",
            lambda doc: doc.update(aligned_size=[1654, 1170, 3]),
        ),
        (
            "aligned_size as strings",
            lambda doc: doc.update(aligned_size=["1654", "1170"]),
        ),
        (
            "collapse rule admits a third value",
            lambda doc: doc["mask_convention"].update(values=[0, 128, 255]),
        ),
        (
            "alignment is not the shared one",
            lambda doc: doc.update(alignment="resize"),
        ),
        (
            "page missing image_id",
            lambda doc: doc["pages"][0].pop("image_id"),
        ),
        (
            "page carries mask_path",
            lambda doc: doc["pages"][0].update(mask_path="data/groundtruth/ARMS/000.png"),
        ),
        (
            "bundle root absolute",
            lambda doc: doc["image_bundle"].update(root="D:/data/raw"),
        ),
        (
            "input_image_identity truncated",
            lambda doc: doc["pages"][0].update(input_image_identity="abc"),
        ),
    ],
)
def test_validator_rejects_a_violation(corrupt, schema, label, change):
    """Each mutation must be caught, and every error must name its location."""
    errors = _validate(corrupt(change), schema)

    assert errors, f"validator accepted: {label}"
    assert all(error.startswith("$") for error in errors), errors
    assert all(": " in error for error in errors), errors
