"""Unit tests for provenance completeness (T007, FR-056, FR-018b, SC-014).

The required-field list is read out of ``contracts/provenance.schema.json``
itself rather than restated here, so the test cannot drift from the contract:
every required key at every level is removed in turn and the refusal must name
that key's schema path.  A record with a gap is refused, never accepted with
the gap (SC-014).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from manga_text_seg.provenance import check_completeness, require_complete, ProvenanceError

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "specs"
    / "002-deep-learning-segmentation-benchmark"
    / "contracts"
    / "provenance.schema.json"
)


def _schema() -> dict:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _required_paths(schema: dict, prefix: str = "") -> list[str]:
    """Every *reachable* required key in the schema as a dotted path.

    Descent stops at an object that is not itself required (``deviation``), so
    the parametrisation never asks for a path a conforming record need not have.
    """
    required = schema.get("required", [])
    paths = [f"{prefix}{key}" for key in required]
    for key in required:
        sub = schema.get("properties", {}).get(key, {})
        if isinstance(sub, dict) and sub.get("required"):
            paths.extend(_required_paths(sub, prefix=f"{prefix}{key}."))
    return paths


def _delete(record: dict, path: str) -> dict:
    """Return a copy of ``record`` with the dotted ``path`` removed."""
    out = copy.deepcopy(record)
    parts = path.split(".")
    node = out
    for part in parts[:-1]:
        node = node[part]
    del node[parts[-1]]
    return out


REQUIRED_PATHS = _required_paths(_schema())


class TestRequiredFieldTable:
    """The completeness table matches the contract's own required lists."""

    def test_contract_has_required_fields(self) -> None:
        assert REQUIRED_PATHS, "provenance.schema.json declares no required keys"
        assert "method" in REQUIRED_PATHS
        assert "repository.revision" in REQUIRED_PATHS
        assert "checkpoint.sha256" in REQUIRED_PATHS
        assert "device.name" in REQUIRED_PATHS


class TestCompleteRecordPasses:
    """A record carrying every required field is accepted."""

    def test_complete_record_has_no_missing_paths(self, default_provenance) -> None:
        assert check_completeness(default_provenance) == []

    def test_require_complete_accepts_complete_record(self, default_provenance) -> None:
        require_complete(default_provenance)


class TestEveryRequiredFieldIsEnforced:
    """Removing any one required field refuses the record, naming that path."""

    @pytest.mark.parametrize("path", REQUIRED_PATHS)
    def test_removed_field_is_named(self, default_provenance, path: str) -> None:
        incomplete = _delete(default_provenance, path)
        missing = check_completeness(incomplete)
        assert path in missing, (
            f"removing '{path}' was not reported; got {missing}"
        )

    @pytest.mark.parametrize("path", REQUIRED_PATHS)
    def test_refusal_names_the_path(self, default_provenance, path: str) -> None:
        incomplete = _delete(default_provenance, path)
        with pytest.raises(ProvenanceError) as excinfo:
            require_complete(incomplete)
        assert path in str(excinfo.value), (
            f"refusal did not name '{path}': {excinfo.value}"
        )

    def test_removing_the_whole_repository_object_is_named(self, default_provenance) -> None:
        incomplete = copy.deepcopy(default_provenance)
        del incomplete["repository"]
        assert "repository" in check_completeness(incomplete)


class TestRevisionFormat:
    """A branch name, tag or short SHA is not a revision and is refused."""

    @pytest.mark.parametrize(
        "revision",
        ["main", "v1.0", "de8f148", "de8f148c78978d70ad0e0ae3242566da6d70f3a", "Z" * 40],
    )
    def test_non_sha_revision_refused(self, default_provenance, revision: str) -> None:
        record = copy.deepcopy(default_provenance)
        record["repository"]["revision"] = revision
        missing = check_completeness(record)
        assert "repository.revision" in missing, (
            f"revision {revision!r} was accepted; got {missing}"
        )

    def test_full_sha_revision_accepted(self, default_provenance) -> None:
        record = copy.deepcopy(default_provenance)
        record["repository"]["revision"] = "de8f148c78978d70ad0e0ae3242566da6d70f3a5"
        assert check_completeness(record) == []

    @pytest.mark.parametrize("sha256", ["deadbeef", "", "z" * 64])
    def test_non_sha256_checkpoint_hash_refused(self, default_provenance, sha256: str) -> None:
        record = copy.deepcopy(default_provenance)
        record["checkpoint"]["sha256"] = sha256
        assert "checkpoint.sha256" in check_completeness(record)

    def test_checkpoint_size_must_be_positive(self, default_provenance) -> None:
        record = copy.deepcopy(default_provenance)
        record["checkpoint"]["size_bytes"] = 0
        assert "checkpoint.size_bytes" in check_completeness(record)


class TestLoadEvidence:
    """FR-018b: a record without positive load evidence is inadmissible."""

    def test_unevidenced_record_is_refused(self, default_provenance) -> None:
        record = copy.deepcopy(default_provenance)
        record["checkpoint_load_evidence"]["evidenced"] = False
        missing = check_completeness(record)
        assert "checkpoint_load_evidence.evidenced" in missing

    def test_blank_detail_is_refused(self, default_provenance) -> None:
        record = copy.deepcopy(default_provenance)
        record["checkpoint_load_evidence"]["detail"] = ""
        assert "checkpoint_load_evidence.detail" in check_completeness(record)


class TestNonObjectRecord:
    """A record that is not an object is refused rather than crashing."""

    def test_non_mapping_refused(self) -> None:
        with pytest.raises(ProvenanceError):
            require_complete(["not", "a", "record"])
