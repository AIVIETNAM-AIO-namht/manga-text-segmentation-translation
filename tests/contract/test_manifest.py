"""Contract tests for manifest schema (manifest.schema.json).

Hand-rolled validators — no jsonschema dependency (Constitution Principle III).
Validates the serialized manifest output of build_manifest/save_manifest
against the frozen contract.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from manga_text_seg.manifest import build_manifest, save_manifest, load_manifest
from manga_text_seg.imaging import MASK_ENCODINGS

# --- Contract constants (from contracts/manifest.schema.json) ---

STEM_PATTERN = re.compile(r"^[0-9]{3}$")
MASK_ENCODING_VALUES = set(MASK_ENCODINGS)  # {"magenta-black", "magenta-only", "near-black-only", "all-white-empty"}
EXPECTED_PAIR_COUNT = 390


# --- Test helpers ---

def _validate_pair(pair: dict, idx: int) -> None:
    """Validate a single pair dict against the manifest schema contract."""
    required_keys = {"manga", "stem", "raw_path", "mask_path", "mask_encoding"}
    missing = required_keys - set(pair.keys())
    assert not missing, f"Pair {idx} missing required keys: {missing}"

    assert isinstance(pair["manga"], str), f"Pair {idx}: manga must be string"
    assert isinstance(pair["stem"], str), f"Pair {idx}: stem must be string"
    assert isinstance(pair["raw_path"], str), f"Pair {idx}: raw_path must be string"
    assert isinstance(pair["mask_path"], str), f"Pair {idx}: mask_path must be string"
    assert isinstance(pair["mask_encoding"], str), f"Pair {idx}: mask_encoding must be string"

    assert STEM_PATTERN.match(pair["stem"]), (
        f"Pair {idx}: stem '{pair['stem']}' does not match pattern ^[0-9]{{3}}$"
    )
    assert pair["mask_encoding"] in MASK_ENCODING_VALUES, (
        f"Pair {idx}: mask_encoding '{pair['mask_encoding']}' "
        f"not in {sorted(MASK_ENCODING_VALUES)}"
    )


def _validate_manifest_dict(data: dict) -> None:
    """Validate a full manifest dict against manifest.schema.json."""
    # Top-level required keys
    assert "run_id" in data, "Missing required key: run_id"
    assert "pairs" in data, "Missing required key: pairs"
    assert isinstance(data["run_id"], str), "run_id must be string"
    assert isinstance(data["pairs"], list), "pairs must be array"

    # Pair count constraint: minItems=390, maxItems=390
    n = len(data["pairs"])
    assert n == EXPECTED_PAIR_COUNT, (
        f"Expected exactly {EXPECTED_PAIR_COUNT} pairs, got {n}"
    )

    # Validate each pair
    for i, pair in enumerate(data["pairs"]):
        assert isinstance(pair, dict), f"Pair {i} must be object"
        _validate_pair(pair, i)


# --- Tests ---

class TestManifestContractRoundTrip:
    """build_manifest → save → JSON round-trip satisfies the contract."""

    @pytest.fixture()
    def manifest_dict(self, tmp_path: Path) -> dict:
        """Build, save, and reload a manifest from the real dataset."""
        gt_root = Path("data/groundtruth/post-processed")
        raw_root = Path("data/raw/Manga109s_released_2026_05_21/images")
        if not gt_root.is_dir() or not raw_root.is_dir():
            pytest.skip("Real dataset not present")

        manifest = build_manifest(gt_root, raw_root, run_id="contract-test")
        manifest_path = tmp_path / "manifest.json"
        save_manifest(manifest, manifest_path)

        with open(manifest_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def test_top_level_required_keys(self, manifest_dict: dict) -> None:
        """Contract: top-level must have run_id and pairs."""
        assert "run_id" in manifest_dict
        assert "pairs" in manifest_dict

    def test_pair_count_exactly_390(self, manifest_dict: dict) -> None:
        """Contract: pairs array minItems=390, maxItems=390."""
        assert len(manifest_dict["pairs"]) == 390

    def test_all_pairs_have_required_keys(self, manifest_dict: dict) -> None:
        """Contract: each pair requires manga, stem, raw_path, mask_path, mask_encoding."""
        required = {"manga", "stem", "raw_path", "mask_path", "mask_encoding"}
        for i, pair in enumerate(manifest_dict["pairs"]):
            missing = required - set(pair.keys())
            assert not missing, f"Pair {i} missing: {missing}"

    def test_stem_pattern(self, manifest_dict: dict) -> None:
        """Contract: stem matches ^[0-9]{3}$."""
        for i, pair in enumerate(manifest_dict["pairs"]):
            assert STEM_PATTERN.match(pair["stem"]), (
                f"Pair {i}: stem '{pair['stem']}' violates pattern"
            )

    def test_mask_encoding_enum(self, manifest_dict: dict) -> None:
        """Contract: mask_encoding is one of the four enum values."""
        for i, pair in enumerate(manifest_dict["pairs"]):
            assert pair["mask_encoding"] in MASK_ENCODING_VALUES, (
                f"Pair {i}: invalid mask_encoding '{pair['mask_encoding']}'"
            )

    def test_full_manifest_validation(self, manifest_dict: dict) -> None:
        """Contract: complete manifest validates against all schema rules."""
        _validate_manifest_dict(manifest_dict)


class TestManifestContractReproducibility:
    """Re-running build_manifest produces byte-identical JSON (FR-004/FR-005)."""

    def test_deterministic_output(self, tmp_path: Path) -> None:
        """Two successive build+save cycles produce identical JSON."""
        gt_root = Path("data/groundtruth/post-processed")
        raw_root = Path("data/raw/Manga109s_released_2026_05_21/images")
        if not gt_root.is_dir() or not raw_root.is_dir():
            pytest.skip("Real dataset not present")

        p1 = tmp_path / "m1.json"
        p2 = tmp_path / "m2.json"

        m = build_manifest(gt_root, raw_root, run_id="det-test")
        save_manifest(m, p1)
        save_manifest(m, p2)

        assert p1.read_bytes() == p2.read_bytes(), "Manifest not byte-identical across runs"

    def test_load_manifest_roundtrip(self, tmp_path: Path) -> None:
        """save_manifest → load_manifest preserves all fields."""
        gt_root = Path("data/groundtruth/post-processed")
        raw_root = Path("data/raw/Manga109s_released_2026_05_21/images")
        if not gt_root.is_dir() or not raw_root.is_dir():
            pytest.skip("Real dataset not present")

        m = build_manifest(gt_root, raw_root, run_id="rt-test")
        p = tmp_path / "manifest.json"
        save_manifest(m, p)
        loaded = load_manifest(p)

        assert loaded.run_id == m.run_id
        assert loaded.total == m.total
        assert len(loaded.pairs) == len(m.pairs)
        for orig, loaded_p in zip(m.pairs, loaded.pairs):
            assert orig.manga == loaded_p.manga
            assert orig.stem == loaded_p.stem
            assert orig.mask_encoding == loaded_p.mask_encoding


class TestManifestContractEdgeCases:
    """Validate contract behavior on synthetic inputs."""

    def test_invalid_stem_rejected(self) -> None:
        """Stem not matching ^[0-9]{3}$ should fail validation."""
        bad_pair = {"manga": "Test", "stem": "00a", "raw_path": "/x",
                    "mask_path": "/y", "mask_encoding": "magenta-black"}
        assert not STEM_PATTERN.match(bad_pair["stem"])

    def test_invalid_mask_encoding_rejected(self) -> None:
        """Mask encoding outside enum should fail validation."""
        bad_encoding = "invalid-encoding"
        assert bad_encoding not in MASK_ENCODING_VALUES

    def test_missing_required_key_rejected(self) -> None:
        """Pair missing mask_encoding should fail validation."""
        incomplete = {"manga": "Test", "stem": "001", "raw_path": "/x", "mask_path": "/y"}
        required = {"manga", "stem", "raw_path", "mask_path", "mask_encoding"}
        assert not required.issubset(incomplete.keys())
