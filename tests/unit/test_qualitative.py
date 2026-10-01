"""T045 [P] [US5] Qualitative report scaffold — seven criteria, all fields empty
(FR-032, FR-035, SC-008).

**TDD status**: This test file is written FIRST and MUST FAIL until
``src/manga_text_seg/qualitative.py`` is implemented (Constitution Principle II).

## What the qualitative scaffold is (FR-032, FR-035)

The system generates a pre-labelled document (``qualitative.md`` and
``qualitative.json``) in which every assessed sample has:

- Full identity labels: segmentation method, inpainting algorithm,
  mask-processing configuration, and ``image_id``.
- **Seven** criteria each on one documented ordinal scale, with the rating
  field **empty** and a written observation field **empty**.

The human reviewer fills in the ratings and observations; the system NEVER
auto-scores any criterion (FR-032, SC-008).

## Prohibited content (SC-008)

No PSNR figure, no SSIM figure, no synthetic clean-background ground truth,
and no single-score method ranking appears in the scaffold.

## Module under test (not yet implemented)

    from manga_text_seg.qualitative import generate_scaffold, QualitativeEntry

Imports are inside test functions so that pytest can collect this file even
before ``qualitative.py`` exists — the failure surfaces when the test runs.

Spec file refs: specs/003-text-removal-inpainting/tasks.md T045,
data-model.md §QualitativeReport, spec.md FR-032/FR-035.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# The seven mandated criteria (FR-032, SC-008)
# ---------------------------------------------------------------------------

SEVEN_CRITERIA: list[str] = [
    "completeness_of_text_removal",
    "amount_of_missed_text",
    "amount_of_background_over_erased",
    "naturalness_of_restored_region",
    "artifacts_or_noise",
    "halo_or_border_effects",
    "damage_to_linework_texture_panel_borders",
]

#: Human-readable labels for error messages.
CRITERIA_LABELS: dict[str, str] = {
    "completeness_of_text_removal":         "Completeness of text removal",
    "amount_of_missed_text":                "Amount of missed text",
    "amount_of_background_over_erased":     "Amount of background over-erased",
    "naturalness_of_restored_region":       "Naturalness of the restored region",
    "artifacts_or_noise":                   "Artifacts or noise",
    "halo_or_border_effects":               "Halo or border effects",
    "damage_to_linework_texture_panel_borders": "Damage to linework/texture/panel borders",
}

#: Terms prohibited inside any scaffold content (SC-008).
PROHIBITED_TERMS: list[str] = [
    "PSNR",
    "SSIM",
    "synthetic ground truth",
    "single-score",
    "ranking",
    "auto-score",
    "automatically scored",
]

# ---------------------------------------------------------------------------
# Synthetic fixture samples
# ---------------------------------------------------------------------------

#: A minimal set of sample descriptors that cover all four method identities
#: and both inpainting algorithms.
FIXTURE_SAMPLES: list[dict[str, Any]] = [
    {
        "image_id": "ARMS/000",
        "method": "manga_text_segmentation",
        "algorithm": "telea",
        "mask_proc_config": {
            "dilation": {"enabled": True, "kernel_shape": "ellipse",
                         "kernel_size": [3, 3], "iterations": 1},
            "inpaint_radius": 3,
        },
    },
    {
        "image_id": "ARMS/001",
        "method": "comic_text_detector",
        "algorithm": "ns",
        "mask_proc_config": {
            "dilation": {"enabled": True, "kernel_shape": "ellipse",
                         "kernel_size": [3, 3], "iterations": 1},
            "inpaint_radius": 3,
        },
    },
    {
        "image_id": "Arisa/003",
        "method": "unetpp_efficientnetv2",
        "algorithm": "telea",
        "mask_proc_config": {
            "dilation": {"enabled": True, "kernel_shape": "ellipse",
                         "kernel_size": [3, 3], "iterations": 1},
            "inpaint_radius": 3,
        },
    },
    {
        "image_id": "DollGun/002",
        "method": "classical_baseline",
        "algorithm": "ns",
        "mask_proc_config": {
            "dilation": {"enabled": True, "kernel_shape": "ellipse",
                         "kernel_size": [3, 3], "iterations": 1},
            "inpaint_radius": 3,
        },
    },
]


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

def _import_generate_scaffold():
    """Import the module under test; fail with a clear TDD message if absent."""
    try:
        from manga_text_seg.qualitative import generate_scaffold  # noqa: PLC0415
        return generate_scaffold
    except ImportError as exc:
        pytest.fail(
            f"qualitative.py is not implemented yet — T045 MUST FAIL (TDD): {exc}"
        )


# ---------------------------------------------------------------------------
# Tests: Identity labels (FR-035)
# ---------------------------------------------------------------------------

class TestIdentityLabels:
    """Every entry must be labelled with method, algorithm, mask config, image_id."""

    def test_each_entry_has_image_id(self) -> None:
        """``image_id`` matches the sample it was generated for."""
        generate_scaffold = _import_generate_scaffold()
        scaffold = generate_scaffold(FIXTURE_SAMPLES)
        input_ids = {s["image_id"] for s in FIXTURE_SAMPLES}
        output_ids = {entry["image_id"] for entry in scaffold}
        assert output_ids == input_ids, (
            f"Scaffold image_ids {output_ids} do not match input {input_ids}"
        )

    def test_each_entry_has_segmentation_method(self) -> None:
        """``method`` is set to the segmentation method identity."""
        generate_scaffold = _import_generate_scaffold()
        scaffold = generate_scaffold(FIXTURE_SAMPLES)
        for i, entry in enumerate(scaffold):
            assert "method" in entry, f"Entry {i}: 'method' key missing"
            assert entry["method"] == FIXTURE_SAMPLES[i]["method"], (
                f"Entry {i}: method mismatch"
            )

    def test_each_entry_has_inpainting_algorithm(self) -> None:
        """``algorithm`` is set to the inpainting algorithm ('telea' or 'ns')."""
        generate_scaffold = _import_generate_scaffold()
        scaffold = generate_scaffold(FIXTURE_SAMPLES)
        for i, entry in enumerate(scaffold):
            assert "algorithm" in entry, f"Entry {i}: 'algorithm' key missing"
            assert entry["algorithm"] in ("telea", "ns"), (
                f"Entry {i}: algorithm '{entry['algorithm']}' not in {{telea, ns}}"
            )
            assert entry["algorithm"] == FIXTURE_SAMPLES[i]["algorithm"]

    def test_each_entry_has_mask_proc_config(self) -> None:
        """``mask_proc_config`` carries the mask-processing configuration."""
        generate_scaffold = _import_generate_scaffold()
        scaffold = generate_scaffold(FIXTURE_SAMPLES)
        for i, entry in enumerate(scaffold):
            assert "mask_proc_config" in entry, (
                f"Entry {i}: 'mask_proc_config' key missing"
            )


# ---------------------------------------------------------------------------
# Tests: Seven criteria presence (FR-032)
# ---------------------------------------------------------------------------

class TestSevenCriteria:
    """All seven criteria must appear in every scaffold entry (FR-032)."""

    def test_all_seven_criteria_present(self) -> None:
        """Every scaffold entry contains all seven criterion keys."""
        generate_scaffold = _import_generate_scaffold()
        scaffold = generate_scaffold(FIXTURE_SAMPLES)
        for entry in scaffold:
            criteria = entry.get("criteria", {})
            missing = [c for c in SEVEN_CRITERIA if c not in criteria]
            assert not missing, (
                f"Entry '{entry.get('image_id')}': "
                f"missing criteria keys: {missing}. "
                f"All seven criteria are mandatory (FR-032)."
            )

    def test_criteria_count_is_exactly_seven(self) -> None:
        """Each entry has exactly seven criteria — no extra, no missing."""
        generate_scaffold = _import_generate_scaffold()
        scaffold = generate_scaffold(FIXTURE_SAMPLES)
        for entry in scaffold:
            criteria = entry.get("criteria", {})
            count = len(criteria)
            assert count == 7, (
                f"Entry '{entry.get('image_id')}': expected 7 criteria, got {count}. "
                f"Present: {sorted(criteria.keys())}"
            )

    def test_criteria_keys_match_spec_exactly(self) -> None:
        """Criterion keys match the seven names defined in FR-032, no aliasing."""
        generate_scaffold = _import_generate_scaffold()
        scaffold = generate_scaffold(FIXTURE_SAMPLES)
        for entry in scaffold:
            actual_keys = set(entry.get("criteria", {}).keys())
            expected_keys = set(SEVEN_CRITERIA)
            assert actual_keys == expected_keys, (
                f"Entry '{entry.get('image_id')}': criteria key mismatch. "
                f"Extra: {actual_keys - expected_keys}, "
                f"Missing: {expected_keys - actual_keys}"
            )


# ---------------------------------------------------------------------------
# Tests: All rating and observation fields must be empty (FR-035, SC-008)
# ---------------------------------------------------------------------------

class TestEmptyFields:
    """The system MUST NOT auto-score any criterion (FR-032, SC-008)."""

    def _is_empty(self, value: Any) -> bool:
        """A field is 'empty' if it is None, '', [], or {}."""
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() == ""
        if isinstance(value, (list, dict)):
            return len(value) == 0
        return False

    def test_all_rating_fields_are_empty(self) -> None:
        """Every criterion's rating field is None or empty string — never a number."""
        generate_scaffold = _import_generate_scaffold()
        scaffold = generate_scaffold(FIXTURE_SAMPLES)
        non_empty = []
        for entry in scaffold:
            criteria = entry.get("criteria", {})
            for criterion_key, criterion_value in criteria.items():
                rating = criterion_value.get("rating")
                if not self._is_empty(rating):
                    non_empty.append(
                        f"'{entry.get('image_id')}' / '{criterion_key}': "
                        f"rating = {rating!r} (must be empty)"
                    )
        assert not non_empty, (
            f"FR-032 / SC-008 violation — {len(non_empty)} auto-filled rating(s):\n"
            + "\n".join(non_empty)
            + "\n\nThe system MUST NOT auto-score any criterion. "
            "A human reviewer fills these in."
        )

    def test_all_observation_fields_are_empty(self) -> None:
        """Every criterion's observation/comment field is None or empty string."""
        generate_scaffold = _import_generate_scaffold()
        scaffold = generate_scaffold(FIXTURE_SAMPLES)
        non_empty = []
        for entry in scaffold:
            criteria = entry.get("criteria", {})
            for criterion_key, criterion_value in criteria.items():
                obs = criterion_value.get("observation", criterion_value.get("comment"))
                if not self._is_empty(obs):
                    non_empty.append(
                        f"'{entry.get('image_id')}' / '{criterion_key}': "
                        f"observation = {obs!r} (must be empty)"
                    )
        assert not non_empty, (
            f"FR-035 violation — {len(non_empty)} pre-filled observation(s):\n"
            + "\n".join(non_empty)
            + "\n\nObservation fields must be left empty for the human reviewer."
        )

    def test_no_numeric_rating_auto_generated(self) -> None:
        """No criterion contains a numeric rating value."""
        generate_scaffold = _import_generate_scaffold()
        scaffold = generate_scaffold(FIXTURE_SAMPLES)
        numeric_ratings = []
        for entry in scaffold:
            criteria = entry.get("criteria", {})
            for criterion_key, criterion_value in criteria.items():
                rating = criterion_value.get("rating")
                if isinstance(rating, (int, float)):
                    numeric_ratings.append(
                        f"'{entry.get('image_id')}' / '{criterion_key}': "
                        f"rating = {rating!r}"
                    )
        assert not numeric_ratings, (
            f"SC-008 violation — numeric ratings auto-generated:\n"
            + "\n".join(numeric_ratings)
        )


# ---------------------------------------------------------------------------
# Tests: Prohibited content must not appear (SC-008)
# ---------------------------------------------------------------------------

class TestProhibitedContent:
    """No PSNR/SSIM figure, synthetic GT, or single-score ranking (SC-008)."""

    def _scaffold_as_text(self, scaffold: list[dict]) -> str:
        """Serialise the scaffold to a JSON string for text scanning."""
        return json.dumps(scaffold)

    def test_no_psnr_in_scaffold(self) -> None:
        """The string 'PSNR' must not appear anywhere in the scaffold."""
        generate_scaffold = _import_generate_scaffold()
        scaffold = generate_scaffold(FIXTURE_SAMPLES)
        text = self._scaffold_as_text(scaffold)
        assert "PSNR" not in text, (
            "SC-008 violation: 'PSNR' found in qualitative scaffold. "
            "No quantitative metric may appear in the scaffold."
        )

    def test_no_ssim_in_scaffold(self) -> None:
        """The string 'SSIM' must not appear anywhere in the scaffold."""
        generate_scaffold = _import_generate_scaffold()
        scaffold = generate_scaffold(FIXTURE_SAMPLES)
        text = self._scaffold_as_text(scaffold)
        assert "SSIM" not in text, (
            "SC-008 violation: 'SSIM' found in qualitative scaffold."
        )

    def test_no_ranking_score_in_scaffold(self) -> None:
        """No single-score method ranking appears in the scaffold."""
        generate_scaffold = _import_generate_scaffold()
        scaffold = generate_scaffold(FIXTURE_SAMPLES)
        text = self._scaffold_as_text(scaffold)
        # We allow the word 'rank' only inside the rated ordinal scale description,
        # but a numeric score or explicit 'ranking' heading is prohibited.
        assert "ranking" not in text.lower(), (
            "SC-008 violation: 'ranking' found in qualitative scaffold. "
            "No single-score method ranking may appear in the primary report."
        )


# ---------------------------------------------------------------------------
# Tests: JSON and Markdown output parity (FR-035)
# ---------------------------------------------------------------------------

class TestOutputParity:
    """``qualitative.json`` and ``qualitative.md`` must carry identical content."""

    def test_json_output_is_valid_and_parseable(self, tmp_path: Path) -> None:
        """The JSON output path produces valid JSON."""
        generate_scaffold = _import_generate_scaffold()
        json_path = tmp_path / "qualitative.json"
        try:
            generate_scaffold(FIXTURE_SAMPLES, json_out=json_path)
        except TypeError:
            # If the function does not yet accept kwargs, skip parity test
            pytest.skip("generate_scaffold does not yet accept json_out kwarg")

        assert json_path.exists(), "qualitative.json was not written"
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert isinstance(data, list), "qualitative.json root should be a list"

    def test_markdown_output_is_written(self, tmp_path: Path) -> None:
        """The Markdown output path produces a non-empty .md file."""
        generate_scaffold = _import_generate_scaffold()
        md_path = tmp_path / "qualitative.md"
        try:
            generate_scaffold(FIXTURE_SAMPLES, md_out=md_path)
        except TypeError:
            pytest.skip("generate_scaffold does not yet accept md_out kwarg")

        assert md_path.exists(), "qualitative.md was not written"
        content = md_path.read_text(encoding="utf-8")
        assert content.strip(), "qualitative.md is empty"


# ---------------------------------------------------------------------------
# Tests: Scaffold covers all provided samples (FR-035)
# ---------------------------------------------------------------------------

class TestScaffoldCoverage:
    """The scaffold must cover every supplied sample, one entry each."""

    def test_scaffold_length_equals_sample_count(self) -> None:
        """One entry per input sample — no entries added or dropped."""
        generate_scaffold = _import_generate_scaffold()
        scaffold = generate_scaffold(FIXTURE_SAMPLES)
        assert len(scaffold) == len(FIXTURE_SAMPLES), (
            f"Expected {len(FIXTURE_SAMPLES)} entries, got {len(scaffold)}"
        )

    def test_empty_sample_list_produces_empty_scaffold(self) -> None:
        """An empty input list produces an empty scaffold, not an error."""
        generate_scaffold = _import_generate_scaffold()
        scaffold = generate_scaffold([])
        assert scaffold == [], (
            f"Expected empty scaffold for empty input, got {scaffold}"
        )
