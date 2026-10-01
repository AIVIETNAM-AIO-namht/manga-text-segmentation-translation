"""T048 [P] [US6] Ablation outputs in separated namespace, untouched benchmark (FR-024, SC-011).

**TDD status**: This test file is written FIRST and MUST FAIL until
``src/manga_text_seg/ablation.py`` is implemented (Constitution Principle II).

## What an ablation run is (FR-024, SC-011)

The system supports supplementary sweeps over dilation configurations and
inpaint radii.  Ablation outputs MUST:

- Live in a clearly separated namespace: ``outputs/inpainting/ablation/<run_id>/``
- Be labelled as ablation (``"ablation": true`` in ``run.json``)
- Record their own configuration (``varied`` params + ``baseline_config``)
- Be excluded from the main benchmark comparison

## What an ablation run MUST NOT do

- Write anything into the main benchmark tree
  (``outputs/inpainting/<run_id>/`` without the ``ablation/`` segment)
- Modify any existing benchmark run directory
- Contaminate selection or board outputs with ablation data

## Module under test (not yet implemented)

    from manga_text_seg.ablation import run_ablation, AblationResult

All test imports are inside test functions so that the file can be *collected*
by pytest even before ``ablation.py`` exists (the import error surfaces as
a FAIL only when the test runs, not at collection time).

Spec file refs: specs/003-text-removal-inpainting/tasks.md T048,
spec.md FR-024, spec.md SC-011, data-model.md §AblationRun.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The ablation output root, relative to project output root.
#: FR-024: ``outputs/inpainting/ablation/<run_id>/``
ABLATION_NAMESPACE = "ablation"

#: The main benchmark output root, relative to project output root.
#: FR-024: ``outputs/inpainting/<run_id>/``
MAIN_BENCHMARK_NAMESPACE = "inpainting"

#: Four method identities (FR-026).
METHODS: list[str] = [
    "classical_baseline",
    "manga_text_segmentation",
    "comic_text_detector",
    "unetpp_efficientnetv2",
]

#: Sample ablation configurations to sweep.
ABLATION_CONFIGS: list[dict[str, Any]] = [
    {
        "dilation": {"kernel_shape": "ellipse", "kernel_size": [3, 3],
                     "iterations": 1},
        "inpaint_radius": 3,
    },
    {
        "dilation": {"kernel_shape": "ellipse", "kernel_size": [5, 5],
                     "iterations": 1},
        "inpaint_radius": 3,
    },
    {
        "dilation": {"kernel_shape": "ellipse", "kernel_size": [7, 7],
                     "iterations": 2},
        "inpaint_radius": 5,
    },
]

#: Baseline configuration used in the main benchmark (for comparison).
BASELINE_CONFIG: dict[str, Any] = {
    "dilation": {"kernel_shape": "ellipse", "kernel_size": [3, 3],
                 "iterations": 1},
    "inpaint_radius": 3,
}

#: A fixed run ID for the ablation run.
ABLATION_RUN_ID = "ablation_sweep_001"

#: A fixed run ID for a pre-existing main benchmark run.
MAIN_RUN_ID = "main_benchmark_001"


# ---------------------------------------------------------------------------
# Helper: create a fake main benchmark run tree
# ---------------------------------------------------------------------------

def _create_main_benchmark_tree(output_root: Path, run_id: str) -> Path:
    """Create a minimal main benchmark run tree with a run.json.

    Returns the path to the run directory.
    """
    run_dir = output_root / MAIN_BENCHMARK_NAMESPACE / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write a minimal run.json for the main benchmark
    run_json = {
        "run_id": run_id,
        "ablation": False,
        "config": BASELINE_CONFIG,
    }
    (run_dir / "run.json").write_text(
        json.dumps(run_json, indent=2), encoding="utf-8"
    )

    # Create a dummy output file for each method
    for method in METHODS:
        method_dir = run_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        (method_dir / "ARMS_000.png").write_bytes(b"\x89PNG_DUMMY")
        metadata = {"image_id": "ARMS/000", "method": method, "status": "ok"}
        (method_dir / "ARMS_000.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    return run_dir


def _snapshot_directory(directory: Path) -> dict[str, bytes]:
    """Take a byte-level snapshot of all files in a directory tree."""
    snapshot: dict[str, bytes] = {}
    if directory.exists():
        for f in sorted(directory.rglob("*")):
            if f.is_file():
                snapshot[str(f.relative_to(directory))] = f.read_bytes()
    return snapshot


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

def _import_run_ablation():
    """Import the module under test; fail with a clear TDD message if absent."""
    try:
        from manga_text_seg.ablation import run_ablation  # noqa: PLC0415
        return run_ablation
    except ImportError as exc:
        pytest.fail(
            f"ablation.py is not implemented yet — T048 MUST FAIL (TDD): {exc}"
        )


# ---------------------------------------------------------------------------
# Tests: Output path lives under ablation namespace (FR-024, SC-011)
# ---------------------------------------------------------------------------

class TestAblationNamespace:
    """Ablation outputs MUST live under ``outputs/inpainting/ablation/<run_id>/``."""

    def test_ablation_output_dir_is_under_ablation_namespace(
        self, tmp_path: Path,
    ) -> None:
        """The ablation run creates its directory under the ablation namespace."""
        run_ablation = _import_run_ablation()
        output_root = tmp_path / "outputs"
        result = run_ablation(
            output_root=output_root,
            run_id=ABLATION_RUN_ID,
            configs=ABLATION_CONFIGS,
            baseline_config=BASELINE_CONFIG,
        )
        ablation_dir = result.get("ablation_dir", result.get("output_dir"))
        ablation_path = Path(ablation_dir)
        # The ablation path must contain the 'ablation' segment
        rel_path = ablation_path.relative_to(output_root)
        parts = rel_path.parts
        assert ABLATION_NAMESPACE in parts, (
            f"Ablation output directory '{ablation_path}' does not contain "
            f"'{ABLATION_NAMESPACE}' segment. FR-024 requires ablation outputs "
            f"to live under outputs/inpainting/ablation/<run_id>/. "
            f"Relative parts: {parts}"
        )

    def test_ablation_output_never_in_main_tree(
        self, tmp_path: Path,
    ) -> None:
        """No ablation file appears directly under the main benchmark tree.

        The main benchmark tree is ``outputs/inpainting/<run_id>/`` (without
        the ``ablation/`` segment). The ablation output must be strictly
        separate (FR-024).
        """
        run_ablation = _import_run_ablation()
        output_root = tmp_path / "outputs"
        # Create a main benchmark tree first
        main_dir = _create_main_benchmark_tree(output_root, MAIN_RUN_ID)

        run_ablation(
            output_root=output_root,
            run_id=ABLATION_RUN_ID,
            configs=ABLATION_CONFIGS,
            baseline_config=BASELINE_CONFIG,
        )

        # Check no new files appeared in the main benchmark directory
        main_tree = output_root / MAIN_BENCHMARK_NAMESPACE / MAIN_RUN_ID
        ablation_tree = (
            output_root / MAIN_BENCHMARK_NAMESPACE / ABLATION_NAMESPACE
        )
        # Walk the main tree and ensure no file has 'ablation' in it
        for f in main_tree.rglob("*"):
            if f.is_file():
                rel = f.relative_to(main_tree)
                assert ABLATION_NAMESPACE not in str(rel).lower(), (
                    f"Ablation-related file '{f}' found inside the main "
                    f"benchmark tree '{main_tree}'. This violates FR-024."
                )

    def test_each_config_gets_its_own_subdirectory(
        self, tmp_path: Path,
    ) -> None:
        """Each ablation configuration produces a distinct output subdirectory."""
        run_ablation = _import_run_ablation()
        output_root = tmp_path / "outputs"
        result = run_ablation(
            output_root=output_root,
            run_id=ABLATION_RUN_ID,
            configs=ABLATION_CONFIGS,
            baseline_config=BASELINE_CONFIG,
        )
        config_dirs = result.get("config_dirs", [])
        assert len(config_dirs) == len(ABLATION_CONFIGS), (
            f"Expected {len(ABLATION_CONFIGS)} config directories, "
            f"got {len(config_dirs)}"
        )
        # All directories must be distinct
        assert len(set(str(d) for d in config_dirs)) == len(config_dirs), (
            f"Config directories are not all unique: {config_dirs}"
        )


# ---------------------------------------------------------------------------
# Tests: run.json metadata for ablation (TEAM-ASSIGNMENT T048 §2)
# ---------------------------------------------------------------------------

class TestAblationRunMetadata:
    """``run.json`` of an ablation run records ablation-specific fields."""

    def test_run_json_has_ablation_flag_true(
        self, tmp_path: Path,
    ) -> None:
        """``run.json`` has ``"ablation": true`` (TEAM-ASSIGNMENT T048 §2)."""
        run_ablation = _import_run_ablation()
        output_root = tmp_path / "outputs"
        result = run_ablation(
            output_root=output_root,
            run_id=ABLATION_RUN_ID,
            configs=ABLATION_CONFIGS,
            baseline_config=BASELINE_CONFIG,
        )
        ablation_dir = Path(
            result.get("ablation_dir", result.get("output_dir"))
        )
        run_json_path = ablation_dir / "run.json"
        assert run_json_path.exists(), (
            f"run.json not found at {run_json_path}"
        )
        with open(run_json_path, encoding="utf-8") as fh:
            run_data = json.load(fh)
        assert run_data.get("ablation") is True, (
            f"run.json 'ablation' field should be True, "
            f"got {run_data.get('ablation')!r}. "
            "Ablation runs must be explicitly labelled (FR-024)."
        )

    def test_run_json_records_varied_parameters(
        self, tmp_path: Path,
    ) -> None:
        """``run.json`` records which parameters were varied (T048 §2)."""
        run_ablation = _import_run_ablation()
        output_root = tmp_path / "outputs"
        result = run_ablation(
            output_root=output_root,
            run_id=ABLATION_RUN_ID,
            configs=ABLATION_CONFIGS,
            baseline_config=BASELINE_CONFIG,
        )
        ablation_dir = Path(
            result.get("ablation_dir", result.get("output_dir"))
        )
        run_json_path = ablation_dir / "run.json"
        with open(run_json_path, encoding="utf-8") as fh:
            run_data = json.load(fh)
        assert "varied" in run_data, (
            "run.json missing 'varied' field — the ablation run must record "
            "which parameters were swept (T048 §2)."
        )
        # 'varied' should be non-empty (we provided multiple configs)
        varied = run_data["varied"]
        assert varied, (
            f"run.json 'varied' field is empty: {varied!r}. "
            "It should list the parameters that were swept."
        )

    def test_run_json_records_baseline_config(
        self, tmp_path: Path,
    ) -> None:
        """``run.json`` records the baseline configuration for comparison (T048 §2)."""
        run_ablation = _import_run_ablation()
        output_root = tmp_path / "outputs"
        result = run_ablation(
            output_root=output_root,
            run_id=ABLATION_RUN_ID,
            configs=ABLATION_CONFIGS,
            baseline_config=BASELINE_CONFIG,
        )
        ablation_dir = Path(
            result.get("ablation_dir", result.get("output_dir"))
        )
        run_json_path = ablation_dir / "run.json"
        with open(run_json_path, encoding="utf-8") as fh:
            run_data = json.load(fh)
        assert "baseline_config" in run_data, (
            "run.json missing 'baseline_config' field — the ablation run must "
            "record the baseline configuration being compared against (T048 §2)."
        )

    def test_run_json_has_run_id(
        self, tmp_path: Path,
    ) -> None:
        """``run.json`` records the ablation run ID."""
        run_ablation = _import_run_ablation()
        output_root = tmp_path / "outputs"
        result = run_ablation(
            output_root=output_root,
            run_id=ABLATION_RUN_ID,
            configs=ABLATION_CONFIGS,
            baseline_config=BASELINE_CONFIG,
        )
        ablation_dir = Path(
            result.get("ablation_dir", result.get("output_dir"))
        )
        run_json_path = ablation_dir / "run.json"
        with open(run_json_path, encoding="utf-8") as fh:
            run_data = json.load(fh)
        assert run_data.get("run_id") == ABLATION_RUN_ID, (
            f"run.json 'run_id' should be '{ABLATION_RUN_ID}', "
            f"got {run_data.get('run_id')!r}"
        )


# ---------------------------------------------------------------------------
# Tests: Main benchmark tree is untouched (T048 §3, SC-011)
# ---------------------------------------------------------------------------

class TestMainBenchmarkUntouched:
    """The main benchmark run directory must not be modified by an ablation run."""

    def test_main_tree_files_unchanged_after_ablation(
        self, tmp_path: Path,
    ) -> None:
        """Every file in the main benchmark tree is byte-identical before and after.

        This is the core invariant of T048 §3: ablation must NEVER modify
        existing benchmark data.
        """
        run_ablation = _import_run_ablation()
        output_root = tmp_path / "outputs"
        main_dir = _create_main_benchmark_tree(output_root, MAIN_RUN_ID)

        # Snapshot before
        snapshot_before = _snapshot_directory(main_dir)
        assert len(snapshot_before) > 0, (
            "Main benchmark tree has no files (fixture setup error)"
        )

        # Run the ablation
        run_ablation(
            output_root=output_root,
            run_id=ABLATION_RUN_ID,
            configs=ABLATION_CONFIGS,
            baseline_config=BASELINE_CONFIG,
        )

        # Snapshot after
        snapshot_after = _snapshot_directory(main_dir)

        # File set must be identical (no additions, no deletions)
        assert set(snapshot_before.keys()) == set(snapshot_after.keys()), (
            f"Main benchmark file set changed after ablation. "
            f"Before: {sorted(snapshot_before.keys())}, "
            f"After: {sorted(snapshot_after.keys())}"
        )

        # Each file must be byte-identical
        for filename, content_before in snapshot_before.items():
            content_after = snapshot_after[filename]
            assert content_before == content_after, (
                f"File '{filename}' in main benchmark tree was modified by "
                f"the ablation run. This violates T048 §3 / SC-011."
            )

    def test_no_new_files_in_main_tree(
        self, tmp_path: Path,
    ) -> None:
        """No new files are created inside the main benchmark tree."""
        run_ablation = _import_run_ablation()
        output_root = tmp_path / "outputs"
        main_dir = _create_main_benchmark_tree(output_root, MAIN_RUN_ID)

        files_before = set(
            str(f.relative_to(main_dir)) for f in main_dir.rglob("*")
            if f.is_file()
        )

        run_ablation(
            output_root=output_root,
            run_id=ABLATION_RUN_ID,
            configs=ABLATION_CONFIGS,
            baseline_config=BASELINE_CONFIG,
        )

        files_after = set(
            str(f.relative_to(main_dir)) for f in main_dir.rglob("*")
            if f.is_file()
        )

        new_files = files_after - files_before
        assert not new_files, (
            f"New file(s) appeared in main benchmark tree after ablation: "
            f"{new_files}. Ablation must not write to the main tree (T048 §3)."
        )


# ---------------------------------------------------------------------------
# Tests: Configuration recording per ablation config
# ---------------------------------------------------------------------------

class TestAblationConfigRecording:
    """Each ablation configuration records its exact parameters (FR-024)."""

    def test_each_config_has_recorded_metadata(
        self, tmp_path: Path,
    ) -> None:
        """Each configuration directory contains a metadata file with its config."""
        run_ablation = _import_run_ablation()
        output_root = tmp_path / "outputs"
        result = run_ablation(
            output_root=output_root,
            run_id=ABLATION_RUN_ID,
            configs=ABLATION_CONFIGS,
            baseline_config=BASELINE_CONFIG,
        )
        config_dirs = result.get("config_dirs", [])
        for config_dir in config_dirs:
            config_dir = Path(config_dir)
            # Look for a config or metadata JSON file
            json_files = list(config_dir.glob("*.json"))
            assert len(json_files) > 0, (
                f"Configuration directory '{config_dir}' has no JSON metadata "
                "file. Each ablation config must record its parameters (FR-024)."
            )

    def test_ablation_configs_match_input(
        self, tmp_path: Path,
    ) -> None:
        """The number of ablation configuration outputs matches the input count."""
        run_ablation = _import_run_ablation()
        output_root = tmp_path / "outputs"
        result = run_ablation(
            output_root=output_root,
            run_id=ABLATION_RUN_ID,
            configs=ABLATION_CONFIGS,
            baseline_config=BASELINE_CONFIG,
        )
        config_dirs = result.get("config_dirs", [])
        assert len(config_dirs) == len(ABLATION_CONFIGS), (
            f"Expected {len(ABLATION_CONFIGS)} ablation config outputs, "
            f"got {len(config_dirs)}"
        )


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------

class TestAblationEdgeCases:
    """Edge cases for ablation runs."""

    def test_single_config_ablation_still_separated(
        self, tmp_path: Path,
    ) -> None:
        """Even a single-config ablation stays in the ablation namespace."""
        run_ablation = _import_run_ablation()
        output_root = tmp_path / "outputs"
        single_config = [ABLATION_CONFIGS[0]]
        result = run_ablation(
            output_root=output_root,
            run_id=ABLATION_RUN_ID,
            configs=single_config,
            baseline_config=BASELINE_CONFIG,
        )
        ablation_dir = Path(
            result.get("ablation_dir", result.get("output_dir"))
        )
        rel = ablation_dir.relative_to(output_root)
        assert ABLATION_NAMESPACE in rel.parts, (
            f"Single-config ablation output '{ablation_dir}' is not under "
            f"the '{ABLATION_NAMESPACE}' namespace."
        )

    def test_ablation_with_empty_configs_raises_or_returns_empty(
        self, tmp_path: Path,
    ) -> None:
        """An ablation with zero configs either raises ValueError or returns
        an empty config_dirs list — but never writes to the main tree."""
        run_ablation = _import_run_ablation()
        output_root = tmp_path / "outputs"
        main_dir = _create_main_benchmark_tree(output_root, MAIN_RUN_ID)
        snapshot_before = _snapshot_directory(main_dir)

        try:
            result = run_ablation(
                output_root=output_root,
                run_id=ABLATION_RUN_ID,
                configs=[],
                baseline_config=BASELINE_CONFIG,
            )
            # If it succeeds, config_dirs must be empty
            config_dirs = result.get("config_dirs", [])
            assert config_dirs == [], (
                f"Empty configs should produce empty config_dirs, "
                f"got {config_dirs}"
            )
        except (ValueError, KeyError):
            pass  # Raising is acceptable for empty configs

        # Main tree must be untouched regardless
        snapshot_after = _snapshot_directory(main_dir)
        assert snapshot_before == snapshot_after, (
            "Main benchmark tree was modified even though ablation had "
            "zero configs."
        )
