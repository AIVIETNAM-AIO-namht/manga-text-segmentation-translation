"""T075 recursive, artifact-wide determinism checks (FR-058, SC-012)."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from manga_text_seg.config import load_translation_config
from manga_text_seg.translation.pipeline import run_batch
from manga_text_seg.translation.translate import FixtureTranslationProvider

ROOT = Path(__file__).resolve().parents[2]
_TIMING_KEYS = {
    "stage_timings",
    "processing_time_seconds",
    "ocr_time_seconds",
    "translation_time_seconds",
    "render_time_seconds",
}
_TIMESTAMP = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[T ][0-2]\d:[0-5]\d:[0-6]\d"
    r"(?:\.\d+)?(?:Z|[+-][0-2]\d:?\d{2})?)?\b"
)
_UUID = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-8][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
)
_RANDOM_ID_KEY = re.compile(r"(?:random|nonce|uuid|guid|generated[_-]?id)", re.I)
_ABSOLUTE_PATH = re.compile(r"(?:[A-Za-z]:\\|/(?:Users|home|tmp|var)/)")


def _config(bundle, output_root: Path, cache_root: Path):
    base = load_translation_config(ROOT / "configs" / "translation.json")
    provider = replace(
        base.provider,
        name="deterministic-fixture",
        model="fixture-v1",
        version="1",
        secret_env_var=None,
        deterministic=True,
        cache=replace(base.provider.cache, root=cache_root, enabled=False),
    )
    return replace(
        base,
        manifest_path=bundle["manifest"],
        output_roots=replace(
            base.output_roots, translation=output_root, cache=cache_root
        ),
        provider=provider,
    )


def _run(config, bundle, run_id):
    return run_batch(
        config,
        run_id,
        repo_root=ROOT,
        provider=FixtureTranslationProvider({"fixture OCR": "fixture translation"}),
        recognizer=lambda _image: "fixture OCR",
        prediction_root=bundle["prediction_mask"].parents[2],
        inpainting_root=bundle["inpainted"].parents[3],
    )


def _normalize_allowed(value: Any, *, pointer: str, relative_path: str) -> Any:
    """Normalize only documented elapsed values and each run's unique ID."""
    if relative_path == "run.json" and pointer == "$/run_id":
        return "<RUN_ID>"
    if relative_path == "errors.json" and pointer == "$/run_id":
        return "<RUN_ID>"
    if relative_path.endswith("/metadata.json") and pointer == "$/run_id":
        return "<RUN_ID>"
    key = pointer.rsplit("/", 1)[-1]
    if key in _TIMING_KEYS:
        return "<ELAPSED_TIME>"
    if isinstance(value, dict):
        return {
            item_key: _normalize_allowed(
                item,
                pointer=f"{pointer}/{item_key}",
                relative_path=relative_path,
            )
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _normalize_allowed(
                item,
                pointer=f"{pointer}/{index}",
                relative_path=relative_path,
            )
            for index, item in enumerate(value)
        ]
    return value


def _walk(value: Any, pointer: str = "$"):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk(child, f"{pointer}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{pointer}/{index}")
    else:
        yield pointer, value


def _clock_or_uuid_findings(value: Any, artifact: str) -> list[str]:
    findings = []
    for pointer, key in _mapping_keys(value):
        if _RANDOM_ID_KEY.search(key):
            findings.append(
                f"{artifact}#{pointer}: random/UUID identifier field {key!r}"
            )
    for pointer, scalar in _walk(value):
        if not isinstance(scalar, str):
            continue
        if _TIMESTAMP.search(scalar):
            findings.append(f"{artifact}#{pointer}: timestamp {scalar!r}")
        if _UUID.search(scalar):
            findings.append(f"{artifact}#{pointer}: UUID-like value {scalar!r}")
    return findings


def _mapping_keys(value: Any, pointer: str = "$"):
    if isinstance(value, dict):
        for key, child in value.items():
            child_pointer = f"{pointer}/{key}"
            yield child_pointer, str(key)
            yield from _mapping_keys(child, child_pointer)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _mapping_keys(child, f"{pointer}/{index}")


def _absolute_path_values(value: Any, artifact: str) -> dict[str, str]:
    """Record path-bearing values so cross-run machine/temp paths are compared."""
    return {
        f"{artifact}#{pointer}": scalar
        for pointer, scalar in _walk(value)
        if isinstance(scalar, str) and _ABSOLUTE_PATH.search(scalar)
    }


def _json_artifacts(run_dir: Path) -> dict[str, Any]:
    artifacts = {}
    for path in sorted(run_dir.rglob("*.json")):
        relative = path.relative_to(run_dir).as_posix()
        artifacts[relative] = json.loads(path.read_text(encoding="utf-8"))
    return artifacts


def _normalized_tree(run_dir: Path) -> dict[str, bytes]:
    normalized = {}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir).as_posix()
        if path.suffix.lower() == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            value = _normalize_allowed(value, pointer="$", relative_path=relative)
            normalized[relative] = (
                json.dumps(value, indent=2, ensure_ascii=False) + "\n"
            ).encode("utf-8")
        else:
            normalized[relative] = path.read_bytes()
    return normalized


def _assert_tree_equal(first: dict[str, bytes], second: dict[str, bytes]) -> None:
    assert set(first) == set(second), (
        f"artifact set differs; only RUN A={sorted(set(first) - set(second))}, "
        f"only RUN B={sorted(set(second) - set(first))}"
    )
    for relative in sorted(first):
        if first[relative] != second[relative]:
            raise AssertionError(f"deterministic artifact differs: {relative}")


def test_nested_nondeterminism_diagnostic_identifies_artifact_and_key() -> None:
    nested = {
        "regions": [
            {
                "provenance": {
                    "observed_at": "2026-10-04T12:30:45Z",
                    "random_id": "e84d",
                }
            }
        ]
    }
    findings = _clock_or_uuid_findings(nested, "ARMS/001/metadata.json")

    assert findings == [
        "ARMS/001/metadata.json#$/regions/0/provenance/random_id: "
        "random/UUID identifier field 'random_id'",
        "ARMS/001/metadata.json#$/regions/0/provenance/observed_at: "
        "timestamp '2026-10-04T12:30:45Z'",
    ]


def test_equivalent_deterministic_fixture_runs_match_recursively(
    tmp_path, translation_bundle
):
    bundle = translation_bundle()
    config = _config(bundle, tmp_path / "outputs", tmp_path / "cache")
    first = _run(config, bundle, "determinism-a")
    second = _run(config, bundle, "determinism-b")

    assert first.run_record.provider["deterministic"] is True
    assert first.run_record.run_id == "determinism-a"
    assert second.run_record.run_id == "determinism-b"

    first_json = _json_artifacts(first.run_dir)
    second_json = _json_artifacts(second.run_dir)
    assert set(first_json) == set(second_json), "JSON artifact paths differ"
    for artifact in sorted(first_json):
        assert not _clock_or_uuid_findings(
            first_json[artifact], artifact
        ), "unexpected timestamp/UUID in deterministic output: " + "; ".join(
            _clock_or_uuid_findings(first_json[artifact], artifact)
        )

    # Resolved configuration legitimately contains absolute paths. They remain
    # part of the comparison, and their exact pointer/value pairs must agree.
    first_paths = {
        item: value
        for artifact, document in first_json.items()
        for item, value in _absolute_path_values(document, artifact).items()
    }
    second_paths = {
        item: value
        for artifact, document in second_json.items()
        for item, value in _absolute_path_values(document, artifact).items()
    }
    assert first_paths == second_paths, "absolute/temp/machine path values differ"

    _assert_tree_equal(
        _normalized_tree(first.run_dir), _normalized_tree(second.run_dir)
    )
