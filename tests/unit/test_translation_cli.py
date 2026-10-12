"""T078 user-facing Spec 004 CLI invocation tests."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

from manga_text_seg.config import load_translation_config
from manga_text_seg.translation.pipeline import run_intake

ROOT = Path(__file__).resolve().parents[2]


def _write_cli_config(tmp_path: Path, bundle: dict) -> Path:
    raw = json.loads((ROOT / "configs" / "translation.json").read_text())
    raw["manifest_path"] = str(bundle["manifest"])
    raw["output_roots"]["translation"] = str(tmp_path / "outputs")
    raw["output_roots"]["cache"] = str(tmp_path / "cache")
    raw["provider"]["name"] = "fixture"
    raw["provider"]["secret_env_var"] = None
    raw["provider"]["deterministic"] = True
    raw["provider"]["cache"]["root"] = str(tmp_path / "cache")
    # The CLI derives repo_root from config_path.parent.parent. Give it a
    # repo-shaped parent holding the committed Spec 2 identity records, which
    # intake resolves relative to repo_root.
    fixture_repo = tmp_path / "fixture-repo"
    (fixture_repo / "benchmark").mkdir(parents=True)
    (fixture_repo / "configs").mkdir()
    for name in ("page-list.json", "image-identity.json"):
        shutil.copyfile(ROOT / "benchmark" / name, fixture_repo / "benchmark" / name)
    config_path = fixture_repo / "configs" / "translation.json"
    config_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return config_path


def _invoke_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.setdefault("MANGA_TEXT_SEG_INTEGRATION", "1")
    return subprocess.run(
        ["uv", "run", "--locked", "--python", "3.12", "manga-text-seg", *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _seed_valid_run(tmp_path: Path, translation_bundle):
    bundle = translation_bundle()
    config_path = _write_cli_config(tmp_path, bundle)
    config = load_translation_config(config_path)
    config = replace(
        config,
        provider=replace(
            config.provider,
            cache=replace(config.provider.cache, root=tmp_path / "cache"),
        ),
    )
    result = run_intake(
        config,
        "cli-valid-run",
        repo_root=ROOT,
        image_ids=(bundle["image_id"],),
        prediction_root=bundle["prediction_mask"].parents[2],
        inpainting_root=bundle["inpainted"].parents[3],
    )
    return bundle, config_path, result


def test_cli_valid_stage_invocation_writes_and_updates_real_artifacts(
    tmp_path: Path, translation_bundle
) -> None:
    bundle, config_path, result = _seed_valid_run(tmp_path, translation_bundle)

    completed = _invoke_cli(
        "translate",
        "--config",
        str(config_path),
        "--run",
        "cli-valid-run",
        "--image-id",
        bundle["image_id"],
        "--stage",
        "regions",
    )

    assert completed.returncode == 0, completed.stderr
    assert f"Re-ran regions for {bundle['image_id']}" in completed.stdout
    page_dir = result.pages[0].page_dir
    regions = json.loads((page_dir / "regions.json").read_text(encoding="utf-8"))
    metadata = json.loads((page_dir / "metadata.json").read_text(encoding="utf-8"))
    assert regions["regions"][0]["region_id"] == metadata["regions"][0]["region_id"]
    assert metadata["image_id"] == bundle["image_id"]
    assert metadata["run_id"] == "cli-valid-run"
    assert metadata["segmentation_method"] == "unetpp_efficientnetv2"


def test_cli_rejects_missing_required_run_before_creating_outputs(
    tmp_path: Path,
) -> None:
    completed = _invoke_cli("translate", "--config", str(tmp_path / "missing.json"))

    assert completed.returncode == 2
    assert "--run" in completed.stderr
    assert not (tmp_path / "outputs").exists()


def test_cli_rejects_nonexistent_config_without_fabricating_artifacts(
    tmp_path: Path,
) -> None:
    completed = _invoke_cli(
        "translate",
        "--config",
        str(tmp_path / "does-not-exist.json"),
        "--run",
        "nonexistent-config-run",
        "--stage",
        "intake",
    )

    assert completed.returncode == 2
    assert "Configuration error:" in completed.stderr
    assert not list(tmp_path.rglob("run.json"))


def test_cli_duplicate_intake_refuses_without_changing_existing_run(
    tmp_path: Path, translation_bundle
) -> None:
    bundle, config_path, result = _seed_valid_run(tmp_path, translation_bundle)
    before = _tree_bytes(result.run_dir)

    completed = _invoke_cli(
        "translate",
        "--config",
        str(config_path),
        "--run",
        "cli-valid-run",
        "--image-id",
        bundle["image_id"],
        "--stage",
        "intake",
    )

    assert completed.returncode == 2
    assert "Refusing to overwrite:" in completed.stderr
    assert _tree_bytes(result.run_dir) == before


def test_cli_explicit_overwrite_is_scoped_to_target_run(
    tmp_path: Path, translation_bundle
) -> None:
    bundle, config_path, result = _seed_valid_run(tmp_path, translation_bundle)
    config = load_translation_config(config_path)
    sibling = run_intake(
        config,
        "cli-sibling-run",
        repo_root=ROOT,
        image_ids=(bundle["image_id"],),
        prediction_root=bundle["prediction_mask"].parents[2],
        inpainting_root=bundle["inpainted"].parents[3],
    )
    sibling_before = _tree_bytes(sibling.run_dir)
    target_before = _tree_bytes(result.run_dir)

    completed = _invoke_cli(
        "translate",
        "--config",
        str(config_path),
        "--run",
        "cli-valid-run",
        "--image-id",
        bundle["image_id"],
        "--stage",
        "intake",
        "--overwrite",
    )

    assert completed.returncode == 0, completed.stderr
    assert _tree_bytes(result.run_dir) != target_before
    assert _tree_bytes(sibling.run_dir) == sibling_before
    overwritten_record = json.loads(
        (result.run_dir / "run.json").read_text(encoding="utf-8")
    )
    assert overwritten_record["run_id"] == "cli-valid-run"
    assert overwritten_record["page_statuses"][bundle["image_id"]] == (
        "missing_inpainted_image"
    )
