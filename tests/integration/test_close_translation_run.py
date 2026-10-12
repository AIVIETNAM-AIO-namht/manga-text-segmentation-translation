"""An interrupted run can be closed from the pages it finished.

``run_batch`` writes ``run.json`` only after the last manifest page, so a
stopped run keeps its page artifacts but has no run record. ``close_run``
rebuilds that record from the finished pages; this checks the rebuild equals
what ``run_batch`` itself would have written.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

from manga_text_seg.config import load_translation_config
from manga_text_seg.translation.pipeline import run_batch
from manga_text_seg.translation.translate import FixtureTranslationProvider

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.close_translation_run import close_run  # noqa: E402


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


def test_close_run_matches_a_completed_run(tmp_path, translation_bundle):
    bundle = translation_bundle()
    config = _config(bundle, tmp_path / "translation", tmp_path / "cache")

    completed = _run(config, bundle, "completed")
    expected = json.loads((completed.run_dir / "run.json").read_text(encoding="utf-8"))

    # Simulate an interrupted run: page artifacts survive, the record does not.
    (completed.run_dir / "run.json").unlink()
    (completed.run_dir / "errors.json").unlink()

    closed_dir = close_run(config, "completed", repo_root=ROOT)

    assert closed_dir == completed.run_dir
    assert json.loads((closed_dir / "run.json").read_text(encoding="utf-8")) == expected
    assert (closed_dir / "errors.json").exists()


def test_close_run_refuses_a_run_that_is_already_closed(tmp_path, translation_bundle):
    bundle = translation_bundle()
    config = _config(bundle, tmp_path / "translation", tmp_path / "cache")
    completed = _run(config, bundle, "already-closed")

    try:
        close_run(config, "already-closed", repo_root=ROOT)
    except ValueError as exc:
        assert "already closed" in str(exc)
    else:
        raise AssertionError("closing a closed run must be refused")


def test_close_run_reports_unfinished_pages(tmp_path, translation_bundle):
    bundle = translation_bundle()
    config = _config(bundle, tmp_path / "translation", tmp_path / "cache")
    completed = _run(config, bundle, "unfinished")

    (completed.run_dir / "run.json").unlink()
    metadata_path = (
        completed.run_dir / config.segmentation_method / bundle["image_id"] / "metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["status"] = "failed"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    try:
        close_run(config, "unfinished", repo_root=ROOT)
    except ValueError as exc:
        assert "did not finish cleanly" in str(exc)
    else:
        raise AssertionError("a failed page must block the close")
