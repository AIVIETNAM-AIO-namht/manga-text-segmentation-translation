"""T059 run isolation and default non-overwrite integration checks."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from manga_text_seg.config import load_translation_config
from manga_text_seg.translation.pipeline import run_batch
from manga_text_seg.translation.runs import create_run_directory
from manga_text_seg.translation.translate import FixtureTranslationProvider

ROOT = Path(__file__).resolve().parents[2]


def _config(bundle, tmp_path):
    base = load_translation_config(ROOT / "configs" / "translation.json")
    cache = tmp_path / "cache"
    provider_config = replace(
        base.provider,
        name="fixture",
        model="fixture",
        version="1",
        secret_env_var=None,
        cache=replace(base.provider.cache, root=cache),
    )
    return replace(
        base,
        manifest_path=bundle["manifest"],
        output_roots=replace(
            base.output_roots, translation=tmp_path / "outputs", cache=cache
        ),
        provider=provider_config,
    )


def _run(config, bundle, run_id):
    provider = FixtureTranslationProvider({"fixture OCR": "translated text"})
    return run_batch(
        config,
        run_id,
        repo_root=ROOT,
        provider=provider,
        recognizer=lambda image: "fixture OCR",
        prediction_root=bundle["prediction_mask"].parents[2],
        inpainting_root=bundle["inpainted"].parents[3],
    )


def test_distinct_runs_coexist_and_duplicate_refusal_preserves_existing_tree(
    tmp_path, translation_bundle
):
    bundle = translation_bundle()
    config = _config(bundle, tmp_path)
    first = _run(config, bundle, "run-one")
    original = (first.run_dir / "run.json").read_bytes()
    first_png = next(first.run_dir.rglob("rendered.png")).read_bytes()
    first_digest = hashlib.sha256(original + first_png).hexdigest()

    second = _run(config, bundle, "run-two")
    assert first.run_dir.is_dir() and second.run_dir.is_dir()
    assert (
        hashlib.sha256(
            (first.run_dir / "run.json").read_bytes()
            + next(first.run_dir.rglob("rendered.png")).read_bytes()
        ).hexdigest()
        == first_digest
    )

    with pytest.raises(FileExistsError):
        _run(config, bundle, "run-one")
    assert (
        hashlib.sha256(
            (first.run_dir / "run.json").read_bytes()
            + next(first.run_dir.rglob("rendered.png")).read_bytes()
        ).hexdigest()
        == first_digest
    )


def test_explicit_overwrite_targets_only_selected_run(tmp_path, translation_bundle):
    bundle = translation_bundle()
    config = _config(bundle, tmp_path)
    first = _run(config, bundle, "run-one")
    other = _run(config, bundle, "run-two")
    other_before = (other.run_dir / "run.json").read_bytes()
    overwritten = run_batch(
        config,
        "run-one",
        repo_root=ROOT,
        provider=FixtureTranslationProvider({"fixture OCR": "new result"}),
        recognizer=lambda image: "fixture OCR",
        prediction_root=bundle["prediction_mask"].parents[2],
        inpainting_root=bundle["inpainted"].parents[3],
        overwrite=True,
    )
    assert overwritten.run_dir == first.run_dir
    assert other.run_dir.joinpath("run.json").read_bytes() == other_before


def test_existing_run_helper_refuses_without_overwrite(tmp_path):
    target = create_run_directory(tmp_path, "stable")
    marker = target / "marker"
    marker.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        create_run_directory(tmp_path, "stable")
    assert marker.read_bytes() == b"preserve"
