"""T076: secret-value audit across a completed synthetic translation run."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from manga_text_seg.config import load_translation_config
from manga_text_seg.translation.pipeline import run_batch
from manga_text_seg.translation.translate import FixtureTranslationProvider

ROOT = Path(__file__).resolve().parents[2]
SECRET_NAME = "MANGA_TRANSLATE_TEST_SECRET"
SECRET_VALUE = "sentinel-secret-value-must-not-leak"


def test_secret_value_never_appears_in_run_outputs_cache_or_logs(
    tmp_path, translation_bundle, monkeypatch, capsys
):
    bundle = translation_bundle()
    mask = np.zeros((1170, 1654), dtype=np.uint8)
    mask[100:130, 100:150] = 255
    mask[220:250, 200:260] = 255
    assert cv2.imwrite(str(bundle["prediction_mask"]), mask)
    monkeypatch.setenv(SECRET_NAME, SECRET_VALUE)

    base = load_translation_config(ROOT / "configs" / "translation.json")
    cache = tmp_path / "cache"
    provider_config = replace(
        base.provider,
        name="fixture",
        model="fixture-model",
        version="1",
        secret_env_var=SECRET_NAME,
        cache=replace(base.provider.cache, root=cache, enabled=True, bypass=False),
    )
    config = replace(
        base,
        manifest_path=bundle["manifest"],
        output_roots=replace(
            base.output_roots, translation=tmp_path / "outputs", cache=cache
        ),
        provider=provider_config,
    )
    recognizer_results = iter(("first OCR", "second OCR"))
    provider = FixtureTranslationProvider(
        {
            "first OCR": "safe translation",
            "second OCR": RuntimeError(f"provider rejected credential {SECRET_VALUE}"),
        }
    )
    result = run_batch(
        config,
        "secret-audit",
        repo_root=ROOT,
        provider=provider,
        recognizer=lambda _crop: next(recognizer_results),
        prediction_root=bundle["prediction_mask"].parents[2],
        inpainting_root=bundle["inpainted"].parents[3],
    )

    output_files = [path for path in result.run_dir.rglob("*") if path.is_file()]
    cache_files = [path for path in cache.rglob("*") if path.is_file()]
    captured = capsys.readouterr()
    content = "\n".join(
        [
            path.read_bytes().decode("utf-8", errors="ignore")
            for path in output_files + cache_files
        ]
        + [captured.out, captured.err]
    )
    assert SECRET_VALUE not in content
    assert SECRET_NAME in content
    run_record = json.loads((result.run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_record["translation_provider"]["name"] == "fixture"
    assert run_record["translation_provider"]["model"] == "fixture-model"
    assert run_record["translation_provider"]["version"] == "1"
    assert run_record["translation_provider"]["secret_env_var"] == SECRET_NAME
