"""Fixture-only equivalents of quickstart scenarios 1–3 and 5–8."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from manga_text_seg.config import load_translation_config
from manga_text_seg.translation.pipeline import (
    export_intermediates,
    rerun_stage,
    run_intake,
)
from manga_text_seg.translation.translate import FixtureTranslationProvider

ROOT = Path(__file__).resolve().parents[2]


def test_quickstart_stages_consume_fixture_artifacts_in_order(
    tmp_path, translation_bundle
):
    bundle = translation_bundle()
    source_sidecar = json.loads(bundle["inpainted_sidecar"].read_text(encoding="utf-8"))
    source_sidecar["dilation"] = {
        "enabled": True,
        "kernel_shape": "ellipse",
        "kernel_size": [5, 3],
        "iterations": 2,
    }
    bundle["inpainted_sidecar"].write_text(
        json.dumps(source_sidecar, indent=2) + "\n", encoding="utf-8"
    )
    base = load_translation_config(ROOT / "configs" / "translation.json")
    cache = tmp_path / "cache"
    provider_config = replace(
        base.provider,
        name="fixture",
        model="fixture",
        version="1",
        secret_env_var=None,
        cache=replace(base.provider.cache, root=cache, enabled=False),
    )
    config = replace(
        base,
        manifest_path=bundle["manifest"],
        output_roots=replace(
            base.output_roots, translation=tmp_path / "outputs", cache=cache
        ),
        provider=provider_config,
    )
    intake = run_intake(
        config,
        "quickstart-fixture",
        repo_root=ROOT,
        image_ids=(bundle["image_id"],),
        prediction_root=bundle["prediction_mask"].parents[2],
        inpainting_root=bundle["inpainted"].parents[3],
    )
    page_dir = next(intake.run_dir.rglob("metadata.json")).parent
    output_metadata = json.loads(
        (page_dir / "metadata.json").read_text(encoding="utf-8")
    )
    inpainting_record = output_metadata["inpainting_method"]
    assert inpainting_record["algorithm"] == source_sidecar["algorithm"]
    assert inpainting_record["inpaint_algorithm"] == source_sidecar["inpaint_algorithm"]
    assert inpainting_record["inpaint_radius"] == source_sidecar["inpaint_radius"]
    assert inpainting_record["dilation"] == source_sidecar["dilation"]

    regions = rerun_stage("regions", bundle["image_id"], intake.run_dir)
    assert regions.status == "ok"
    assert (page_dir / "regions.json").is_file()
    ocr = rerun_stage(
        "ocr",
        bundle["image_id"],
        intake.run_dir,
        recognizer=lambda _crop: "quickstart OCR",
    )
    assert ocr.results[0].text == "quickstart OCR"
    translated = rerun_stage(
        "translate",
        bundle["image_id"],
        intake.run_dir,
        provider=FixtureTranslationProvider(
            {"quickstart OCR": "quickstart translation"}
        ),
    )
    assert translated[0].translated_text == "quickstart translation"
    rendered = rerun_stage("render", bundle["image_id"], intake.run_dir)
    assert rendered.image_path == page_dir / "rendered.png"
    assert (page_dir / "metadata.json").is_file()

    exported = export_intermediates(
        intake.run_dir,
        bundle["image_id"],
        tmp_path / "exports" / "quickstart-page",
    )
    assert (exported / "rendered.png").read_bytes() == (
        page_dir / "rendered.png"
    ).read_bytes()
    assert (page_dir / "rendered.png").is_file()
