"""T077: static and fixture-output checks against vendoring and GT use."""

from __future__ import annotations

import ast
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from manga_text_seg.config import load_translation_config
from manga_text_seg.manifest import load_manifest
from manga_text_seg.translation.pipeline import run_batch
from manga_text_seg.translation.translate import FixtureTranslationProvider

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_SUFFIXES = {
    ".pth",
    ".pt",
    ".onnx",
    ".ckpt",
    ".safetensors",
    ".weights",
}
FORBIDDEN_IMPORTS = {
    "manga_text_seg.availability",
    "manga_text_seg.run",
    "manga_text_seg.runs",
    "manga_text_seg.methods",
    "manga_text_seg.adapters",
    "torch",
    "torchvision",
    "ultralytics",
}


def test_translation_feature_does_not_vendor_models_or_read_ground_truth(
    tmp_path, translation_bundle
):
    source_root = ROOT / "src" / "manga_text_seg" / "translation"
    for source in source_root.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = {node.module}
            else:
                continue
            assert not any(
                name == forbidden or name.startswith(forbidden + ".")
                for name in imported
                for forbidden in FORBIDDEN_IMPORTS
            ), source

    forbidden_dirs = {"vendor", "third_party", "checkpoints", "weights"}
    assert not any(
        part.lower() in forbidden_dirs
        for path in source_root.rglob("*")
        for part in path.parts
    )
    assert not any(
        path.suffix.lower() in FORBIDDEN_SUFFIXES for path in source_root.rglob("*")
    )

    bundle = translation_bundle()
    # Two synthetic prediction components; the Spec 1 GT path remains absent and
    # is never supplied to the translation pipeline.
    mask = np.zeros((1170, 1654), dtype=np.uint8)
    mask[90:120, 100:150] = 255
    mask[210:240, 250:300] = 255
    assert cv2.imwrite(str(bundle["prediction_mask"]), mask)
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
    pair = load_manifest(bundle["manifest"]).pairs[0]
    assert not pair.mask_path.exists()
    result = run_batch(
        config,
        "no-gt-run",
        repo_root=ROOT,
        provider=FixtureTranslationProvider({"fixture OCR": "fixture translation"}),
        recognizer=lambda _crop: "fixture OCR",
        prediction_root=bundle["prediction_mask"].parents[2],
        inpainting_root=bundle["inpainted"].parents[3],
    )
    page_dir = next(result.run_dir.rglob("metadata.json")).parent
    assert (page_dir / "mask.png").read_bytes() == bundle[
        "prediction_mask"
    ].read_bytes()
    assert not any(
        path.suffix.lower() in FORBIDDEN_SUFFIXES
        for path in result.run_dir.rglob("*")
        if path.is_file()
    )


def test_translation_pipeline_never_opens_protected_fixture_path(
    tmp_path, translation_bundle
):
    """Observe Python file-open attempts against an isolated protected tree."""
    bundle = translation_bundle()
    isolated_repo = tmp_path / "isolated-repository"
    benchmark = isolated_repo / "benchmark"
    benchmark.mkdir(parents=True)
    for name in ("page-list.json", "image-identity.json"):
        shutil.copyfile(ROOT / "benchmark" / name, benchmark / name)

    protected_root = isolated_repo / "data" / "no-need-to-read"
    protected_root.mkdir(parents=True)
    protected_probe = protected_root / "must-not-open.txt"
    protected_probe.write_text("isolated sentinel", encoding="utf-8")
    protected_root_resolved = protected_root.resolve()
    attempted_opens = []

    def deny_protected_open(event, arguments):
        if event != "open" or not arguments:
            return
        raw_path = arguments[0]
        if not isinstance(raw_path, (str, bytes, os.PathLike)):
            return
        try:
            candidate = Path(os.fsdecode(raw_path)).resolve()
            candidate.relative_to(protected_root_resolved)
        except (OSError, ValueError):
            return
        attempted_opens.append(str(candidate))
        raise PermissionError(f"protected fixture path cannot be opened: {candidate}")

    sys.addaudithook(deny_protected_open)

    # Prove the guard observes and blocks the relevant architecture-level
    # operation, then clear this deliberate calibration attempt before pipeline.
    import pytest

    with pytest.raises(PermissionError, match="protected fixture path"):
        protected_probe.read_text(encoding="utf-8")
    assert attempted_opens == [str(protected_probe.resolve())]
    attempted_opens.clear()

    base = load_translation_config(ROOT / "configs" / "translation.json")
    cache = tmp_path / "cache-protected-open-test"
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
            base.output_roots,
            translation=tmp_path / "outputs-protected-open-test",
            cache=cache,
        ),
        provider=provider_config,
    )
    result = run_batch(
        config,
        "protected-open-audit",
        repo_root=isolated_repo,
        provider=FixtureTranslationProvider({"fixture OCR": "fixture translation"}),
        recognizer=lambda _crop: "fixture OCR",
        prediction_root=bundle["prediction_mask"].parents[2],
        inpainting_root=bundle["inpainted"].parents[3],
    )

    assert attempted_opens == []
    assert result.pages[0].status == "success"
    assert (result.pages[0].page_dir / "rendered.png").is_file()
