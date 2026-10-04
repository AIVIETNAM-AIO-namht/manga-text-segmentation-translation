"""T013 consumer-level tests for the Spec 004 addressing contract."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from manga_text_seg.config import load_translation_config
from manga_text_seg.translation.pipeline import run_intake
from manga_text_seg.translation.runs import page_directory

ROOT = Path(__file__).resolve().parents[3]


def _fixture_run(tmp_path: Path, translation_bundle):
    bundle = translation_bundle()
    base = load_translation_config(ROOT / "configs" / "translation.json")
    config = replace(
        base,
        manifest_path=bundle["manifest"],
        output_roots=replace(
            base.output_roots,
            translation=tmp_path / "outputs",
            cache=tmp_path / "cache",
        ),
        provider=replace(
            base.provider,
            secret_env_var=None,
            cache=replace(base.provider.cache, root=tmp_path / "cache"),
        ),
    )
    result = run_intake(
        config,
        "addressing-run",
        repo_root=ROOT,
        image_ids=(bundle["image_id"],),
        prediction_root=bundle["prediction_mask"].parents[2],
        inpainting_root=bundle["inpainted"].parents[3],
    )
    return bundle, result


def test_consumer_resolves_and_reads_the_addressed_page_without_an_index(
    tmp_path: Path, translation_bundle
) -> None:
    bundle, result = _fixture_run(tmp_path, translation_bundle)
    image_id = bundle["image_id"]

    # The consumer is given only the three contract address fields and the
    # configured output root. It does not read the manifest or an index.
    bundle["manifest"].unlink()
    address = ("addressing-run", "classical_baseline", image_id)
    resolved = page_directory(tmp_path / "outputs", *address)

    assert resolved == result.pages[0].page_dir
    assert f"{resolved.parent.name}/{resolved.name}" == image_id
    assert (resolved / "mask.png").read_bytes() == bundle[
        "prediction_mask"
    ].read_bytes()
    metadata = json.loads((resolved / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["image_id"] == image_id
    assert metadata["run_id"] == address[0]
    assert metadata["segmentation_method"] == address[1]
    assert not (result.run_dir / "index.json").exists()
    assert not (result.run_dir / "manifest.json").exists()


def test_each_wrong_contract_field_fails_to_resolve(
    tmp_path: Path, translation_bundle
) -> None:
    bundle, _ = _fixture_run(tmp_path, translation_bundle)
    bundle["manifest"].unlink()
    image_id = bundle["image_id"]
    page_list = json.loads(
        (ROOT / "benchmark" / "page-list.json").read_text(encoding="utf-8")
    )
    wrong_image_id = next(
        item["image_id"] for item in page_list["pages"] if item["image_id"] != image_id
    )

    wrong_addresses = [
        ("wrong-run", "classical_baseline", image_id),
        ("addressing-run", "comic_text_detector", image_id),
        ("addressing-run", "classical_baseline", wrong_image_id),
    ]
    for address in wrong_addresses:
        resolved = page_directory(tmp_path / "outputs", *address)
        assert not resolved.exists(), f"unexpected artifact resolved at {resolved}"


def test_nonexistent_address_and_invalid_method_fail_explicitly(tmp_path: Path) -> None:
    unresolved = page_directory(
        tmp_path / "outputs", "unknown-run", "classical_baseline", "ARMS/001"
    )
    assert not unresolved.exists()
    with pytest.raises(ValueError, match="unsupported segmentation method"):
        page_directory(
            tmp_path / "outputs", "unknown-run", "unknown-method", "ARMS/001"
        )
