"""T012: Spec 2 identity records are required run inputs, never placeholders."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from manga_text_seg.config import load_translation_config
from manga_text_seg.translation.pipeline import _identity_records, run_intake

ROOT = Path(__file__).resolve().parents[3]


def _config(bundle, tmp_path):
    base = load_translation_config(ROOT / "configs" / "translation.json")
    return replace(
        base,
        manifest_path=bundle["manifest"],
        output_roots=replace(
            base.output_roots,
            translation=tmp_path / "outputs" / "translation",
            cache=tmp_path / "cache",
        ),
    )


def test_identity_records_quote_the_committed_spec2_sources(translation_bundle):
    bundle = translation_bundle()
    image_id = bundle["image_id"]
    page_list = json.loads((ROOT / "benchmark" / "page-list.json").read_text())
    image_records = json.loads((ROOT / "benchmark" / "image-identity.json").read_text())

    page_identity, input_identities = _identity_records(ROOT, (image_id,))

    page = next(item for item in page_list["pages"] if item["image_id"] == image_id)
    assert page_identity == page_list["page_list_identity"]
    assert input_identities == {image_id: image_records[image_id]["sha256"]}
    assert input_identities[image_id] == page["input_image_identity"]


@pytest.mark.parametrize("identity_state", ["missing", "malformed"])
def test_unavailable_or_malformed_identity_refuses_before_run_creation(
    tmp_path, translation_bundle, identity_state
):
    bundle = translation_bundle()
    config = _config(bundle, tmp_path)
    repo = tmp_path / "repo"
    benchmark = repo / "benchmark"
    benchmark.mkdir(parents=True)
    if identity_state == "malformed":
        (benchmark / "page-list.json").write_text(
            json.dumps(
                {
                    "page_list_identity": "not-a-sha256",
                    "pages": [
                        {
                            "image_id": bundle["image_id"],
                            "input_image_identity": "bad",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (benchmark / "image-identity.json").write_text(
            json.dumps({bundle["image_id"]: {"sha256": "bad"}}), encoding="utf-8"
        )

    with pytest.raises(ValueError, match="Spec 2"):
        run_intake(
            config,
            f"identity-{identity_state}",
            repo_root=repo,
            image_ids=(bundle["image_id"],),
        )

    assert not config.output_roots.translation.exists()
