"""The classical admitted view: the transform fills what Spec 003/004 require.

``tools/relayout_classical_admitted_view.py`` bridges Spec 1's flat tree to the
``masks/``+``metadata/`` layout Spec 003's contract reads. The transform is the
part that can silently rot, so it gets the check; the byte-copy is asserted in
the script itself and by the repository-wide check below.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "relayout_classical_admitted_view",
    ROOT / "tools" / "relayout_classical_admitted_view.py",
)
relayout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(relayout)

SPEC1_SIDECAR = {
    "image_id": "ARMS/000",
    "method": "adaptive",
    "threshold": "auto",
    "alignment": {"strategy": "crop-topleft", "post_size": [1170, 1654]},
    "inference_time_seconds": 0.01,
    "status": "ok",
}


def test_transform_overrides_method_and_status_and_quotes_the_identities():
    document = relayout._transform_sidecar(
        SPEC1_SIDECAR,
        image_id="ARMS/000",
        page_identity="page-hash",
        image_identity="image-hash",
    )

    assert document["method"] == "classical_baseline"
    assert document["status"] == "success"
    assert document["output_size"] == [1654, 1170]
    assert document["page_list_identity"] == "page-hash"
    assert document["input_image_identity"] == "image-hash"
    # Spec 1's own record is kept, not discarded.
    assert document["threshold"] == "auto"
    assert document["alignment"] == SPEC1_SIDECAR["alignment"]
    assert SPEC1_SIDECAR["method"] == "adaptive"  # the source is untouched


def test_transform_leaves_the_spec1_sidecar_unmodified():
    relayout._transform_sidecar(
        SPEC1_SIDECAR,
        image_id="ARMS/000",
        page_identity="page-hash",
        image_identity="image-hash",
    )

    assert "page_list_identity" not in SPEC1_SIDECAR
    assert "output_size" not in SPEC1_SIDECAR


@pytest.mark.skipif(
    not relayout._method_root(ROOT).is_dir(),
    reason="the admitted view is generated output (outputs/ is gitignored)",
)
def test_built_view_is_accepted_by_spec003_intake():
    assert relayout.check(ROOT) == []


def test_every_built_sidecar_carries_the_five_required_fields():
    metadata_root = relayout._method_root(ROOT) / "metadata"
    if not metadata_root.is_dir():
        pytest.skip("the admitted view is generated output (outputs/ is gitignored)")

    page_identity, identities = relayout._benchmark(ROOT)
    sidecars = sorted(metadata_root.rglob("*.json"))
    assert sidecars

    for sidecar_path in sidecars:
        document = json.loads(sidecar_path.read_text(encoding="utf-8"))
        image_id = document["image_id"]
        assert document["method"] == "classical_baseline"
        assert document["status"] == "success"
        assert document["output_size"] == [1654, 1170]
        assert document["page_list_identity"] == page_identity
        assert document["input_image_identity"] == identities[image_id]
