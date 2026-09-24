"""One evaluation procedure, two ways in (T017, FR-053(e), FR-057, research.md R12).

A mask that arrives from a runner as a hand-off and the same mask produced in
process by an adapter must score identically — because they are scored by the
same code, in this project, on this machine.  If these two ever diverge, one of
them is computing metrics somewhere else, which FR-057 forbids.

Synthetic throughout: the ``standin`` adapter thresholds the page, so the mask
is deterministic and no checkpoint is loaded (FR-050, FR-051).
"""
from __future__ import annotations

import numpy as np

from manga_text_seg.adapters import create
from manga_text_seg.align import align
from manga_text_seg.imaging import load_gt_mask, load_mask
from manga_text_seg.metrics import METRIC_KEYS, compute_metrics
from manga_text_seg.normalize import normalize_mask
from manga_text_seg.receipt import admit


def _adapter_masks(manifest) -> dict[str, np.ndarray]:
    """Run the standin adapter over every page, exactly as ``run.py`` would."""
    adapter = create("standin")
    masks = {}
    for pair in manifest.pairs:
        result = adapter.infer(pair.raw_path)
        assert result.status == "ok", f"{pair.image_id}: {result.error}"
        masks[pair.image_id] = result.mask
    return masks


def test_metric_matches_the_adapter_path(exported, make_handoff, tmp_path):
    manifest = exported["manifest"]
    page_list = exported["page_list"]
    masks = _adapter_masks(manifest)

    handoff = make_handoff(page_list, masks)
    rows = admit(
        handoff,
        page_list=page_list,
        identity_record=exported["identity_record"],
        run_dir=tmp_path / "run",
        manifest=manifest,
    )

    by_id = {row["image_id"]: row for row in rows}
    assert set(by_id) == set(masks)

    for pair in manifest.pairs:
        prediction = masks[pair.image_id]
        aligned = align(normalize_mask(load_gt_mask(pair.mask_path)), prediction)
        expected = compute_metrics(aligned.mask, aligned.page)

        for key in METRIC_KEYS:
            assert by_id[pair.image_id][key] == expected[key], (
                f"{pair.image_id}: {key} differs between the two paths"
            )


def test_admission_stores_the_mask_it_was_handed(exported, make_handoff, tmp_path):
    """The metric equality above only means anything if the bytes survived.

    A lossy mask write would change the score for a reason that has nothing to
    do with the two paths meeting.
    """
    manifest = exported["manifest"]
    page_list = exported["page_list"]
    masks = _adapter_masks(manifest)

    handoff = make_handoff(page_list, masks)
    admit(
        handoff,
        page_list=page_list,
        identity_record=exported["identity_record"],
        run_dir=tmp_path / "run",
        manifest=manifest,
    )

    for page in page_list["pages"]:
        stored = tmp_path / "run" / "standin" / "masks" / page["manga"] / f"{page['stem']}.png"
        assert np.array_equal(load_mask(stored), masks[page["image_id"]])
