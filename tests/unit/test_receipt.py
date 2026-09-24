"""Receipt validation and admission (T016, FR-050a, FR-058, SC-014).

The five behaviours from ``contracts/returned-result.md`` §"Test obligations"
that belong to the receipt itself: a stale list is refused, a mismatched image
is refused, a provenance gap is refused, a defective mask is refused without
being repaired, and a refusal leaves the run untouched.

Every case is synthetic — no checkpoint, no download, no network (FR-050).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from manga_text_seg.adapters.manga_text_segmentation import fold_for_image_id
from manga_text_seg.receipt import RUN_RECORD_NAME, ReceiptError, admit, validate_handoff

RUN_ID = "testrun"
STALE_IDENTITY = "0" * 64

#: The one method the contract requires a per-page fold attribution of
#: (FR-013a).  Spelled here rather than imported so the test fails if the
#: constant in the module under test is renamed out from under it.
METHOD_A = "manga-text-segmentation"


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    """A run directory under a temporary ``deliverables/`` tree."""
    return tmp_path / "deliverables" / RUN_ID


@pytest.fixture()
def page_list(exported) -> dict:
    return exported["page_list"]


def _masks(page_list: dict, aligned_mask: np.ndarray) -> dict[str, np.ndarray]:
    return {page["image_id"]: aligned_mask.copy() for page in page_list["pages"]}


def _snapshot(root: Path) -> dict[str, str]:
    """Every file under ``root`` as ``relative path -> sha256``; empty if absent."""
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _admit(exported, handoff: Path, run_dir: Path):
    return admit(
        handoff,
        page_list=exported["page_list"],
        identity_record=exported["identity_record"],
        run_dir=run_dir,
        manifest=exported["manifest"],
    )


def _renamed(provenance: dict, method: str) -> dict:
    return {**provenance, "method": method}


# -- A complete hand-off is admitted -----------------------------------------


def test_admits_a_complete_handoff(exported, page_list, aligned_mask, make_handoff, run_dir):
    handoff = make_handoff(page_list, _masks(page_list, aligned_mask))

    rows = _admit(exported, handoff, run_dir)

    method_dir = run_dir / "standin"
    assert (method_dir / "provenance.json").is_file()
    assert (method_dir / "errors.json").is_file()
    assert (method_dir / "metrics" / "per-page.csv").is_file()
    assert (method_dir / "metrics" / "summary.json").is_file()
    for page in page_list["pages"]:
        manga, stem = page["manga"], page["stem"]
        assert (method_dir / "masks" / manga / f"{stem}.png").is_file()
        assert (method_dir / "metadata" / manga / f"{stem}.json").is_file()

    assert [row["image_id"] for row in rows] == [p["image_id"] for p in page_list["pages"]]


def test_validate_handoff_accepts_a_clean_handoff(exported, page_list, aligned_mask, make_handoff):
    handoff = make_handoff(page_list, _masks(page_list, aligned_mask))

    validate_handoff(
        handoff,
        page_list=page_list,
        identity_record=exported["identity_record"],
    )  # does not raise


# -- 1. A stale page list is refused, not aligned by name --------------------


def test_refuses_a_stale_page_list(exported, page_list, aligned_mask, make_handoff, run_dir):
    handoff = make_handoff(
        page_list,
        _masks(page_list, aligned_mask),
        sidecar=lambda image_id, doc: {**doc, "page_list_identity": STALE_IDENTITY},
    )

    with pytest.raises(ReceiptError) as caught:
        _admit(exported, handoff, run_dir)

    assert caught.value.check == "page-list-identity"
    assert caught.value.field == "page_list_identity"
    assert not _snapshot(run_dir), "a refusal must write nothing"


# -- 2. A mismatched input image is refused, naming the page -----------------


def test_refuses_a_mismatched_input_image_identity(
    exported, page_list, aligned_mask, make_handoff, run_dir
):
    target = page_list["pages"][1]["image_id"]

    handoff = make_handoff(
        page_list,
        _masks(page_list, aligned_mask),
        sidecar=lambda image_id, doc: (
            {**doc, "input_image_identity": "1" * 64} if image_id == target else doc
        ),
    )

    with pytest.raises(ReceiptError) as caught:
        _admit(exported, handoff, run_dir)

    assert caught.value.check == "input-image-identity"
    assert caught.value.image_id == target
    assert not _snapshot(run_dir)


# -- 3. A provenance gap is refused, naming the field -----------------------


def test_refuses_missing_checkpoint_load_evidence(
    exported, page_list, aligned_mask, make_handoff, default_provenance, run_dir
):
    def drop_evidence(record: dict) -> dict:
        record.pop("checkpoint_load_evidence")
        return record

    handoff = make_handoff(
        page_list,
        _masks(page_list, aligned_mask),
        provenance=drop_evidence(default_provenance),
    )

    with pytest.raises(ReceiptError) as caught:
        _admit(exported, handoff, run_dir)

    assert caught.value.check == "provenance"
    assert caught.value.field == "checkpoint_load_evidence"
    assert not _snapshot(run_dir)


def test_refuses_unevidenced_checkpoint_load(
    exported, page_list, aligned_mask, make_handoff, default_provenance, run_dir
):
    """FR-018b: plausible masks are not evidence that the weights were loaded."""
    provenance = {
        **default_provenance,
        "checkpoint_load_evidence": {
            **default_provenance["checkpoint_load_evidence"],
            "evidenced": False,
        },
    }
    handoff = make_handoff(page_list, _masks(page_list, aligned_mask), provenance=provenance)

    with pytest.raises(ReceiptError) as caught:
        _admit(exported, handoff, run_dir)

    assert caught.value.check == "provenance"
    assert caught.value.field == "checkpoint_load_evidence.evidenced"


# -- 4. A defective mask is refused and not repaired ------------------------


def test_refuses_a_wrong_sized_mask_without_resizing(
    exported, page_list, aligned_mask, make_handoff, run_dir
):
    target = page_list["pages"][0]
    masks = _masks(page_list, aligned_mask)
    masks[target["image_id"]] = np.zeros((64, 64), np.uint8)

    handoff = make_handoff(page_list, masks)

    with pytest.raises(ReceiptError) as caught:
        _admit(exported, handoff, run_dir)

    assert caught.value.check == "mask"
    assert caught.value.image_id == target["image_id"]
    assert not _snapshot(run_dir)

    returned = handoff / "masks" / target["manga"] / f"{target['stem']}.png"
    from manga_text_seg.imaging import load_mask

    assert load_mask(returned).shape == (64, 64), "the returned mask must be untouched"


def test_refuses_an_out_of_convention_value(
    exported, page_list, aligned_mask, make_handoff, run_dir
):
    target = page_list["pages"][0]
    masks = _masks(page_list, aligned_mask)
    masks[target["image_id"]] = np.full(aligned_mask.shape, 128, np.uint8)

    handoff = make_handoff(page_list, masks)

    with pytest.raises(ReceiptError) as caught:
        _admit(exported, handoff, run_dir)

    assert caught.value.check == "mask-values"
    assert caught.value.image_id == target["image_id"]
    assert not _snapshot(run_dir)


# -- 6. Method A's fold attribution is the fold that held its book out -------
#
# The fixture pages are ARMS/000 and ARMS/001; ARMS was held out of fold 2.


def _as_method_a(provenance: dict) -> dict:
    return {**provenance, "method": METHOD_A}


def _attributing(image_id: str, doc: dict) -> dict:
    """A sidecar that names the fold entitled to score this page (FR-013a)."""
    return {**doc, "fold_attribution": f"fold_{fold_for_image_id(image_id)}"}


def test_admits_method_a_when_every_page_names_its_entitled_fold(
    exported, page_list, aligned_mask, make_handoff, default_provenance, run_dir
):
    """The happy path check 6 exists to permit, not only to refuse."""
    handoff = make_handoff(
        page_list,
        _masks(page_list, aligned_mask),
        method=METHOD_A,
        provenance=_as_method_a(default_provenance),
        sidecar=_attributing,
    )

    _admit(exported, handoff, run_dir)

    assert (run_dir / METHOD_A / "provenance.json").is_file()


def test_refuses_a_page_scored_by_a_checkpoint_that_trained_on_its_book(
    exported, page_list, aligned_mask, make_handoff, default_provenance, run_dir
):
    """FR-013a: the fold must be the one whose training split held the book out.

    This is the delivered Method A run's defect replayed: every one of its 390
    pages is attributed to fold 0, which trained on 36 of the 45 books.  A
    presence-and-range check cannot see that; only the derived map can.
    """
    target = page_list["pages"][1]["image_id"]

    handoff = make_handoff(
        page_list,
        _masks(page_list, aligned_mask),
        method=METHOD_A,
        provenance=_as_method_a(default_provenance),
        sidecar=lambda image_id, doc: (
            {**doc, "fold_attribution": "fold_0"} if image_id == target else _attributing(image_id, doc)
        ),
    )

    with pytest.raises(ReceiptError) as caught:
        _admit(exported, handoff, run_dir)

    assert caught.value.check == "fold-attribution"
    assert caught.value.image_id == target
    assert caught.value.field == "fold_attribution"
    assert "fold 0" in caught.value.reason
    assert "fold 2" in caught.value.reason, "the refusal names the fold that was entitled"
    assert not _snapshot(run_dir)


def test_refuses_a_method_a_page_with_no_fold_attribution(
    exported, page_list, aligned_mask, make_handoff, default_provenance, run_dir
):
    """``make_handoff`` writes ``fold_attribution: None`` unless told otherwise."""
    handoff = make_handoff(
        page_list,
        _masks(page_list, aligned_mask),
        method=METHOD_A,
        provenance=_as_method_a(default_provenance),
    )

    with pytest.raises(ReceiptError) as caught:
        _admit(exported, handoff, run_dir)

    assert caught.value.check == "fold-attribution"
    assert not _snapshot(run_dir)


@pytest.mark.parametrize("spelling", ["2", "fold.2", "fold_", "fold_2.pkl"])
def test_refuses_a_fold_attribution_that_is_not_the_documented_spelling(
    exported, page_list, aligned_mask, make_handoff, default_provenance, run_dir, spelling
):
    """ONBOARDING §6 spells it ``"fold_<n>"``; anything else is not coerced into it."""
    handoff = make_handoff(
        page_list,
        _masks(page_list, aligned_mask),
        method=METHOD_A,
        provenance=_as_method_a(default_provenance),
        sidecar=lambda image_id, doc: {**doc, "fold_attribution": spelling},
    )

    with pytest.raises(ReceiptError) as caught:
        _admit(exported, handoff, run_dir)

    assert caught.value.check == "fold-attribution"
    assert not _snapshot(run_dir)


def test_a_fold_attribution_is_not_required_of_methods_that_have_no_folds(
    exported, page_list, aligned_mask, make_handoff, run_dir
):
    """Check 6 is Method A's alone; the stand-in must still be admissible."""
    handoff = make_handoff(page_list, _masks(page_list, aligned_mask))

    _admit(exported, handoff, run_dir)

    assert (run_dir / "standin" / "provenance.json").is_file()


# -- 5. A refusal does not damage the run -----------------------------------


def test_a_refusal_leaves_admitted_methods_byte_identical(
    exported, page_list, aligned_mask, make_handoff, default_provenance, run_dir
):
    first = make_handoff(page_list, _masks(page_list, aligned_mask))
    _admit(exported, first, run_dir)
    after_first = _snapshot(run_dir)

    second = make_handoff(
        page_list,
        _masks(page_list, aligned_mask),
        method="second",
        provenance=_renamed(default_provenance, "second"),
    )
    _admit(exported, second, run_dir)
    after_second = _snapshot(run_dir)

    assert set(after_second) > set(after_first)
    # The run record is the run's bookkeeping, not an admitted result: FR-059
    # requires it to grow as methods arrive.  What must not move is the
    # already-admitted method's own files.
    moved = {
        k: v
        for k, v in after_second.items()
        if k in after_first and not k.endswith(RUN_RECORD_NAME)
    }
    assert moved == {k: v for k, v in after_first.items() if not k.endswith(RUN_RECORD_NAME)}

    third = make_handoff(
        page_list,
        _masks(page_list, aligned_mask),
        sidecar=lambda image_id, doc: {**doc, "page_list_identity": STALE_IDENTITY},
    )
    with pytest.raises(ReceiptError):
        _admit(exported, third, run_dir)

    assert _snapshot(run_dir) == after_second, "already-admitted methods must not move"


# -- An operator mistake is caught before anything is written ---------------


def test_refuses_a_method_that_disagrees_with_its_provenance(
    exported, page_list, aligned_mask, make_handoff, default_provenance, run_dir
):
    handoff = make_handoff(
        page_list,
        _masks(page_list, aligned_mask),
        method="alpha",
        provenance=_renamed(default_provenance, "beta"),
    )

    with pytest.raises(ReceiptError) as caught:
        admit(
            handoff,
            page_list=page_list,
            identity_record=exported["identity_record"],
            run_dir=run_dir,
            manifest=exported["manifest"],
            method="alpha",
        )

    assert caught.value.check == "method"
    assert not _snapshot(run_dir)


def test_sidecars_survive_admission_verbatim(exported, page_list, aligned_mask, make_handoff, run_dir):
    """The runner's sidecar is the record of what the runner did, gaps and all."""
    handoff = make_handoff(page_list, _masks(page_list, aligned_mask))

    _admit(exported, handoff, run_dir)

    for page in page_list["pages"]:
        manga, stem = page["manga"], page["stem"]
        source = json.loads((handoff / "metadata" / manga / f"{stem}.json").read_text("utf-8"))
        stored = json.loads(
            (run_dir / "standin" / "metadata" / manga / f"{stem}.json").read_text("utf-8")
        )
        assert stored == source
        assert stored["page_list_identity"] == page_list["page_list_identity"]
        assert stored["input_image_identity"] == page["input_image_identity"]
