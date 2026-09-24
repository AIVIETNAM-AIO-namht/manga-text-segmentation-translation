"""Exporting the distributable page list (T014, FR-054, FR-054a, quickstart §1).

The page list is what leaves this project. Every assertion here is an assertion
about what a runner can and cannot learn from it: 390 ordered pages, the same
bytes on every re-export, and no ground truth in any form.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from manga_text_seg.pagelist import (
    BUNDLE_ROOT,
    IMAGE_IDENTITY_NAME,
    PAGE_LIST_NAME,
    build_page_list,
    export,
    load_identity_record,
    load_page_list,
    page_list_identity,
    sha256_file,
    write_page_list,
)

from ..conftest import FIXTURE_PAGES

#: Keys that would carry ground truth into a runner's hands (FR-054a).
FORBIDDEN_KEYS = {"gt_root", "raw_root", "mask_path", "mask_encoding"}
#: Substrings whose presence anywhere in the document is a leak.
FORBIDDEN_TEXT = ("groundtruth", "ground-truth", "no-need-to-read")


def _walk(node, path="$"):
    """Yield ``(path, key, value)`` for every mapping entry in the document."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield f"{path}.{key}", key, value
            yield from _walk(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, f"{path}[{index}]")


# -- Shape and order ---------------------------------------------------------


def test_page_list_has_one_page_per_pair_in_order(real_page_list, real_manifest):
    pages = real_page_list["pages"]

    assert len(pages) == real_manifest.total == 390
    assert [p["image_id"] for p in pages] == [
        p.image_id for p in real_manifest.pairs
    ]
    assert [p["image_id"] for p in pages] == sorted(
        p["image_id"] for p in pages
    ), "pages must be sorted (manga, stem) — the schema says so"


def test_page_list_records_the_uniform_aligned_size(real_page_list):
    assert real_page_list["aligned_size"] == [1654, 1170]
    assert real_page_list["alignment"] == "crop-topleft"
    assert real_page_list["generated_at_run"] == "default"


# -- Nothing about the answer key -------------------------------------------


def test_page_list_carries_no_ground_truth(real_page_list):
    flat = json.dumps(real_page_list, ensure_ascii=False).lower()

    for where, key, _ in _walk(real_page_list):
        assert key not in FORBIDDEN_KEYS, f"ground-truth key leaked at {where}"
    for needle in FORBIDDEN_TEXT:
        assert needle not in flat, f"ground-truth reference leaked: {needle!r}"

    for page in real_page_list["pages"]:
        assert set(page) == {
            "manga",
            "stem",
            "image_id",
            "image_ref",
            "input_image_identity",
        }


def test_image_ref_stays_inside_the_bundle(real_page_list):
    """`image_ref` is relative to the bundle root, which is relative to the list.

    Both are relative so the pair can be relocated (research.md R2); neither may
    be absolute, and neither may point outside the bundle.
    """
    assert real_page_list["image_bundle"]["root"] == BUNDLE_ROOT

    for page in real_page_list["pages"]:
        ref = page["image_ref"]
        assert not ref.startswith(("/", "\\")), ref
        assert ":" not in ref and ".." not in ref, ref


# -- Determinism (FR-054: every runner consumes the same list instance) -------


def test_re_export_is_byte_identical(real_manifest, tmp_path):
    first_doc = build_page_list(real_manifest)
    second_doc = build_page_list(real_manifest)

    first = write_page_list(first_doc, tmp_path / "a")
    second = write_page_list(second_doc, tmp_path / "b")

    assert first.read_bytes() == second.read_bytes()
    assert first_doc["page_list_identity"] == second_doc["page_list_identity"]


def test_identity_is_the_canonical_serialisation_without_the_field(real_page_list):
    """The identity is recomputable, and it covers the whole document.

    Recomputable so a receipt can verify it without trusting the file it came
    in; covering the whole document so an edit that drops a page changes it.
    """
    doc = dict(real_page_list)
    assert doc["page_list_identity"] == page_list_identity(doc)

    # Not a hash of the file as written: the field is excluded, so the value
    # does not depend on itself.
    body = {k: v for k, v in doc.items() if k != "page_list_identity"}
    with_field = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
    assert doc["page_list_identity"] != hashlib.sha256(
        with_field.encode("utf-8")
    ).hexdigest()
    assert page_list_identity(body) == page_list_identity(doc), (
        "omitting an absent field must be a no-op"
    )


# -- The bundle and its identity record (T019, FR-054a) -----------------------


def test_export_writes_the_bundle_and_its_identity_record(
    tmp_path, two_page_manifest
):
    from manga_text_seg.manifest import load_manifest

    manifest = load_manifest(two_page_manifest)
    benchmark_dir = tmp_path / "benchmark"

    path = export(manifest, benchmark_dir)
    doc = load_page_list(benchmark_dir)
    record = load_identity_record(benchmark_dir)

    assert path == benchmark_dir / PAGE_LIST_NAME
    assert (benchmark_dir / IMAGE_IDENTITY_NAME).is_file()
    assert len(doc["pages"]) == FIXTURE_PAGES
    assert len(record) == FIXTURE_PAGES

    for pair in manifest.pairs:
        page = next(p for p in doc["pages"] if p["image_id"] == pair.image_id)
        bundle_file = benchmark_dir / doc["image_bundle"]["root"] / page["image_ref"]

        assert bundle_file.is_file(), f"missing bundle file for {pair.image_id}"
        assert bundle_file.read_bytes() == pair.raw_path.read_bytes()
        assert page["input_image_identity"] == sha256_file(pair.raw_path)
        assert record[pair.image_id] == {
            "sha256": page["input_image_identity"],
            "size_bytes": pair.raw_path.stat().st_size,
        }


def test_export_refuses_a_non_uniform_dataset(monkeypatch, two_page_manifest):
    """One aligned size is recorded, so a mixed-size dataset cannot be exported.

    Asserted across every page rather than read off the first: publishing the
    first page's size would silently misdescribe every page after it (FR-024a).
    """
    import manga_text_seg.pagelist as pagelist
    from manga_text_seg.manifest import load_manifest
    from manga_text_seg.pagelist import PageListError

    sizes = iter([(1170, 1654), (1200, 1654)])
    monkeypatch.setattr(pagelist, "load_raw_page", lambda path: np.zeros(next(sizes), np.uint8))

    with pytest.raises(PageListError, match="aligned size"):
        build_page_list(load_manifest(two_page_manifest))
