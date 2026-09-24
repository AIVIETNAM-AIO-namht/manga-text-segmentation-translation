"""The four-panel cases and the case report over an admitted run.

T042, T043 (US4 scenarios 2 and 4; FR-040, FR-041).

The mean hides the failure modes; these are the two artefacts that show them.
One is pictorial: for a configurable number of cases, the raw page, the
ground-truth mask, the prediction mask and the prediction-versus-ground-truth
overlay, on one grid — the same four panels the classical sweep renders, read
from the admitted run's own masks instead of a sweep's output directory.

The other is written: representative good cases and representative failure
cases per method, the failures carrying the reason the runner recorded.  Both
are built from an *existing* run directory — nothing here re-runs inference,
which is what makes US4's independent test runnable against a hand-off that
arrived days ago.

The case *selection* is the part ONBOARDING §8 is explicit about: not the first
pages of the manifest, but the best pages, the middle ones, and the failures.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2

from manga_text_seg.config import Config
from manga_text_seg.imaging import load_raw_page
from manga_text_seg.manifest import save_manifest
from manga_text_seg.receipt import admit
from manga_text_seg.visualize import (
    CASE_DIR,
    _case_entries,
    render_cases,
    write_case_report,
)

#: One page's recorded failure, in the shape ``data-model.md`` gives a
#: ``DLFailureRecord`` entry.  ``error`` and ``reason`` differ on purpose: a
#: reader that collapses them into one field must be caught, not pass by luck.
RECORDED_FAILURE = {
    "image_id": None,  # filled in per test: it must be a page of this page list
    "error": "CUDA out of memory",
    "stage": "inference",
    "reason": "the page's text layer is larger than this method's configured tile",
}


def _masks(manifest) -> dict:
    """A deterministic binary mask per page: no adapter, no checkpoint (FR-050)."""
    return {
        pair.image_id: (load_raw_page(pair.raw_path) > 127).astype("uint8") * 255
        for pair in manifest.pairs
    }


def _timed(image_id: str, doc: dict) -> dict:
    doc["inference_time_seconds"] = 2.5
    return doc


def _provenance_for(default_provenance: dict, method: str) -> dict:
    record = json.loads(json.dumps(default_provenance))
    record["method"] = method
    return record


def _admit(exported, make_handoff, default_provenance, run_dir, method, masks, *, failures=None):
    """Admit one hand-off, optionally returning a non-empty failure list."""
    handoff = make_handoff(
        exported["page_list"],
        masks,
        method=method,
        provenance=_provenance_for(default_provenance, method),
        sidecar=_timed,
    )
    if failures is not None:
        (handoff / "errors.json").write_text(
            json.dumps(failures, indent=2) + "\n", encoding="utf-8"
        )
    return admit(
        handoff,
        page_list=exported["page_list"],
        identity_record=exported["identity_record"],
        run_dir=run_dir,
        manifest=exported["manifest"],
    )


def _cfg(tmp_path: Path, manifest) -> Config:
    """The config the case step runs under.

    Its ``output_dir`` holds the manifest the run was scored from, which is the
    only thing the renderer asks it for — the case count comes from the same
    ``visualization`` block the classical sweep uses (FR-033, reused by FR-040).
    """
    output_root = tmp_path / "out"
    save_manifest(manifest, output_root / "default" / "manifest.json")
    return Config(
        raw_root=manifest.raw_root,
        gt_root=manifest.gt_root,
        output_root=output_root,
        run_id="default",
        viz_n=20,
    )


def _awaited(run_dir: Path) -> list[str]:
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    return record["methods_awaited"]


def test_every_case_is_rendered_as_a_four_panel_composite(
    exported, make_handoff, default_provenance, tmp_path
):
    """US4 scenario 2: a configurable number of cases, four panels each.

    The prediction is read from the admitted run's own ``masks/`` — the bytes
    the runner returned, scored and stored — not from a sweep's output
    directory, and inference is not re-run to produce it.
    """
    run_dir = tmp_path / "run"
    failed_id = exported["page_list"]["pages"][0]["image_id"]
    _admit(
        exported,
        make_handoff,
        default_provenance,
        run_dir,
        "standin-a",
        _masks(exported["manifest"]),
        failures=[dict(RECORDED_FAILURE, image_id=failed_id)],
    )

    rendered = render_cases(_cfg(tmp_path, exported["manifest"]), run_dir)

    for page in exported["page_list"]["pages"]:
        composite_path = (
            run_dir / "standin-a" / CASE_DIR / f"{page['manga']}_{page['stem']}.png"
        )
        assert composite_path.is_file(), f"{page['image_id']}: no composite rendered"
        composite = cv2.imread(str(composite_path))
        assert composite is not None
        assert composite.ndim == 3, "the grid is the four panels in colour, not one mask"

    assert rendered["standin-a"] == len(exported["page_list"]["pages"])

    # A method that did not run has no cases and no zero: ``None``, so nothing
    # downstream can print "0 composites" as though it had been rendered.
    awaited = _awaited(run_dir)
    assert awaited, "the fixture run must still await something"
    for method in awaited:
        assert rendered[method] is None


def test_the_case_report_separates_good_cases_from_failures(
    exported, make_handoff, default_provenance, tmp_path
):
    """US4 scenario 4: representative good cases and failure cases, per method.

    The failures carry the runner's own reason, read back from the record
    rather than summarised.  A method that did not run is reported as not run,
    with its reason and with no failure list at all (scenario 5).
    """
    run_dir = tmp_path / "run"
    failed_id = exported["page_list"]["pages"][0]["image_id"]
    _admit(
        exported,
        make_handoff,
        default_provenance,
        run_dir,
        "standin-a",
        _masks(exported["manifest"]),
        failures=[dict(RECORDED_FAILURE, image_id=failed_id)],
    )

    path = write_case_report(_cfg(tmp_path, exported["manifest"]), run_dir)
    assert path.name == "cases.json" and path.is_file()

    document = json.loads(path.read_text(encoding="utf-8"))
    entry = document["methods"]["standin-a"]
    assert entry["ran"] is True

    cases = entry["cases"]
    kinds = [case["case_kind"] for case in cases]
    assert set(kinds) <= {"good", "average", "failure"}
    # One kind per page.  Two pages, one of them failed, leaves one scored page:
    # one good case and no average one, because there is no middle to sit in.
    # Listing the same page as both would be the overlap the singular enum in
    # data-model.md forbids — ``test_the_three_case_kinds_are_three_different_pages``
    # exercises the middle with a pool large enough to have one.
    page_ids = [case["image_id"] for case in cases]
    assert len(page_ids) == len(set(page_ids)), "a page carries one case_kind"

    good = [case for case in cases if case["case_kind"] == "good"]
    assert [case["f1"] for case in good] == sorted(
        (case["f1"] for case in good), reverse=True
    )

    failure = [case for case in cases if case["case_kind"] == "failure"][0]
    assert failure["image_id"] == failed_id
    assert failure["error"] == RECORDED_FAILURE["error"]
    assert failure["reason"] == RECORDED_FAILURE["reason"]
    # No F1: the page did not come back, so it has no score to show — a zero
    # would read as a score (FR-036's rule, on the case list).
    assert failure["f1"] is None

    awaited = _awaited(run_dir)
    for method in awaited:
        absent = document["methods"][method]
        assert absent["ran"] is False
        assert absent["reason"]
        assert absent["cases"] == []
        assert absent["failures"] is None, (
            f"{method} did not run, so the report must not present an empty "
            "failure list as though it had failed nothing"
        )

    # The readable half says the same thing, for a reader who will not open JSON.
    markdown = path.with_suffix(".md").read_text(encoding="utf-8")
    assert "Did not run" in markdown
    assert RECORDED_FAILURE["reason"] in markdown


def _rows(*f1_by_stem: tuple[str, float]) -> list[dict]:
    return [
        {"manga": "ARMS", "stem": stem, "method": "standin-a", "f1": f1}
        for stem, f1 in f1_by_stem
    ]


def test_the_three_case_kinds_are_three_different_pages():
    """The middle case, and the rule that a failed page is not a candidate for it.

    ``000`` carries the best F1 *and* a recorded failure, so it would be picked
    as the good case if the ranked pool were the whole per-page file — which it
    is not: a page the runner reported failed is not a case that came back.  With
    the case count at one, the three kinds are three distinct pages.
    """
    rows = _rows(("000", 0.90), ("001", 0.80), ("002", 0.70),
                 ("003", 0.60), ("004", 0.50))
    failures = [dict(RECORDED_FAILURE, image_id="ARMS/000")]

    cases = _case_entries(rows, failures, n=1)
    by_kind = {case["case_kind"]: case["image_id"] for case in cases}

    # Ranked pool is 001, 002, 003, 004; with n=1 the average window opens at
    # (4 - 1) // 2 = 1, so the middle case is the second-best page.
    assert by_kind == {"good": "ARMS/001", "average": "ARMS/002", "failure": "ARMS/000"}

    # A count wider than the pool picks every scored page once, not twice: the
    # good window swallows the pool and the average window has nothing left.
    wide = _case_entries(rows, failures, n=20)
    picked = [case["image_id"] for case in wide]
    assert len(picked) == len(set(picked)), "a page carries one case_kind"
    assert set(picked) == {"ARMS/000", "ARMS/001", "ARMS/002", "ARMS/003", "ARMS/004"}
