"""The success and failure lists, and what a failure is a failure *of*.

T041 (US4 scenarios 1, 3 and 5; FR-039, FR-041).

US4 is the diagnostic layer: the mean hides which pages broke which method, and
why.  Two of the three behaviours here are about a distinction the numbers alone
cannot make.

The first is scope.  A page that failed inside a method that ran is a *result*:
that method was attempted, this page of it did not come back, and the rest of
its pages are still scored.  A method that never returned is not a result at
all — and the difference has to be visible, because "zero failed pages" is a
claim about a method, while an absent method has made no claim about anything.
So an absent method's lists are ``None``, never ``[]``: an empty list is an
answer, and this method did not give one (FR-036's rule, applied to the failure
list instead of the metric).

The second is that a failure carries its *recorded* reason.  The runner wrote
why, in its own words, in ``errors.json``; the receipt copied that file into the
run verbatim, and it is read back verbatim.  Nothing here composes a reason, and
nothing fills in a missing one — a method whose runner recorded only an error
and no reason is read as having recorded only an error (Rule 1).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from manga_text_seg.benchmark import assemble_comparison, case_lists
from manga_text_seg.imaging import load_raw_page
from manga_text_seg.receipt import admit
from manga_text_seg.report import load_metrics_csv, write_metrics_csv

#: The pages the two-page fixture export hands out.
FIXTURE_PAGES = 2

#: One page's recorded failure, in the shape ``data-model.md`` gives a
#: ``DLFailureRecord`` entry.  ``error`` is what went wrong; ``reason`` is why,
#: and the two are deliberately different strings so a reader that collapses
#: them into one field is caught rather than passing by coincidence.
RECORDED_FAILURE = {
    "image_id": None,  # filled in per test: it must be a page of this page list
    "error": "CUDA out of memory",
    "stage": "inference",
    "reason": "the page's text layer is larger than this method's configured tile",
}


def _baseline(tmp_path: Path, *, failed: int = 0) -> Path:
    """A one-method classical baseline, plus the sweep's failure list.

    Spec 1's sweep writes a row to ``metrics.csv`` only when the pair succeeded
    and an entry to ``failures.json`` only when it did not, so the two files
    already *are* the two lists — the counts are read off them, not derived.
    """
    rows = [
        {
            "manga": "ARMS",
            "stem": f"{index:03d}",
            "method": "otsu",
            "iou": 0.10,
            "precision": 0.20,
            "recall": 0.30,
            "f1": 0.25,
            "tp": 10,
            "fp": 5,
            "fn": 3,
            "inference_time_seconds": 0.01,
        }
        for index in range(FIXTURE_PAGES)
    ]
    path = tmp_path / "classical" / "metrics.csv"
    write_metrics_csv(rows, path)

    failures = {
        "otsu": [
            {"image_id": f"ARMS/failed-{index}", "error": "decode failed"}
            for index in range(failed)
        ]
    }
    (path.parent / "failures.json").write_text(
        json.dumps({"run_id": "classical", "failures": failures}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


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
    """Admit one hand-off, optionally returning a non-empty failure list.

    ``make_handoff`` writes an empty ``errors.json`` with no hook, so a test that
    needs recorded failures writes them into the hand-off itself — which is
    exactly where a runner would have put them.
    """
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


def _failed_page_id(exported) -> str:
    return exported["page_list"]["pages"][0]["image_id"]


def _rows(comparison: dict) -> dict:
    return {row["method"]: row for row in comparison["rows"]}


def test_each_method_has_a_success_list_and_a_failure_list_with_reasons(
    exported, make_handoff, default_provenance, tmp_path
):
    """US4 scenario 1: the two lists exist, and every failure carries its reason."""
    run_dir = tmp_path / "run"
    failed_id = _failed_page_id(exported)
    _admit(
        exported,
        make_handoff,
        default_provenance,
        run_dir,
        "standin-a",
        _masks(exported["manifest"]),
        failures=[dict(RECORDED_FAILURE, image_id=failed_id)],
    )

    lists = case_lists(run_dir)["standin-a"]

    assert lists["method"] == "standin-a"
    assert lists["run_id"] == run_dir.name
    assert failed_id in lists["failed"][0]["image_id"]
    assert lists["failed"][0]["error"] == RECORDED_FAILURE["error"]
    # The reason is the runner's own sentence, read back unchanged.  It is not
    # derived from ``error`` and not replaced by a summary of it.
    assert lists["failed"][0]["reason"] == RECORDED_FAILURE["reason"]
    assert lists["failed"][0]["stage"] == RECORDED_FAILURE["stage"]

    # The two lists partition the pages that were scored: no page is in both,
    # and every scored page is in one of them.
    scored = [
        f"{row['manga']}/{row['stem']}"
        for row in load_metrics_csv(run_dir / "standin-a" / "metrics" / "per-page.csv")
    ]
    assert sorted(lists["successful"] + [e["image_id"] for e in lists["failed"]]) == sorted(scored)
    assert not set(lists["successful"]) & {e["image_id"] for e in lists["failed"]}


def test_a_failed_page_is_failed_for_its_own_method_only_and_the_rest_are_scored(
    exported, make_handoff, default_provenance, tmp_path
):
    """US4 scenario 3: one page's error is contained to one method, one page.

    The remaining pages of that hand-off are still scored — a failure is a
    result about a page, not a reason to abandon the run — and a second method
    admitted alongside it is untouched.
    """
    run_dir = tmp_path / "run"
    failed_id = _failed_page_id(exported)
    masks = _masks(exported["manifest"])

    _admit(
        exported,
        make_handoff,
        default_provenance,
        run_dir,
        "standin-a",
        masks,
        failures=[dict(RECORDED_FAILURE, image_id=failed_id)],
    )
    _admit(exported, make_handoff, default_provenance, run_dir, "standin-b", masks)

    lists = case_lists(run_dir)

    # The failing method: one page failed, the other was still scored.
    assert len(lists["standin-a"]["failed"]) == 1
    assert len(lists["standin-a"]["successful"]) == FIXTURE_PAGES - 1

    # The other method returned everything, and says so with real empty lists —
    # it ran, so an empty failure list is its answer, not a missing one.
    assert lists["standin-b"]["successful"] is not None
    assert len(lists["standin-b"]["successful"]) == FIXTURE_PAGES
    assert lists["standin-b"]["failed"] == []


def test_the_report_separates_a_failed_page_from_a_method_that_never_ran(
    exported, make_handoff, default_provenance, tmp_path
):
    """US4 scenario 5: an absent method is not a method that failed nothing.

    Both halves are asserted together, because the whole point is that they are
    distinguishable: the method that ran has lists, and the method that did not
    has ``None`` — not ``[]``, which would read as "failed nothing".
    """
    run_dir = tmp_path / "run"
    _admit(
        exported,
        make_handoff,
        default_provenance,
        run_dir,
        "standin-a",
        _masks(exported["manifest"]),
    )

    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["methods_awaited"], "the fixture run must still await something"

    lists = case_lists(run_dir)
    ran = lists["standin-a"]
    assert ran["scope"] == "page-level"
    assert ran["successful"] is not None and ran["failed"] == []

    for method in record["methods_awaited"]:
        absent = lists[method]
        assert absent["scope"] == "method-level"
        assert absent["successful"] is None
        assert absent["failed"] is None, (
            f"{method} did not run, so it must not present an empty failure list "
            "as though it had failed nothing"
        )
        assert absent["reason"]


def test_the_summary_counts_successful_and_failed_pages_per_method(
    exported, make_handoff, default_provenance, tmp_path
):
    """FR-039 at the summary level: the counts a reader compares methods by.

    The baseline's counts come from Spec 1's ``failures.json`` rather than from
    the absence of a row — a baseline shown as having failed nothing would be a
    claim about the data that nobody made.
    """
    run_dir = tmp_path / "run"
    failed_id = _failed_page_id(exported)
    _admit(
        exported,
        make_handoff,
        default_provenance,
        run_dir,
        "standin-a",
        _masks(exported["manifest"]),
        failures=[dict(RECORDED_FAILURE, image_id=failed_id)],
    )

    comparison = assemble_comparison(
        baseline_metrics=_baseline(tmp_path, failed=1), run_dir=run_dir
    )
    rows = _rows(comparison)

    assert rows["standin-a"]["successful_pages"] == FIXTURE_PAGES - 1
    assert rows["standin-a"]["failed_pages"] == 1

    # The classical sweep's own counts: its CSV rows are its successes and its
    # failures.json entries are its failures.
    assert rows["otsu"]["successful_pages"] == FIXTURE_PAGES
    assert rows["otsu"]["failed_pages"] == 1

    # A method that did not run has no counts at all — the same rule its metrics
    # follow, for the same reason (FR-036).
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    for method in record["methods_awaited"]:
        assert rows[method]["status"] == "not_run"
        assert rows[method].get("successful_pages") is None
        assert rows[method].get("failed_pages") is None


def test_a_recorded_failure_outside_the_page_list_is_not_counted(
    exported, make_handoff, default_provenance, tmp_path
):
    """The sixty orphan ground-truth masks are not failures of this benchmark.

    A runner that lists a page this run never handed out has recorded something
    real, but it is not a page of this benchmark: it must not be reported as a
    failure, and it must not push the success count negative
    (``spec.md:376``).
    """
    run_dir = tmp_path / "run"
    _admit(
        exported,
        make_handoff,
        default_provenance,
        run_dir,
        "standin-a",
        _masks(exported["manifest"]),
        failures=[dict(RECORDED_FAILURE, image_id="ARMS/not-a-page-of-this-list")],
    )

    lists = case_lists(run_dir)["standin-a"]
    assert len(lists["successful"]) == FIXTURE_PAGES

    rows = _rows(
        assemble_comparison(baseline_metrics=_baseline(tmp_path), run_dir=run_dir)
    )
    assert rows["standin-a"]["successful_pages"] == FIXTURE_PAGES
    assert rows["standin-a"]["failed_pages"] == 0


def test_a_runner_wrapped_failure_list_is_read_as_returned(
    exported, make_handoff, default_provenance, tmp_path
):
    """The recorded shape is the runner's, and the project does not correct it.

    One runner on this project writes its failures under a wrapper carrying the
    method and a count.  The list inside it is the failure list; the wrapper is
    not a failure, and reading it as one would invent a page.
    """
    run_dir = tmp_path / "run"
    failed_id = _failed_page_id(exported)
    _admit(
        exported,
        make_handoff,
        default_provenance,
        run_dir,
        "standin-a",
        _masks(exported["manifest"]),
        failures={
            "method": "standin-a",
            "error_count": 1,
            "errors": [dict(RECORDED_FAILURE, image_id=failed_id)],
        },
    )

    lists = case_lists(run_dir)["standin-a"]
    assert [entry["image_id"] for entry in lists["failed"]] == [failed_id]
    assert len(lists["successful"]) == FIXTURE_PAGES - 1


@pytest.mark.parametrize("field", ["error", "reason", "stage"])
def test_a_missing_field_is_read_as_missing_not_composed(
    exported, make_handoff, default_provenance, tmp_path, field
):
    """Rule 1: nothing is silently corrected, not even in the reader.

    A runner that recorded an error but no reason is read as having recorded no
    reason.  Filling the gap would put the project's words in the runner's
    record.
    """
    run_dir = tmp_path / "run"
    failed_id = _failed_page_id(exported)
    entry = dict(RECORDED_FAILURE, image_id=failed_id)
    del entry[field]
    _admit(
        exported,
        make_handoff,
        default_provenance,
        run_dir,
        "standin-a",
        _masks(exported["manifest"]),
        failures=[entry],
    )

    recorded = case_lists(run_dir)["standin-a"]["failed"][0]
    assert recorded[field] is None
