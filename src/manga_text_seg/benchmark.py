"""The combined comparison: the classical baseline plus every admitted method.

T037 (FR-034, FR-037, FR-059; US3 scenarios 1 and 2).

Nothing here runs inference.  The classical baseline is read from Spec 1's
persisted ``metrics.csv`` and each DL method from the ``metrics/per-page.csv``
its receipt already wrote, so the comparison is assembled from results that
exist — and admitting a later result cannot move an earlier row, because no
earlier row is recomputed (FR-059).

This module is also the one place that knows *which* method a row belongs to,
and that is deliberate.  FR-051 forbids method-specific branching in the shared
evaluation and rendering path, but two facts a row must carry are per method
and cannot be derived: the device that produced a timing (FR-060) and the
contamination disclosure that has to sit beside a score (SC-011).  They are
resolved here, and ``report.py`` renders whatever string it is handed.

The disclosure is three different statements, not one flag.  Method A is
de-contaminated by construction, method C is fully contaminated, and method B's
extent is unknowable — SC-011 requires the extent to be quantified where it is
known and stated as unknown where it is not, so a single boolean would be a
false answer for two of the three.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .availability import probe_device
from .report import load_metrics_csv

RUN_RECORD_NAME = "run.json"

#: Where a method's admitted per-page rows live, relative to the run directory.
METRICS_NAME = Path("metrics") / "per-page.csv"

#: The four accuracy metrics, mean and standard deviation alike (FR-037).
METRIC_KEYS = ("iou", "precision", "recall", "f1")

#: The baseline trains on nothing, so there is nothing for the evaluation
#: dataset to overlap with — the one method with a settled answer.
_NO_OVERLAP = (
    "no overlap: this method is the classical baseline and learns no "
    "parameters from any dataset, so it has no training data that could "
    "overlap with the evaluation set"
)

#: A method the project did not configure.  Stated as unknown rather than
#: assumed clean: not having looked is not the same as having found nothing.
_UNIDENTIFIED = (
    "contamination extent unknown: this method's training data was not "
    "identified, so no overlap with the evaluation dataset could be "
    "established"
)

#: The three configured methods, each with its own answer (SC-011).
_DISCLOSURES = {
    "manga-text-segmentation": (
        "de-contaminated by construction: leave-one-fold-out attribution "
        "(FR-013a) means each page is scored only by the checkpoint whose fold "
        "held that page's book out, so no scored page was trained on"
    ),
    "comic-text-detector": (
        "contamination extent unknown: the training split behind the released "
        "checkpoint is not published, so the extent of any overlap with the "
        "evaluation dataset cannot be established either way"
    ),
    "unetpp-efficientnetv2": (
        "fully contaminated: the released checkpoint was trained on all 45 "
        "books of the evaluation ground truth, so every score it reports is a "
        "training-set score"
    ),
}

#: ponytail: the recorded availability verdict is the richer reason, and the
#: CLI already prints it from ``benchmark/availability.json``.  Reading it here
#: would mean handing this function a repository root it does not otherwise
#: need, so the row says what the run itself knows and the verdict stays where
#: it is recorded.  Add the lookup when a reader wants both in one place.
_AWAITED_REASON = "not run: no result for this method has been admitted to this run"


def _disclosure(method: str) -> str:
    """This method's contamination status, quantified or declared unknown."""
    return _DISCLOSURES.get(method, _UNIDENTIFIED)


def _scored(
    method: str,
    pages: list[dict],
    device: dict | None,
    disclosure: str,
    *,
    successful_pages: int,
    failed_pages: int,
) -> dict:
    """One method's row: four metrics with mean and std, its timing, and counts."""
    frame = pd.DataFrame(pages)
    times = frame["inference_time_seconds"]
    return {
        "method": method,
        "status": "scored",
        "metrics": {
            key: {
                "mean": float(frame[key].mean()),
                "std": float(frame[key].std(ddof=0)),
            }
            for key in METRIC_KEYS
        },
        "timing": {
            "mean_seconds": float(times.mean()),
            "total_seconds": float(times.sum()),
        },
        "successful_pages": successful_pages,
        "failed_pages": failed_pages,
        "device": device,
        "reason": None,
        "disclosure": disclosure,
    }


def _awaited(method: str) -> dict:
    """A method the run still awaits: present, explained, and carrying no number.

    FR-036 forbids a placeholder, so the metric and timing fields are null
    rather than zero — a zero is a score, and this method has not produced one.
    """
    return {
        "method": method,
        "status": "not_run",
        "metrics": None,
        "timing": None,
        "device": None,
        "reason": _AWAITED_REASON,
        "disclosure": _disclosure(method),
    }


def assemble_comparison(baseline_metrics: Path, run_dir: Path) -> dict:
    """Assemble the run's comparison from results that already exist.

    ``baseline_metrics`` is Spec 1's ``metrics.csv``; ``run_dir`` is the
    directory each admitted method's receipt wrote into, whose ``run.json``
    names what was admitted and what is still awaited.
    """
    run_dir = Path(run_dir)
    record = json.loads((run_dir / RUN_RECORD_NAME).read_text(encoding="utf-8"))

    rows = []
    # The baseline is every classical method in that one file, and the classical
    # methods share the one device — this environment's, which is the only
    # device anything here can be attributed to (FR-060).
    baseline_metrics = Path(baseline_metrics)
    baseline = pd.DataFrame(load_metrics_csv(baseline_metrics))
    baseline_failures = _baseline_failures(baseline_metrics)
    device = probe_device()
    for method, group in baseline.groupby("method", sort=True):
        # Spec 1's sweep writes a row to ``metrics.csv`` only when the pair
        # succeeded and an entry to ``failures.json`` only when it did not, so
        # the two files are the two lists already — nothing is subtracted here.
        rows.append(
            _scored(
                str(method),
                group.to_dict(orient="records"),
                device,
                _NO_OVERLAP,
                successful_pages=len(group),
                failed_pages=len(baseline_failures.get(str(method), [])),
            )
        )

    devices = record.get("method_device", {})
    for method in record.get("methods_admitted", []):
        pages = load_metrics_csv(run_dir / method / METRICS_NAME)
        scored_ids = _scored_page_ids(run_dir / method)
        # Intersected, not subtracted blindly: a runner that lists a page this
        # run never handed out has recorded something real, but it is not a page
        # of this benchmark, and it must not turn the success count negative or
        # be reported as a failure of a page that was never evaluated
        # (spec.md:376, the sixty ground-truth masks with no source image).
        failed = len({entry["image_id"] for entry in _failure_entries(run_dir / method)} & set(scored_ids))
        rows.append(
            _scored(
                method,
                pages,
                devices.get(method) or device,
                _disclosure(method),
                successful_pages=len(scored_ids) - failed,
                failed_pages=failed,
            )
        )

    for method in record.get("methods_awaited", []):
        rows.append(_awaited(method))

    rows.sort(key=lambda row: row["method"])
    return {
        "run_id": record["run_id"],
        "page_list_identity": record["page_list_identity"],
        "rows": rows,
    }


def write_comparison(comparison: dict, path: Path) -> Path:
    """Persist the comparison, so the run holds the table it reported."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


# --- The per-method success and failure lists (T041; FR-039, US4) ------------

#: The fields a failure entry carries.  ``error`` and ``reason`` are not the same
#: thing and neither is derived from the other: a runner that recorded only one
#: of them is read as having recorded only one of them (Rule 1, "nothing is
#: silently corrected").
_FAILURE_FIELDS = ("image_id", "error", "stage", "reason")

#: A method that ran failed pages, if any.  A method that did not run has no
#: lists at all — see ``case_lists``.
_PAGE_SCOPE = "page-level"
_METHOD_SCOPE = "method-level"


def _failure_entries(method_dir: Path) -> list[dict]:
    """The failures this method's hand-off recorded, verbatim and re-shaped.

    The receipt copies the runner's ``errors.json`` into the run as returned, so
    two shapes are already on disk: a bare list, and a runner's own wrapper
    carrying the list under ``errors``.  Both are read; neither is corrected.
    """
    document = json.loads((method_dir / "errors.json").read_text(encoding="utf-8"))
    if isinstance(document, dict):
        document = document.get("errors") or []
    return [{field: entry.get(field) for field in _FAILURE_FIELDS} for entry in document]


def _scored_page_ids(method_dir: Path) -> list[str]:
    """Every page this method was scored on, as ``manga/stem``.

    The persisted per-page rows identify a page by its two path parts rather
    than by ``image_id``, so the join is made here and in one place.
    """
    return [
        f"{row['manga']}/{row['stem']}"
        for row in load_metrics_csv(method_dir / METRICS_NAME)
    ]


def _baseline_failures(baseline_metrics: Path) -> dict[str, list[dict]]:
    """Spec 1's ``failures.json``, keyed by method.

    The classical sweep already records which pages it could not process, and a
    baseline shown as having failed nothing would be a claim about the data that
    nobody made.  The file is absent whenever the sweep had no failures to
    write, which is the ordinary case and reads as no failures.
    """
    path = Path(baseline_metrics).parent / "failures.json"
    if not path.is_file():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    return {str(method): list(entries) for method, entries in (document.get("failures") or {}).items()}


def case_lists(run_dir: Path) -> dict[str, dict]:
    """Per method: the pages that succeeded, the pages that failed, and why.

    FR-039, read off the run as it stands.  A method that ran has two real lists
    — one of them may be empty, because for a method that ran an empty failure
    list is an answer.  A method that did not run has no lists at all: ``None``,
    not ``[]``, so nothing downstream can read its absence as "failed nothing"
    (US4 scenario 5, the same rule FR-036 applies to its metrics).
    """
    run_dir = Path(run_dir)
    record = json.loads((run_dir / RUN_RECORD_NAME).read_text(encoding="utf-8"))
    run_id = record["run_id"]

    lists: dict[str, dict] = {}
    for method in record.get("methods_admitted", []):
        method_dir = run_dir / method
        failures = _failure_entries(method_dir)
        failed = {entry["image_id"] for entry in failures}
        lists[method] = {
            "method": method,
            "run_id": run_id,
            "scope": _PAGE_SCOPE,
            "successful": [
                page_id
                for page_id in _scored_page_ids(method_dir)
                if page_id not in failed
            ],
            "failed": failures,
            "reason": None,
            # ponytail: the availability verdict holds the remediation guidance
            # and lives outside the run; add the lookup when a reader wants it
            # beside the failure rather than in the availability report.
            "remediation": None,
        }

    for method in record.get("methods_awaited", []):
        lists[method] = {
            "method": method,
            "run_id": run_id,
            "scope": _METHOD_SCOPE,
            "successful": None,
            "failed": None,
            "reason": _AWAITED_REASON,
            "remediation": None,
        }
    return lists
