"""Receipt validation and admission (FR-058, FR-059, FR-057).

A runner returns prediction masks and provenance, not scores.  This module is
the door they come through: six checks in the order fixed by
``contracts/returned-result.md``, then — and only on a full pass — the copy into
the run and the evaluation, which happens here, on this machine, through
``metrics.compute_metrics``.

Nothing is ever silently corrected: a refused hand-off is not resized,
re-thresholded, cropped or dropped page by page, and nothing is written into
the run before every check has passed (FR-058 rule 1 and 3).
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from . import adapters, report
from .adapters.manga_text_segmentation import fold_for_image_id
from .align import align
from .imaging import load_gt_mask
from .manifest import Manifest
from .metrics import compute_metrics
from .normalize import normalize_mask
from .provenance import check_completeness, deviation_refusal

#: The values FR-004 admits; anything else is a different mask convention.
MASK_VALUES = {0, 255}

#: The run record's filename, beside the per-method directories in a run
#: (T024).  Beside and not inside: the run's contents are the methods it has
#: admitted, and the record is about all of them (FR-044).
RUN_RECORD_NAME = "run.json"

#: FR-047: what re-running a method over the same page list is *not* pinned to,
#: listed rather than left implicit.  Neither is a defect — the page list,
#: alignment convention, mask format and evaluation procedure are fixed, so the
#: evaluated image list and the metric definitions are the same; only the
#: environment the masks came out of differs, and that is recorded per method.
NONDETERMINISM_SOURCES = (
    "each method's environment: interpreter, package versions, device and "
    "thread scheduling are recorded per method, never once for the run "
    "(FR-044, FR-052), so the same method re-run elsewhere may return "
    "different masks",
    "Method A's leave-one-fold-out attribution: five checkpoints are released, "
    "so which one scores a page follows from the page's book rather than from "
    "anything this run fixes (FR-013a)",
)

#: Sidecar fields lifted into the run's per-method configuration snapshot.
#: FR-024a and FR-024b: the padding multiple that was inverted, the threshold
#: that actually scored, and the collapse rule that produced the binary mask
#: all belong in the run record, because a reader cannot re-derive them from
#: the mask and the sidecar is a runner's file (FR-052), not this project's.
_CONFIGURATION_KEYS = (
    "input_size",
    "threshold",
    "padding_multiple",
    "effective_threshold",
    "preprocessing",
    "postprocessing",
    "label_collapse",
)

#: Methods whose every page must carry a fold attribution (FR-013a).  A
#: hand-off can arrive with no adapter loaded, so this is a constant rather
#: than a lookup into the adapter registry.
FOLD_ATTRIBUTION_METHODS = frozenset({"manga-text-segmentation"})

#: The folds the five released checkpoints cover.
FOLDS = frozenset(range(5))

#: ONBOARDING §6 spells the attribution ``"fold_<n>"``, and the delivered
#: Method A hand-off uses that spelling throughout.  Anything else — ``"2"``,
#: ``"fold.2"``, ``"fold_2.pkl"`` — is not coerced into it: a runner that
#: returned a different spelling returned something this project did not ask
#: for, and FR-058 rule 1 forbids silently correcting it.
FOLD_ATTRIBUTION_PREFIX = "fold_"


def _attributed_fold(value: object) -> int | None:
    """The fold an attribution names, or ``None`` if it is not ``"fold_<n>"``."""
    if not isinstance(value, str) or not value.startswith(FOLD_ATTRIBUTION_PREFIX):
        return None
    digits = value[len(FOLD_ATTRIBUTION_PREFIX) :]
    return int(digits) if digits.isdigit() else None


class ReceiptError(ValueError):
    """A hand-off was refused, naming the check and the field or page that failed."""

    def __init__(
        self,
        check: str,
        reason: str,
        *,
        field: str | None = None,
        image_id: str | None = None,
    ) -> None:
        super().__init__(f"[{check}] {reason}")
        self.check = check
        self.reason = reason
        self.field = field
        self.image_id = image_id


def _read_json(path: Path) -> dict | None:
    """The parsed document, or ``None`` when it is absent or unreadable.

    Absent and malformed are the same answer on purpose: both mean the runner
    did not return the record, and the refusal names the field either way.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _sidecar_path(handoff: Path, page: dict) -> Path:
    return handoff / "metadata" / page["manga"] / f"{page['stem']}.json"


def _mask_path(handoff: Path, page: dict) -> Path:
    return handoff / "masks" / page["manga"] / f"{page['stem']}.png"


def _read_mask(path: Path) -> np.ndarray | None:
    """The mask exactly as stored, or ``None`` if it is not there.

    Read through cv2 rather than ``imaging.load_mask`` so checks 3 and 4 stay
    separable: ``load_mask`` folds "single channel" and "values ⊆ {0,255}" into
    one error naming the path, and the contract requires the *check* that failed
    to be nameable.
    """
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return image


def validate_handoff(
    handoff: Path,
    *,
    page_list: dict,
    identity_record: dict,
) -> None:
    """Run the FR-058 checks in order; raise on the first failure.

    Returns ``None`` when the hand-off is admissible.  Writes nothing.
    """
    handoff = Path(handoff)
    pages = page_list["pages"]
    width, height = page_list["aligned_size"]

    sidecars = {page["image_id"]: _read_json(_sidecar_path(handoff, page)) for page in pages}

    # 1. The result answers *this* page list, not a stale one.
    expected_list = page_list["page_list_identity"]
    for page in pages:
        record = sidecars[page["image_id"]]
        quoted = record.get("page_list_identity") if record else None
        if quoted != expected_list:
            raise ReceiptError(
                "page-list-identity",
                f"{page['image_id']}: page list identity {quoted!r} is not the "
                f"identity of the committed export ({expected_list}); a result "
                f"produced against another list is refused, not aligned by name",
                field="page_list_identity",
                image_id=page["image_id"],
            )

    # 2. ...applied to *these* images.
    for page in pages:
        image_id = page["image_id"]
        entry = identity_record.get(image_id)
        if entry is None:
            raise ReceiptError(
                "input-image-identity",
                f"{image_id}: not in the committed identity record",
                field="input_image_identity",
                image_id=image_id,
            )
        quoted = (sidecars[image_id] or {}).get("input_image_identity")
        if quoted != entry["sha256"]:
            raise ReceiptError(
                "input-image-identity",
                f"{image_id}: quoted input identity {quoted!r} is not the "
                f"identity of the distributed image ({entry['sha256']})",
                field="input_image_identity",
                image_id=image_id,
            )

    # 3. Every mask is there, single-channel, and the aligned size.
    masks: dict[str, np.ndarray] = {}
    for page in pages:
        image_id = page["image_id"]
        mask = _read_mask(_mask_path(handoff, page))
        if mask is None:
            raise ReceiptError(
                "mask",
                f"{image_id}: no mask at {_mask_path(handoff, page)}",
                image_id=image_id,
            )
        if mask.ndim != 2 or mask.dtype != np.uint8:
            raise ReceiptError(
                "mask",
                f"{image_id}: expected single-channel uint8, got shape "
                f"{mask.shape} dtype {mask.dtype}",
                image_id=image_id,
            )
        if mask.shape != (height, width):
            raise ReceiptError(
                "mask",
                f"{image_id}: mask is {mask.shape[1]}x{mask.shape[0]}, the page "
                f"list records {width}x{height}; refused, not resized",
                image_id=image_id,
            )
        masks[image_id] = mask

    # 4. ...in the shared value convention.
    for page in pages:
        image_id = page["image_id"]
        values = set(np.unique(masks[image_id]).tolist())
        if not values <= MASK_VALUES:
            raise ReceiptError(
                "mask-values",
                f"{image_id}: mask carries {sorted(values - MASK_VALUES)}, "
                f"outside the convention {sorted(MASK_VALUES)}; refused, not "
                f"re-thresholded",
                image_id=image_id,
            )

    # 5. The provenance record is complete (FR-056, SC-014).
    provenance = _read_json(handoff / "provenance.json")
    if provenance is None:
        raise ReceiptError(
            "provenance",
            "no readable provenance.json in the hand-off",
            field="provenance",
        )
    missing = check_completeness(provenance)
    if missing:
        raise ReceiptError(
            "provenance",
            f"provenance record is incomplete: {', '.join(missing)}; refused "
            f"rather than accepted with gaps (FR-056)",
            field=missing[0],
        )

    # 6. The runner's deviations, recorded rather than absorbed (FR-055).
    #
    # A workaround that changed the mask makes the result a *different* method,
    # so it is refused here rather than reported under the pinned method's name.
    # A workaround that left the mask alone passes and stays recorded.
    deviation = deviation_refusal(provenance)
    if deviation is not None:
        raise ReceiptError("deviation", deviation, field="deviation.changes_output")

    # 7. Method A's leave-one-fold-out attribution (FR-013a).
    #
    # Upstream held every book out of exactly one of its five folds, so for any
    # page at most one released checkpoint never trained on it.  The sidecar
    # names that fold; this check derives it from the published seed and refuses
    # the page when the two disagree.  The delivered Method A run is refused
    # here, correctly: it attributes all 390 pages to fold 0, whose training
    # split contained 36 of the 45 books.
    if provenance.get("method") in FOLD_ATTRIBUTION_METHODS:
        for page in pages:
            image_id = page["image_id"]
            quoted = (sidecars[image_id] or {}).get("fold_attribution")
            claimed = _attributed_fold(quoted)
            if claimed is None or claimed not in FOLDS:
                raise ReceiptError(
                    "fold-attribution",
                    f"{image_id}: fold_attribution {quoted!r} is not one of the "
                    f"five released folds, spelled {FOLD_ATTRIBUTION_PREFIX}<n> "
                    f"for n in {sorted(FOLDS)}; Method A is inadmissible without "
                    f"it (FR-013a)",
                    field="fold_attribution",
                    image_id=image_id,
                )

            entitled = fold_for_image_id(image_id)
            if entitled is None:
                # FR-013a: "a mapping that cannot be reproduced and verified
                # makes method A's result inadmissible — reported unavailable
                # with the reason (FR-018) — not approximated."
                raise ReceiptError(
                    "fold-attribution",
                    f"{image_id}: the book this page belongs to is not in the "
                    f"upstream split, so no fold can be verified as having held "
                    f"it out; Method A is inadmissible for this page rather "
                    f"than approximated (FR-013a)",
                    field="fold_attribution",
                    image_id=image_id,
                )
            if claimed != entitled:
                raise ReceiptError(
                    "fold-attribution",
                    f"{image_id}: scored by fold {claimed}, but this page's book "
                    f"was held out of fold {entitled} — the fold {claimed} "
                    f"checkpoint trained on it, so the page is not a held-out "
                    f"prediction (FR-013a)",
                    field="fold_attribution",
                    image_id=image_id,
                )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def update_run_record(
    run_dir: Path,
    *,
    page_list: dict,
    method: str,
    provenance: dict,
    sidecars: list[dict],
) -> dict:
    """Merge one admitted method into the run's record and write it back (T024).

    Every field is per method, keyed by method name, because a run is assembled
    from results produced in different environments (FR-044, FR-052) — there is
    no run-level device or package version to record, and writing one would be
    the assumption FR-044 forbids.  Nothing already in the record is rewritten:
    admitting a second method adds a key and leaves the first alone (FR-059).
    """
    run_dir = Path(run_dir)
    path = run_dir / RUN_RECORD_NAME
    record = _read_json(path)

    if record is None:
        record = {
            "run_id": run_dir.name,
            "page_list_identity": page_list["page_list_identity"],
            "methods_admitted": [],
            "methods_awaited": [],
            "method_configuration": {},
            "method_provenance": {},
            "method_device": {},
            "admission_timestamps": {},
            "started_at": _now(),
            "ended_at": None,
            "nondeterminism_sources": list(NONDETERMINISM_SOURCES),
        }
    elif record.get("page_list_identity") != page_list["page_list_identity"]:
        # Results produced against two different page lists are not one run:
        # FR-044 records the identity they were *all* produced against.
        raise ReceiptError(
            "run-record",
            f"run {record['run_id']!r} holds results produced against page list "
            f"{record['page_list_identity']}, but this hand-off answers "
            f"{page_list['page_list_identity']}; admit it into its own run",
            field="page_list_identity",
        )

    admitted = set(record["methods_admitted"]) | {method}
    record["methods_admitted"] = sorted(admitted)
    # FR-049: the run says which methods it still awaits.  The registry is the
    # only list of what could arrive; a method with no adapter yet is simply
    # not awaited by this project, which is why T025/T039/T040 widen it.
    record["methods_awaited"] = sorted(set(adapters.available()) - admitted)
    record["method_provenance"][method] = provenance
    record["method_device"][method] = provenance.get("device")
    record["admission_timestamps"][method] = _now()
    record["method_configuration"][method] = {
        key: sidecars[0][key] for key in _CONFIGURATION_KEYS if key in sidecars[0]
    }
    # FR-047: "ended" is a claim that every awaited method has returned.  It is
    # made only when that is true, and un-made if the registry later grows.
    record["ended_at"] = _now() if not record["methods_awaited"] else None

    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def admit(
    handoff: Path,
    *,
    page_list: dict,
    identity_record: dict,
    run_dir: Path,
    manifest: Manifest,
    method: str | None = None,
) -> list[dict]:
    """Validate a hand-off, then admit it into ``run_dir`` and score it.

    Returns one row per page, carrying ``image_id`` and the metric keys, so a
    caller can compare the two paths that meet here (FR-057).
    """
    handoff = Path(handoff)
    provenance = _read_json(handoff / "provenance.json") or {}
    declared = provenance.get("method")

    # An operator mistake, not a seventh contract check: catch it before the
    # six checks so a mislabelled directory is named as such rather than
    # reported as whatever it happens to fail first.
    if method is not None and declared != method:
        raise ReceiptError(
            "method",
            f"--method {method!r} but the hand-off's provenance declares "
            f"{declared!r}; refusing before anything is written",
            field="method",
        )

    validate_handoff(handoff, page_list=page_list, identity_record=identity_record)

    method_dir = Path(run_dir) / declared
    (method_dir / "masks").mkdir(parents=True, exist_ok=True)
    (method_dir / "metadata").mkdir(parents=True, exist_ok=True)

    masks: dict[str, np.ndarray] = {}
    for page in page_list["pages"]:
        image_id = page["image_id"]
        source_mask = _mask_path(handoff, page)
        destination_mask = method_dir / "masks" / page["manga"] / f"{page['stem']}.png"
        destination_mask.parent.mkdir(parents=True, exist_ok=True)
        # A byte copy: the mask is scored as returned, and T017 asserts the
        # stored bytes still equal what the runner handed over.
        shutil.copy2(source_mask, destination_mask)
        masks[image_id] = _read_mask(destination_mask)

        source_meta = _sidecar_path(handoff, page)
        destination_meta = method_dir / "metadata" / page["manga"] / f"{page['stem']}.json"
        destination_meta.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_meta, destination_meta)

    shutil.copy2(handoff / "provenance.json", method_dir / "provenance.json")
    errors_source = handoff / "errors.json"
    if errors_source.is_file():
        shutil.copy2(errors_source, method_dir / "errors.json")
    else:
        (method_dir / "errors.json").write_text("[]\n", encoding="utf-8")

    # Only now — after every mask and record is stored — does the project score
    # the result, here, with the one shared procedure.
    rows = []
    by_id = {pair.image_id: pair for pair in manifest.pairs}
    sidecars = []
    for page in page_list["pages"]:
        image_id = page["image_id"]
        pair = by_id[image_id]
        ground_truth = normalize_mask(load_gt_mask(pair.mask_path))
        aligned = align(ground_truth, masks[image_id], page_list["alignment"])

        row = {
            "manga": page["manga"],
            "stem": page["stem"],
            "method": declared,
        }
        row.update(compute_metrics(aligned.mask, aligned.page))
        sidecar = _read_json(method_dir / "metadata" / page["manga"] / f"{page['stem']}.json") or {}
        sidecars.append(sidecar)
        row["inference_time_seconds"] = sidecar.get("inference_time_seconds")
        row["image_id"] = image_id
        rows.append(row)

    report.write_metrics_csv(rows, method_dir / "metrics" / "per-page.csv")
    report.write_summaries_json(rows, method_dir / "metrics" / "summary.json", Path(run_dir).name)

    # Last, so a run that failed to score never claims to have admitted anything.
    update_run_record(
        Path(run_dir),
        page_list=page_list,
        method=declared,
        provenance=provenance,
        sidecars=sidecars,
    )
    return rows
