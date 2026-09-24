"""Visualization: 4-panel comparison composites (FR-032, FR-033).

Panels: raw page | ground-truth mask | prediction mask | overlay.
Selection modes (FR-033):
  - ``first-N`` : first N pairs in deterministic manifest order (default)
  - ``best-N``  : N pairs ranked by F1, best first
  - ``worst-N`` : N pairs ranked by F1, worst first

Runs from persisted metrics.csv WITHOUT re-running inference
(contract output-layout.md rule 4).  Output:
  outputs/segmentation/<run_id>/viz/<method>/<mode>-<n>/<manga>_<stem>.png
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .align import align
from .benchmark import METRICS_NAME, case_lists
from .config import Config
from .imaging import load_gt_mask, load_mask, load_raw_page
from .manifest import Manifest, load_manifest
from .normalize import normalize_mask
from .report import load_metrics_csv

OVERLAY_GT_COLOR = (0, 255, 0)      # green: GT only
OVERLAY_PRED_COLOR = (255, 0, 0)    # blue: prediction only
OVERLAY_BOTH_COLOR = (0, 255, 255)  # yellow: overlap
OVERLAY_BG_COLOR = (40, 40, 40)

# Directory labels for selection modes (FR-033). quickstart.md documents the
# composite path as viz/<method>/<mode>-<n>/ with the SHORT label
# ("first-20", "best-10", ...), NOT the raw config value "first-N" — so the
# config mode string is mapped to its label here.
MODE_LABELS = {
    "first-N": "first",
    "best-N": "best",
    "worst-N": "worst",
}


def _selection_image_ids(
    cfg: Config,
    manifest: Manifest,
    metrics_by_method: dict[str, dict[str, float]],
    method: str,
) -> list[tuple[str, str]]:
    """Select (manga, stem) pairs for one method per FR-033."""
    n = cfg.viz_n
    if cfg.viz_mode == "first-N":
        selected = [(p.manga, p.stem) for p in manifest.pairs[:n]]
    else:
        f1_by_id = metrics_by_method.get(method, {})
        ranked = sorted(
            f1_by_id.items(),
            key=lambda kv: (kv[1], kv[0]),
            reverse=(cfg.viz_mode == "best-N"),
        )
        selected = []
        for image_id, _ in ranked[:n]:
            manga, stem = image_id.split("/", 1)
            selected.append((manga, stem))
    return selected


def _build_overlay(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """GT-green / pred-blue / overlap-yellow overlay (FR-032 panel 4)."""
    h, w = gt.shape
    overlay = np.full((h, w, 3), OVERLAY_BG_COLOR, dtype=np.uint8)
    gt_bool = gt == 255
    pred_bool = pred == 255
    overlay[gt_bool & ~pred_bool] = OVERLAY_GT_COLOR
    overlay[pred_bool & ~gt_bool] = OVERLAY_PRED_COLOR
    overlay[gt_bool & pred_bool] = OVERLAY_BOTH_COLOR
    return overlay


def _render_composite(row_img: np.ndarray, gt: np.ndarray,
                      pred: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Assemble a 2x2 grid of the four panels, resized to a common short side."""
    panels = [row_img, gt, pred, overlay]
    short_side = 500
    scaled = []
    for panel in panels:
        if panel.ndim == 2:
            panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        h, w = panel.shape[:2]
        scale = short_side / max(h, w)
        scaled.append(cv2.resize(panel, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA))
    ph = max(p.shape[0] for p in scaled)
    pw = max(p.shape[1] for p in scaled)
    padded = [cv2.copyMakeBorder(p, 0, ph - p.shape[0], 0, pw - p.shape[1],
                                 cv2.BORDER_CONSTANT, value=OVERLAY_BG_COLOR)
              for p in scaled]
    top = np.hstack([padded[0], padded[1]])
    bottom = np.hstack([padded[2], padded[3]])
    return np.vstack([top, bottom])


def _panels(
    cfg: Config,
    manifest: Manifest,
    method: str,
    manga: str,
    stem: str,
    *,
    pred_dir: Path | None = None,
) -> list[np.ndarray]:
    """Load the four panels for one pair (or fewer if a file is missing).

    ``pred_dir`` overrides where the prediction is looked for.  The classical
    sweep writes predictions under ``cfg.output_dir/<method>``; an admitted run
    keeps them under its own ``masks/``.  It is a location, not a method, so
    nothing here branches on a method's name (FR-051).
    """
    pair = next((p for p in manifest.pairs
                 if p.manga == manga and p.stem == stem), None)
    panels = []
    if pair is not None:
        panels.append(load_raw_page(pair.raw_path))
        gt = normalize_mask(load_gt_mask(pair.mask_path))
        pred_root = Path(pred_dir) if pred_dir is not None else cfg.output_dir / method
        pred_path = pred_root / manga / f"{stem}.png"
        if pred_path.is_file():
            pred = load_mask(pred_path)
            # GT masks are 6x2 px larger than raw pages in this dataset (R2);
            # align GT to the prediction's geometry (crop-topleft, FR-015) so
            # the overlay and panels share the exact dimensions the sweep's
            # metrics were computed on. Without this, _build_overlay would
            # broadcast-crash on mismatched GT/pred shapes.
            aligned = align(gt, pred, cfg.alignment_strategy)
            panels.append(aligned.mask)
            panels.append(aligned.page)
        else:
            panels.append(gt)
    return panels


def render_composites(cfg: Config) -> dict:
    """Render composites for every configured method (FR-032, FR-033)."""
    manifest_path = cfg.output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json not found at {manifest_path}; run 'discover' first")
    manifest = load_manifest(manifest_path)

    metrics_path = cfg.output_dir / "metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics.csv not found at {metrics_path}; run 'sweep' first")
    rows = load_metrics_csv(metrics_path)

    metrics_by_method: dict[str, dict[str, float]] = {}
    for row in rows:
        metrics_by_method.setdefault(row["method"], {})[
            f"{row['manga']}/{row['stem']}"] = float(row["f1"])

    rendered: dict[str, int] = {}
    out_root = cfg.output_dir / "viz"
    for entry in cfg.methods:
        method = entry["name"]
        selected = _selection_image_ids(cfg, manifest, metrics_by_method, method)
        count = 0
        for manga, stem in selected:
            panels = _panels(cfg, manifest, method, manga, stem)
            if len(panels) == 3:
                overlay = _build_overlay(panels[1], panels[2])
                composite = _render_composite(panels[0], panels[1], panels[2], overlay)
                out_dir = out_root / method / f"{MODE_LABELS[cfg.viz_mode]}-{cfg.viz_n}"
                out_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_dir / f"{manga}_{stem}.png"), composite)
                count += 1
        rendered[method] = count
    return rendered


# --- US4: the case report over an admitted run (T042, T043; FR-040, FR-041) ---
#
# A different artefact from ``render_composites`` above.  That one illustrates a
# classical sweep from ``cfg.output_dir``; this one illustrates an admitted DL
# run from ``deliverables/<run_id>/``, whose masks are the runner's returned
# predictions.  Same four panels, same overlay, same composite grid — the only
# difference is where the prediction is read from and which cases are picked.
#
# The prediction location is a path, not a method name: nothing below branches
# on which method it is rendering (FR-051, US1 scenario 6).

#: The three case kinds the report must cover (ONBOARDING §8: a good case, an
#: average one, a failure — and never only the first pages of the manifest).
CASE_KINDS = ("good", "average", "failure")


def _ranked_cases(rows: list[dict]) -> list[dict]:
    """Every scored page of one method, best F1 first.

    Ties break on ``image_id`` so the same run always picks the same cases.
    """
    ranked = sorted(rows, key=lambda row: (-float(row["f1"]), f"{row['manga']}/{row['stem']}"))
    return [{"image_id": f"{row['manga']}/{row['stem']}", "f1": float(row["f1"])} for row in ranked]


def _case_entries(rows: list[dict], failures: list[dict], n: int) -> list[dict]:
    """The good, average and failure cases for one method (FR-041).

    ``good`` is the best ``n`` by F1, ``average`` the middle ``n``, ``failure``
    the pages the hand-off recorded as failed.  A page the runner reported
    failed is not a candidate for either ranked kind — it is not a case that
    came back — and a page picked for both kinds is listed once, as ``good``:
    a method with few pages has fewer cases, not the same page twice.
    """
    failed_ids = {entry.get("image_id") for entry in failures}
    scored = [row for row in rows
              if f"{row['manga']}/{row['stem']}" not in failed_ids]
    ranked = _ranked_cases(scored)
    middle = max((len(ranked) - n) // 2, 0)

    entries: list[dict] = []
    seen: set[str] = set()
    for kind, window in (("good", ranked[:n]), ("average", ranked[middle:middle + n])):
        for case in window:
            if case["image_id"] in seen:
                continue
            seen.add(case["image_id"])
            entries.append({**case, "case_kind": kind})
    entries.extend(
        {"image_id": entry.get("image_id"), "f1": None, "case_kind": "failure",
         "error": entry.get("error"), "stage": entry.get("stage"),
         "reason": entry.get("reason")}
        for entry in failures
    )
    return entries


def _composite_path(out_dir: Path, image_id: str) -> Path:
    manga, stem = image_id.split("/", 1)
    return out_dir / f"{manga}_{stem}.png"


#: Where a method's cases land, under its own directory in the run.  Named for
#: what it holds rather than for a selection mode: the cases are the best, the
#: middle and the failed pages, not the first N of anything (ONBOARDING §8).
CASE_DIR = "visualizations"


def render_cases(cfg: Config, run_dir: Path) -> dict:
    """Render the four-panel cases for every admitted method (FR-040).

    Reads the admitted masks and the persisted per-page metrics; inference is
    never re-run (US4's independent test runs this against an existing run).
    Returns ``{method: count}`` for the methods that ran, and for a method that
    did not, ``{method: None}`` — it has no cases to render and no zero to show.
    """
    run_dir = Path(run_dir)
    manifest_path = cfg.output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json not found at {manifest_path}; run 'discover' first")
    manifest = load_manifest(manifest_path)

    lists = case_lists(run_dir)
    rendered: dict[str, int | None] = {}

    for method, lists_for_method in lists.items():
        if lists_for_method["successful"] is None:
            rendered[method] = None
            continue

        rows = load_metrics_csv(run_dir / method / METRICS_NAME)
        cases = _case_entries(rows, lists_for_method["failed"], cfg.viz_n)
        pred_dir = run_dir / method / "masks"
        out_dir = run_dir / method / CASE_DIR

        written: set[Path] = set()
        for case in cases:
            image_id = case["image_id"]
            if not image_id:
                # A failure recorded without a page id names no page, so there
                # is no page to illustrate; it is still reported (Rule 1).
                continue
            manga, stem = image_id.split("/", 1)
            panels = _panels(cfg, manifest, method, manga, stem, pred_dir=pred_dir)
            if len(panels) != 3:
                continue
            overlay = _build_overlay(panels[1], panels[2])
            composite = _render_composite(panels[0], panels[1], panels[2], overlay)
            out_dir.mkdir(parents=True, exist_ok=True)
            path = _composite_path(out_dir, image_id)
            cv2.imwrite(str(path), composite)
            written.add(path)
        # The count is composites on disk, not case entries: two entries naming
        # one page are one file, and reporting two would be a lie.
        rendered[method] = len(written)
    return rendered


def write_case_report(cfg: Config, run_dir: Path, path: Path | None = None) -> Path:
    """Write the case report: good cases and failure cases per method (FR-041).

    Machine-readable JSON beside a readable Markdown summary, both written from
    the admitted results and the recorded reasons.  A method that did not run is
    reported as not run, with its reason, and with no failure count — an absent
    method is never shown as having failed nothing (US4 scenario 5).
    """
    run_dir = Path(run_dir)
    lists = case_lists(run_dir)
    run_id = next(iter(lists.values()))["run_id"] if lists else run_dir.name

    methods: dict[str, dict] = {}
    for method, lists_for_method in lists.items():
        if lists_for_method["successful"] is None:
            methods[method] = {
                "method": method,
                "scope": lists_for_method["scope"],
                "ran": False,
                "reason": lists_for_method["reason"],
                "cases": [],
                "failures": None,
            }
            continue
        rows = load_metrics_csv(run_dir / method / METRICS_NAME)
        cases = _case_entries(rows, lists_for_method["failed"], cfg.viz_n)
        methods[method] = {
            "method": method,
            "scope": lists_for_method["scope"],
            "ran": True,
            "reason": None,
            "cases": [
                {
                    **case,
                    "composite": (
                        f"{CASE_DIR}/{_composite_path(Path('.'), case['image_id']).as_posix()}"
                        if case["image_id"] else None
                    ),
                }
                for case in cases
            ],
            "failures": lists_for_method["failed"],
        }

    path = Path(path) if path is not None else run_dir / "cases.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"run_id": run_id, "methods": methods}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    path.with_suffix(".md").write_text(_case_markdown(document), encoding="utf-8")
    return path


def _case_markdown(document: dict) -> str:
    """The human-readable half of the case report."""
    lines = [f"# Representative cases — run {document['run_id']}", ""]
    for method, entry in sorted(document["methods"].items()):
        lines += [f"## {method}", ""]
        if not entry["ran"]:
            lines += [
                f"**Did not run.** {entry['reason']}",
                "",
                "No page of this method failed, because no page of this method ran.",
                "",
            ]
            continue
        for kind in CASE_KINDS:
            cases = [case for case in entry["cases"] if case["case_kind"] == kind]
            lines += [f"### {kind.capitalize()} cases", ""]
            if not cases:
                lines += ["None.", ""]
                continue
            lines += ["| page | F1 | composite | reason |", "| --- | --- | --- | --- |"]
            for case in cases:
                f1 = "—" if case["f1"] is None else f"{case['f1']:.3f}"
                composite = case.get("composite") or "—"
                reason = (case.get("reason") or "—").replace("|", "\\|")
                lines.append(f"| {case['image_id']} | {f1} | {composite} | {reason} |")
            lines.append("")
    return "\n".join(lines)