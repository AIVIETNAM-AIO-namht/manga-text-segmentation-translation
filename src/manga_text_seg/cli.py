"""Command-line interface for manga text segmentation.

Subcommands (FR-036: all paths come from a single JSON config file):
  discover  — build the (manga, stem) manifest from the dataset
  validate  — validate manifest pairs and report orphans
  run       — run one segmentation method on all pairs
  sweep     — run every configured method across the dataset
  report    — summarize per-method metrics, or a run's cross-method comparison
  visualize — render comparison composites, or a run's four-panel cases
  export    — write the distributable page list and the input images (FR-054)
  admit     — validate and admit a returned hand-off (FR-058, FR-059)
  status    — pre-flight availability verdicts, and admitted vs awaited (FR-016, FR-049)
  inpaint   — remove text from pages with TELEA / NS, from admitted masks (Spec 003)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ConfigError, load_config
from .manifest import build_manifest, load_manifest, save_manifest

#: The exported artefacts and the run store sit at fixed places in the repo
#: (benchmark/README.md, quickstart §4) rather than in a config file: they are
#: where a runner is *told* to look, and a location a runner has to guess is a
#: location that drifts.  Spec 1's own default run is the manifest's home.
BENCHMARK_DIR = "benchmark"
DELIVERABLES_DIR = "deliverables"
DEFAULT_MANIFEST = "outputs/segmentation/default/manifest.json"

#: Where the pre-flight verdicts are recorded.  Beside the exported page list,
#: because it is the same kind of thing: an artefact this project can read back
#: later without redoing the work that produced it (FR-022).
AVAILABILITY_RECORD = "availability.json"

#: ``inpaint`` reads Spec 003's own config, not Spec 1's ``config.json``.
DEFAULT_INPAINT_CONFIG = "configs/inpainting.json"


def _common_parser(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "-c",
        "--config",
        default="config.json",
        help="Path to JSON config (default: config.json)",
    )


def _output_dir(config_path: str) -> Path:
    """Resolve the output directory for this run from the config file."""
    cfg = load_config(config_path)
    return cfg.output_dir


def _repo_root(config_path: str) -> Path:
    """The repository root, by the rule ``load_config`` already resolves paths with."""
    return Path(config_path).resolve().parent.parent


def cmd_discover(config_path: str) -> int:
    """Build manifest and write it to <output>/manifest.json (FR-004/FR-005)."""
    cfg = load_config(config_path)
    manifest = build_manifest(cfg.gt_root, cfg.raw_root, cfg.run_id)
    out = cfg.output_dir / "manifest.json"
    save_manifest(manifest, out)
    orphan_count = len(manifest.orphans)
    print(
        f"Discovered {manifest.total} valid pairs across {len(manifest.manga_names)} manga."
    )
    print(f"Wrote manifest: {out}")
    if orphan_count:
        print(f"Warning: {orphan_count} GT masks have no matching raw page.")
    return 0


def cmd_validate(config_path: str, manifest_path: str | None = None) -> int:
    from .validation import validate_manifest, write_validation_report, report_issues

    cfg = load_config(config_path)
    if manifest_path is None:
        manifest_path = str(cfg.output_dir / "manifest.json")
    manifest = load_manifest(Path(manifest_path))
    report = validate_manifest(manifest)

    if report.total_issues == 0:
        print(f"Manifest OK: {manifest.total} pairs, no issues.")
        return 0

    issues = report_issues(report)
    print(f"Manifest has {report.total_issues} issue(s):")
    for line in issues:
        print(f"  - {line}")

    report_path = write_validation_report(report, cfg.output_dir)
    print(f"Report written: {report_path}")
    return 1


def cmd_run(config_path: str, method: str) -> int:
    from .run import run_method_on_manifest

    cfg = load_config(config_path)
    manifest = load_manifest(cfg.output_dir / "manifest.json")
    summary = run_method_on_manifest(cfg, manifest, method)
    print(f"Method '{method}': {summary['ok']} ok, {summary['failed']} failed.")
    if summary["failed"]:
        print("Failures (see sidecars):")
        for f in summary["failures"]:
            print(f"  - {f['image_id']}: {f['error']}")
    return 0 if summary["failed"] == 0 else 1


def cmd_sweep(config_path: str) -> int:
    from .run import sweep_all_methods

    cfg = load_config(config_path)
    manifest = load_manifest(cfg.output_dir / "manifest.json")
    summary = sweep_all_methods(cfg, manifest)
    print(
        f"Sweep complete: {summary['total_rows']} rows for "
        f"{len(summary['methods'])} method(s) across {summary['total_pairs']} pairs."
    )
    for method_name, totals in summary["methods"].items():
        print(f"  {method_name}: {totals['ok']} ok, {totals['failed']} failed")
    return 0


def cmd_report(config_path: str, run_id: str | None = None) -> int:
    """Per-method summaries, or a run's cross-method comparison (FR-037).

    Without ``--run`` this is Spec 1's report over the classical ``metrics.csv``.
    With ``--run`` it is the combined comparison — the classical baseline plus
    every method the run has admitted — assembled from results that already
    exist, so admitting a later result cannot move an earlier row (FR-059).
    """
    from .report import build_report

    cfg = load_config(config_path)
    if run_id is None:
        path = build_report(cfg)
        print(f"Report written: {path}")
        return 0

    from .benchmark import assemble_comparison, write_comparison
    from .report import write_comparison_charts, write_comparison_csv

    run_dir = _repo_root(config_path) / DELIVERABLES_DIR / run_id
    comparison = assemble_comparison(
        baseline_metrics=cfg.output_dir / "metrics.csv", run_dir=run_dir
    )
    table = write_comparison_csv(comparison, run_dir / "comparison.csv")
    charts = write_comparison_charts(comparison, run_dir / "charts")
    write_comparison(comparison, run_dir / "comparison.json")

    print(f"Comparison for run '{comparison['run_id']}':")
    for row in comparison["rows"]:
        if row["metrics"] is None:
            print(f"  not run: {row['method']} — {row['reason']}")
            continue
        metrics = row["metrics"]
        device = (row.get("device") or {}).get("name", "unknown device")
        print(
            f"  {row['method']}: IoU {metrics['iou']['mean']:.3f} "
            f"P {metrics['precision']['mean']:.3f} "
            f"R {metrics['recall']['mean']:.3f} "
            f"F1 {metrics['f1']['mean']:.3f} — on {device}"
        )
    print(f"Table: {table}")
    for path in charts:
        print(f"Chart: {path}")
    return 0


def cmd_visualize(config_path: str, run_id: str | None = None) -> int:
    """Comparison composites, or a run's four-panel cases (FR-040).

    Without ``--run`` this renders Spec 1's selection over the classical sweep's
    output directory.  With ``--run`` it renders the cases of a run that already
    exists, reading back the masks the run admitted — inference is never re-run
    (US4's independent test runs against a hand-off that arrived days ago).
    """
    from .visualize import MODE_LABELS, render_composites

    cfg = load_config(config_path)
    if run_id is None:
        rendered = render_composites(cfg)
        mode = MODE_LABELS[cfg.viz_mode]
        n = cfg.viz_n
        print(f"Rendered '{mode}-{n}' composites:")
        for method, count in rendered.items():
            print(f"  {method}: {count} composite(s)")
        return 0

    from .visualize import render_cases, write_case_report

    run_dir = _repo_root(config_path) / DELIVERABLES_DIR / run_id
    rendered = render_cases(cfg, run_dir)
    report = write_case_report(cfg, run_dir)

    print(f"Rendered cases for run '{run_id}':")
    for method, count in rendered.items():
        if count is None:
            # A method that did not run has no cases and no zero: printing
            # "0 composite(s)" would read as a method that ran and rendered
            # none (FR-036's rule, on the case list).
            print(f"  {method}: did not run — no cases rendered")
        else:
            print(f"  {method}: {count} composite(s)")
    print(f"Case report: {report}")
    return 0


def cmd_scorecard(config_path: str) -> int:
    from .report import build_report, build_scorecard

    cfg = load_config(config_path)
    path = build_scorecard(cfg)
    print(f"Scorecard written: {path}")
    return 0


def cmd_export(config_path: str) -> int:
    """Write the distributable page list and image bundle (FR-054, FR-054a)."""
    from .pagelist import BUNDLE_ROOT, IMAGE_IDENTITY_NAME, export

    cfg = load_config(config_path)
    manifest = load_manifest(cfg.output_dir / "manifest.json")
    benchmark_dir = _repo_root(config_path) / BENCHMARK_DIR

    path = export(manifest, benchmark_dir)
    print(f"Exported {manifest.total} page(s) from run '{manifest.run_id}'.")
    print(f"Page list: {path}")
    print(f"Image identity record: {benchmark_dir / IMAGE_IDENTITY_NAME}")
    print(f"Image bundle: {benchmark_dir / BUNDLE_ROOT}")
    return 0


def cmd_admit(
    config_path: str,
    run_id: str,
    method: str,
    handoff: str,
    manifest_path: str | None = None,
) -> int:
    """Validate a returned hand-off, then admit and score it (FR-058, FR-059).

    The config is read for the repository root only: the page list, the identity
    record and the manifest come from the exported artefacts, so what gets
    admitted is what was actually distributed, not what a config now says.
    """
    from .pagelist import load_identity_record, load_page_list
    from .receipt import ReceiptError, admit

    repo_root = _repo_root(config_path)
    benchmark_dir = repo_root / BENCHMARK_DIR
    manifest_file = (
        Path(manifest_path) if manifest_path else repo_root / DEFAULT_MANIFEST
    )

    try:
        rows = admit(
            Path(handoff),
            page_list=load_page_list(benchmark_dir),
            identity_record=load_identity_record(benchmark_dir),
            run_dir=repo_root / DELIVERABLES_DIR / run_id,
            manifest=load_manifest(manifest_file),
            method=method,
        )
    except ReceiptError as exc:
        where = f" ({exc.field})" if exc.field else ""
        page = f" on {exc.image_id}" if exc.image_id else ""
        print(f"Refused [{exc.check}]{where}{page}: {exc.reason}", file=sys.stderr)
        return 2

    print(f"Admitted {len(rows)} page(s) for '{method}' into run '{run_id}'.")
    print(f"Masks: {repo_root / DELIVERABLES_DIR / run_id / method / 'masks'}")
    return 0


def cmd_status(config_path: str, run_id: str | None = None) -> int:
    """Pre-flight availability, and what a run has actually admitted (US2).

    Without ``--run`` this is the check itself: it produces a verdict per method
    in *this* environment, prints it, and records it so a later status retrieves
    it without re-checking (FR-022).  With ``--run`` it reports the run's
    admitted and awaited methods from what was recorded — a method that did not
    run is named with its reason and gets no numbers (FR-018, FR-036, FR-049).
    """
    from .availability import check_all, load_verdicts, save_verdicts
    from .config import load_dl_config

    repo_root = _repo_root(config_path)
    benchmark_dir = repo_root / BENCHMARK_DIR
    record = benchmark_dir / AVAILABILITY_RECORD

    if run_id is None:
        verdicts = check_all(
            load_dl_config(config_path, repo_root=repo_root), repo_root=repo_root
        )
        save_verdicts(record, verdicts)
        for verdict in verdicts:
            environment = verdict["environment"]
            device = environment["device"]
            print(f"{verdict['method']}: {verdict['verdict'].upper()}")
            print(
                f"  checked in: Python {environment['interpreter']} on "
                f"{device['type']} ({device['name']})"
            )
            if verdict["reason"]:
                print(f"  reason: {verdict['reason']}")
            for step in verdict["remediation"]:
                print(f"  fix: {step}")
        print(f"Recorded: {record}")
        return 0

    try:
        verdicts = {verdict["method"]: verdict for verdict in load_verdicts(record)}
    except (FileNotFoundError, ValueError) as exc:
        print(f"No usable verdicts: {exc}", file=sys.stderr)
        return 2

    run_file = repo_root / DELIVERABLES_DIR / run_id / "run.json"
    try:
        record_doc = json.loads(run_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(
            f"Run '{run_id}' has no readable record at {run_file}: {exc}",
            file=sys.stderr,
        )
        return 2

    admitted = list(record_doc.get("methods_admitted", []))
    awaited = list(record_doc.get("methods_awaited", []))

    print(f"Run '{run_id}':")
    for method in admitted:
        print(f"  admitted: {method}")
    for method in awaited:
        verdict = verdicts.get(method)
        if verdict is None:
            print(f"  not run: {method} — no verdict was recorded for it")
        else:
            print(f"  not run: {method} — {verdict['verdict']}: {verdict['reason']}")
    return 0


def _mask_origin(method: str, source_root: Path, manifest) -> tuple[str, str | None]:
    """``(experiment_id, model_repository)`` for a method's masks (sample metadata).

    ``classical_baseline`` comes from Spec 1: its experiment is the manifest's own
    run and it has no upstream repository. The deep-learning identities come from
    their admitted run: the run id is the directory above the method's root and
    the repository is read from the ``provenance.json`` recorded with the masks.
    """
    from .runs import RunError

    if method == "classical_baseline":
        return manifest.run_id, None
    provenance_file = source_root / "provenance.json"
    try:
        provenance = json.loads(provenance_file.read_text(encoding="utf-8"))
        repository = provenance["repository"]["url"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RunError(
            f"cannot tell which repository produced the '{method}' masks: "
            f"no readable provenance at {provenance_file} ({exc})"
        ) from exc
    return source_root.parent.name, repository


def cmd_inpaint(
    config_path: str,
    run_id: str,
    method: str | None,
    image_id: str | None = None,
    algorithm: str | None = None,
    overwrite: bool = False,
    all_methods: bool = False,
) -> int:
    """Remove text from pages using one method's admitted masks (Spec 003, US2).

    ``--image-id`` processes a single prediction mask; without it every page of
    Spec 1's manifest is processed for ``method``. ``--algorithm`` restricts the
    run to TELEA or NS; without it both run (FR-019). A sample that fails is
    recorded in ``errors.json`` and printed, and the run carries on (FR-011); a
    run that completes exits 0 whatever it recorded. Exit 2 is a refusal: an
    existing run id without ``--overwrite``, a method with no admitted masks, or
    an invalid argument.
    """
    from .config import load_inpaint_config
    from .inpaint import process_batch
    from .intake import source_available
    from .runs import (
        INPAINT_METHOD_ORDER, InpaintRun, MethodSource, RunError, create_run,
        write_performance_summary, write_run_record,
    )
    from .pagelist import load_identity_record, load_page_list

    repo_root = _repo_root(config_path)
    config = load_inpaint_config(config_path, repo_root=repo_root)
    manifest = load_manifest(repo_root / DEFAULT_MANIFEST)

    if all_methods:
        methods = tuple(m for m in INPAINT_METHOD_ORDER if m in config.methods
                        and source_available(config.methods[m].source_root))
    elif method in config.methods:
        methods = (method,)
    else:
        print(f"Refused: unknown or missing method '{method}'.", file=sys.stderr)
        return 2
    if not methods:
        print(
            "Refused: no configured method has admitted masks.",
            file=sys.stderr,
        )
        return 2
    if not all_methods and not source_available(config.methods[methods[0]].source_root):
        print(f"Refused: '{methods[0]}' has no admitted masks.", file=sys.stderr)
        return 2

    if image_id is None:
        pages = sorted((pair.manga, pair.stem) for pair in manifest.pairs)
    else:
        parts = image_id.split("/")
        if len(parts) != 2 or not all(parts):
            print(
                f"Refused: --image-id '{image_id}' must look like <manga>/<page_id>.",
                file=sys.stderr,
            )
            return 2
        pages = [(parts[0], parts[1])]

    try:
        origins = {
            name: _mask_origin(name, config.methods[name].source_root, manifest)
            for name in methods
        }
        benchmark_dir = repo_root / BENCHMARK_DIR
        page_identity = load_page_list(benchmark_dir)["page_list_identity"]
        image_identities = load_identity_record(benchmark_dir)
        run_dir = create_run(config.output_root, run_id, overwrite=overwrite)
    except RunError as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2

    try:
        method_sources = {
            name: MethodSource(
                str(config.methods[name].source_root),
                origins[name][0],
                "highest IoU of the six Spec 1 methods (research R1)"
                if name == "classical_baseline" else None,
            )
            for name in methods
        }
        batch = process_batch(
            manifest, pages, methods=methods, run_id=run_id, config=config,
            origins=origins, algorithms=(algorithm,) if algorithm else None,
        )
        batch["errors"].write(run_dir / "errors.json")
        write_performance_summary(run_dir, batch["timings"], batch["counts"])
        page_ids = {f"{manga}/{stem}" for manga, stem in pages}
        selected_identities = {
            image_id: image_identities[image_id]
            for image_id in sorted(page_ids) if image_id in image_identities
        }
        write_run_record(run_dir, InpaintRun(
            run_id=run_id, method_sources=method_sources,
            mask_processing_config=config.mask_processing,
            inpainting_config=config.inpainting,
            page_list_identity=page_identity,
            input_image_identity=selected_identities,
            counts=batch["counts"],
        ))
    except RunError as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2
    processed = sum(next(iter(v.values())).succeeded for v in batch["counts"].values())
    rejected = sum(next(iter(v.values())).failed for v in batch["counts"].values())
    print(
        f"Run '{run_id}', methods {', '.join(methods)}: "
        f"{processed} method-pages processed, {rejected} rejected."
    )
    print(f"Output: {run_dir}")
    print(f"Errors: {run_dir / 'errors.json'}")
    return 0


def cmd_boards(config_path: str, run_id: str, method: str) -> int:
    """Select representative pages from persisted metrics and render boards."""
    import csv
    import json

    import cv2

    from .boards import write_board
    from .config import load_inpaint_config
    from .runs import INPAINT_ALGORITHMS, artifact_paths, sample_paths
    from .selection import SELECTION_RULES, select_samples, write_selection

    repo_root = _repo_root(config_path)
    config = load_inpaint_config(config_path, repo_root=repo_root)
    if method not in config.methods:
        raise ValueError(f"Unknown inpainting method {method!r}")
    run_dir = config.output_root / run_id
    run_record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if method not in run_record.get("methods", []):
        raise ValueError(f"Run {run_id!r} contains no outputs for {method!r}")

    if method == "classical_baseline":
        metric_path = repo_root / "outputs" / "segmentation" / "default" / "metrics.csv"
        rows = list(csv.DictReader(metric_path.open(encoding="utf-8", newline="")))
        resolved = run_record["method_sources"][method]["resolved_identity"]
        rows = [row for row in rows if row.get("method") == resolved]
    else:
        source_root = Path(run_record["method_sources"][method]["source_root"])
        metric_path = source_root.parent / "metrics" / "per-page.csv"
        rows = list(csv.DictReader(metric_path.open(encoding="utf-8", newline="")))

    metrics = []
    for row in rows:
        image_id = row.get("image_id") or f"{row['manga']}/{row['stem']}"
        metrics.append({
            "image_id": image_id,
            "iou": float(row["iou"]),
            "f1": float(row["f1"]),
            "fp": int(float(row["fp"])),
            "fn": int(float(row["fn"])),
        })
    config_record = {
        "dilation": run_record["mask_processing_config"]["dilation"],
        "inpaint_radius": run_record["inpainting_config"]["radius"],
    }
    selections: dict[str, dict[str, dict]] = {}
    board_count = 0
    for rule in SELECTION_RULES:
        result = select_samples(
            {method: metrics}, rule, config.selection_n,
            manual_flags=config.manual_artifact_flags,
        )
        selections[rule] = result
        for entry in result[method]["selected"]:
            image_id = entry["image_id"]
            panel_paths = {}
            for algorithm in INPAINT_ALGORITHMS:
                paths = artifact_paths(sample_paths(
                    config.output_root, run_id, method, algorithm, image_id
                ))
                panel_paths[algorithm] = paths
            def load(path):
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise FileNotFoundError(f"Required board input is missing or unreadable: {path}")
                return image
            sample = {
                "image_id": image_id,
                "method": method,
                "algorithm": "telea+ns",
                "mask_proc_config": config_record,
                "panels": {
                    "original_page": load(panel_paths["telea"].page),
                    "raw_prediction_mask": load(panel_paths["telea"].raw_mask),
                    "dilated_mask": load(panel_paths["telea"].processed_mask),
                    "telea_result": load(panel_paths["telea"].inpainted),
                    "ns_result": load(panel_paths["ns"].inpainted),
                },
            }
            manga, stem = image_id.split("/", 1)
            destination = run_dir / "boards" / method / rule / f"{manga}_{stem}.png"
            write_board(sample, destination)
            board_count += 1
    write_selection(run_dir / "selection.json", selections)
    print(f"Wrote {board_count} boards for '{method}' in run '{run_id}'.")
    print(f"Selection: {run_dir / 'selection.json'}")
    return 0


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="manga-text-seg",
        description="Manga text segmentation toolkit",
    )
    subparsers = parser.add_subparsers(dest="command")

    discover = subparsers.add_parser(
        "discover", help="Build the (manga, stem) manifest"
    )
    _common_parser(discover)

    validate = subparsers.add_parser(
        "validate", help="Validate manifest and report orphans"
    )
    _common_parser(validate)
    validate.add_argument(
        "--manifest", default=None, help="Manifest path (default: from config)"
    )

    run = subparsers.add_parser("run", help="Run one segmentation method on all pairs")
    _common_parser(run)
    run.add_argument("--method", required=True, help="Method name to run (e.g., otsu)")

    sweep = subparsers.add_parser("sweep", help="Run all configured methods")
    _common_parser(sweep)

    report = subparsers.add_parser("report", help="Summarize per-method metrics")
    _common_parser(report)
    report.add_argument(
        "--run", default=None, help="Compare every method admitted to this run"
    )

    visualize = subparsers.add_parser("visualize", help="Render comparison composites")
    _common_parser(visualize)
    visualize.add_argument(
        "--run", default=None, help="Render the four-panel cases of this run"
    )

    scorecard = subparsers.add_parser(
        "scorecard", help="Export per-method scorecard CSV"
    )
    _common_parser(scorecard)

    export_cmd = subparsers.add_parser(
        "export", help="Write the distributable page list and input images"
    )
    _common_parser(export_cmd)

    admit_cmd = subparsers.add_parser(
        "admit", help="Validate and admit a returned hand-off"
    )
    _common_parser(admit_cmd)
    admit_cmd.add_argument("--run", required=True, help="Run id to admit into")
    admit_cmd.add_argument(
        "--method", required=True, help="Method the hand-off declares"
    )
    admit_cmd.add_argument(
        "--manifest", default=None, help="Manifest path (default: Spec 1's default run)"
    )
    admit_cmd.add_argument(
        "handoff", help="Directory holding masks/, metadata/, provenance.json"
    )

    status_cmd = subparsers.add_parser(
        "status", help="Pre-flight availability verdicts, and admitted vs awaited"
    )
    _common_parser(status_cmd)
    status_cmd.add_argument(
        "--run", default=None, help="Report admitted vs awaited for this run id"
    )

    inpaint_cmd = subparsers.add_parser(
        "inpaint", help="Remove text from pages using a method's admitted masks"
    )
    _common_parser(inpaint_cmd)
    inpaint_cmd.set_defaults(config=DEFAULT_INPAINT_CONFIG)
    inpaint_cmd.add_argument("--run", required=True, help="Run id to write into")
    inpaint_cmd.add_argument(
        "--method", default=None, help="Segmentation method whose masks to use"
    )
    inpaint_cmd.add_argument("--all-methods", action="store_true", help="Process every configured method with admitted masks")
    inpaint_cmd.add_argument(
        "--image-id", default=None, help="Process only this page (<manga>/<page_id>)"
    )
    inpaint_cmd.add_argument(
        "--algorithm",
        choices=["telea", "ns"],
        default=None,
        help="Run only this algorithm (default: both)",
    )
    inpaint_cmd.add_argument(
        "--overwrite", action="store_true", help="Replace an existing run with this id"
    )

    boards_cmd = subparsers.add_parser("boards", help="Select samples and render comparison boards")
    _common_parser(boards_cmd)
    boards_cmd.set_defaults(config=DEFAULT_INPAINT_CONFIG)
    boards_cmd.add_argument("--run", required=True, help="Inpainting run id")
    boards_cmd.add_argument("--method", required=True, help="Segmentation method to render")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "discover":
            code = cmd_discover(args.config)
        elif args.command == "validate":
            code = cmd_validate(args.config, args.manifest)
        elif args.command == "run":
            code = cmd_run(args.config, args.method)
        elif args.command == "sweep":
            code = cmd_sweep(args.config)
        elif args.command == "report":
            code = cmd_report(args.config, args.run)
        elif args.command == "visualize":
            code = cmd_visualize(args.config, args.run)
        elif args.command == "scorecard":
            code = cmd_scorecard(args.config)
        elif args.command == "export":
            code = cmd_export(args.config)
        elif args.command == "admit":
            code = cmd_admit(
                args.config, args.run, args.method, args.handoff, args.manifest
            )
        elif args.command == "status":
            code = cmd_status(args.config, args.run)
        elif args.command == "inpaint":
            code = cmd_inpaint(
                args.config,
                args.run,
                args.method,
                args.image_id,
                args.algorithm,
                args.overwrite,
                args.all_methods,
            )
        elif args.command == "boards":
            code = cmd_boards(args.config, args.run, args.method)
        else:  # pragma: no cover - argparse prevents unknown commands
            parser.error(f"Unknown command: {args.command}")
            return
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(2)
    except FileNotFoundError as exc:
        print(f"Missing file: {exc}", file=sys.stderr)
        sys.exit(2)

    sys.exit(code)


if __name__ == "__main__":
    main()
