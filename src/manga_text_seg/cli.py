"""Command-line interface for manga text segmentation.

Subcommands (FR-036: all paths come from a single JSON config file):
  discover  — build the (manga, stem) manifest from the dataset
  validate  — validate manifest pairs and report orphans
  run       — run one segmentation method on all pairs
  sweep     — run every configured method across the dataset
  report    — summarize per-method metrics into a report
  visualize — render side-by-side comparison composites
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, load_config
from .manifest import build_manifest, load_manifest, save_manifest


def _common_parser(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "-c", "--config",
        default="config.json",
        help="Path to JSON config (default: config.json)",
    )


def _output_dir(config_path: str) -> Path:
    """Resolve the output directory for this run from the config file."""
    cfg = load_config(config_path)
    return cfg.output_dir


def cmd_discover(config_path: str) -> int:
    """Build manifest and write it to <output>/manifest.json (FR-004/FR-005)."""
    cfg = load_config(config_path)
    manifest = build_manifest(cfg.gt_root, cfg.raw_root, cfg.run_id)
    out = cfg.output_dir / "manifest.json"
    save_manifest(manifest, out)
    orphan_count = len(manifest.orphans)
    print(f"Discovered {manifest.total} valid pairs across {len(manifest.manga_names)} manga.")
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
    print(f"Sweep complete: {summary['total_rows']} rows for "
          f"{len(summary['methods'])} method(s) across {summary['total_pairs']} pairs.")
    for method_name, totals in summary["methods"].items():
        print(f"  {method_name}: {totals['ok']} ok, {totals['failed']} failed")
    return 0


def cmd_report(config_path: str) -> int:
    from .report import build_report

    cfg = load_config(config_path)
    path = build_report(cfg)
    print(f"Report written: {path}")
    return 0


def cmd_visualize(config_path: str) -> int:
    from .visualize import MODE_LABELS, render_composites

    cfg = load_config(config_path)
    rendered = render_composites(cfg)
    mode = MODE_LABELS[cfg.viz_mode]
    n = cfg.viz_n
    print(f"Rendered '{mode}-{n}' composites:")
    for method, count in rendered.items():
        print(f"  {method}: {count} composite(s)")
    return 0


def cmd_scorecard(config_path: str) -> int:
    from .report import build_report, build_scorecard

    cfg = load_config(config_path)
    path = build_scorecard(cfg)
    print(f"Scorecard written: {path}")
    return 0


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="manga-text-seg",
        description="Manga text segmentation toolkit",
    )
    subparsers = parser.add_subparsers(dest="command")

    discover = subparsers.add_parser("discover", help="Build the (manga, stem) manifest")
    _common_parser(discover)

    validate = subparsers.add_parser("validate", help="Validate manifest and report orphans")
    _common_parser(validate)
    validate.add_argument("--manifest", default=None, help="Manifest path (default: from config)")

    run = subparsers.add_parser("run", help="Run one segmentation method on all pairs")
    _common_parser(run)
    run.add_argument("--method", required=True, help="Method name to run (e.g., otsu)")

    sweep = subparsers.add_parser("sweep", help="Run all configured methods")
    _common_parser(sweep)

    report = subparsers.add_parser("report", help="Summarize per-method metrics")
    _common_parser(report)

    visualize = subparsers.add_parser("visualize", help="Render comparison composites")
    _common_parser(visualize)

    scorecard = subparsers.add_parser("scorecard", help="Export per-method scorecard CSV")
    _common_parser(scorecard)

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
            code = cmd_report(args.config)
        elif args.command == "visualize":
            code = cmd_visualize(args.config)
        elif args.command == "scorecard":
            code = cmd_scorecard(args.config)
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