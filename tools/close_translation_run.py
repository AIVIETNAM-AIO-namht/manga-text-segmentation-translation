"""Close a translation run from the pages it already produced.

``run_batch`` writes ``run.json`` only after every manifest page finishes, so
an interrupted run leaves page artifacts with no run record. This rebuilds
that record from the pages that did finish, without re-running OCR or
translation. The manifest page list is unchanged: the run is recorded as
covering exactly the pages it processed.

    python tools/close_translation_run.py configs/translation.json spec004-vilao \
        --segmentation-method unetpp_efficientnetv2 \
        --inpainting-run spec003-main --target-lang vi
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from manga_text_seg.config import TranslationConfig, load_translation_config  # noqa: E402
from manga_text_seg.manifest import load_manifest  # noqa: E402
from manga_text_seg.translation.pipeline import (  # noqa: E402
    _identity_records,
    _jsonable,
)
from manga_text_seg.translation.runs import (  # noqa: E402
    ErrorReport,
    TranslationRun,
    write_run_json,
)

# A page whose metadata says "ok" finished the whole chain; run_batch records
# it as "success". Anything else is not a finished page.
_COMPLETE_STATUSES = {"success", "no_regions"}


def _rendering_record(config: TranslationConfig) -> dict[str, Any]:
    rendering = config.rendering
    return {
        "font_family": rendering.font_family,
        "font_path": str(rendering.font_path),
        "fallback_font_path": (
            str(rendering.fallback_font_path) if rendering.fallback_font_path else None
        ),
        "font_size": rendering.font_size,
        "min_font_size": rendering.min_font_size,
        "color": list(rendering.color),
        "stroke_width": rendering.stroke_width,
        "stroke_fill": list(rendering.stroke_fill),
        "background": list(rendering.background) if rendering.background else None,
        "wrap": rendering.wrap,
        "line_spacing": rendering.line_spacing,
        "align_h": rendering.align_h,
        "align_v": rendering.align_v,
        "text_direction": rendering.text_direction,
        "expansion_allowance": rendering.expansion_allowance,
    }


def _completed_pages(
    config: TranslationConfig, run_dir: Path
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Manifest pages with a finished metadata.json, in manifest order."""
    page_root = run_dir / config.segmentation_method
    ids: list[str] = []
    statuses: dict[str, str] = {}
    for pair in load_manifest(config.manifest_path).pairs:
        metadata_path = page_root / pair.image_id / "metadata.json"
        if not metadata_path.exists():
            continue
        status = json.loads(metadata_path.read_text(encoding="utf-8"))["status"]
        ids.append(pair.image_id)
        statuses[pair.image_id] = "success" if status == "ok" else status
    return tuple(ids), statuses


def close_run(
    config: TranslationConfig, run_id: str, *, repo_root: str | Path
) -> Path:
    """Write run.json for a run whose pages finished but which never closed."""
    run_dir = config.output_roots.translation / run_id
    if not run_dir.is_dir():
        raise ValueError(f"run directory does not exist: {run_dir}")
    if (run_dir / "run.json").exists():
        raise ValueError(f"run is already closed: {run_dir / 'run.json'}")

    ids, statuses = _completed_pages(config, run_dir)
    if not ids:
        raise ValueError(
            f"no completed pages found under {run_dir / config.segmentation_method}"
        )
    incomplete = {i: s for i, s in statuses.items() if s not in _COMPLETE_STATUSES}
    if incomplete:
        raise ValueError(f"pages did not finish cleanly: {incomplete}")

    page_list_identity, identities = _identity_records(Path(repo_root), ids)
    run_record = TranslationRun(
        run_id=run_id,
        segmentation_method=config.segmentation_method,
        inpainting_source={
            "run_id": config.inpainting_run_id,
            "algorithm": config.inpainting_algorithm,
        },
        target_language=config.target_language,
        ocr_engine={"name": config.ocr.engine, "version": None, "deterministic": True},
        provider={
            "name": config.provider.name,
            "model": config.provider.model,
            "version": config.provider.version,
            "deterministic": config.provider.deterministic,
            "secret_env_var": config.provider.secret_env_var,
        },
        rendering=_rendering_record(config),
        resolved_config={
            "manifest_path": str(config.manifest_path),
            "segmentation_method": config.segmentation_method,
            "inpainting_source": {
                "run_id": config.inpainting_run_id,
                "algorithm": config.inpainting_algorithm,
            },
            "target_language": config.target_language,
            "region_extraction": _jsonable(config.region_extraction),
            "ocr": _jsonable(config.ocr),
            "provider": _jsonable(config.provider),
            "rendering": _jsonable(config.rendering),
            "output_roots": _jsonable(config.output_roots),
        },
        input_image_ids=ids,
        page_statuses=statuses,
        page_list_identity=page_list_identity,
        input_image_identity=identities,
        region_extraction_config=_jsonable(config.region_extraction),
        counts={
            "pages_attempted": len(ids),
            "pages_succeeded": len(ids),
            "pages_failed": 0,
        },
    )
    write_run_json(run_dir, run_record)
    ErrorReport(run_id).write(run_dir)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="translation config JSON")
    parser.add_argument("run_id", help="run id under the translation output root")
    parser.add_argument("--segmentation-method", required=True)
    parser.add_argument("--inpainting-run", required=True)
    parser.add_argument("--target-lang", required=True)
    args = parser.parse_args(argv)

    config = replace(
        load_translation_config(args.config),
        segmentation_method=args.segmentation_method,
        inpainting_run_id=args.inpainting_run,
        target_language=args.target_lang,
    )
    repo_root = Path(args.config).resolve().parent.parent
    run_dir = close_run(config, args.run_id, repo_root=repo_root)
    print(f"closed {args.run_id} -> {run_dir / 'run.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
