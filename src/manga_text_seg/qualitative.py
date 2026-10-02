"""Generate the human-filled qualitative review scaffold (Spec 003, US5).

This module only labels existing inpainting metadata; it never evaluates images
or assigns ratings. Ratings and observations belong to the human reviewer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


ORDINAL_SCALE = {
    "1": "Poor / severe issue",
    "2": "Fair / noticeable issue",
    "3": "Acceptable / moderate",
    "4": "Good / minor issue",
    "5": "Excellent / no visible issue",
}

CRITERIA = {
    "completeness_of_text_removal": "Completeness of text removal",
    "amount_of_missed_text": "Amount of missed text",
    "amount_of_background_over_erased": "Amount of background over-erased",
    "naturalness_of_restored_region": "Naturalness of the restored region",
    "artifacts_or_noise": "Artifacts or noise",
    "halo_or_border_effects": "Halo or border effects",
    "damage_to_linework_texture_panel_borders": "Damage to linework, texture and panel borders",
}


def generate_scaffold(
    samples: Iterable[dict[str, Any]],
    *,
    json_out: str | Path | None = None,
    md_out: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Build entries for samples; optionally write machine and reviewer formats.

    Input descriptors must contain image_id, method, algorithm and
    mask_proc_config. A descriptor may also include board_reference.
    """
    entries = []
    for sample in samples:
        entry = {
            "image_id": sample["image_id"],
            "method": sample["method"],
            "algorithm": sample["algorithm"],
            "mask_proc_config": sample["mask_proc_config"],
            "board_reference": sample.get("board_reference", ""),
            "criteria": {
                key: {"label": label, "scale": ORDINAL_SCALE, "rating": None, "observation": ""}
                for key, label in CRITERIA.items()
            },
        }
        entries.append(entry)

    if json_out is not None:
        path = Path(json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if md_out is not None:
        path = Path(md_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Qualitative inpainting review", "", "Ratings and observations are intentionally blank for human reviewers.", "", "Ordinal scale:"]
        lines.extend(f"- **{number}** — {label}" for number, label in ORDINAL_SCALE.items())
        for entry in entries:
            lines.extend(["", f"## {entry['image_id']} — {entry['method']} / {entry['algorithm']}", "", f"- Mask processing: `{json.dumps(entry['mask_proc_config'], ensure_ascii=False, sort_keys=True)}`", f"- Comparison board: `{entry['board_reference']}`", "", "| Criterion | Rating (1–5) | Observation |", "|---|---|---|"])
            lines.extend(f"| {criterion['label']} |  |  |" for criterion in entry["criteria"].values())
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return entries


def generate_run_scaffold(run_dir: str | Path) -> tuple[Path, Path, int]:
    """Collect successful sample sidecars and write the two scaffold files.

    Existing outputs are never replaced because they may contain reviewer work.
    """
    root = Path(run_dir)
    json_path, md_path = root / "qualitative.json", root / "qualitative.md"
    existing = [path for path in (json_path, md_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing qualitative scaffold: {', '.join(map(str, existing))}")

    samples = []
    for metadata_path in sorted(root.glob("*/*/*/*.json")):
        if metadata_path.name in {"run.json", "performance.json", "errors.json"}:
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("status") != "ok":
            continue
        image_id = metadata.get("image_id")
        if not image_id:
            continue
        samples.append({
            "image_id": image_id,
            "method": metadata["method"],
            "algorithm": metadata["algorithm"],
            "mask_proc_config": {
                "dilation": metadata.get("dilation"),
                "inpaint_radius": metadata.get("inpaint_radius"),
            },
            "board_reference": f"boards/{metadata['method']}/<selection-rule>/{image_id.replace('/', '_')}.png",
        })
    generate_scaffold(samples, json_out=json_path, md_out=md_path)
    return json_path, md_path, len(samples)
