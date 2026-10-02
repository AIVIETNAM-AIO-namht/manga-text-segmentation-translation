"""Ablation run scaffolding and configuration provenance (Spec 003, US6)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


def _changed_fields(configs: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Describe every setting whose value changes across the sweep."""
    flattened: dict[str, list[Any]] = {}
    for config in configs:
        for section, values in config.items():
            nested = values if isinstance(values, dict) else {"": values}
            for key, value in nested.items():
                name = f"{section}.{key}" if key else section
                flattened.setdefault(name, []).append(value)
    return {
        key: values
        for key, values in flattened.items()
        if len({json.dumps(value, sort_keys=True) for value in values}) > 1
    }


def run_ablation(
    *,
    output_root: str | Path,
    run_id: str,
    configs: Iterable[dict[str, Any]],
    baseline_config: dict[str, Any],
) -> dict[str, Any]:
    """Create an isolated ablation run with one directory per configuration.

    ``output_root`` is the configured inpainting root. Thus this function
    addresses ``<output_root>/ablation/<run_id>`` and cannot write into the
    main benchmark namespace. Existing run IDs are never replaced.
    """
    if not run_id or run_id in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("run_id must be a non-empty path-safe identifier")
    configs = list(configs)
    if not configs:
        raise ValueError("an ablation sweep requires at least one configuration")
    if any(not isinstance(item, dict) for item in configs):
        raise ValueError("each ablation configuration must be an object")

    ablation_dir = Path(output_root) / "ablation" / run_id
    ablation_dir.mkdir(parents=True, exist_ok=False)
    config_dirs = []
    for index, config in enumerate(configs, start=1):
        config_dir = ablation_dir / f"config-{index:03d}"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        config_dirs.append(config_dir)

    record = {
        "run_id": run_id,
        "ablation": True,
        "varied": _changed_fields(configs),
        "baseline_config": baseline_config,
        "configurations": [
            {"id": path.name, "config": config}
            for path, config in zip(config_dirs, configs)
        ],
    }
    (ablation_dir / "run.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "ablation_dir": ablation_dir,
        "output_dir": ablation_dir,
        "config_dirs": config_dirs,
        "varied": record["varied"],
    }
