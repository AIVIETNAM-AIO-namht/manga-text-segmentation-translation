"""Manifest: deterministic pair matching and JSON persistence.

FR-004: discover valid (manga, stem) pairs from filesystem, producing a
manifest containing 390 entries sorted by (manga, stem). FR-005: persist
as JSON; re-loading from the same files reproduces the manifest byte-for-byte.

Per-pair serialization follows contracts/manifest.schema.json exactly:
``raw_path``, ``mask_path`` and ``mask_encoding`` (one of
magenta-black | magenta-only | near-black-only | all-white-empty).

The manifest is the canonical contract between discovery and downstream
processing — once built, all consumers iterate in manifest order.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .discovery import discover_masks, discover_raw_pages, SourceFile
from .imaging import detect_mask_encoding


@dataclass(frozen=True)
class Pair:
    manga: str
    stem: str
    mask_path: Path
    raw_path: Path
    mask_encoding: str

    @property
    def image_id(self) -> str:
        return f"{self.manga}/{self.stem}"


@dataclass(frozen=True)
class Manifest:
    run_id: str
    gt_root: Path
    raw_root: Path
    pairs: list[Pair] = field(default_factory=list)

    # Derived summaries
    @property
    def total(self) -> int:
        return len(self.pairs)

    @property
    def manga_names(self) -> list[str]:
        return sorted(set(p.manga for p in self.pairs))

    @property
    def orphans(self) -> list[SourceFile]:
        """GT masks with no matching raw page."""
        mask_keys = {(p.manga, p.stem) for p in self.pairs}
        all_masks = discover_masks(self.gt_root)
        return [m for m in all_masks if (m.manga, m.stem) not in mask_keys]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "gt_root": str(self.gt_root),
            "raw_root": str(self.raw_root),
            "total": self.total,
            "manga_names": self.manga_names,
            "pairs": [
                {"manga": p.manga, "stem": p.stem,
                 "raw_path": str(p.raw_path),
                 "mask_path": str(p.mask_path),
                 "mask_encoding": p.mask_encoding}
                for p in self.pairs
            ],
        }


def build_manifest(
    gt_root: Path,
    raw_root: Path,
    run_id: str,
) -> Manifest:
    """Build the manifest from filesystem (FR-004/FR-005).

    Match by exact (manga, stem) across masks and raw pages.
    Ordering is deterministic: sorted by (manga, stem).
    Each pair carries its GT encoding classified from the mask file
    (mask_encoding, manifest.schema.json enum).
    """
    masks = discover_masks(gt_root)
    raws = discover_raw_pages(raw_root)

    raw_lookup: dict[tuple[str, str], SourceFile] = {
        (r.manga, r.stem): r for r in raws
    }

    pairs: list[Pair] = []
    for mask in masks:
        key = (mask.manga, mask.stem)
        if key in raw_lookup:
            pairs.append(Pair(
                manga=mask.manga,
                stem=mask.stem,
                mask_path=mask.path,
                raw_path=raw_lookup[key].path,
                mask_encoding=detect_mask_encoding(mask.path),
            ))

    pairs.sort(key=lambda p: (p.manga, p.stem))
    return Manifest(run_id=run_id, gt_root=gt_root, raw_root=raw_root, pairs=pairs)


def save_manifest(manifest: Manifest, path: Path) -> None:
    """Write manifest to JSON (FR-005). Path structure is deterministic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest.to_dict(), fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def load_manifest(path: Path) -> Manifest:
    """Load manifest from a previously-saved JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    pairs = [
        Pair(
            manga=p["manga"],
            stem=p["stem"],
            mask_path=Path(p["mask_path"]),
            raw_path=Path(p["raw_path"]),
            mask_encoding=p["mask_encoding"],
        )
        for p in data["pairs"]
    ]
    return Manifest(
        run_id=data["run_id"],
        gt_root=Path(data["gt_root"]),
        raw_root=Path(data["raw_root"]),
        pairs=pairs,
    )