"""Dataset discovery: index ground-truth masks and raw pages.

FR-001/FR-002: recursively scan the ground-truth root and the raw-images root,
keying every file by (manga, stem). Ordering is deterministic: results are
sorted by (manga name, page stem), never by filesystem-walk order, so the
manifest is byte-identical across runs (FR-004).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MASK_SUFFIXES = {".png"}
RAW_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class SourceFile:
    manga: str
    stem: str
    path: Path

    @property
    def image_id(self) -> str:
        return f"{self.manga}/{self.stem}"


def _scan(root: Path, suffixes: set[str]) -> list[SourceFile]:
    """Scan ``root`` for files with the given suffixes, preserving manga+stem keys."""
    found: list[SourceFile] = []
    if not root.is_dir():
        return found
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if entry.is_dir():
            for file in sorted(entry.iterdir(), key=lambda p: p.name):
                if file.suffix.lower() in suffixes:
                    found.append(SourceFile(entry.name, file.stem, file))
    # Deterministic order: (manga, stem)
    return sorted(found, key=lambda f: (f.manga, f.stem))


def discover_masks(gt_root: Path) -> list[SourceFile]:
    """List all ground-truth masks under ``gt_root`` (FR-001)."""
    return _scan(gt_root, MASK_SUFFIXES)


def discover_raw_pages(raw_root: Path) -> list[SourceFile]:
    """List all raw pages under ``raw_root`` (FR-001/FR-002)."""
    return _scan(raw_root, RAW_SUFFIXES)