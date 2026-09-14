"""Dataset validation: check pairs, categorize issues, write report (FR-006, FR-007).

FR-007 categories:
  1. masks missing a source image        (orphans)
  2. source images without a mask        (reverse orphans)
  3. corrupt / undecodable files
  4. filename mismatches                 (case-insensitive stem near-matches)
  5. manga-folder mismatches             (stem found under a different manga folder)

FR-008: validation NEVER deletes or silently drops data — it only reports.
Known expected state of the shipped dataset: 390 valid pairs, 60 orphans, 0 corrupt.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from .discovery import SourceFile, discover_masks, discover_raw_pages
from .manifest import Manifest

OUTPUT_FILENAME = "validation-report.json"


def _decodes(path: Path) -> bool:
    """Probe whether an image file decodes (FR-006)."""
    try:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        return img is not None
    except Exception:  # noqa: BLE001 - any decode error means "not decodable"
        return False


@dataclass(frozen=True)
class ValidationReport:
    """Categorized validation findings (FR-007)."""

    run_id: str
    pairs: int
    orphans: list[SourceFile] = field(default_factory=list)
    reverse_orphans: list[SourceFile] = field(default_factory=list)
    corrupt: list[Path] = field(default_factory=list)
    filename_mismatches: list[dict[str, Any]] = field(default_factory=list)
    manga_folder_mismatches: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_issues(self) -> int:
        return (
            len(self.orphans)
            + len(self.reverse_orphans)
            + len(self.corrupt)
            + len(self.filename_mismatches)
            + len(self.manga_folder_mismatches)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "valid_pairs": self.pairs,
            "summary": {
                "masks_missing_source": len(self.orphans),
                "sources_without_mask": len(self.reverse_orphans),
                "corrupt_files": len(self.corrupt),
                "filename_mismatches": len(self.filename_mismatches),
                "manga_folder_mismatches": len(self.manga_folder_mismatches),
                "total_issues": self.total_issues,
            },
            "categories": {
                "masks_missing_source": [
                    {"manga": m.manga, "stem": m.stem, "path": str(m.path)}
                    for m in self.orphans
                ],
                "sources_without_mask": [
                    {"manga": m.manga, "stem": m.stem, "path": str(m.path)}
                    for m in self.reverse_orphans
                ],
                "corrupt_files": [str(p) for p in self.corrupt],
                "filename_mismatches": self.filename_mismatches,
                "manga_folder_mismatches": self.manga_folder_mismatches,
            },
        }


def validate_manifest(manifest: Manifest) -> ValidationReport:
    """Validate every candidate pair and categorize issues (FR-006, FR-007)."""
    all_masks = discover_masks(manifest.gt_root)
    all_pages = discover_raw_pages(manifest.raw_root)

    pair_keys = {(p.manga, p.stem) for p in manifest.pairs}
    mask_keys = {(m.manga, m.stem) for m in all_masks}
    page_keys = {(p.manga, p.stem) for p in all_pages}

    orphans = [m for m in all_masks if (m.manga, m.stem) not in pair_keys]
    reverse_orphans = [p for p in all_pages if (p.manga, p.stem) not in pair_keys]

    # FR-006: every paired file must decode.
    corrupt: list[Path] = []
    for pair in manifest.pairs:
        if not _decodes(pair.mask_path):
            corrupt.append(pair.mask_path)
        if not _decodes(pair.raw_path):
            corrupt.append(pair.raw_path)

    # FR-007 category 4: case-insensitive stem near-matches between
    # orphan masks and pages (suggests filename case mismatch).
    filename_mismatches: list[dict[str, Any]] = []
    stem_to_manga: dict[str, list[str]] = {}
    for p in all_pages:
        stem_to_manga.setdefault(p.stem.lower(), []).append(p.manga)
    for m in orphans:
        buddies = stem_to_manga.get(m.stem.lower(), [])
        for manga in buddies:
            if manga != m.manga:
                filename_mismatches.append({"mask": f"{m.manga}/{m.stem}",
                                            "page": f"{manga}/{m.stem}"})

    # FR-007 category 5: a page stem that exists as an orphan mask under a
    # different manga folder (true manga-folder mismatch, not case variance).
    manga_folder_mismatches: list[dict[str, Any]] = []
    for p in reverse_orphans:
        for m in all_masks:
            if m.stem.lower() == p.stem.lower() and m.manga != p.manga:
                manga_folder_mismatches.append({"mask": f"{m.manga}/{m.stem}",
                                                "page": f"{p.manga}/{p.stem}"})
                break

    return ValidationReport(
        run_id=manifest.run_id,
        pairs=manifest.total,
        orphans=orphans,
        reverse_orphans=reverse_orphans,
        corrupt=corrupt,
        filename_mismatches=filename_mismatches,
        manga_folder_mismatches=manga_folder_mismatches,
    )


def write_validation_report(report: ValidationReport, output_dir: Path) -> Path:
    """Persist the validation report next to the manifest (output-layout.md)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / OUTPUT_FILENAME
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path


def report_issues(report: ValidationReport) -> list[str]:
    """Human-readable issue lines; empty when the dataset is clean."""
    lines = []
    if report.orphans:
        lines.append(f"{len(report.orphans)} mask(s) missing a source image")
    if report.reverse_orphans:
        lines.append(f"{len(report.reverse_orphans)} source image(s) without a mask")
    if report.corrupt:
        lines.append(f"{len(report.corrupt)} corrupt/undecodable file(s)")
    if report.filename_mismatches:
        lines.append(f"{len(report.filename_mismatches)} filename mismatch(es)")
    if report.manga_folder_mismatches:
        lines.append(f"{len(report.manga_folder_mismatches)} manga-folder mismatch(es)")
    return lines