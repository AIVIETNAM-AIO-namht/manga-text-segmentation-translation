"""Run addressing and scaffolding for Spec 003 (inpainting).

* T007 — addressing: where a sample's artifacts live, computed from
  ``(run_id, method, algorithm, image_id)`` alone, with no index file and no
  manifest lookup (FR-029, contracts/inpainted-output.md).
* T008 — scaffolding: creating a run's directory, and refusing to overwrite a
  previous run unless overwrite was explicitly requested (FR-027, SC-006).

Later tasks (T009-T011, T021, T027) append writers to this module.
"""

from __future__ import annotations

import json
import platform
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from manga_text_seg.config import (
    INPAINT_ALGORITHMS,
    INPAINT_METHOD_IDENTITIES,
    INPAINT_OUTPUT_FORMATS,
    DilationConfig,
    InpaintingConfig,
    MaskProcessingConfig,
)

#: Subdirectory of the inpainting output root that holds ablation runs (FR-024),
#: kept apart so they can never be mistaken for main-benchmark runs (FR-017).
ABLATION_DIRNAME = "ablation"


class RunError(ValueError):
    """Raised when a run cannot be addressed or created as requested."""


@dataclass(frozen=True)
class SamplePaths:
    """Where one sample's inpainted image and metadata sidecar live."""

    image_path: Path
    metadata_path: Path


def _validate_run_id(run_id: str) -> None:
    if not run_id or not run_id.strip():
        raise RunError("run_id must be a non-empty string")
    if any(sep in run_id for sep in ("/", "\\")) or run_id in {".", ".."}:
        raise RunError(f"run_id {run_id!r} must not contain path separators")


def _run_dir(output_root: Path, run_id: str, ablation: bool) -> Path:
    _validate_run_id(run_id)
    base = Path(output_root) / ABLATION_DIRNAME if ablation else Path(output_root)
    return base / run_id


def sample_paths(
    output_root: Path,
    run_id: str,
    method: str,
    algorithm: str,
    image_id: str,
    *,
    ablation: bool = False,
) -> SamplePaths:
    """Resolve a sample's addresses from its identity alone (FR-029).

    ``<output_root>/<run_id>/<method>/<algorithm>/<manga>/<page_id>.png`` and
    the ``.json`` sibling. ``<manga>/<page_id>`` is ``image_id`` split on its
    separator, so the path spells the ``image_id`` back out.
    """
    if method not in INPAINT_METHOD_IDENTITIES:
        raise RunError(
            f"Unknown method {method!r}; expected one of "
            f"{sorted(INPAINT_METHOD_IDENTITIES)}"
        )
    if algorithm not in INPAINT_ALGORITHMS:
        raise RunError(
            f"Unknown algorithm {algorithm!r}; expected one of "
            f"{sorted(INPAINT_ALGORITHMS)}"
        )
    parts = image_id.split("/")
    if len(parts) != 2 or not all(parts):
        raise RunError(f"image_id {image_id!r} must have the form '<manga>/<page_id>'")
    manga, page_id = parts

    sample_dir = _run_dir(output_root, run_id, ablation) / method / algorithm / manga
    return SamplePaths(
        image_path=sample_dir / f"{page_id}.png",
        metadata_path=sample_dir / f"{page_id}.json",
    )


def create_run(
    output_root: Path,
    run_id: str,
    *,
    overwrite: bool = False,
    ablation: bool = False,
) -> Path:
    """Create a run's directory and return it.

    A run ID that already exists is refused unless ``overwrite`` is explicitly
    True (FR-027, SC-006). This is what makes the gitignored
    ``outputs/inpainting/`` safe to leave untracked. Only the named run's own
    directory is ever touched — never another run's.
    """
    run_dir = _run_dir(output_root, run_id, ablation)
    if run_dir.exists():
        if not overwrite:
            raise RunError(
                f"Run {run_id!r} already exists at {run_dir}; "
                "pass overwrite=True to replace it explicitly"
            )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    return run_dir


# --------------------------------------------------------------------------- #
# SampleMetadata writer (T009, FR-025)
# --------------------------------------------------------------------------- #

ALIGNMENT_RULES = {"pass-through", "crop-topleft"}
SAMPLE_STATUSES = {"ok", "failed"}
_IMAGE_ID_PATTERN = re.compile(r"^[^/]+/[0-9]{3}$")


@dataclass(frozen=True)
class AlignmentRecord:
    """Pre/post alignment sizes and the rule applied (FR-015, FR-018).

    Sizes are ``[width, height]``, matching ``sample-metadata.schema.json``.
    """

    rule: str
    pre_size: tuple[int, int]
    post_size: tuple[int, int]


@dataclass(frozen=True)
class SampleMetadata:
    """The per-sample sidecar. Field list is normative (contract C1); it is
    the schema's required list plus ``error_message``. Field order here is the
    order written to disk, which keeps the output byte-stable (research R3).
    """

    image_id: str
    method: str
    algorithm: str
    model_repository: str | None
    experiment_id: str
    input_image_path: str
    prediction_mask_path: str
    mask_size: tuple[int, int]
    alignment: AlignmentRecord
    dilation: DilationConfig
    inpaint_algorithm: str
    inpaint_radius: float
    output_format: str
    processing_time_seconds: float
    status: str
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not _IMAGE_ID_PATTERN.match(self.image_id):
            raise RunError(f"image_id {self.image_id!r} must be '<manga>/<NNN>'")
        if self.method not in INPAINT_METHOD_IDENTITIES:
            raise RunError(f"Unknown method {self.method!r}")
        if self.algorithm not in INPAINT_ALGORITHMS:
            raise RunError(f"Unknown algorithm {self.algorithm!r}")
        if self.inpaint_algorithm != f"INPAINT_{self.algorithm.upper()}":
            raise RunError(
                f"algorithm {self.algorithm!r} and inpaint_algorithm "
                f"{self.inpaint_algorithm!r} disagree"
            )
        if self.output_format not in INPAINT_OUTPUT_FORMATS:
            raise RunError(f"Unknown output_format {self.output_format!r}")
        if self.alignment.rule not in ALIGNMENT_RULES:
            raise RunError(f"Unknown alignment rule {self.alignment.rule!r}")
        if self.status not in SAMPLE_STATUSES:
            raise RunError(f"Unknown status {self.status!r}")
        if self.status == "failed" and not self.error_message:
            raise RunError("A failed sample must carry an error_message")
        d = self.dilation
        if d.enabled and (
            d.kernel_shape is None or d.kernel_size is None or d.iterations is None
        ):
            raise RunError(
                "dilation is enabled but kernel_shape, kernel_size and iterations "
                "are not all set"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_sample_metadata(path: Path, metadata: SampleMetadata) -> None:
    """Write one sidecar as UTF-8 JSON with a trailing newline.

    No timestamp is written: ``processing_time_seconds`` is the only
    non-deterministic field, and the determinism comparison excludes it (R3).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# ErrorReport (T010, FR-011, FR-012)
# --------------------------------------------------------------------------- #

#: The nine FR-012 categories in spec order, plus ``missing_metadata`` from
#: contracts/intake-validation.md check 7 (FR-012 says "at minimum").
ERROR_CATEGORIES = (
    "missing_source_image",
    "missing_prediction_mask",
    "image_id_mismatch",
    "wrong_size",
    "empty_mask",
    "non_binary_mask",
    "corrupt_image",
    "inpainting_failure",
    "output_write_failure",
    "missing_metadata",
)


@dataclass(frozen=True)
class ErrorEntry:
    image_id: str
    category: str
    reason: str
    method: str | None = None


class MaskRejection(RunError):
    """One input refused on receipt (FR-010, FR-012).

    Raised by intake and mask processing, caught by the caller, which records it
    and carries on with the remaining samples (FR-011). ``image_id`` is filled in
    by whoever knows the sample's name; it is what the error report is keyed on.
    """

    def __init__(self, category: str, reason: str, image_id: str | None = None) -> None:
        if category not in ERROR_CATEGORIES:
            raise RunError(
                f"Unknown error category {category!r}; expected one of "
                f"{list(ERROR_CATEGORIES)}"
            )
        super().__init__(f"{category}: {reason}")
        self.category = category
        self.reason = reason
        self.image_id = image_id


class ErrorReport:
    """Accumulates every failure of one run. A failing sample never stops the
    batch (FR-011); this object is the record of each one."""

    def __init__(self) -> None:
        self._entries: list[ErrorEntry] = []

    def add(
        self,
        image_id: str,
        category: str,
        reason: str,
        method: str | None = None,
    ) -> None:
        if category not in ERROR_CATEGORIES:
            raise RunError(
                f"Unknown error category {category!r}; expected one of "
                f"{list(ERROR_CATEGORIES)}"
            )
        if method is not None and method not in INPAINT_METHOD_IDENTITIES:
            raise RunError(f"Unknown method {method!r}")
        self._entries.append(ErrorEntry(image_id, category, reason, method))

    def add_rejection(self, rejection: MaskRejection, *, method: str) -> None:
        """Record a ``MaskRejection`` under the method it happened in (T021)."""
        if rejection.image_id is None:
            raise RunError(
                f"Rejection {rejection.category!r} carries no image_id; "
                "it cannot be reported"
            )
        self.add(rejection.image_id, rejection.category, rejection.reason, method)

    def __len__(self) -> int:
        return len(self._entries)

    def to_dict(self) -> dict[str, Any]:
        # Fixed category order, sorted entries: output never depends on the
        # order failures happened to occur in (research R3).
        by_category: dict[str, Any] = {}
        for category in ERROR_CATEGORIES:
            entries = sorted(
                (e for e in self._entries if e.category == category),
                key=lambda e: (e.image_id, e.method or "", e.reason),
            )
            by_category[category] = {
                "count": len(entries),
                "entries": [asdict(e) for e in entries],
            }
        return {"total": len(self._entries), "by_category": by_category}

    def write(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


# --------------------------------------------------------------------------- #
# InpaintRun record (T011, FR-027)
# --------------------------------------------------------------------------- #

#: The FR-026 order. ``INPAINT_METHOD_IDENTITIES`` is a set and has no order;
#: anything written to disk iterates this tuple instead (research R3).
INPAINT_METHOD_ORDER = (
    "classical_baseline",
    "manga_text_segmentation",
    "comic_text_detector",
    "unetpp_efficientnetv2",
)
RUN_RECORD_FILENAME = "run.json"


@dataclass(frozen=True)
class MethodSource:
    """Where one identity's masks were read from, and what it resolved to.

    For ``classical_baseline`` ``resolved_identity`` is the Spec 1 method
    chosen and ``reason`` says why (research R1); for a DL identity it is the
    admitted ``run_id`` under ``deliverables/``.
    """

    source_root: str
    resolved_identity: str
    reason: str | None = None


@dataclass(frozen=True)
class SampleCounts:
    attempted: int
    succeeded: int
    failed: int

    def __post_init__(self) -> None:
        if min(self.attempted, self.succeeded, self.failed) < 0:
            raise RunError("sample counts must not be negative")
        if self.attempted != self.succeeded + self.failed:
            raise RunError(
                f"attempted ({self.attempted}) must equal succeeded "
                f"({self.succeeded}) + failed ({self.failed})"
            )


@dataclass(frozen=True)
class InpaintRun:
    """One identified execution, persisted as ``<run_dir>/run.json``.

    ``page_list_identity`` and ``input_image_identity`` are quoted verbatim
    from ``benchmark/`` and never recomputed here (FR-054a).
    """

    run_id: str
    method_sources: dict[str, MethodSource]
    mask_processing_config: MaskProcessingConfig
    inpainting_config: InpaintingConfig
    page_list_identity: Any
    input_image_identity: Any
    counts: dict[str, dict[str, SampleCounts]]
    ablation: bool = False

    def __post_init__(self) -> None:
        _validate_run_id(self.run_id)
        if not self.method_sources:
            raise RunError("A run must process at least one method")
        for method in self.method_sources:
            if method not in INPAINT_METHOD_IDENTITIES:
                raise RunError(f"Unknown method {method!r}")
        for method, by_algorithm in self.counts.items():
            if method not in self.method_sources:
                raise RunError(
                    f"counts given for method {method!r} that was not processed"
                )
            for algorithm in by_algorithm:
                if algorithm not in INPAINT_ALGORITHMS:
                    raise RunError(f"Unknown algorithm {algorithm!r} in counts")

    def to_dict(self) -> dict[str, Any]:
        methods = [m for m in INPAINT_METHOD_ORDER if m in self.method_sources]
        return {
            "run_id": self.run_id,
            "methods": methods,
            "method_sources": {m: asdict(self.method_sources[m]) for m in methods},
            "mask_processing_config": asdict(self.mask_processing_config),
            "inpainting_config": asdict(self.inpainting_config),
            "page_list_identity": self.page_list_identity,
            "input_image_identity": self.input_image_identity,
            "counts": {
                m: {a: asdict(self.counts[m][a]) for a in sorted(self.counts[m])}
                for m in methods
                if m in self.counts
            },
            "ablation": self.ablation,
        }


def write_run_record(run_dir: Path, run: InpaintRun) -> Path:
    """Write ``run.json`` into ``run_dir`` and return its path.

    No timestamp: the run ID is the only time-identifying field (research R3).
    """
    path = Path(run_dir) / RUN_RECORD_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(run.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


PERFORMANCE_FILENAME = "performance.json"


def write_performance_summary(
    run_dir: Path,
    timings: dict[str, dict[str, list[float]]],
    counts: dict[str, dict[str, SampleCounts]],
    *,
    runtime: dict[str, Any] | None = None,
) -> Path:
    """Write deterministic method × algorithm timing and count aggregates.

    Timings contain successful per-sample algorithm durations. Failed counts
    come from the batch counters, so rejected inputs remain visible even though
    they never produced a metadata sidecar.
    """
    from .availability import probe_device

    groups: dict[str, dict[str, Any]] = {}
    for method in INPAINT_METHOD_ORDER:
        if method not in counts:
            continue
        groups[method] = {}
        for algorithm in sorted(counts[method]):
            values = timings.get(method, {}).get(algorithm, [])
            count = counts[method][algorithm]
            groups[method][algorithm] = {
                "mean_processing_time_seconds": (
                    sum(values) / len(values) if values else None
                ),
                "sample_count": len(values),
                "failure_count": count.failed,
            }
    document = {
        "groups": groups,
        "runtime": runtime or {
            "python": platform.python_version(),
            "device": probe_device(),
        },
    }
    path = Path(run_dir) / PERFORMANCE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Artifact writers (T027, FR-006, FR-025, FR-028)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ArtifactPaths:
    """The five files beside one sample's inpainted page (inpainted-output.md).

    Every one sits in the sample's own ``<method>/<algorithm>/<manga>/``
    directory and is named from the ``<page_id>`` stem.
    """

    inpainted: Path
    metadata: Path
    raw_mask: Path
    processed_mask: Path
    page: Path


def artifact_paths(paths: SamplePaths) -> ArtifactPaths:
    """Derive the raw-mask, processed-mask and page-copy addresses from
    ``sample_paths``' result, so they can never disagree with it."""
    directory = paths.image_path.parent
    stem = paths.image_path.stem
    return ArtifactPaths(
        inpainted=paths.image_path,
        metadata=paths.metadata_path,
        raw_mask=directory / f"{stem}.raw.png",
        processed_mask=directory / f"{stem}.mask.png",
        page=directory / f"{stem}.page.png",
    )


def _write_image(path: Path, image: np.ndarray, what: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        written = cv2.imwrite(str(path), image)
    except (cv2.error, OSError) as error:
        raise MaskRejection("output_write_failure", f"{what}: {error}") from error
    if not written:
        raise MaskRejection(
            "output_write_failure", f"{what} could not be written to {path}"
        )


def write_raw_mask_copy(source: Path, destination: Path) -> None:
    """Copy the admitted mask byte for byte; the source is never moved or altered
    (FR-006, contract rule 2)."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(source, destination)
    except OSError as error:
        raise MaskRejection(
            "output_write_failure", f"raw mask copy to {destination}: {error}"
        ) from error


def write_processed_mask(path: Path, mask: np.ndarray) -> None:
    """Write the mask actually handed to ``cv2.inpaint`` (contract rule 3)."""
    _write_image(path, mask, "processed mask")


def write_page_copy(path: Path, page: np.ndarray) -> None:
    """Write the original page image, losslessly, beside the outputs."""
    _write_image(path, page, "page copy")


def write_inpainted_page(path: Path, image: np.ndarray, output_format: str) -> None:
    """Write the inpainted page in the configured lossless format (FR-028).

    ``sample_paths`` fixes the ``.png`` suffix, and OpenCV encodes by suffix, so
    any other ``output_format`` would leave a file whose name lies about its
    content. That is refused rather than written.
    """
    if output_format not in INPAINT_OUTPUT_FORMATS:
        raise RunError(f"Unknown output_format {output_format!r}")
    if Path(path).suffix.lstrip(".") != output_format:
        raise RunError(
            f"output_format {output_format!r} does not match the address "
            f"{Path(path).name!r}; addresses are fixed by inpainted-output.md"
        )
    _write_image(path, image, "inpainted page")
