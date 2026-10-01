"""OpenCV inpainting for Spec 003 (US2).

* T026 - ``inpaint_page``: ``INPAINT_TELEA`` / ``INPAINT_NS`` on a page and a
  mask, radius from ``InpaintingConfig`` (FR-019, FR-021).
* T028 - ``process_sample``: one valid mask through validation, alignment,
  dilation and both algorithms, into the artifact tree (FR-025).
* T029 - determinism: no wall-clock value in any per-sample file, and the
  algorithms are always visited in sorted order.
* T030 - timing: each algorithm's own run time, recorded separately (FR-022).

Nothing here looks at which method produced the mask, so one code path serves
all four identities and per-method tuning is impossible by construction
(FR-021).
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from manga_text_seg.config import INPAINT_ALGORITHMS, InpaintConfig, InpaintingConfig
from manga_text_seg.imaging import load_raw_page
from manga_text_seg.intake import intake_mask
from manga_text_seg.manifest import Manifest
from manga_text_seg.maskproc import dilate_mask
from manga_text_seg.runs import (
    MaskRejection,
    RunError,
    SampleCounts,
    SampleMetadata,
    artifact_paths,
    sample_paths,
    write_inpainted_page,
    write_page_copy,
    write_processed_mask,
    write_raw_mask_copy,
    write_sample_metadata,
)

#: Directory form of the algorithm (``telea``, ``ns``) -> the OpenCV constant.
_ALGORITHM_FLAGS = {"telea": cv2.INPAINT_TELEA, "ns": cv2.INPAINT_NS}
assert set(_ALGORITHM_FLAGS) == set(INPAINT_ALGORITHMS)  # config and code agree


def inpaint_page(
    page: np.ndarray,
    mask: np.ndarray,
    algorithm: str,
    inpainting: InpaintingConfig,
) -> np.ndarray:
    """Return ``page`` with the pixels under ``mask`` filled in by ``algorithm``.

    ``mask`` is the mask *after* normalization, alignment and dilation: this
    function inpaints exactly what it is handed and never dilates. Both inputs
    are left untouched. ``algorithm`` is ``"telea"`` or ``"ns"``; the radius
    comes from ``inpainting`` alone.
    """
    if algorithm not in _ALGORITHM_FLAGS:
        raise RunError(
            f"Unknown algorithm {algorithm!r}; expected one of "
            f"{sorted(_ALGORITHM_FLAGS)}"
        )
    if page.shape[:2] != mask.shape[:2]:
        raise RunError(
            f"page is {page.shape[1]}x{page.shape[0]} but mask is "
            f"{mask.shape[1]}x{mask.shape[0]}"
        )
    return cv2.inpaint(page, mask, inpainting.radius, _ALGORITHM_FLAGS[algorithm])


def _resolve_algorithms(
    config: InpaintConfig, requested: tuple[str, ...] | None
) -> tuple[str, ...]:
    """The algorithms to run, validated and in sorted order (T029).

    ``None`` means the configured set, which in the main benchmark is both
    (FR-019). Order never depends on how the caller or the config listed them.
    """
    chosen = config.inpainting.algorithms if requested is None else requested
    unknown = [a for a in chosen if a not in INPAINT_ALGORITHMS]
    if unknown:
        raise RunError(
            f"Unknown algorithm {unknown[0]!r}; expected one of "
            f"{sorted(INPAINT_ALGORITHMS)}"
        )
    if not chosen:
        raise RunError("At least one inpainting algorithm is required")
    return tuple(sorted(dict.fromkeys(chosen)))


def _timed_inpaint(
    page: np.ndarray,
    mask: np.ndarray,
    algorithm: str,
    inpainting: InpaintingConfig,
) -> tuple[np.ndarray, float]:
    """Run one algorithm and return its result with its own run time (T030).

    The clock covers ``inpaint_page`` only: reading, validating and dilating the
    mask are shared by both algorithms and are not attributed to either.
    """
    started = time.perf_counter()
    image = inpaint_page(page, mask, algorithm, inpainting)
    return image, time.perf_counter() - started


def process_sample(
    manifest: Manifest,
    manga: str,
    stem: str,
    *,
    method: str,
    run_id: str,
    config: InpaintConfig,
    experiment_id: str,
    model_repository: str | None,
    algorithms: tuple[str, ...] | None = None,
    ablation: bool = False,
) -> dict[str, SampleMetadata]:
    """Process one mask into the artifact tree; return each algorithm's metadata.

    The mask is read from ``config.methods[method].source_root`` and taken
    through intake, then dilated as ``config`` says. Every algorithm is run on
    that same page and mask; only when all of them have succeeded is anything
    written, so a sample that fails leaves no artifact behind. For each
    algorithm the five files of ``contracts/inpainted-output.md`` are written.

    Raises ``MaskRejection`` (with ``image_id`` set) for any per-sample failure:
    an input refused by intake, an ``inpainting_failure`` or an
    ``output_write_failure``. The caller records it and carries on (FR-011).
    Raises ``RunError`` for a call that is itself wrong (unknown method or
    algorithm).
    """
    if method not in config.methods:
        raise RunError(f"Method {method!r} is not configured")
    chosen = _resolve_algorithms(config, algorithms)
    image_id = f"{manga}/{stem}"

    accepted = intake_mask(config.methods[method].source_root, manifest, manga, stem)
    processed_mask, dilation = dilate_mask(
        accepted.mask, config.mask_processing.dilation
    )
    page = load_raw_page(accepted.page_path)

    outcomes: dict[str, tuple[np.ndarray, float]] = {}
    for algorithm in chosen:
        try:
            outcomes[algorithm] = _timed_inpaint(
                page, processed_mask, algorithm, config.inpainting
            )
        except (cv2.error, RunError) as error:
            raise MaskRejection(
                "inpainting_failure", f"{algorithm}: {error}", image_id
            ) from error

    metadata: dict[str, SampleMetadata] = {}
    try:
        for algorithm in chosen:
            image, seconds = outcomes[algorithm]
            files = artifact_paths(
                sample_paths(
                    config.output_root,
                    run_id,
                    method,
                    algorithm,
                    image_id,
                    ablation=ablation,
                )
            )
            write_page_copy(files.page, page)
            write_raw_mask_copy(accepted.mask_path, files.raw_mask)
            write_processed_mask(files.processed_mask, processed_mask)
            write_inpainted_page(
                files.inpainted, image, config.inpainting.output_format
            )
            record = SampleMetadata(
                image_id=image_id,
                method=method,
                algorithm=algorithm,
                model_repository=model_repository,
                experiment_id=experiment_id,
                input_image_path=str(accepted.page_path),
                prediction_mask_path=str(accepted.mask_path),
                mask_size=accepted.raw_size,
                alignment=accepted.alignment,
                dilation=dilation,
                inpaint_algorithm=f"INPAINT_{algorithm.upper()}",
                inpaint_radius=config.inpainting.radius,
                output_format=config.inpainting.output_format,
                processing_time_seconds=seconds,
                status="ok",
            )
            write_sample_metadata(files.metadata, record)
            metadata[algorithm] = record
    except MaskRejection as rejection:
        rejection.image_id = image_id
        raise
    except OSError as error:
        raise MaskRejection(
            "output_write_failure", f"writing the sample's files: {error}", image_id
        ) from error
    return metadata


def process_batch(
    manifest: Manifest,
    pages: list[tuple[str, str]],
    *,
    methods: tuple[str, ...],
    run_id: str,
    config: InpaintConfig,
    origins: dict[str, tuple[str, str | None]],
    algorithms: tuple[str, ...] | None = None,
) -> dict:
    """Process a page batch while isolating each page's intake/inpaint failure.

    Returns the error report, method × algorithm counters and successful
    per-algorithm timings for run-level writers. Page order is preserved from
    the caller, which supplies the manifest's stable manga/stem ordering.
    """
    from manga_text_seg.runs import ErrorReport

    chosen_algorithms = _resolve_algorithms(config, algorithms)
    report = ErrorReport()
    counts: dict[str, dict[str, SampleCounts]] = {}
    timings: dict[str, dict[str, list[float]]] = {}
    for method in methods:
        attempted = succeeded = failed = 0
        method_timings = {algorithm: [] for algorithm in chosen_algorithms}
        for manga, stem in pages:
            attempted += 1
            experiment_id, model_repository = origins[method]
            try:
                outcome = process_sample(
                    manifest, manga, stem, method=method, run_id=run_id,
                    config=config, experiment_id=experiment_id,
                    model_repository=model_repository, algorithms=chosen_algorithms,
                )
            except MaskRejection as rejection:
                failed += 1
                report.add_rejection(rejection, method=method)
            else:
                succeeded += 1
                for algorithm, metadata in outcome.items():
                    method_timings[algorithm].append(metadata.processing_time_seconds)
        counts[method] = {
            algorithm: SampleCounts(attempted, succeeded, failed)
            for algorithm in chosen_algorithms
        }
        timings[method] = method_timings
    return {"errors": report, "counts": counts, "timings": timings}
