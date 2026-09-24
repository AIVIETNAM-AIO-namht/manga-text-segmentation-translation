"""The deep-learning adapter interface (T012, FR-008-FR-011).

An adapter is the boundary at which one external repository is called. It is a
segmentation method — ``segment(image) -> mask`` is exactly Spec 1's frozen
interface, so the metric and reporting modules accept it unmodified — plus the
four things FR-010 requires the boundary to hand back: the input size, the
inference time, model metadata, and an error status instead of an exception.

Everything method-specific (preprocessing, thresholds, tensor layout, framework
calls, geometry mapping) lives in a subclass and its configuration, never in
``metrics.py``, ``report.py``, ``visualize.py`` or ``cli.py`` (FR-011).
"""
from __future__ import annotations

import platform
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InferenceResult:
    """What one ``infer`` call hands back (FR-010).

    ``mask`` is ``None`` exactly when ``status`` is ``"failed"``: a failure is
    reported, never raised, so one bad page cannot abort a 390-page run
    (FR-031's isolation, carried into the DL path).
    """

    status: str
    mask: np.ndarray | None
    input_size: tuple[int, int] | None
    inference_time_seconds: float
    metadata: dict[str, Any]
    error: str | None = None


class DLMethodAdapter(ABC):
    """Base class for the three deep-learning methods (FR-013).

    Configuration arrives once, at construction (``configs/dl.json`` through
    the registry): device, threshold, checkpoint path, padding. ``segment``
    takes nothing but the image, so a caller cannot reconfigure a method
    per page.
    """

    def __init__(self, config: dict | None = None) -> None:
        self._config: dict[str, Any] = dict(config or {})
        self._device: dict[str, Any] = self._config.get("device") or {
            "type": "cpu",
            # FR-060: a bare 'cpu' is not a name. Record what actually ran.
            "name": platform.processor() or platform.machine() or "unknown",
        }
        checkpoint = self._config.get("checkpoint")
        self._checkpoint: str | None = str(checkpoint) if checkpoint else None

    # -- The frozen segmentation interface (Spec 1 method-interface.md) ------

    @property
    @abstractmethod
    def name(self) -> str:
        """Registry name, e.g. ``manga-text-segmentation``."""

    @abstractmethod
    def segment(self, image: np.ndarray) -> np.ndarray:
        """Grayscale uint8 ``[H, W]`` in -> binary uint8 ``[H, W]`` out.

        Values are exclusively ``{0, 255}`` and the shape is unchanged: a method
        never resizes its input, because alignment happens after inference
        (FR-024a).
        """

    # -- The boundary record (FR-010) ---------------------------------------

    def metadata(self) -> dict[str, Any]:
        """Model metadata for the page sidecar (FR-010, FR-011)."""
        return {
            "name": self.name,
            "device": dict(self._device),
            "checkpoint": self._checkpoint,
        }

    def infer(self, image: "np.ndarray | str | Path") -> InferenceResult:
        """Run :meth:`segment`, returning the mask and the boundary record.

        A path is accepted as well as an array (FR-009) and read as the raw
        page the classical baseline read, so both paths meet the same bytes
        (FR-054a).
        """
        try:
            array = self._as_array(image)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            return self._failure(None, exc)

        started = time.perf_counter()
        try:
            mask = self.segment(array)
        except Exception as exc:  # noqa: BLE001 - FR-010 error status
            return self._failure(array.shape[:2], exc, time.perf_counter() - started)
        elapsed = time.perf_counter() - started

        return InferenceResult(
            status="ok",
            mask=mask,
            input_size=(int(array.shape[0]), int(array.shape[1])),
            inference_time_seconds=round(elapsed, 4),
            metadata=self.metadata(),
        )

    # -- Internals -----------------------------------------------------------

    @staticmethod
    def _as_array(image: "np.ndarray | str | Path") -> np.ndarray:
        """Accept an in-memory image or a path, and refuse anything else."""
        if isinstance(image, np.ndarray):
            array = image
        elif isinstance(image, (str, Path)):
            from ..imaging import load_raw_page

            array = load_raw_page(image)
        else:
            raise TypeError(
                f"expected an image array or path, got {type(image).__name__}"
            )

        if array.ndim != 2:
            raise ValueError(
                f"expected a single-channel image, got shape {array.shape}"
            )
        if array.dtype != np.uint8:
            raise ValueError(f"expected a uint8 image, got {array.dtype}")
        return array

    def _failure(
        self,
        input_size: tuple[int, int] | None,
        exc: Exception,
        elapsed: float = 0.0,
    ) -> InferenceResult:
        return InferenceResult(
            status="failed",
            mask=None,
            input_size=input_size,
            inference_time_seconds=round(elapsed, 4),
            metadata=self.metadata(),
            error=f"{type(exc).__name__}: {exc}",
        )
