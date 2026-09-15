"""Composable multi-stage pipeline method (FR-020 last bullet).

Research R4: a pipeline is an ordered composition of the classical stages,
described entirely in config.  Each stage is a ``{method, params}`` dict fed to
``create``; stages run in order, each consuming the previous stage's binary
output (the first stage consumes the preprocessed grayscale page).  With no
config (or no ``stages``), the pipeline defaults to a single Otsu stage so the
method always returns a valid binary mask.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import BaseSegmentationMethod, create, register

DEFAULT_STAGES = [{"method": "otsu"}]


class PipelineMethod(BaseSegmentationMethod):
    """Binary segmentation by chaining registered methods in sequence."""

    def __init__(self, config: dict) -> None:
        stages = config.get("stages") or DEFAULT_STAGES
        if not isinstance(stages, list):
            raise ValueError(
                f"pipeline: 'stages' must be a list, got {type(stages).__name__}"
            )
        # Validate the stage list up front (contract Rule 3: config at create
        # time) by instantiating every stage factory once.
        self._stages: list[BaseSegmentationMethod] = []
        for index, entry in enumerate(stages):
            if not isinstance(entry, dict) or "method" not in entry:
                raise ValueError(
                    f"pipeline: stage {index} must be a {{'method', 'params'}} dict, "
                    f"got {entry!r}"
                )
            self._stages.append(
                create(str(entry["method"]), dict(entry.get("params") or {}))
            )

    @property
    def name(self) -> str:
        return "pipeline"

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Feed the image through every stage in order; always binary output."""
        result: Any = image
        for stage in self._stages:
            result = stage.segment(np.ascontiguousarray(result, dtype=np.uint8))
        return np.asarray(result, dtype=np.uint8)


register("pipeline", PipelineMethod)