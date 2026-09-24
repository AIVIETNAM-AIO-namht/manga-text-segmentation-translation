"""Synthetic stand-in adapter (T013, FR-050, FR-051).

Not one of the three benchmarked methods (FR-013). It exists so the adapter
interface, the receipt path and the CLI can be tested end to end without a
checkpoint, a framework or a network call: ``segment`` is a fixed threshold, so
the same image always yields the same mask.

Registered under ``standin`` — deliberately not a registry name from
``configs/dl.json``, so it can never be mistaken for a real method's result.
"""
from __future__ import annotations

import numpy as np

from . import register
from .base import DLMethodAdapter

#: Fixed threshold, mid-grey. Deterministic by construction: no RNG, no model.
STANDIN_THRESHOLD = 127


class StandinAdapter(DLMethodAdapter):
    """Deterministic threshold adapter standing in for a real model."""

    @property
    def name(self) -> str:
        return "standin"

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Threshold at mid-grey; text = 255, background = 0."""
        if image.ndim != 2:
            raise ValueError(
                f"standin: expected single-channel image, got shape {image.shape}"
            )
        return np.where(image > STANDIN_THRESHOLD, 255, 0).astype(np.uint8)


register("standin", StandinAdapter)
