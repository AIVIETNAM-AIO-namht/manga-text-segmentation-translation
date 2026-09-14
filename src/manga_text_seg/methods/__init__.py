"""Segmentation method registry (FR-035, contracts/method-interface.md).

Frozen for Spec 2: deep-learning adapters implement this exact interface.
Adding a method = one new file under ``methods/`` + one ``register`` call.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np

Factory = Callable[[dict], "BaseSegmentationMethod"]


class BaseSegmentationMethod(ABC):
    """Interface every method must implement.

    Rules (per contract):
      1. ``segment`` takes single-channel uint8 [H, W] and returns single-channel
         uint8 [H, W] with values exclusively in {0, 255} (background = 0,
         text = 255).
      2. Methods MUST NOT read the GT mask, the manifest, or the filesystem.
      3. All parameters arrive via ``config`` at ``create`` time.
      4. A method never resizes its input.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def segment(self, image: np.ndarray) -> np.ndarray:
        """Grayscale uint8 [H, W] in -> binary uint8 [H, W] out, values in {0, 255}."""
        ...


class UnknownMethodError(KeyError):
    """Raised when ``create`` is asked for a method that is not registered."""


_REGISTRY: dict[str, Factory] = {}


def register(name: str, factory: Factory) -> None:
    """Register a method factory under a name (idempotent overwrite)."""
    _REGISTRY[name] = factory


def create(name: str, config: Optional[dict] = None) -> BaseSegmentationMethod:
    """Instantiate a registered method with the given parameter dict."""
    if name not in _REGISTRY:
        raise UnknownMethodError(
            f"Unknown segmentation method: '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](config or {})


def available() -> list[str]:
    """Return the sorted, registered method names."""
    return sorted(_REGISTRY)


# Import method modules last, so their module-level ``register`` calls resolve
# against the fully-defined registry API above (each module does
# ``from . import register`` at import time).  Adding a method = one new file
# under ``methods/`` + one ``register`` line + the import here (quickstart §5).
#
# The six classical methods (spec.md FR-020) plus the composable pipeline.
from . import adaptive  # noqa: E402,F401  (register("adaptive", AdaptiveMethod))
from . import components  # noqa: E402,F401  (register("components", ComponentsMethod))
from . import edges  # noqa: E402,F401  (register("edges", EdgesMethod))
from . import morphology  # noqa: E402,F401  (register("morphology", MorphologyMethod))
from . import mser  # noqa: E402,F401  (register("mser", MserMethod))
from . import otsu  # noqa: E402,F401  (register("otsu", OtsuMethod))
from . import pipeline  # noqa: E402,F401  (register("pipeline", PipelineMethod))