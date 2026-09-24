"""Deep-learning adapter registry (FR-051, contracts/returned-result.md).

Mirrors ``manga_text_seg.methods``: adding a DL method = one new file under
``adapters/`` + one ``register`` call. This package is the *only* place
method-specific behaviour may live — ``metrics.py``, ``report.py`` and
``visualize.py`` MUST NOT branch on method identity (US1 scenario 6).
"""
from __future__ import annotations

from typing import Callable, Optional

from .base import DLMethodAdapter, InferenceResult

Factory = Callable[[dict], DLMethodAdapter]


class UnknownAdapterError(KeyError):
    """Raised when ``create`` is asked for an adapter that is not registered."""


_REGISTRY: dict[str, Factory] = {}


def register(name: str, factory: Factory) -> None:
    """Register an adapter factory under a name (idempotent overwrite)."""
    _REGISTRY[name] = factory


def create(name: str, config: Optional[dict] = None) -> "DLMethodAdapter":
    """Instantiate a registered adapter with the given configuration mapping."""
    if name not in _REGISTRY:
        raise UnknownAdapterError(
            f"Unknown DL adapter: '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](config or {})


def available() -> list[str]:
    """Return the sorted, registered adapter names."""
    return sorted(_REGISTRY)


# Import adapter modules last, so their module-level ``register`` calls resolve
# against the fully-defined registry API above (each module does
# ``from . import register`` at import time).
from . import comic_text_detector  # noqa: E402,F401
from . import manga_text_segmentation  # noqa: E402,F401
from . import standin  # noqa: E402,F401  (register("standin", StandinAdapter))
from . import unetpp_efficientnetv2  # noqa: E402,F401
