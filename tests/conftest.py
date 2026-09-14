"""Pytest bootstrap: make ``manga_text_seg`` importable from the src layout.

The package uses a ``src/`` layout (pyproject.toml).  This conftest inserts
``src/`` at the front of ``sys.path`` so the test suite runs without a prior
``pip install -e ".[dev]"``.  Harmless when the package is already installed —
the repo's own source is then preferred, which is exactly what the tests assert
against.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))