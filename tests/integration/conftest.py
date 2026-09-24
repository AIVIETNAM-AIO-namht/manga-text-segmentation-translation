"""The opt-in boundary around real checkpoints (T045, FR-051).

Everything in this directory may load a real large pretrained model, download a
checkpoint, or import a framework this project does not depend on.  The default
suite must not: FR-051 says integration tests that do so *may be separated and
made explicitly opt-in*, and this directory is that separation.

The guard is a single hook.  It always tags the directory's tests with the
``integration`` marker, so ``-m "not integration"`` is a second lever, and it
skips them unless ``MANGA_TEXT_SEG_INTEGRATION`` is set.  Nothing here is
collected as a pass by default — a skipped test is visibly skipped, never a
green that ran nothing.

The hook is session-global even though this file is directory-scoped, so it
filters on path: without that filter it would skip the entire suite.  The
``tests/unit`` tests are the ones that must keep running.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

#: Set to any non-empty value to run this directory's tests.  Named for the
#: package rather than a bare ``INTEGRATION`` so an unrelated variable in a CI
#: environment cannot turn the real-checkpoint path on by accident.
OPT_IN_ENV = "MANGA_TEXT_SEG_INTEGRATION"

_HERE = Path(__file__).resolve().parent


def pytest_collection_modifyitems(config, items):
    opt_in = bool(os.environ.get(OPT_IN_ENV))
    marker = pytest.mark.integration
    skip = pytest.mark.skip(
        reason=f"opt-in integration test: set {OPT_IN_ENV}=1 to run (FR-051)"
    )
    for item in items:
        if _HERE not in Path(str(item.path)).resolve().parents:
            continue
        item.add_marker(marker)
        if not opt_in:
            item.add_marker(skip)
