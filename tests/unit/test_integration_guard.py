"""The opt-in boundary actually holds (T045, FR-051).

The guard in ``tests/integration/conftest.py`` is a collection hook: get its path
filter wrong and it either skips the whole suite or skips nothing, and both fail
silently — a suite that collected nothing looks green.  So the hook is called
directly here, with the two attributes it uses, on both sides of the boundary.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration import conftest

UNIT_TEST = Path(__file__).resolve()
INTEGRATION_TEST = UNIT_TEST.parent.parent / "integration" / "test_real_checkpoint.py"


class _Item:
    """The two attributes the hook touches: where the test lives, and its marks."""

    def __init__(self, path: Path):
        self.path = path
        self.markers: list = []

    def add_marker(self, marker) -> None:
        self.markers.append(marker)


def _skipped(item: _Item) -> bool:
    return any(mark.name == "skip" for mark in item.markers)


def test_the_default_suite_skips_the_integration_directory(monkeypatch):
    monkeypatch.delenv(conftest.OPT_IN_ENV, raising=False)
    inside, outside = _Item(INTEGRATION_TEST), _Item(UNIT_TEST)

    conftest.pytest_collection_modifyitems(None, [inside, outside])

    assert _skipped(inside), "a real-checkpoint test ran without being asked to"
    assert not _skipped(outside), "the guard swallowed a unit test"
    assert all(mark.name == "integration" for mark in inside.markers if mark.name != "skip")


def test_the_opt_in_environment_variable_is_what_lets_it_through(monkeypatch):
    monkeypatch.setenv(conftest.OPT_IN_ENV, "1")
    inside = _Item(INTEGRATION_TEST)

    conftest.pytest_collection_modifyitems(None, [inside])

    assert not _skipped(inside)
    assert any(mark.name == "integration" for mark in inside.markers), (
        "an opted-in test still needs the marker, so -m 'not integration' works"
    )


def test_the_marker_is_registered(pytestconfig):
    """An unregistered marker is only a warning, and a warning is not a boundary."""
    assert any(line.startswith("integration:") for line in pytestconfig.getini("markers"))
