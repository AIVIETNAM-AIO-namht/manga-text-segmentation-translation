"""Recorded verdicts survive the process that produced them (T028, FR-022).

US2 scenario 5: a check that has run leaves recorded verdicts, reasons and
producing environments that are retrievable later *without re-running the
check*. The test proves "without re-running" the only way that is not a claim:
it saves the verdicts, then replaces the checking function with one that raises
if it is called at all, then reads the verdicts back.

The environment stamp is the load-bearing part (FR-017). A verdict is true of
the environment that produced it and of no other, so a stored verdict that lost
its environment is not evidence — it is refused, not defaulted.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from manga_text_seg import availability
from manga_text_seg.availability import (
    VerdictStoreError,
    check_all,
    load_verdicts,
    save_verdicts,
)
from manga_text_seg.config import load_dl_config

REPO_ROOT = Path(__file__).resolve().parents[2]
DL_CONFIG = REPO_ROOT / "configs" / "dl.json"


def _config():
    return load_dl_config(DL_CONFIG, repo_root=REPO_ROOT)


@pytest.fixture
def verdicts():
    return check_all(_config(), repo_root=REPO_ROOT)


def _explode(*args, **kwargs):
    raise AssertionError("the availability check was re-run; FR-022 forbids it")


class TestRoundTrip:
    def test_recorded_verdicts_are_read_back_without_re_running_the_check(
        self, tmp_path, verdicts, monkeypatch
    ):
        path = tmp_path / "availability.json"
        save_verdicts(path, verdicts)

        monkeypatch.setattr(availability, "check_method", _explode)
        monkeypatch.setattr(availability, "check_all", _explode)

        restored = load_verdicts(path)

        assert [verdict["method"] for verdict in restored] == [
            verdict["method"] for verdict in verdicts
        ]
        assert restored == verdicts

    def test_the_reason_and_the_producing_environment_survive(self, tmp_path, verdicts):
        path = tmp_path / "availability.json"
        save_verdicts(path, verdicts)

        restored = {verdict["method"]: verdict for verdict in load_verdicts(path)}

        for verdict in verdicts:
            stored = restored[verdict["method"]]
            assert stored["verdict"] == verdict["verdict"]
            assert stored["reason"] == verdict["reason"]
            assert stored["environment"] == verdict["environment"]
            assert stored["remediation"] == verdict["remediation"]
            assert stored["checked_at"] == verdict["checked_at"]

    def test_the_stored_file_is_plain_json_an_operator_can_read(self, tmp_path, verdicts):
        path = tmp_path / "availability.json"
        save_verdicts(path, verdicts)

        document = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(document, dict)
        assert "verdicts" in document
        assert isinstance(document["verdicts"], list)

    def test_reading_verdicts_does_not_rewrite_the_file(self, tmp_path, verdicts):
        path = tmp_path / "availability.json"
        save_verdicts(path, verdicts)
        before = path.read_bytes()

        load_verdicts(path)

        assert path.read_bytes() == before


class TestRefusals:
    """A stored verdict without its environment is not evidence (FR-017)."""

    def test_a_stored_verdict_with_no_environment_is_refused(self, tmp_path, verdicts):
        path = tmp_path / "availability.json"
        save_verdicts(path, verdicts)

        document = json.loads(path.read_text(encoding="utf-8"))
        del document["verdicts"][0]["environment"]
        path.write_text(json.dumps(document), encoding="utf-8")

        with pytest.raises(VerdictStoreError) as excinfo:
            load_verdicts(path)
        assert "environment" in str(excinfo.value)

    def test_a_stored_verdict_with_no_reason_is_refused(self, tmp_path, verdicts):
        path = tmp_path / "availability.json"
        save_verdicts(path, verdicts)

        document = json.loads(path.read_text(encoding="utf-8"))
        del document["verdicts"][0]["reason"]
        path.write_text(json.dumps(document), encoding="utf-8")

        with pytest.raises(VerdictStoreError) as excinfo:
            load_verdicts(path)
        assert "reason" in str(excinfo.value)

    def test_a_missing_store_is_reported_not_defaulted_to_empty(self, tmp_path):
        """An absent record means the check has not run — not that all is well."""
        with pytest.raises(FileNotFoundError):
            load_verdicts(tmp_path / "nothing-here.json")

    def test_a_corrupt_store_is_reported_not_silently_empty(self, tmp_path):
        path = tmp_path / "availability.json"
        path.write_text("{not json", encoding="utf-8")

        with pytest.raises(VerdictStoreError):
            load_verdicts(path)


class TestCheckedAt:
    def test_the_check_is_timestamped(self, tmp_path, verdicts):
        for verdict in verdicts:
            assert verdict["checked_at"]
            # An ISO-8601 timestamp, so a later reader can order two checks.
            assert "T" in verdict["checked_at"]

    def test_saving_twice_replaces_rather_than_appends(self, tmp_path, verdicts):
        path = tmp_path / "availability.json"
        save_verdicts(path, verdicts)
        save_verdicts(path, verdicts)

        assert len(load_verdicts(path)) == len(verdicts)
