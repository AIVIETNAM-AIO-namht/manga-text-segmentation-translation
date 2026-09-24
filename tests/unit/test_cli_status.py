"""``status`` — pre-flight verdicts, and admitted vs awaited (T033, FR-049).

US2's checkpoint is that runnability is known before inference starts and that
the verdicts survive the process that produced them.  The command has three
paths and this covers each: the check itself (which records what it found), the
``--run`` report read back out of that record, and a run whose record is absent.

The repository root is a temporary directory, so nothing here writes into the
real ``benchmark/`` — the command's own path rule is what the config's location
encodes, and the test exercises it rather than bypassing it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from manga_text_seg.cli import AVAILABILITY_RECORD, BENCHMARK_DIR, cmd_status

REPO_ROOT = Path(__file__).resolve().parents[2]
DL_CONFIG = REPO_ROOT / "configs" / "dl.json"


def _repo(tmp_path: Path) -> Path:
    """A throwaway repository root with the real DL config inside it."""
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "dl.json").write_text(
        DL_CONFIG.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return root


def _config_path(root: Path) -> str:
    return str(root / "configs" / "dl.json")


def _write_run(root: Path, run_id: str, admitted, awaited) -> None:
    run_dir = root / "deliverables" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps({"methods_admitted": admitted, "methods_awaited": awaited}),
        encoding="utf-8",
    )


def test_the_check_reports_every_method_and_records_the_verdicts(tmp_path, capsys):
    root = _repo(tmp_path)

    assert cmd_status(_config_path(root)) == 0

    out = capsys.readouterr().out
    assert "manga-text-segmentation: UNAVAILABLE" in out
    assert "comic-text-detector: UNAVAILABLE" in out
    assert "unetpp-efficientnetv2: UNAVAILABLE" in out
    # FR-019: what to obtain, from where, how large.
    assert "comictextdetector.pt" in out
    assert "79948869 bytes" in out
    assert "https://github.com/dmMaze/comic-text-detector" in out
    # FR-017: each verdict is stamped with the environment that produced it.
    assert "checked in: Python" in out

    recorded = json.loads(
        (root / BENCHMARK_DIR / AVAILABILITY_RECORD).read_text(encoding="utf-8")
    )
    assert [verdict["method"] for verdict in recorded["verdicts"]] == [
        "manga-text-segmentation",
        "comic-text-detector",
        "unetpp-efficientnetv2",
    ]


def test_the_run_report_names_awaited_methods_with_their_reasons(tmp_path, capsys):
    """FR-036/FR-018: "not run" with a reason, and no numbers for them."""
    root = _repo(tmp_path)
    cmd_status(_config_path(root))
    capsys.readouterr()
    _write_run(root, "default", ["manga-text-segmentation"], ["comic-text-detector"])

    assert cmd_status(_config_path(root), "default") == 0

    out = capsys.readouterr().out
    assert "admitted: manga-text-segmentation" in out
    assert "not run: comic-text-detector" in out
    assert "comictextdetector.pt" in out
    # No synthetic value: nothing that could be read as a metric.
    for token in ("iou", "dice", "f1", "0.0"):
        assert token not in out.lower()


def test_a_run_with_no_record_is_refused_rather_than_invented(tmp_path, capsys):
    root = _repo(tmp_path)
    cmd_status(_config_path(root))
    capsys.readouterr()

    assert cmd_status(_config_path(root), "nosuchrun") == 2
    assert "no readable record" in capsys.readouterr().err


def test_a_run_report_before_any_check_names_the_missing_verdicts(tmp_path, capsys):
    """FR-022's other edge: the run exists but the check has not been run here."""
    root = _repo(tmp_path)
    _write_run(root, "default", [], ["comic-text-detector"])

    assert cmd_status(_config_path(root), "default") == 2
    assert "No usable verdicts" in capsys.readouterr().err


@pytest.mark.parametrize("run_id", [None, "default"])
def test_the_repo_root_is_where_the_config_says_it_is(tmp_path, run_id):
    """The record lands beside the page list, never in the current directory."""
    root = _repo(tmp_path)
    _write_run(root, "default", [], [])
    cmd_status(_config_path(root), run_id)
    if run_id is None:
        assert (root / BENCHMARK_DIR / AVAILABILITY_RECORD).is_file()
