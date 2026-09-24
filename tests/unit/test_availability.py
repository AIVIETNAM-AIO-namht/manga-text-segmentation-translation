"""The per-method pre-flight verdict (T027, US2 scenarios 1, 2 and 4).

US2's point is that runnability is known *before* inference starts. Scenario 1:
with no checkpoints installed every method reports unavailable, naming the
missing artefact, where to obtain it and how large it is. Scenario 2: an absent
or incompatible dependency is named as a version conflict here, rather than
surfacing mid-benchmark. Scenario 4: a checkpoint that is present but truncated
or of the wrong kind is unavailable with the mismatch described.

Every assertion is made against this machine's real filesystem and this
interpreter's real package set — ``checkpoints/`` is absent and no deep-learning
runtime is importable, which is exactly scenario 1's precondition. Nothing here
downloads, hashes a 344 MB file, or loads a checkpoint (FR-051).

The configuration is perturbed rather than restated: a copy of ``configs/dl.json``
is written to ``tmp_path`` with one field changed, the same pattern
``test_config_dl.py`` uses, so the tests cannot drift from the shipped config.
"""
from __future__ import annotations

import importlib.util
import json
import platform
import subprocess
from pathlib import Path

import pytest

from manga_text_seg.availability import check_all, check_method
from manga_text_seg.config import load_dl_config

REPO_ROOT = Path(__file__).resolve().parents[2]
DL_CONFIG = REPO_ROOT / "configs" / "dl.json"
METHOD_NAMES = ["manga-text-segmentation", "comic-text-detector", "unetpp-efficientnetv2"]

#: The method used where a single small checkpoint keeps the test cheap.
SMALL = "comic-text-detector"


def _write_dl(tmp_path: Path, mutate=None) -> Path:
    """A perturbed copy of ``configs/dl.json``, written and returned."""
    doc = json.loads(DL_CONFIG.read_text(encoding="utf-8"))
    if mutate is not None:
        mutate(doc)
    path = tmp_path / "dl.json"
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def _config(tmp_path: Path, mutate=None):
    """The perturbed configuration, with paths resolved against the real repo."""
    return load_dl_config(_write_dl(tmp_path, mutate), repo_root=REPO_ROOT)


def _checkout(root: Path, requirements: str) -> Path:
    """A directory standing in for a method's external repository (FR-055)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "requirements.txt").write_text(requirements, encoding="utf-8")
    return root


def _point_at_checkout(doc: dict, method: str, path: Path) -> None:
    doc["methods"][method]["repository"]["path"] = str(path)


def _point_checkpoint_at(doc: dict, method: str, path: Path, size: int, sha) -> None:
    checkpoint = doc["methods"][method]["checkpoints"][0]
    checkpoint["path"] = str(path)
    checkpoint["size_bytes"] = size
    checkpoint["sha256"] = sha


class TestScenario1NothingIsInstalled:
    """Scenario 1: no checkpoints ⇒ unavailable, naming artefact, source, size."""

    def test_each_method_names_the_missing_artefact_its_size_and_where_to_get_it(
        self, tmp_path
    ):
        config = _config(tmp_path)
        verdict = check_method(
            config.methods[SMALL], repo_root=REPO_ROOT
        )

        assert verdict["method"] == SMALL
        assert verdict["verdict"] == "unavailable"

        status = verdict["checkpoint_status"]
        assert status["present"] is False
        assert status["missing_artefact"] == "comictextdetector.pt"
        assert status["size_bytes"] == 79948869
        assert status["obtain_from"] == "https://github.com/dmMaze/comic-text-detector"

        # FR-016: each failure with a human-readable reason naming the artefact.
        assert "comictextdetector.pt" in verdict["reason"]

        # FR-019: the remediation carries the source and the expected size.
        remediation = " ".join(verdict["remediation"])
        assert "comictextdetector.pt" in remediation
        assert "79948869" in remediation
        assert "https://github.com/dmMaze/comic-text-detector" in remediation

    def test_the_shipped_configuration_reports_every_method_unavailable_here(
        self, tmp_path
    ):
        """The real pre-flight, on the real config, against this real machine."""
        verdicts = check_all(_config(tmp_path), repo_root=REPO_ROOT)

        assert [verdict["method"] for verdict in verdicts] == METHOD_NAMES
        for verdict in verdicts:
            assert verdict["verdict"] == "unavailable"
            assert verdict["reason"]
            assert verdict["remediation"]

    def test_method_a_reports_the_fold_it_is_missing(self, tmp_path):
        """FR-013a: Method A needs all five folds before any page is inferred."""
        config = _config(tmp_path)
        verdict = check_method(config.methods["manga-text-segmentation"], repo_root=REPO_ROOT)

        assert verdict["verdict"] == "unavailable"
        missing = verdict["checkpoint_status"]["missing_artefact"]
        assert "fold" in missing


class TestScenario2Dependencies:
    """Scenario 2: the version conflict is named here, not at inference time."""

    def test_a_dependency_that_is_not_installed_is_a_named_conflict(self, tmp_path):
        checkout = _checkout(tmp_path / "checkout", "torch==1.4.0\nfastai==1.0.60\n")
        config = _config(tmp_path, lambda doc: _point_at_checkout(doc, SMALL, checkout))
        verdict = check_method(config.methods[SMALL], repo_root=REPO_ROOT)

        status = verdict["dependency_status"]
        assert status["satisfied"] is False
        assert "torch" in status["conflict"]
        assert "1.4.0" in status["conflict"]
        assert verdict["verdict"] == "unavailable"

    def test_a_dependency_at_another_version_names_both_versions(self, tmp_path):
        """numpy is installed here, so this is a real mismatch, not a mock."""
        checkout = _checkout(tmp_path / "checkout", "numpy==1.0.0\n")
        config = _config(tmp_path, lambda doc: _point_at_checkout(doc, SMALL, checkout))
        verdict = check_method(config.methods[SMALL], repo_root=REPO_ROOT)

        conflict = verdict["dependency_status"]["conflict"]
        assert verdict["dependency_status"]["satisfied"] is False
        assert "numpy" in conflict
        assert "1.0.0" in conflict
        # The observed version is named too, so an operator can see the gap.
        assert "2." in conflict

    def test_dependencies_that_are_satisfied_do_not_fail_the_verdict(self, tmp_path):
        checkout = _checkout(tmp_path / "checkout", "numpy\n")
        config = _config(tmp_path, lambda doc: _point_at_checkout(doc, SMALL, checkout))
        verdict = check_method(config.methods[SMALL], repo_root=REPO_ROOT)

        assert verdict["dependency_status"]["satisfied"] is True
        assert verdict["dependency_status"]["conflict"] is None
        assert verdict["environment"]["packages"]["numpy"]

    def test_a_pinned_stack_that_will_not_resolve_names_the_interpreter_recheck(
        self, tmp_path
    ):
        """FR-021: an unresolvable pin is re-checked elsewhere, not written off."""
        checkout = _checkout(tmp_path / "checkout", "torch==1.4.0\n")
        config = _config(tmp_path, lambda doc: _point_at_checkout(doc, SMALL, checkout))
        verdict = check_method(config.methods[SMALL], repo_root=REPO_ROOT)

        assert any("re-check" in step for step in verdict["remediation"])
        assert platform.python_version() in " ".join(verdict["remediation"])


class TestScenario4TheCheckpointIsThereButWrong:
    """Scenario 4: present but truncated, or of the wrong architecture."""

    def test_a_truncated_checkpoint_describes_the_size_mismatch(self, tmp_path):
        path = tmp_path / "ckpt" / "comictextdetector.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"PK\x03\x04" + b"\x00" * 100)

        config = _config(
            tmp_path,
            lambda doc: _point_checkpoint_at(doc, SMALL, path, 79948869, None),
        )
        verdict = check_method(config.methods[SMALL], repo_root=REPO_ROOT)

        assert verdict["verdict"] == "unavailable"
        assert verdict["checkpoint_status"]["present"] is True
        assert verdict["checkpoint_status"]["missing_artefact"] is None
        assert "comictextdetector.pt" in verdict["reason"]
        assert "79948869" in verdict["reason"]
        assert str(path.stat().st_size) in verdict["reason"]

    def test_a_checkpoint_of_the_wrong_kind_describes_the_mismatch(self, tmp_path):
        """Right size, wrong container: a text file where a torch zip belongs."""
        path = tmp_path / "ckpt" / "comictextdetector.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a checkpoint" * 2)

        config = _config(
            tmp_path,
            lambda doc: _point_checkpoint_at(
                doc, SMALL, path, path.stat().st_size, None
            ),
        )
        verdict = check_method(config.methods[SMALL], repo_root=REPO_ROOT)

        assert verdict["verdict"] == "unavailable"
        assert verdict["checkpoint_status"]["present"] is True
        assert "comictextdetector.pt" in verdict["reason"]

    def test_a_checkpoint_whose_digest_disagrees_is_unavailable(self, tmp_path):
        """FR-016's integrity half: the configured digest is checked, not assumed."""
        path = tmp_path / "ckpt" / "comictextdetector.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"PK\x03\x04" + b"\x00" * 100)

        config = _config(
            tmp_path,
            lambda doc: _point_checkpoint_at(
                doc, SMALL, path, path.stat().st_size, "f" * 64
            ),
        )
        verdict = check_method(config.methods[SMALL], repo_root=REPO_ROOT)

        assert verdict["verdict"] == "unavailable"
        assert "ffffffff" in verdict["reason"] or "f" * 64 in verdict["reason"]


class TestScenario7AnAcceleratorThisMachineLacks:
    """FR-021: re-checked in a suitable environment before being declared unavailable."""

    @pytest.mark.skipif(
        importlib.util.find_spec("torch") is not None,
        reason="a CUDA runtime may be present, so the accelerator may not be missing",
    )
    def test_a_cuda_device_this_machine_lacks_names_the_recheck(self, tmp_path):
        def _require_cuda(doc: dict) -> None:
            doc["methods"][SMALL]["device"] = {"type": "cuda", "name": "Tesla T4"}

        config = _config(tmp_path, _require_cuda)
        verdict = check_method(config.methods[SMALL], repo_root=REPO_ROOT)

        assert verdict["verdict"] == "unavailable"
        assert "Tesla T4" in verdict["reason"]
        assert verdict["environment"]["device"]["type"] == "cpu"
        assert any("re-check" in step for step in verdict["remediation"])


class TestPinnedRevision:
    """FR-016: third-party code availability and pinned revision."""

    def test_a_checkout_at_the_pinned_revision_satisfies_the_repository_check(
        self, tmp_path
    ):
        checkout = _checkout(tmp_path / "checkout", "numpy\n")
        subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.email=fixture@example.invalid",
                "-c",
                "user.name=fixture",
                "commit",
                "-q",
                "--allow-empty",
                "-m",
                "pinned",
            ],
            cwd=checkout,
            check=True,
        )
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        def _pin(doc: dict) -> None:
            _point_at_checkout(doc, SMALL, checkout)
            doc["methods"][SMALL]["repository"]["revision"] = head

        config = _config(tmp_path, _pin)
        verdict = check_method(config.methods[SMALL], repo_root=REPO_ROOT)

        status = verdict["repository_status"]
        assert status["present"] is True
        assert status["revision_matches"] is True

    def test_a_checkout_at_another_revision_names_both(self, tmp_path):
        checkout = _checkout(tmp_path / "checkout", "numpy\n")
        subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.email=fixture@example.invalid",
                "-c",
                "user.name=fixture",
                "commit",
                "-q",
                "--allow-empty",
                "-m",
                "other",
            ],
            cwd=checkout,
            check=True,
        )
        config = _config(tmp_path, lambda doc: _point_at_checkout(doc, SMALL, checkout))
        verdict = check_method(config.methods[SMALL], repo_root=REPO_ROOT)

        status = verdict["repository_status"]
        assert status["present"] is True
        assert status["revision_matches"] is False
        assert "440b978563c71b758e31aaa315d100faba1efa2f" in status["detail"]


class TestVerdictShape:
    """The nine attributes ``data-model.md`` fixes, plus the FR-016 reason."""

    def test_every_verdict_carries_its_producing_environment(self, tmp_path):
        verdicts = check_all(_config(tmp_path), repo_root=REPO_ROOT)

        for verdict in verdicts:
            assert set(verdict) >= {
                "method",
                "verdict",
                "environment",
                "checkpoint_status",
                "repository_status",
                "dependency_status",
                "license_status",
                "remediation",
                "checked_at",
            }
            assert verdict["verdict"] in {"available", "unavailable"}
            assert verdict["environment"]["interpreter"]
            assert verdict["environment"]["device"]["type"] in {"cpu", "cuda"}
            # FR-060: a bare 'cpu' is not a name.
            assert verdict["environment"]["device"]["name"]
            assert verdict["checked_at"]

    def test_both_licences_are_recorded_separately(self, tmp_path):
        config = _config(tmp_path)
        verdict = check_method(config.methods[SMALL], repo_root=REPO_ROOT)

        status = verdict["license_status"]
        assert status["code"] == "GPL-3.0"
        assert status["weights"] == "GPL-3.0"
        assert status["restriction"]

    def test_a_method_with_no_declared_licence_records_that_as_the_restriction(
        self, tmp_path
    ):
        """``none`` is a recorded position, and the restriction says what it means."""
        config = _config(tmp_path)
        verdict = check_method(config.methods["unetpp-efficientnetv2"], repo_root=REPO_ROOT)

        status = verdict["license_status"]
        assert status["code"] == "none"
        assert status["weights"] == "none"
        assert status["restriction"]

    def test_an_unknown_method_is_not_silently_verdict_free(self, tmp_path):
        """A method in the config but with no adapter still gets a verdict."""
        def _add(doc: dict) -> None:
            doc["methods"]["not-registered"] = json.loads(
                json.dumps(doc["methods"][SMALL])
            )

        verdicts = check_all(_config(tmp_path, _add), repo_root=REPO_ROOT)
        assert "not-registered" in [verdict["method"] for verdict in verdicts]
