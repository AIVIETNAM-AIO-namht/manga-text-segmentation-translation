"""Per-method pre-flight availability verdicts (FR-016–FR-022, US2).

``config.py`` already validates everything that is *static* about a method: the
revision is 40 hex characters, the checkpoint records a size, the device type is
one of two spellings, both licences are present.  What it cannot know is whether
any of it is true of *this* machine.  That is this module's whole job: is the
checkpoint file actually there at the recorded size and digest, is the
third-party checkout present at the pinned revision, are the pinned dependencies
importable at compatible versions, what device does this environment actually
have, and what should an operator do about each failure.

FR-017 is why every verdict carries an ``environment`` block: the runtimes live
outside this project (FR-055, FR-052), so a verdict obtained here says nothing
about a machine that will actually run the method.  The stamp is what lets a
reader tell the two apart.  A verdict is produced in under a minute and never
loads a checkpoint (SC-007); nothing under the out-of-scope data directory is
opened (SC-009).
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from .config import DLConfig, DLMethod

#: A torch artefact is a zip container; a plain pickle starts with a protocol
#: opcode.  Both are legitimate, so the check accepts either and refuses a file
#: that is neither — the right byte count is not evidence of the right file.
_ZIP_MAGIC = b"PK\x03\x04"
_PICKLE_OPCODE = b"\x80"

#: ``name``, optionally ``name==version``.  Markers, extras and hashes are not
#: interpreted: the pinned stack is what upstream's ``requirements.txt`` names,
#: and inventing a resolution for the rest would be a claim this check cannot
#: support.
_REQUIREMENT = re.compile(r"^([A-Za-z0-9._-]+)\s*(?:==\s*([^\s;#]+))?")


def probe_device() -> dict[str, str]:
    """The compute device this environment actually has (FR-016, FR-060).

    FR-060: a bare ``"cpu"`` is not a name.  The device is probed rather than
    assumed from the configuration, because the configuration says what the
    method *wants* and the verdict has to record what it will *get*.
    """
    name = platform.processor() or platform.machine() or "unknown"
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch
        except Exception:  # a broken torch is not this check's failure to report
            return {"type": "cpu", "name": name}
        if torch.cuda.is_available():
            return {"type": "cuda", "name": torch.cuda.get_device_name(0)}
    return {"type": "cpu", "name": name}


def _installed(name: str) -> str | None:
    """The installed version of ``name``, or ``None`` when it is not installed."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _requirements(checkout: Path | None) -> list[tuple[str, str | None]]:
    """The pinned dependencies a method's checkout declares, in file order."""
    if checkout is None:
        return []
    try:
        lines = (checkout / "requirements.txt").read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    pinned: list[tuple[str, str | None]] = []
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = _REQUIREMENT.match(line)
        if match is not None:
            pinned.append((match.group(1), match.group(2)))
    return pinned


def _container_ok(path: Path) -> bool:
    with path.open("rb") as handle:
        head = handle.read(4)
    return head.startswith(_ZIP_MAGIC) or head[:1] == _PICKLE_OPCODE


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(checkout: Path) -> str | None:
    """The checkout's HEAD, or ``None`` when it is not a readable git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=checkout,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _licence_status(method: DLMethod) -> dict[str, Any]:
    """Both licences, and what each one means for reuse (FR-016).

    ``none`` is a recorded position rather than a missing value: upstream
    published no licence, so reuse is unverified, and the restriction says so
    instead of leaving a reader to read the absence as permission.
    """
    code, weights = method.code_license, method.weight_license
    if "none" in (code, weights):
        which = " and ".join(
            label
            for label, value in (("code", code), ("weights", weights))
            if value == "none"
        )
        restriction = (
            f"no licence is declared upstream for the {which}; redistribution and "
            f"reuse are unverified, so this is recorded as declared rather than "
            f"read as permission"
        )
    elif code != weights:
        restriction = (
            f"code is {code} and weights are {weights}; the two differ, so both "
            f"terms apply to whatever is redistributed"
        )
    else:
        restriction = (
            f"{code} — the identifier is recorded as declared by the upstream "
            f"project, not as a determination that reuse is permitted here"
        )
    return {"code": code, "weights": weights, "restriction": restriction}


def check_method(method: DLMethod, *, repo_root: Path) -> dict[str, Any]:
    """One method's verdict, produced here and stamped with this environment.

    Every failing check contributes its own reason and its own remediation step
    rather than the first one aborting the rest (FR-016, FR-019): an operator
    fixing a machine wants the whole list, not one item per round trip.
    """
    root = Path(repo_root)
    device = probe_device()
    reasons: list[str] = []
    remediation: list[str] = []

    # FR-021: the method wants an accelerator this environment does not have.
    # Reported with the re-check instruction rather than as a plain verdict,
    # because this machine is not the machine that will run it.
    wanted = method.device.get("type")
    if wanted is not None and wanted != device["type"]:
        reasons.append(
            f"the method is configured for device {wanted} "
            f"({method.device.get('name')}), but this environment has "
            f"{device['type']} ({device['name']})"
        )
        remediation.append(
            f"re-check this method in an environment with a {wanted} device "
            f"({method.device.get('name')}) before recording it as unavailable "
            f"here (FR-021)"
        )

    # FR-016: checkpoint presence and integrity.  FR-018a: the weights actually
    # loaded are the configured ones, which starts with the file on disk being
    # the file the configuration describes.
    present = True
    missing_artefact: str | None = None
    for checkpoint in method.checkpoints:
        path = Path(checkpoint.path)
        if not path.is_file():
            present = False
            if missing_artefact is None:
                missing_artefact = checkpoint.identity
            reasons.append(f"checkpoint {checkpoint.identity} is not present at {path}")
            remediation.append(
                f"obtain {checkpoint.identity} ({checkpoint.size_bytes} bytes) from "
                f"{method.repository['url']} and place it at {path}"
            )
            continue

        observed_size = path.stat().st_size
        if observed_size != checkpoint.size_bytes:
            reasons.append(
                f"checkpoint {checkpoint.identity} at {path} is {observed_size} "
                f"bytes, but the configuration records {checkpoint.size_bytes}; it "
                f"is truncated or is not the pinned artefact"
            )
            remediation.append(
                f"re-obtain {checkpoint.identity} ({checkpoint.size_bytes} bytes) "
                f"from {method.repository['url']}"
            )
            continue

        if not _container_ok(path):
            reasons.append(
                f"checkpoint {checkpoint.identity} at {path} is neither a torch "
                f"archive nor a pickle; it is the right size but not the right kind "
                f"of file"
            )
            remediation.append(
                f"re-obtain {checkpoint.identity} from {method.repository['url']}"
            )
            continue

        if checkpoint.sha256:
            observed_digest = _digest(path)
            if observed_digest != checkpoint.sha256:
                reasons.append(
                    f"checkpoint {checkpoint.identity} has sha256 "
                    f"{observed_digest}, but the configuration records "
                    f"{checkpoint.sha256}"
                )
                remediation.append(
                    f"re-obtain {checkpoint.identity} from "
                    f"{method.repository['url']}; the copy here does not match the "
                    f"recorded digest"
                )

    # FR-016: third-party code availability and pinned revision (FR-055 keeps
    # that code out of this project, so its absence is normal and reportable).
    revision = method.repository["revision"]
    checkout: Path | None = None
    declared_path = method.repository.get("path")
    if declared_path is None:
        repository_status = {
            "present": False,
            "revision_matches": False,
            "detail": (
                f"no local checkout is recorded for {method.repository['url']}, so "
                f"the pinned revision {revision} cannot be confirmed here"
            ),
        }
        reasons.append(repository_status["detail"])
        remediation.append(
            f"clone {method.repository['url']} at revision {revision}, point "
            f"repository.path at the checkout, and re-check"
        )
    else:
        checkout = Path(declared_path)
        if not checkout.is_absolute():
            checkout = (root / checkout).resolve()
        head = _git_revision(checkout)
        if head is None:
            repository_status = {
                "present": True,
                "revision_matches": False,
                "detail": (
                    f"{checkout} is not a readable git checkout, so the pinned "
                    f"revision {revision} cannot be confirmed"
                ),
            }
            reasons.append(repository_status["detail"])
        elif head == revision:
            repository_status = {"present": True, "revision_matches": True, "detail": None}
        else:
            repository_status = {
                "present": True,
                "revision_matches": False,
                "detail": (
                    f"{checkout} is at revision {head}, but the configuration pins "
                    f"{revision}"
                ),
            }
            reasons.append(repository_status["detail"])
        if not repository_status["revision_matches"]:
            remediation.append(
                f"check out revision {revision} of {method.repository['url']} at "
                f"{checkout} and re-check"
            )

    # FR-021: version compatibility is settled here, not mid-benchmark.
    packages: dict[str, str] = {}
    conflicts: list[str] = []
    for name, required in _requirements(checkout):
        observed = _installed(name)
        if observed is not None:
            packages[name] = observed
            if required is not None and observed != required:
                conflicts.append(
                    f"{name}=={required} is required but {name} {observed} is installed"
                )
        elif required is None:
            conflicts.append(f"{name} is required but is not installed")
        else:
            conflicts.append(f"{name}=={required} is required but {name} is not installed")

    if conflicts:
        reasons.extend(conflicts)
        remediation.extend(
            f"install the pinned dependency stack before running this method: {conflict}"
            for conflict in conflicts
        )
        # FR-021's other half.  A pinned stack that will not resolve here is the
        # signature of a method pinned to an interpreter this machine does not
        # have — Method A's torch 1.4.0 predates the interpreter below — so the
        # verdict names the environment it was obtained in and says where to
        # re-check, rather than leaving "unavailable" as the last word.
        remediation.append(
            f"if this stack cannot be installed under Python "
            f"{platform.python_version()}, re-check this method in an environment "
            f"with the interpreter its pins were released for (FR-021)"
        )

    verdict = "unavailable" if reasons else "available"
    return {
        "method": method.name,
        "verdict": verdict,
        "reason": "; ".join(reasons),
        "environment": {
            "interpreter": platform.python_version(),
            "device": device,
            "packages": packages,
        },
        "checkpoint_status": {
            "present": present,
            "missing_artefact": missing_artefact,
            "obtain_from": method.repository["url"],
            "size_bytes": method.checkpoints[0].size_bytes,
        },
        "repository_status": repository_status,
        "dependency_status": {
            "satisfied": not conflicts,
            "conflict": "; ".join(conflicts) if conflicts else None,
        },
        "license_status": _licence_status(method),
        "remediation": remediation,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def check_all(config: DLConfig, *, repo_root: Path) -> list[dict[str, Any]]:
    """Every configured method's verdict, in configuration order.

    A method with no adapter still gets a verdict: runnability is a property of
    the environment, and a method this project cannot yet run is exactly the one
    an operator needs a reason for (FR-018).
    """
    return [
        check_method(method, repo_root=repo_root) for method in config.methods.values()
    ]


class VerdictStoreError(ValueError):
    """A stored verdict set could not be read as one (FR-022)."""


#: What a stored verdict must still carry to be evidence at all.  The
#: environment is FR-017's whole point — a verdict is true of the environment
#: that produced it — and the reason is FR-016's.  A record missing either is
#: refused rather than defaulted, because a default would read as a verdict
#: nobody produced.
_REQUIRED_VERDICT_KEYS = ("environment", "reason")


def save_verdicts(path: Path, verdicts: list[dict[str, Any]]) -> None:
    """Write ``verdicts`` to ``path`` as JSON, replacing whatever is there.

    Replacing rather than appending is deliberate: the store holds the *latest*
    check, and a reader that had to pick the newest of several would be reading
    a history this project never agreed to keep.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"verdicts": list(verdicts)}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_verdicts(path: Path) -> list[dict[str, Any]]:
    """The recorded verdicts, read back without re-running any check (FR-022).

    Raises ``FileNotFoundError`` when nothing has been recorded — an absent
    store means the check has not run, which is not the same as a clean verdict
    — and :class:`VerdictStoreError` when what is there cannot be read as a
    verdict set.  Nothing is rewritten: reading a record does not produce one.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"no availability verdicts recorded at {path}; the check has not run"
        )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise VerdictStoreError(f"{path} is not readable JSON: {exc}") from exc

    if not isinstance(document, dict) or not isinstance(document.get("verdicts"), list):
        raise VerdictStoreError(
            f"{path} does not hold a verdict set: expected an object with a "
            f"'verdicts' list"
        )

    for index, verdict in enumerate(document["verdicts"]):
        if not isinstance(verdict, dict):
            raise VerdictStoreError(f"{path}: verdict {index} is not an object")
        for key in _REQUIRED_VERDICT_KEYS:
            if not verdict.get(key):
                raise VerdictStoreError(
                    f"{path}: verdict {index} "
                    f"({verdict.get('method', 'unnamed')}) carries no {key}; a "
                    f"record without it is not evidence of anything (FR-017)"
                )
    return document["verdicts"]
