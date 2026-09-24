"""Provenance completeness checking (FR-056, FR-018a/018b, SC-014).

A returned result carries one provenance record per method.  Completeness is
machine-checked before admission: a record missing any required field, or
carrying a value that does not satisfy the field's own rule, is refused with
the offending field's schema path named — never accepted with the gap filled in
by assumption (SC-014).

The required-field table below mirrors ``contracts/provenance.schema.json``.
It is written out rather than read from the spec tree at runtime: the schema is
a contract for humans and reviewers, and a module that cannot import without
the spec directory present would be a deployment trap.  ``tests/unit/test_provenance.py``
asserts the two agree by deriving its cases from the schema itself, so a drift
between them fails the suite rather than passing quietly.
"""
from __future__ import annotations

import re
from typing import Any

#: Required key per container: ``""`` is the record root. A container that is
#: absent reports the container path; an absent key reports ``container.key``.
REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "": (
        "method",
        "repository",
        "checkpoint",
        "code_license",
        "weight_license",
        "checkpoint_load_evidence",
        "device",
        "interpreter",
        "packages",
        "seed",
        "run_timestamp",
    ),
    "repository": ("url", "revision"),
    "checkpoint": ("identity", "size_bytes", "sha256"),
    "checkpoint_load_evidence": ("evidenced", "method", "detail"),
    "device": ("type", "name"),
}

#: Fields whose value must be a non-empty string.
NON_EMPTY_STRINGS = (
    "method",
    "code_license",
    "weight_license",
    "interpreter",
    "repository.url",
    "checkpoint.identity",
    "checkpoint_load_evidence.method",
    "checkpoint_load_evidence.detail",
    "device.name",
)

REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

DEVICE_TYPES = {"cpu", "cuda"}


class ProvenanceError(ValueError):
    """Raised when a provenance record is incomplete or carries an invalid value."""


#: Sentinel for "this path is absent", distinct from a present ``None``.
_MISSING = object()


def _get(record: dict[str, Any], path: str) -> Any:
    """Return the value at a dotted ``path``, or ``_MISSING`` if any step is absent."""
    node: Any = record
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def check_completeness(record: Any) -> list[str]:
    """Return the schema paths of every required field the record fails.

    An empty list means the record is complete and its values are well formed.
    Returning the whole list rather than raising lets a caller report all the
    gaps at once; :func:`require_complete` raises on the first pass instead,
    which is what admission uses.
    """
    if not isinstance(record, dict):
        raise ProvenanceError(
            f"Provenance record must be an object, got {type(record).__name__}"
        )

    missing: list[str] = []
    for container, keys in REQUIRED_KEYS.items():
        prefix = f"{container}." if container else ""
        if container and _get(record, container) is _MISSING:
            missing.append(container)
            continue
        for key in keys:
            if _get(record, f"{prefix}{key}") is _MISSING:
                missing.append(f"{prefix}{key}")

    for path in NON_EMPTY_STRINGS:
        value = _get(record, path)
        if value is _MISSING:
            continue
        if not isinstance(value, str) or not value.strip():
            missing.append(path)

    revision = _get(record, "repository.revision")
    if revision is not _MISSING and not (
        isinstance(revision, str) and REVISION_PATTERN.match(revision)
    ):
        # A branch name, tag or short SHA is not a revision.
        missing.append("repository.revision")

    sha256 = _get(record, "checkpoint.sha256")
    if sha256 is not _MISSING and not (
        isinstance(sha256, str) and SHA256_PATTERN.match(sha256)
    ):
        missing.append("checkpoint.sha256")

    size = _get(record, "checkpoint.size_bytes")
    if size is not _MISSING and not (
        isinstance(size, int) and not isinstance(size, bool) and size >= 1
    ):
        missing.append("checkpoint.size_bytes")

    evidenced = _get(record, "checkpoint_load_evidence.evidenced")
    if evidenced is not _MISSING and evidenced is not True:
        # FR-018b: positive evidence, or the run is inadmissible for this method.
        missing.append("checkpoint_load_evidence.evidenced")

    device_type = _get(record, "device.type")
    if device_type is not _MISSING and device_type not in DEVICE_TYPES:
        missing.append("device.type")

    packages = _get(record, "packages")
    if packages is not _MISSING and not (
        isinstance(packages, dict)
        and packages
        and all(isinstance(v, str) for v in packages.values())
    ):
        missing.append("packages")

    seed = _get(record, "seed")
    if seed is not _MISSING and not (
        isinstance(seed, int) and not isinstance(seed, bool)
    ):
        missing.append("seed")

    return sorted(set(missing))


def deviation_refusal(record: Any) -> str | None:
    """Why a record is inadmissible as its pinned method, or ``None`` if it is not.

    FR-055 and ``contracts/returned-result.md`` rule 7: a workaround applied in
    the runner's *own* environment is recorded as a deviation from the pinned
    revision, and a deviation that ``changes_output`` makes the result a
    **different method** — not admissible as the pinned one.  A workaround that
    leaves the mask alone is admissible and stays recorded, which is the whole
    point of recording it.

    ``deviation`` is optional in ``contracts/provenance.schema.json``, so this is
    a conditional rule rather than an entry in :data:`REQUIRED_KEYS`: a required
    key there would refuse every hand-off that records no deviation at all.
    """
    if not isinstance(record, dict):
        return None
    deviation = record.get("deviation")
    if deviation is None:
        return None
    if not isinstance(deviation, dict):
        return (
            f"deviation must be an object describing what was changed, got "
            f"{type(deviation).__name__}; a deviation that cannot be read cannot "
            f"be shown not to change the output (FR-055)"
        )
    if not deviation.get("changes_output", False):
        return None
    what = deviation.get("what") or "an unstated change"
    return (
        f"the runner's environment applied a deviation that changes the produced "
        f"mask ({what}); such a result is a different method and is not admissible "
        f"as the pinned one (FR-055, returned-result.md rule 7)"
    )


def require_complete(record: Any) -> None:
    """Raise :class:`ProvenanceError` naming every failing field, or return None.

    FR-056: *"A result whose provenance record is missing or incomplete MUST be
    refused rather than accepted with gaps."*
    """
    missing = check_completeness(record)
    if missing:
        raise ProvenanceError(
            "Provenance record is incomplete; refused rather than accepted with "
            f"gaps (FR-056): {', '.join(missing)}"
        )
