"""T055 [P] No vendored segmentation-model sources or weight files (FR-055, SC-013).

FR-055 (Spec 002): Third-party model source code and pretrained weights MUST NOT
be vendored, copied, patched or committed in any form, any size, in any format.
SC-013 (Spec 003 plan.md §84): zero model source, zero weights — audited by
filesystem search. Nothing in this project can be reused as a model source.

The scan targets only git-tracked files (``git ls-files``), so gitignored
directories such as ``outputs/`` and ``checkpoints/`` are excluded by design.
The test passes on a clean clone and must continue to pass on every commit.

This test is intentionally self-contained: it needs no module under
``src/manga_text_seg/`` and imports only the standard library plus subprocess,
so it runs with ``pytest -o addopts='' -q`` even before the inpainting feature
is implemented (FR-037, FR-039, SC-012).

Implementation status: STANDALONE — PASS on current repo (T055 pre-checkpoint).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Binary weight / checkpoint extensions tracked by FR-055 / SC-013.
#: Covers all common deep-learning serialisation formats regardless of framework.
WEIGHT_EXTENSIONS = frozenset(
    [
        ".pth",   # PyTorch state-dict / full model
        ".pt",    # PyTorch alternative suffix
        ".onnx",  # Open Neural Network Exchange
        ".bin",   # HuggingFace safetensors companion, generic weights
        ".safetensors",  # HuggingFace Safetensors format
        ".pkl",   # Pickle — covers Caffe / sklearn / juvian-style checkpoints
        ".pickle",
        ".ckpt",  # TensorFlow / Lightning checkpoints
        ".h5",    # Keras HDF5
        ".hdf5",
        ".pb",    # TensorFlow frozen graph
        ".tflite",  # TensorFlow Lite
        ".caffemodel",  # Caffe
        ".npy",   # NumPy arrays (large weight matrices stored raw)
        ".npz",   # NumPy compressed arrays
        ".weights",  # Darknet / YOLO
        ".t7",    # Torch7 legacy
        ".mar",   # TorchServe model archive
        ".mlmodel",  # CoreML
    ]
)

#: Source-code directory names that signal a vendored upstream repository.
#: A legitimate file under these names inside ``src/`` would be vendoring.
VENDORED_SOURCE_DIRS = frozenset(
    [
        "comic_text_detector",
        "comic-text-detector",
        "manga_text_segmentation_upstream",
        "Manga-Text-Segmentation",
        "unetpp_efficientnetv2_upstream",
    ]
)

#: File patterns that identify third-party model source code committed inside
#: the *source tree* (``src/``). Checked against the full git-tracked path.
#:
#: Scope: **only ``src/`` paths** — ``deliverables/`` and ``notebooks/`` may
#: legitimately contain folder names matching method identities (they are *data*,
#: not vendored code). FR-055 forbids vendoring code, not naming output dirs.
VENDORED_SOURCE_PATTERNS = [
    re.compile(r"^src/.*comic.text.detector/", re.IGNORECASE),
    re.compile(r"^src/.*Manga.Text.Segmentation/", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_tracked_files() -> list[str]:
    """Return all file paths tracked by git (relative to repo root).

    Uses ``git ls-files`` so that gitignored paths such as
    ``outputs/``, ``checkpoints/``, ``tmp/``, and ``deliverables/*.zip``
    are excluded by design — we only audit what is actually committed.
    """
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNoVendoredWeights:
    """Assert zero weight / checkpoint files in the tracked tree (FR-055, SC-013)."""

    def test_no_weight_files_by_extension(self) -> None:
        """No file tracked by git has an extension that belongs to a weight format.

        A weight file committed at any size — even a 1-byte placeholder — would
        constitute vendoring under FR-055. The test lists all offenders on failure
        so the developer knows exactly what to remove.
        """
        tracked = _git_tracked_files()
        offenders = [
            f for f in tracked if Path(f).suffix.lower() in WEIGHT_EXTENSIONS
        ]
        assert offenders == [], (
            f"FR-055 / SC-013 violation: {len(offenders)} weight / checkpoint file(s) "
            f"found in the tracked tree:\n"
            + "\n".join(f"  {f}" for f in offenders)
            + "\n\nRemove them with `git rm --cached <file>` and add the pattern "
            "to .gitignore to prevent re-addition."
        )


class TestNoVendoredModelSource:
    """Assert no third-party model source code is committed verbatim (FR-055)."""

    def test_no_vendored_source_directories(self) -> None:
        """The tracked tree does not contain a directory associated with an upstream model.

        Vendoring upstream source is prohibited regardless of size: a single
        adapter file is the sanctioned interface (FR-055, Spec 002 §516).
        """
        tracked = _git_tracked_files()
        offenders = [
            f
            for f in tracked
            if any(pattern.search(f) for pattern in VENDORED_SOURCE_PATTERNS)
        ]
        assert offenders == [], (
            f"FR-055 violation: {len(offenders)} file(s) appear to be vendored "
            f"third-party model source:\n"
            + "\n".join(f"  {f}" for f in offenders)
            + "\n\nUpstream code must live in its own repository and be referenced "
            "via the adapter pattern in src/manga_text_seg/adapters/. "
            "Do not copy, patch or embed upstream source in this project."
        )

    def test_src_package_contains_only_own_modules(self) -> None:
        """No subdirectory of ``src/manga_text_seg/`` matches a vendored-source name.

        This is a structural check: a directory named after an upstream project
        inside our own package would be vendoring even if the files themselves
        passed the extension check.
        """
        src_root = REPO_ROOT / "src" / "manga_text_seg"
        if not src_root.exists():
            pytest.skip("src/manga_text_seg/ does not exist yet (pre-scaffold)")

        actual_dirs = {d.name for d in src_root.iterdir() if d.is_dir()}
        forbidden = actual_dirs & VENDORED_SOURCE_DIRS
        assert not forbidden, (
            f"FR-055 violation: directory name(s) inside src/manga_text_seg/ "
            f"match upstream model names: {sorted(forbidden)}. "
            "These must be adapters, not inline copies of the upstream code."
        )


class TestGitStatusSanity:
    """Smoke-check that the git scan itself works correctly."""

    def test_git_ls_files_returns_non_empty_list(self) -> None:
        """``git ls-files`` must return at least the README, so the scan is not a no-op."""
        tracked = _git_tracked_files()
        assert len(tracked) > 0, (
            "git ls-files returned nothing — either the repo has no tracked files "
            "or the subprocess call failed silently. The vendoring check cannot proceed."
        )

    def test_known_source_file_is_tracked(self) -> None:
        """At least one Python file under ``src/`` is tracked (sanity).

        If ``git ls-files`` were not finding tracked files correctly, this
        would fail — catching a broken test environment before the real checks
        produce a false-negative pass.
        """
        tracked = _git_tracked_files()
        py_sources = [f for f in tracked if f.startswith("src/") and f.endswith(".py")]
        assert py_sources, (
            "No .py file under src/ found in git-tracked files. "
            "Either git ls-files is broken or the project has no source yet. "
            "Investigate before relying on the vendoring checks."
        )
