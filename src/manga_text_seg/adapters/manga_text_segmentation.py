"""Method A: ``juvian/Manga-Text-Segmentation``, scored leave-one-fold-out (T025).

The project never witnesses Method A executing: under FR-052 the method runs in
its own environment (Python 3.7, fastai 1.0.60, torch 1.4.0, a fastai-v1
pickle) and returns masks and provenance.  What this module owns is the part
that is the project's own: *which* of the five released fold checkpoints is
entitled to score a given page, and whether that is knowable here at all.

Upstream held every Manga109 book out of exactly one of its five folds, so for
any page at most one released checkpoint never trained on it.  Which one is
computable from the published seed — and scoring all 390 pages with one
checkpoint is not a weaker result but an invalid one (FR-013a, research.md R4).

The derivation reproduces upstream's split without scikit-learn: ``KFold`` is
not a declared project dependency, and ``RandomState(42).shuffle`` on the same
index array reproduces ``KFold(n_splits=5, shuffle=True, random_state=42)``
exactly.

.. note::
   research.md R4 describes the book list as *"the 45 book names sorted and
   split into 5 folds"*.  Upstream does **not** sort it: ``experiments.py``
   holds a hand-maintained literal and indexes into it with
   ``trainFolders.index(...)``, so the literal's order *is* the split.  Four
   ``T`` names sit after every ``Y`` name in it.  Sorting them would move 13 of
   the 45 books to a different fold, silently re-attributing their pages to a
   checkpoint that trained on them — so :data:`TRAIN_FOLDERS` is transcribed in
   upstream's order, not sorted.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from . import register
from .base import DLMethodAdapter, InferenceResult

#: Upstream's published split seed (``experiments.py``).
SEED = 42

#: The five folds it released checkpoints for.
FOLD_COUNT = 5

#: Upstream's ``trainFolders`` literal, in its own order.  The order is
#: load-bearing: the split is derived by indexing into this list.
TRAIN_FOLDERS = (
    "ARMS",
    "AisazuNihaIrarenai",
    "AkkeraKanjinchou",
    "Akuhamu",
    "AosugiruHaru",
    "AppareKappore",
    "Arisa",
    "BEMADER_P",
    "BakuretsuKungFuGirl",
    "Belmondo",
    "BokuHaSitatakaKun",
    "BurariTessenTorimonocho",
    "ByebyeC-BOY",
    "Count3DeKimeteAgeru",
    "DollGun",
    "Donburakokko",
    "DualJustice",
    "EienNoWith",
    "EvaLady",
    "EverydayOsakanaChan",
    "GOOD_KISS_Ver2",
    "GakuenNoise",
    "GarakutayaManta",
    "GinNoChimera",
    "Hamlet",
    "HanzaiKousyouninMinegishiEitarou",
    "HaruichibanNoFukukoro",
    "HarukaRefrain",
    "HealingPlanet",
    "UchiNoNyan'sDiary",
    "UchuKigekiM774",
    "UltraEleven",
    "UnbalanceTokyo",
    "WarewareHaOniDearu",
    "YamatoNoHane",
    "YasasiiAkuma",
    "YouchienBoueigumi",
    "YoumaKourin",
    "YukiNoFuruMachi",
    "YumeNoKayoiji",
    "YumeiroCooking",
    "TotteokiNoABC",
    "ToutaMairimasu",
    "TouyouKidan",
    "TsubasaNoKioku",
)

#: FR-019: what an operator needs to make this method runnable.
REMEDIATION = (
    "install the pinned stack in a Python 3.7 environment: "
    "fastai==1.0.60 --no-deps, torch==1.4.0, torchvision==0.5.0 "
    "(see the method's requirements.txt)",
    "obtain all five fold checkpoints from the upstream v1.0 release "
    "(https://github.com/juvian/Manga-Text-Segmentation/releases/tag/v1.0) "
    "into checkpoints/manga-text-segmentation/, ~344 MB each",
)

#: FR-024a: upstream pads to a multiple of 8 and the padding must be inverted
#: before the mask is returned at the aligned size.
DEFAULT_PADDING_MULTIPLE = 8

#: Upstream's sigmoid cutoff is fixed at 0.5 and is not exposed as a parameter,
#: so this is the value that scores whatever the configuration says.
DEFAULT_THRESHOLD = 0.5

#: The method's training label space: background, easy text, hard text, then
#: three ignore classes.  The GT this project scores against is binary (FR-004),
#: so the method's output has to be collapsed onto it (FR-024b).
LABEL_SPACE = (0, 1, 2, 3, 4, 5)
TEXT_CLASSES = frozenset({1, 2})
IGNORE_CLASSES = frozenset({3, 4, 5})

#: The collapse the runner applies.  Written out rather than derived, because
#: this is the rule that scored the mask; :func:`collapse_verification` is what
#: makes it more than an assumption (FR-024b).
CLASS_TO_BINARY = {0: 0, 1: 255, 2: 255, 3: 0, 4: 0, 5: 0}

LABEL_COLLAPSE_RULE = "argmax over 6 classes; {1,2} -> 255, {0,3,4,5} -> 0"


def collapse_verification() -> dict[str, Any]:
    """FR-024b: verify the collapse, not assume it.

    Two properties, both checkable without ground truth — which never reaches a
    runner (FR-054a): every class in the method's label space is mapped, so no
    class is left unrepresented, and no ignore class is admitted as text.
    """
    unrepresented = sorted(set(LABEL_SPACE) - set(CLASS_TO_BINARY))
    admitted = sorted(cls for cls in IGNORE_CLASSES if CLASS_TO_BINARY.get(cls) == 255)
    return {
        "rule": LABEL_COLLAPSE_RULE,
        "verified": not unrepresented and not admitted,
        "unrepresented_classes": unrepresented,
        "ignore_classes_admitted_as_text": admitted,
    }


class MethodUnavailableError(RuntimeError):
    """Method A cannot run here, and the reason names what is missing."""


def derive_book_folds() -> dict[str, int]:
    """Upstream's book -> held-out fold map, from the published seed.

    The second element of each ``KFold`` tuple is the validation split, so a
    book's fold is the block of the shuffled index array that contains it.
    """
    order = np.arange(len(TRAIN_FOLDERS))
    np.random.RandomState(SEED).shuffle(order)

    per_fold = len(TRAIN_FOLDERS) // FOLD_COUNT
    return {
        TRAIN_FOLDERS[int(index)]: fold
        for fold in range(FOLD_COUNT)
        for index in order[fold * per_fold : (fold + 1) * per_fold]
    }


def fold_for_book(book: str) -> int | None:
    """The fold that held ``book`` out, or ``None`` if it is not a Manga109 book.

    ``None`` is the honest answer and the caller must treat it as inadmissible:
    FR-013a says an unmappable page makes Method A unavailable, not approximated.
    """
    return derive_book_folds().get(book)


def fold_for_image_id(image_id: str) -> int | None:
    """The fold that held out the book ``image_id`` belongs to."""
    return fold_for_book(str(image_id).split("/", 1)[0])


def _field(source: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a raw config dict or a parsed ``DLMethod`` alike."""
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


class MangaTextSegmentationAdapter(DLMethodAdapter):
    """One fold's checkpoint, and the evidence that it may be used at all."""

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        source = config or {}
        #: A configuration was supplied.  The frozen interface (T011)
        #: instantiates every registered adapter with no config at all, and that
        #: one must answer the interface tests; a configured adapter that cannot
        #: run must say so instead of answering with a mask (FR-018).
        self._configured = bool(source)
        self._fold: int | None = _field(source, "fold")
        self._checkpoints: list[dict[str, Any]] = [
            {
                "fold": _field(raw, "fold"),
                "identity": _field(raw, "identity", ""),
                "path": Path(_field(raw, "path", "")),
                "size_bytes": _field(raw, "size_bytes"),
            }
            for raw in (_field(source, "checkpoints") or [])
        ]

    @property
    def name(self) -> str:
        return "manga-text-segmentation"

    def _fold_checkpoint(self) -> dict[str, Any] | None:
        for checkpoint in self._checkpoints:
            if checkpoint["fold"] == self._fold:
                return checkpoint
        return None

    def unavailable_reason(self) -> str | None:
        """Why Method A cannot be run here, or ``None`` if it can (FR-016).

        Checked before any page is inferred, not discovered part-way through
        (FR-021), and every failure carries its remediation (FR-019).
        """
        if self._fold is None:
            return (
                "no fold configured: Method A scores each page with the "
                f"checkpoint of the fold that held its book out (FR-013a), so "
                f"one of {list(range(FOLD_COUNT))} must be selected per adapter"
            )
        if self._fold not in range(FOLD_COUNT):
            return f"fold {self._fold} is not one of the released folds 0-4"

        missing = sorted(
            fold
            for fold in range(FOLD_COUNT)
            if not any(c["fold"] == fold for c in self._checkpoints)
        )
        if missing:
            return (
                f"fold checkpoints {missing} are not configured; all "
                f"{FOLD_COUNT} released folds must be present before any page "
                f"is inferred (FR-013a)"
            )

        checkpoint = self._fold_checkpoint()
        path = checkpoint["path"]
        if not path.is_file():
            return f"checkpoint {checkpoint['identity']} is not at {path}"
        size = checkpoint["size_bytes"]
        if isinstance(size, int) and not isinstance(size, bool):
            observed = path.stat().st_size
            if observed != size:
                return (
                    f"checkpoint {checkpoint['identity']} is {observed} bytes, "
                    f"the configuration records {size}"
                )

        absent = [
            name for name in ("torch", "fastai") if importlib.util.find_spec(name) is None
        ]
        if absent:
            return (
                f"{' and '.join(absent)} not importable in this interpreter; "
                f"Method A runs in its own environment (FR-052)"
            )
        return None

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Refuse to answer when Method A cannot run here.

        The frozen interface (T011) requires a binary ``[H, W]`` answer at the
        input's shape, so an adapter constructed with *no configuration at all*
        returns the conservative one — *no text*.  A configured adapter that
        cannot run is a different thing: FR-018 forbids inventing a result for
        an unavailable method, and a zero mask is exactly such an invention, so
        it raises instead.  :meth:`infer` reports that as ``failed`` rather than
        letting it escape (FR-020).

        ponytail: no inference path is implemented.  Under FR-052 Method A runs
        in its own environment and returns masks; implementing a fastai-v1
        inference path that no environment here can execute would be untested
        code on the critical path.
        """
        if self._configured:
            reason = self.unavailable_reason()
            if reason is not None:
                raise MethodUnavailableError(reason)
        return np.zeros(image.shape[:2], dtype=np.uint8)

    def infer(self, image: "np.ndarray | str | Path") -> InferenceResult:
        """Report Method A unavailable rather than wrapping a zero mask as a result."""
        reason = self.unavailable_reason()
        if reason is None:
            return super().infer(image)

        try:
            array = self._as_array(image)
        except Exception as exc:  # noqa: BLE001 - reported, not raised (FR-010)
            return self._failure(None, exc)
        return self._failure(array.shape[:2], MethodUnavailableError(reason))

    def metadata(self) -> dict[str, Any]:
        """The configuration a reader needs to re-derive the mask (FR-024a/b).

        Declared here rather than read back from the produced mask: the mask is
        binary and has forgotten which multiple was padded away, which threshold
        scored it and which collapse produced it.  The runner writes these into
        its sidecar (FR-052); this surface is where the method says what they
        are, so an operator can compare the two.
        """
        checkpoint = self._fold_checkpoint()
        padding = _field(self._config, "padding") or {}
        return {
            **super().metadata(),
            "checkpoint": checkpoint["identity"] if checkpoint else None,
            "padding_multiple": _field(padding, "multiple", DEFAULT_PADDING_MULTIPLE),
            "effective_threshold": _field(
                self._config, "threshold", DEFAULT_THRESHOLD
            ),
            "preprocessing": {
                "resize": f"pad-to-multiple-{DEFAULT_PADDING_MULTIPLE}",
                "normalise": "imagenet",
            },
            "postprocessing": {"morphology": "none"},
            "label_collapse": collapse_verification(),
        }


register("manga-text-segmentation", MangaTextSegmentationAdapter)
