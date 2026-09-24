"""Method A's leave-one-fold-out attribution (T025, FR-013a, research.md R4).

Every book in Manga109 was held out from exactly one of the five folds the
upstream project released checkpoints for, so for any page at most one released
checkpoint is uncontaminated by it — and which one follows from the published
seed 42.  Using one checkpoint for all 390 pages is not a weaker result but an
invalid one.

The split is reproduced here without scikit-learn: it is installed on this
machine but is not a declared project dependency, and the numpy formulation
reproduces ``KFold(n_splits=5, shuffle=True, random_state=42)`` exactly.

No checkpoint, no framework, no network (FR-050, FR-051).
"""
from __future__ import annotations

import numpy as np
import pytest

from manga_text_seg.adapters import create
from manga_text_seg.adapters.manga_text_segmentation import (
    FOLD_COUNT,
    SEED,
    TRAIN_FOLDERS,
    MethodUnavailableError,
    derive_book_folds,
    fold_for_book,
    fold_for_image_id,
)

#: The map upstream's split produces, read off the derivation and pinned here
#: so a change in the literal or the seed is a failure rather than a silent
#: re-attribution of 390 pages.
EXPECTED_FOLDS = {
    0: {
        "AosugiruHaru", "BakuretsuKungFuGirl", "ByebyeC-BOY",
        "HanzaiKousyouninMinegishiEitarou", "HaruichibanNoFukukoro",
        "TotteokiNoABC", "TouyouKidan", "YasasiiAkuma", "YumeNoKayoiji",
    },
    1: {
        "Akuhamu", "Arisa", "Count3DeKimeteAgeru", "Donburakokko",
        "EienNoWith", "EverydayOsakanaChan", "Hamlet", "ToutaMairimasu",
        "UnbalanceTokyo",
    },
    2: {
        "ARMS", "AppareKappore", "Belmondo", "DualJustice", "HarukaRefrain",
        "UchiNoNyan'sDiary", "UchuKigekiM774", "UltraEleven", "YoumaKourin",
    },
    3: {
        "AisazuNihaIrarenai", "AkkeraKanjinchou", "BurariTessenTorimonocho",
        "GakuenNoise", "GinNoChimera", "WarewareHaOniDearu", "YamatoNoHane",
        "YouchienBoueigumi", "YumeiroCooking",
    },
    4: {
        "BEMADER_P", "BokuHaSitatakaKun", "DollGun", "EvaLady",
        "GOOD_KISS_Ver2", "GarakutayaManta", "HealingPlanet",
        "TsubasaNoKioku", "YukiNoFuruMachi",
    },
}


# -- The derivation ----------------------------------------------------------


def test_the_map_covers_every_book_exactly_once() -> None:
    """Each page is scored by exactly one checkpoint (FR-013a)."""
    folds = derive_book_folds()

    assert len(folds) == len(TRAIN_FOLDERS) == 45
    assert set(folds) == set(TRAIN_FOLDERS)
    assert set(folds.values()) == set(range(FOLD_COUNT))


def test_each_fold_holds_out_nine_books() -> None:
    folds = derive_book_folds()
    sizes = sorted(list(folds.values()).count(f) for f in range(FOLD_COUNT))

    assert sizes == [9] * FOLD_COUNT


def test_the_map_is_the_one_upstreams_split_produces() -> None:
    folds = derive_book_folds()

    for fold, books in EXPECTED_FOLDS.items():
        assert {b for b, f in folds.items() if f == fold} == books


def test_the_published_seed_is_what_is_used() -> None:
    """Seed 42 and a shuffle, not a different seed that happens to look right."""
    assert SEED == 42

    other = np.arange(len(TRAIN_FOLDERS))
    np.random.RandomState(SEED + 1).shuffle(other)
    folds = derive_book_folds()
    by_other_seed = {
        TRAIN_FOLDERS[int(i)]: f
        for f in range(FOLD_COUNT)
        for i in other[f * 9:(f + 1) * 9]
    }
    assert by_other_seed != folds


def test_the_literals_order_is_load_bearing() -> None:
    """Upstream indexes its hardcoded list, so the list's order is the split.

    research.md R4 describes the books as "sorted"; upstream's source does not
    sort them.  Four ``T`` names sit after every ``Y`` name in the literal, and
    sorting them would move 13 of the 45 books to a different fold — silently
    re-attributing their pages to a checkpoint that trained on them.
    """
    folds = derive_book_folds()

    # These four are the ones a sort would move.
    assert folds["TotteokiNoABC"] == 0
    assert folds["ToutaMairimasu"] == 1
    assert folds["TouyouKidan"] == 0
    assert folds["TsubasaNoKioku"] == 4

    sorted_order = sorted(TRAIN_FOLDERS)
    assert sorted_order != TRAIN_FOLDERS, "the literal is not in sorted order"

    indexes = np.arange(len(sorted_order))
    np.random.RandomState(SEED).shuffle(indexes)
    by_sorted = {
        sorted_order[int(i)]: f
        for f in range(FOLD_COUNT)
        for i in indexes[f * 9:(f + 1) * 9]
    }
    assert sum(1 for book in TRAIN_FOLDERS if by_sorted[book] != folds[book]) == 13


def test_the_derivation_is_reproducible() -> None:
    """FR-047: re-running yields the same map, so the same pages are scored."""
    assert derive_book_folds() == derive_book_folds()


# -- Looking a page up -------------------------------------------------------


def test_a_page_resolves_to_its_books_fold() -> None:
    assert fold_for_image_id("ARMS/000") == 2
    assert fold_for_image_id("BEMADER_P/003") == 4
    assert fold_for_book("Akuhamu") == 1


def test_an_unknown_book_is_unmappable_rather_than_approximated() -> None:
    """FR-013a: unmappable means inadmissible, not nearest-match."""
    assert fold_for_book("NotAManga109Book") is None
    assert fold_for_image_id("NotAManga109Book/000") is None


# -- The adapter -------------------------------------------------------------


def _checkpoints() -> list[dict]:
    """Five fold checkpoints as ``configs/dl.json`` spells them."""
    return [
        {
            "identity": f"fold.{fold}.-.final.refined.model.2.pkl",
            "path": f"checkpoints/manga-text-segmentation/fold.{fold}.-.final.refined.model.2.pkl",
            "size_bytes": 344290372,
            "sha256": None,
            "fold": fold,
        }
        for fold in range(FOLD_COUNT)
    ]


def test_the_adapter_is_registered_under_its_registry_name() -> None:
    adapter = create("manga-text-segmentation", {"checkpoints": _checkpoints(), "fold": 2})

    assert adapter.name == "manga-text-segmentation"
    assert adapter.metadata()["checkpoint"] == "fold.2.-.final.refined.model.2.pkl"


def test_all_five_fold_checkpoints_are_required_before_any_page_is_inferred() -> None:
    """FR-013a: verified present up front, not discovered at inference time.

    Only fold 0 is configured, so the reason can name the folds that are absent
    but not their filenames — an adapter cannot describe a checkpoint it was
    never told about, and inventing one would be worse than saying which folds
    are missing.
    """
    adapter = create("manga-text-segmentation", {"checkpoints": _checkpoints()[:1], "fold": 0})

    with pytest.raises(MethodUnavailableError) as caught:
        adapter.segment(np.zeros((32, 64), np.uint8))

    reason = str(caught.value)
    assert "1, 2, 3, 4" in reason, "the reason names which folds are missing"
    assert "FR-013a" in reason


def test_a_missing_checkpoint_is_reported_not_run() -> None:
    """FR-018: an unavailable method is not run and does not abort the benchmark.

    FR-018a in particular: producing a plausible mask from an unloaded model is
    the failure this refuses, so there is no fallback that invents one.
    """
    adapter = create("manga-text-segmentation", {"checkpoints": _checkpoints(), "fold": 2})

    result = adapter.infer(np.zeros((32, 64), np.uint8))

    assert result.status == "failed"
    assert result.mask is None
    assert "fold.2.-.final.refined.model.2.pkl" in (result.error or "")


def test_the_adapter_refuses_a_configuration_with_no_fold() -> None:
    """Which checkpoint scored a page is not something to leave to a default."""
    with pytest.raises(MethodUnavailableError):
        create("manga-text-segmentation", {"checkpoints": _checkpoints()}).segment(
            np.zeros((32, 64), np.uint8)
        )
