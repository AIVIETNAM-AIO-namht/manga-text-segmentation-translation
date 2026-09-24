"""Unit tests for the DL method configuration loader (T008, FR-036, research.md R11).

``configs/dl.json`` is a second, differently-shaped document: it describes the
three deep-learning methods, not the dataset.  It gets its own loader and its
own frozen dataclass, and ``configs/default.json`` keeps loading through
``load_config`` unchanged — that regression is asserted here too.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from manga_text_seg.config import ConfigError, load_config, load_dl_config

REPO_ROOT = Path(__file__).resolve().parents[2]
DL_CONFIG = REPO_ROOT / "configs" / "dl.json"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.json"

METHOD_NAMES = ["manga-text-segmentation", "comic-text-detector", "unetpp-efficientnetv2"]


@pytest.fixture()
def dl_document() -> dict:
    with open(DL_CONFIG, "r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture()
def write_dl(tmp_path: Path):
    """Write a perturbed copy of ``configs/dl.json`` and return its path."""

    def _write(mutate) -> Path:
        with open(DL_CONFIG, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        doc = mutate(doc)
        path = tmp_path / "dl.json"
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return path

    return _write


class TestValidDlConfigLoads:
    """The shipped ``configs/dl.json`` loads and exposes all three methods."""

    def test_all_three_methods_present(self) -> None:
        cfg = load_dl_config(DL_CONFIG)
        assert sorted(cfg.methods) == sorted(METHOD_NAMES)

    def test_repository_revision_is_a_full_sha(self) -> None:
        cfg = load_dl_config(DL_CONFIG)
        for name, method in cfg.methods.items():
            revision = method.repository["revision"]
            assert len(revision) == 40, f"{name}: revision is not a full SHA"
            assert all(c in "0123456789abcdef" for c in revision)

    def test_checkpoint_paths_resolve_against_the_repo_root(self) -> None:
        cfg = load_dl_config(DL_CONFIG, repo_root=REPO_ROOT)
        for method in cfg.methods.values():
            for ckpt in method.checkpoints:
                assert ckpt.path.is_absolute(), f"{ckpt.identity} did not resolve"
                assert str(ckpt.path).startswith(str(REPO_ROOT))

    def test_licences_are_carried_separately(self) -> None:
        cfg = load_dl_config(DL_CONFIG)
        a = cfg.methods["manga-text-segmentation"]
        b = cfg.methods["comic-text-detector"]
        c = cfg.methods["unetpp-efficientnetv2"]
        assert (a.code_license, a.weight_license) == ("MIT", "MIT")
        assert (b.code_license, b.weight_license) == ("GPL-3.0", "GPL-3.0")
        # No published licence is a recorded position, not a missing field.
        assert (c.code_license, c.weight_license) == ("none", "none")

    def test_padding_multiple_is_per_method(self) -> None:
        cfg = load_dl_config(DL_CONFIG)
        assert cfg.methods["manga-text-segmentation"].padding["multiple"] == 8
        assert cfg.methods["comic-text-detector"].padding["multiple"] == 64
        assert cfg.methods["unetpp-efficientnetv2"].padding["multiple"] == 32

    def test_method_a_carries_five_fold_checkpoints(self) -> None:
        cfg = load_dl_config(DL_CONFIG)
        folds = [c.fold for c in cfg.methods["manga-text-segmentation"].checkpoints]
        assert sorted(folds) == [0, 1, 2, 3, 4]

    def test_device_names_the_specific_accelerator(self) -> None:
        cfg = load_dl_config(DL_CONFIG)
        for method in cfg.methods.values():
            assert method.device["name"], "device.name is empty (FR-060)"


class TestDlConfigRefusals:
    """A malformed DL configuration fails fast with a clear ``ConfigError``."""

    def test_missing_methods_key(self, write_dl) -> None:
        path = write_dl(lambda doc: {k: v for k, v in doc.items() if k != "methods"})
        with pytest.raises(ConfigError, match="methods"):
            load_dl_config(path)

    def test_method_without_repository(self, write_dl) -> None:
        def mutate(doc):
            del doc["methods"]["comic-text-detector"]["repository"]
            return doc

        with pytest.raises(ConfigError, match="comic-text-detector"):
            load_dl_config(write_dl(mutate))

    def test_method_without_checkpoints(self, write_dl) -> None:
        def mutate(doc):
            del doc["methods"]["unetpp-efficientnetv2"]["checkpoints"]
            return doc

        with pytest.raises(ConfigError, match="unetpp-efficientnetv2"):
            load_dl_config(write_dl(mutate))

    def test_short_revision_is_refused(self, write_dl) -> None:
        def mutate(doc):
            doc["methods"]["comic-text-detector"]["repository"]["revision"] = "440b978"
            return doc

        with pytest.raises(ConfigError, match="revision"):
            load_dl_config(write_dl(mutate))

    def test_missing_code_license_is_refused(self, write_dl) -> None:
        def mutate(doc):
            del doc["methods"]["comic-text-detector"]["code_license"]
            return doc

        with pytest.raises(ConfigError, match="code_license"):
            load_dl_config(write_dl(mutate))

    def test_missing_weight_license_is_refused(self, write_dl) -> None:
        def mutate(doc):
            del doc["methods"]["manga-text-segmentation"]["weight_license"]
            return doc

        with pytest.raises(ConfigError, match="weight_license"):
            load_dl_config(write_dl(mutate))

    def test_checkpoint_without_sha256_key_is_refused(self, write_dl) -> None:
        def mutate(doc):
            del doc["methods"]["comic-text-detector"]["checkpoints"][0]["sha256"]
            return doc

        with pytest.raises(ConfigError, match="sha256"):
            load_dl_config(write_dl(mutate))

    def test_empty_methods_is_refused(self, write_dl) -> None:
        path = write_dl(lambda doc: {**doc, "methods": {}})
        with pytest.raises(ConfigError, match="methods"):
            load_dl_config(path)

    def test_missing_file_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_dl_config(tmp_path / "absent.json")


class TestNullSha256IsAllowed:
    """A null sha256 is the recorded absence of a published digest, not a gap.

    ``configs/dl.json``'s own ``sha256_note`` says so: four of Method A's five
    release assets have no published digest, and the runner records the value
    they observe in provenance instead.
    """

    def test_method_a_folds_one_to_four_have_null_sha(self, dl_document) -> None:
        ckpts = dl_document["methods"]["manga-text-segmentation"]["checkpoints"]
        assert ckpts[0]["sha256"], "fold 0's published digest should be present"
        assert all(c["sha256"] is None for c in ckpts[1:])

    def test_loader_accepts_null_sha(self) -> None:
        cfg = load_dl_config(DL_CONFIG)
        ckpts = cfg.methods["manga-text-segmentation"].checkpoints
        assert ckpts[0].sha256
        assert ckpts[1].sha256 is None


class TestDefaultConfigStillLoads:
    """The classical configuration is untouched (research.md R11)."""

    def test_default_config_loads_unchanged(self) -> None:
        cfg = load_config(DEFAULT_CONFIG, repo_root=REPO_ROOT)
        assert cfg.run_id == "default"
        assert len(cfg.methods) == 6
        assert cfg.alignment_strategy == "crop-topleft"
        assert cfg.metrics == ["iou", "precision", "recall", "f1"]

    def test_default_config_has_no_dl_methods(self) -> None:
        cfg = load_config(DEFAULT_CONFIG, repo_root=REPO_ROOT)
        names = {m["name"] for m in cfg.methods}
        assert names.isdisjoint(METHOD_NAMES)

    def test_load_config_still_refuses_a_missing_dataset(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"output_root": "o", "run_id": "r"}), encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(path)

    def test_load_config_does_not_accept_the_dl_document(self) -> None:
        with pytest.raises(ConfigError):
            load_config(DL_CONFIG)
