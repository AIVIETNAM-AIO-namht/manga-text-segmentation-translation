"""Unit tests for load_inpaint_config() (T004, FR-038).

Mirrors test_config_dl.py's shape: happy-path tests load the real, committed
configs/inpainting.json; refusal tests load a perturbed copy via write_inpaint.
Written before the implementation exists — must fail (ImportError) until T006.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from manga_text_seg.config import ConfigError, load_inpaint_config

REPO_ROOT = Path(__file__).resolve().parents[2]
INPAINT_CONFIG = REPO_ROOT / "configs" / "inpainting.json"

FOUR_IDENTITIES = {
    "classical_baseline",
    "manga_text_segmentation",
    "comic_text_detector",
    "unetpp_efficientnetv2",
}


@pytest.fixture()
def write_inpaint(tmp_path: Path):
    """Write a perturbed copy of configs/inpainting.json and return its path."""

    def _write(mutate) -> Path:
        with open(INPAINT_CONFIG, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        doc = mutate(doc)
        path = tmp_path / "inpainting.json"
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return path

    return _write


class TestValidInpaintConfigLoads:
    """The shipped configs/inpainting.json loads and exposes all four identities."""

    def test_all_four_identities_present(self) -> None:
        cfg = load_inpaint_config(INPAINT_CONFIG, repo_root=REPO_ROOT)
        assert set(cfg.methods) == FOUR_IDENTITIES

    def test_source_roots_resolve_against_repo_root(self) -> None:
        cfg = load_inpaint_config(INPAINT_CONFIG, repo_root=REPO_ROOT)
        for identity, method in cfg.methods.items():
            assert method.source_root.is_absolute(), f"{identity} did not resolve"
            assert str(method.source_root).startswith(str(REPO_ROOT))

    def test_classical_baseline_resolves_to_adaptive(self) -> None:
        cfg = load_inpaint_config(INPAINT_CONFIG, repo_root=REPO_ROOT)
        expected = (
            REPO_ROOT / "outputs" / "segmentation" / "default" / "adaptive"
        ).resolve()
        assert cfg.methods["classical_baseline"].source_root == expected

    def test_mask_processing_config(self) -> None:
        cfg = load_inpaint_config(INPAINT_CONFIG, repo_root=REPO_ROOT)
        dilation = cfg.mask_processing.dilation
        assert dilation.enabled is True
        assert dilation.kernel_shape == "ellipse"
        assert dilation.kernel_size == (3, 3)
        assert dilation.iterations == 1

    def test_inpainting_config(self) -> None:
        cfg = load_inpaint_config(INPAINT_CONFIG, repo_root=REPO_ROOT)
        assert cfg.inpainting.radius == 3
        assert cfg.inpainting.algorithms == ("telea", "ns")
        assert cfg.inpainting.output_format == "png"

    def test_selection_n(self) -> None:
        cfg = load_inpaint_config(INPAINT_CONFIG, repo_root=REPO_ROOT)
        assert cfg.selection_n == 5


class TestInpaintConfigRefusals:
    """A malformed inpainting configuration fails fast with a clear ConfigError."""

    @pytest.mark.parametrize(
        "missing_key",
        ["methods", "mask_processing", "inpaint", "selection", "output_root"],
    )
    def test_missing_top_level_key(self, write_inpaint, missing_key: str) -> None:
        path = write_inpaint(
            lambda doc: {k: v for k, v in doc.items() if k != missing_key}
        )
        with pytest.raises(ConfigError, match=missing_key):
            load_inpaint_config(path)

    def test_missing_identity(self, write_inpaint) -> None:
        def mutate(doc):
            del doc["methods"]["comic_text_detector"]
            return doc

        with pytest.raises(ConfigError, match="comic_text_detector"):
            load_inpaint_config(write_inpaint(mutate))

    def test_missing_source_root(self, write_inpaint) -> None:
        def mutate(doc):
            del doc["methods"]["classical_baseline"]["source_root"]
            return doc

        with pytest.raises(ConfigError, match="source_root"):
            load_inpaint_config(write_inpaint(mutate))

    def test_missing_file_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_inpaint_config(tmp_path / "absent.json")
