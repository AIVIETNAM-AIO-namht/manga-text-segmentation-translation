"""Contract tests for the deep-learning adapter interface (T011).

Normative for FR-008-FR-011, FR-050, FR-051.  A deep-learning adapter *is* a
segmentation method: it satisfies Spec 1's frozen ``segment(image) -> mask``
interface (``contracts/method-interface.md``) so the metric and reporting
modules accept it unmodified, and it adds the four things FR-010 requires on
top of the mask — input size, inference time, model metadata and an error
status.

Every test here runs against a registered adapter with no checkpoint, no
framework and no network access (FR-051).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from manga_text_seg.adapters import (
    DLMethodAdapter,
    UnknownAdapterError,
    available,
    create,
    register,
    _REGISTRY,
)

# --- Contract constants -----------------------------------------------------

ALL_VALUES_BINARY = {0, 255}

#: Aligned page size as the page-list schema records it: ``[width, height]``.
ALIGNED_SIZE = [1654, 1170]
#: The same size as an ``np.ndarray`` shape is ``(height, width)``.
ALIGNED_SHAPE = (1170, 1654)

#: Metadata keys every adapter must carry for its sidecar (FR-010, FR-011).
REQUIRED_METADATA_KEYS = {"name", "device", "checkpoint"}


def _make_test_image(h: int = 64, w: int = 64) -> np.ndarray:
    """A synthetic grayscale uint8 image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(h, w), dtype=np.uint8)


# --- Registry ---------------------------------------------------------------

class TestAdapterRegistry:
    """register/create/available mirror the classical registry (FR-035, FR-008)."""

    def test_available_returns_sorted_list(self) -> None:
        names = available()
        assert isinstance(names, list)
        assert names == sorted(names)
        assert "standin" in names, "the synthetic adapter must be registered"

    def test_create_known_adapter(self) -> None:
        adapter = create("standin", {})
        assert isinstance(adapter, DLMethodAdapter)
        assert adapter.name == "standin"

    def test_create_unknown_adapter_raises(self) -> None:
        with pytest.raises(UnknownAdapterError):
            create("nonexistent_adapter_xyz")

    def test_create_accepts_no_config(self) -> None:
        adapter = create("standin")
        assert isinstance(adapter, DLMethodAdapter)

    def test_register_adds_to_available(self) -> None:
        class DummyAdapter(DLMethodAdapter):
            @property
            def name(self) -> str:
                return "__test_dummy_adapter__"

            def segment(self, image: np.ndarray) -> np.ndarray:
                return np.zeros(image.shape, dtype=np.uint8)

        register("__test_dummy_adapter__", DummyAdapter)
        assert "__test_dummy_adapter__" in available()
        assert create("__test_dummy_adapter__").name == "__test_dummy_adapter__"

        _REGISTRY.pop("__test_dummy_adapter__", None)


# --- Interface --------------------------------------------------------------

class TestAdapterInterface:
    """Every registered adapter satisfies the frozen segmentation interface."""

    @pytest.fixture(params=sorted(_REGISTRY.keys()))
    def adapter(self, request) -> DLMethodAdapter:
        return create(request.param)

    def test_has_name_property(self, adapter: DLMethodAdapter) -> None:
        assert isinstance(adapter.name, str)
        assert adapter.name

    def test_segment_is_the_frozen_signature(self, adapter: DLMethodAdapter) -> None:
        """Rule 3: nothing but the image is configurable per call."""
        import inspect

        params = list(inspect.signature(adapter.segment).parameters)
        assert params == ["image"], f"segment() should accept only 'image', got {params}"

    def test_segment_returns_single_channel_uint8(self, adapter: DLMethodAdapter) -> None:
        result = adapter.segment(_make_test_image())
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2, f"expected single channel, got {result.ndim}D"
        assert result.dtype == np.uint8, f"expected uint8, got {result.dtype}"

    def test_segment_output_values_are_binary(self, adapter: DLMethodAdapter) -> None:
        result = adapter.segment(_make_test_image())
        values = set(np.unique(result).tolist())
        assert values.issubset(ALL_VALUES_BINARY), (
            f"adapter '{adapter.name}' returned non-binary values: {sorted(values)}"
        )

    def test_segment_never_resizes_its_input(self, adapter: DLMethodAdapter) -> None:
        """Rule 4: alignment happens after inference, never inside it."""
        for shape in [(32, 64), (100, 200), ALIGNED_SHAPE]:
            image = _make_test_image(*shape)
            assert adapter.segment(image).shape == shape

    def test_aligned_size_page_yields_an_aligned_size_mask(
        self, adapter: DLMethodAdapter
    ) -> None:
        """The shape the page list fixes is the shape the mask comes back at."""
        mask = adapter.segment(np.zeros(ALIGNED_SHAPE, dtype=np.uint8))
        assert mask.shape == ALIGNED_SHAPE
        assert [mask.shape[1], mask.shape[0]] == ALIGNED_SIZE


# --- Inference record (FR-010) ---------------------------------------------

class TestInferenceResult:
    """``infer`` returns the mask plus the record FR-010 requires."""

    def test_result_carries_mask_size_time_metadata_and_status(self) -> None:
        adapter = create("standin", {})
        image = _make_test_image(*ALIGNED_SHAPE)
        result = adapter.infer(image)

        assert result.status == "ok"
        assert result.error is None
        assert result.mask is not None and result.mask.shape == ALIGNED_SHAPE
        assert result.input_size == ALIGNED_SHAPE
        assert isinstance(result.inference_time_seconds, float)
        assert result.inference_time_seconds >= 0.0
        assert isinstance(result.metadata, dict)

    def test_result_mask_matches_segment(self) -> None:
        """``infer`` wraps ``segment``; it does not run a second code path."""
        adapter = create("standin", {})
        image = _make_test_image(64, 64)
        assert np.array_equal(adapter.infer(image).mask, adapter.segment(image))

    def test_metadata_names_the_method_and_its_device(self) -> None:
        adapter = create("standin", {"device": {"type": "cpu", "name": "fixture-cpu"}})
        metadata = adapter.metadata()
        assert REQUIRED_METADATA_KEYS.issubset(metadata)
        assert metadata["name"] == adapter.name
        assert metadata["device"]["name"] == "fixture-cpu"

    def test_metadata_is_serialisable_for_the_sidecar(self) -> None:
        """Metadata reaches the sidecar, so it must survive JSON."""
        metadata = create("standin", {}).metadata()
        assert json.loads(json.dumps(metadata)) == metadata

    def test_inference_failure_is_reported_not_raised(self) -> None:
        """FR-010: an error status, so one bad page cannot abort a run."""

        class FailingAdapter(DLMethodAdapter):
            @property
            def name(self) -> str:
                return "__test_failing__"

            def segment(self, image: np.ndarray) -> np.ndarray:
                raise RuntimeError("simulated inference failure")

        result = FailingAdapter({}).infer(_make_test_image())
        assert result.status == "failed"
        assert result.mask is None
        assert "simulated inference failure" in (result.error or "")


# --- No checkpoint, no download (FR-051) -----------------------------------

class TestNoCheckpointRequired:
    """The default suite never loads a checkpoint or touches the network."""

    def test_standin_runs_without_a_checkpoint(self) -> None:
        adapter = create("standin", {})
        assert adapter.metadata()["checkpoint"] is None
        assert adapter.infer(_make_test_image()).status == "ok"

    def test_adapter_accepts_an_image_path(self) -> None:
        """FR-009: an image reference is a path or an in-memory image."""
        from manga_text_seg.imaging import save_mask

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "page.png"
            save_mask(path, _make_test_image(32, 48))
            result = create("standin", {}).infer(path)
            assert result.status == "ok"
            assert result.input_size == (32, 48)
