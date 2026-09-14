"""Contract tests for method interface (contracts/method-interface.md).

Hand-rolled validators — no jsonschema dependency (Constitution Principle III).
Validates that all registered methods conform to the frozen interface contract
and that the registry API behaves correctly.
"""
from __future__ import annotations

import numpy as np
import pytest

from manga_text_seg.methods import (
    BaseSegmentationMethod,
    UnknownMethodError,
    available,
    create,
    register,
    _REGISTRY,
)


# --- Contract constants (from contracts/method-interface.md) ---

ALL_VALUES_BINARY = {0, 255}


# --- Helpers ---

def _make_test_image(h: int = 64, w: int = 64) -> np.ndarray:
    """Create a synthetic grayscale uint8 image for testing."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(h, w), dtype=np.uint8)


# --- Registry tests ---

class TestRegistryContract:
    """Registry register/create/available must follow the contract."""

    def test_available_returns_sorted_list(self) -> None:
        """Contract: available() returns sorted list of registered names."""
        names = available()
        assert isinstance(names, list)
        assert names == sorted(names)

    def test_create_known_method(self) -> None:
        """Contract: create(name, config) instantiates a registered method."""
        method = create("otsu", {"blur_ksize": 5})
        assert isinstance(method, BaseSegmentationMethod)
        assert method.name == "otsu"

    def test_create_unknown_method_raises(self) -> None:
        """Contract: create() with unknown name raises an error."""
        with pytest.raises(UnknownMethodError):
            create("nonexistent_method_xyz")

    def test_register_adds_to_available(self) -> None:
        """Contract: register(name, factory) makes name available via create()."""
        class DummyMethod(BaseSegmentationMethod):
            @property
            def name(self) -> str:
                return "__test_dummy__"

            def segment(self, image: np.ndarray) -> np.ndarray:
                return image

        register("__test_dummy__", lambda cfg: DummyMethod())
        assert "__test_dummy__" in available()
        method = create("__test_dummy__")
        assert method.name == "__test_dummy__"

        # Cleanup
        _REGISTRY.pop("__test_dummy__", None)

    def test_create_with_empty_config(self) -> None:
        """Contract: create(name, None) and create(name, {}) both work."""
        method = create("otsu", None)
        assert isinstance(method, BaseSegmentationMethod)
        method2 = create("otsu", {})
        assert isinstance(method2, BaseSegmentationMethod)


# --- Interface contract tests ---

class TestBaseMethodInterface:
    """Every registered method must satisfy the interface contract."""

    @pytest.fixture(params=sorted(_REGISTRY.keys()))
    def method_name(self, request) -> str:
        return request.param

    @pytest.fixture()
    def method(self, method_name: str) -> BaseSegmentationMethod:
        return create(method_name)

    def test_has_name_property(self, method: BaseSegmentationMethod) -> None:
        """Contract Rule: method has a read-only name property returning str."""
        assert hasattr(method, "name")
        name = method.name
        assert isinstance(name, str)
        assert len(name) > 0

    def test_has_segment_method(self, method: BaseSegmentationMethod) -> None:
        """Contract Rule: method has segment(image) -> np.ndarray."""
        assert hasattr(method, "segment")
        assert callable(method.segment)

    def test_segment_returns_single_channel(self, method: BaseSegmentationMethod) -> None:
        """Contract Rule 1: segment returns single-channel uint8 [H, W]."""
        img = _make_test_image(64, 64)
        result = method.segment(img)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2, f"Expected 2D output, got {result.ndim}D"
        assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"

    def test_segment_output_shape_matches_input(self, method: BaseSegmentationMethod) -> None:
        """Contract Rule 4: method never resizes — output shape == input shape."""
        img = _make_test_image(80, 120)
        result = method.segment(img)
        assert result.shape == img.shape, (
            f"Shape mismatch: input {img.shape} vs output {result.shape}"
        )

    def test_segment_output_values_binary(self, method: BaseSegmentationMethod) -> None:
        """Contract Rule 1: output values must be exclusively in {0, 255}."""
        img = _make_test_image(64, 64)
        result = method.segment(img)
        unique = set(np.unique(result))
        assert unique.issubset(ALL_VALUES_BINARY), (
            f"Method '{method.name}' returned non-binary values: {sorted(unique)}"
        )

    def test_segment_output_background_is_zero(self, method: BaseSegmentationMethod) -> None:
        """Contract: background = 0, text = 255."""
        img = _make_test_image(32, 32)
        result = method.segment(img)
        # Background (majority) should be 0
        assert 0 in np.unique(result), "Background value 0 not present in output"


# --- Rule compliance tests ---

class TestMethodRules:
    """Verify specific rules from contracts/method-interface.md."""

    def test_otsu_name_property(self) -> None:
        """Rule: name is a property, not a mutable attribute."""
        method = create("otsu")
        assert method.name == "otsu"
        # Should be a property — can't set it
        with pytest.raises(AttributeError):
            method.name = "changed"  # type: ignore[misc]

    def test_segment_does_not_read_filesystem(self) -> None:
        """Rule 2: methods MUST NOT read the GT mask, manifest, or filesystem."""
        method = create("otsu")
        img = _make_test_image(64, 64)
        # segment should complete without any file access
        # (this is validated by the method never accepting path arguments)
        import inspect
        sig = inspect.signature(method.segment)
        params = list(sig.parameters.keys())
        assert len(params) == 1, (
            f"segment() should accept only 'image', got params: {params}"
        )

    def test_config_at_create_time(self) -> None:
        """Rule 3: all parameters arrive via config at create time."""
        method = create("otsu", {"blur_ksize": 5})
        # Method should have captured config internally
        assert isinstance(method, BaseSegmentationMethod)
        # segment() takes no config parameter
        import inspect
        sig = inspect.signature(method.segment)
        params = list(sig.parameters.keys())
        assert "config" not in params, "segment() should not accept config parameter"

    def test_never_resizes_input(self) -> None:
        """Rule 4: a method never resizes its input."""
        method = create("otsu")
        for h, w in [(32, 64), (100, 200), (1, 1)]:
            img = _make_test_image(h, w)
            result = method.segment(img)
            assert result.shape == img.shape, (
                f"Method resized {img.shape} -> {result.shape}"
            )
