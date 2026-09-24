"""Method C's padding, its recorded settings, and its checkpoint-load evidence.

T040's check.  The three things this adapter could get wrong and still look
right are the ones tested here: a padding multiple collapsed into the ground
truth's 8 (FR-024a), an optional inference setting applied silently (FR-027,
FR-044), and a checkpoint load inferred from a plausible mask instead of proved
from the state that was installed (FR-018a, FR-018b).
"""
from __future__ import annotations

import numpy as np
import pytest

from manga_text_seg.adapters import create
from manga_text_seg.adapters.manga_text_segmentation import MethodUnavailableError
from manga_text_seg.adapters.unetpp_efficientnetv2 import (
    PADDING_MULTIPLE,
    UnetPlusPlusEfficientNetV2Adapter,
    checkpoint_load_evidence,
    invert_padding,
    pad_to_multiple,
    padding_geometry,
    state_dict_identity,
)

#: The aligned page size.  The whole point of FR-024a is that this is a multiple
#: of the ground truth's 8 and *not* of Method C's 32.
ALIGNED = (1656, 1176)

#: Method C's configured checkpoint, as configs/dl.json records it.
CONFIGURED_SHA256 = "a0a895dc385608554a81ca765b0c62d654462c2bac661abdff63634985ab37fc"


def _config() -> dict:
    """Method C's configuration, shaped as ``config.py`` parses it."""
    return {
        "checkpoints": [
            {
                "identity": "model.pth",
                "path": "checkpoints/unetpp-efficientnetv2/model.pth",
                "size_bytes": 216911417,
                "sha256": CONFIGURED_SHA256,
            }
        ],
        "threshold": 0.5,
        "padding": {"mode": "pad-to-multiple", "multiple": 32},
        "channel_order": "rgb",
    }


class _Tensor:
    """A stand-in for a tensor: the evidence only ever reads ``.shape``."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


def test_method_c_padding_multiple_is_its_own_not_the_ground_truths() -> None:
    """FR-024a: 8 is the ground truth's, 32 is this method's, and they differ."""
    assert PADDING_MULTIPLE == 32
    height, width = ALIGNED
    assert height % 8 == 0 and width % 8 == 0, "the aligned page is a multiple of 8"
    assert height % PADDING_MULTIPLE or width % PADDING_MULTIPLE, (
        "the aligned page must NOT be a multiple of 32, or this padding would be "
        "indistinguishable from the ground truth's and FR-024a would be untestable"
    )


def test_padding_geometry_rounds_up_and_records_both_sizes() -> None:
    geometry = padding_geometry(*ALIGNED)
    assert geometry["padded_size"] == (1664, 1184)
    assert geometry["pad"] == (8, 8)
    assert geometry["source_size"] == ALIGNED
    assert geometry["padding_multiple"] == PADDING_MULTIPLE


def test_padding_is_inverted_to_the_exact_aligned_page() -> None:
    """FR-024, FR-028: the mask comes back at the page's own size, unresized."""
    page = np.full(ALIGNED, 255, dtype=np.uint8)
    geometry = padding_geometry(*ALIGNED)

    padded = pad_to_multiple(page, geometry)
    assert padded.shape == geometry["padded_size"]
    # Only zeros were added, bottom-right: no page pixel was overwritten.
    assert np.array_equal(padded[: ALIGNED[0], : ALIGNED[1]], page)
    assert not padded[ALIGNED[0] :, :].any()
    assert not padded[:, ALIGNED[1] :].any()

    restored = invert_padding(padded, geometry)
    assert restored.shape == ALIGNED
    assert np.array_equal(restored, page)


def test_a_page_already_on_the_stride_is_left_alone() -> None:
    geometry = padding_geometry(64, 32)
    assert geometry["pad"] == (0, 0)
    assert geometry["padded_size"] == (64, 32)


def test_an_empty_page_size_is_refused() -> None:
    with pytest.raises(ValueError, match="positive page size"):
        padding_geometry(0, 1176)


def test_state_dict_identity_separates_key_sets() -> None:
    """The evidence is about which tensors were installed, not their values."""
    first = state_dict_identity({"a.weight": _Tensor((3, 3)), "b.bias": _Tensor((8,))})
    second = state_dict_identity({"a.weight": _Tensor((3, 3)), "c.bias": _Tensor((8,))})
    assert first["keys"] == ["a.weight", "b.bias"]
    assert first["key_count"] == 2
    assert first["key_digest"] != second["key_digest"]
    # The shape is part of the identity: a same-named tensor of another shape is
    # not the published one.
    third = state_dict_identity({"a.weight": _Tensor((5, 5)), "b.bias": _Tensor((8,))})
    assert third["key_digest"] != first["key_digest"]


def test_evidence_is_positive_only_when_a_state_was_actually_installed() -> None:
    """FR-018a: the shape of the evidence the schema names, and nothing weaker."""
    checkpoint = {"identity": "model.pth"}

    loaded = state_dict_identity({"decoder.conv.weight": _Tensor((1, 1, 3, 3))})
    evidenced = checkpoint_load_evidence({"decoder.conv.weight": _Tensor((1, 1, 3, 3))}, checkpoint)
    assert evidenced["evidenced"] is True
    assert loaded["key_digest"] in evidenced["detail"]
    assert evidenced["method"] == "state_dict key set plus checksum comparison"

    # Upstream's loader swallows the missing-file error and carries on with a
    # randomly initialised decoder, so "no state" is the failure this exists for.
    missing = checkpoint_load_evidence(None, checkpoint)
    assert missing["evidenced"] is False
    assert "not evidence" in missing["detail"]

    empty = checkpoint_load_evidence({}, checkpoint)
    assert empty["evidenced"] is False
    assert "untrained decoder" in empty["detail"]


def test_evidence_refuses_a_state_from_a_different_artefact() -> None:
    """A plausible-looking load of the wrong file is still not the pinned one."""
    checkpoint = {"identity": "model.pth"}
    state = {"decoder.conv.weight": _Tensor((1, 1, 3, 3))}

    mismatched = checkpoint_load_evidence(
        state,
        checkpoint,
        observed_sha256="0" * 64,
        configured_sha256=CONFIGURED_SHA256,
    )
    assert mismatched["evidenced"] is False
    assert CONFIGURED_SHA256 in mismatched["detail"]

    matched = checkpoint_load_evidence(
        state,
        checkpoint,
        observed_sha256=CONFIGURED_SHA256,
        configured_sha256=CONFIGURED_SHA256,
    )
    assert matched["evidenced"] is True
    assert CONFIGURED_SHA256 in matched["detail"]


def test_the_configured_adapter_reports_method_c_unavailable_here() -> None:
    """The checkpoint is absent on this machine, so Method C cannot run (FR-018)."""
    adapter = create("unetpp-efficientnetv2", _config())
    reason = adapter.unavailable_reason()
    assert reason and "model.pth" in reason

    with pytest.raises(MethodUnavailableError):
        adapter.segment(np.zeros((64, 64), dtype=np.uint8))

    result = adapter.infer(np.zeros((64, 64), dtype=np.uint8))
    assert result.status == "failed"
    assert result.mask is None
    assert "model.pth" in result.error


def test_an_unconfigured_adapter_answers_the_frozen_interface() -> None:
    """The interface suite instantiates every adapter with no config at all."""
    adapter = create("unetpp-efficientnetv2")
    mask = adapter.segment(np.full((40, 24), 7, dtype=np.uint8))
    assert mask.shape == (40, 24)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 255}


def test_metadata_records_the_geometry_and_the_optional_settings() -> None:
    """FR-024a, FR-025, FR-027: the reader gets the rule, not just a number."""
    adapter = create("unetpp-efficientnetv2", _config())
    metadata = adapter.metadata()

    assert metadata["padding_multiple"] == 32
    assert metadata["preprocessing"]["resize"] == "pad-to-multiple-32"
    assert metadata["preprocessing"]["normalise"] == "imagenet"
    assert metadata["postprocessing"]["morphology"] == "none"
    # FR-044: flip TTA and mixed precision change the result or the timing, so
    # both are recorded rather than left to be guessed.
    assert metadata["preprocessing"]["flip_tta"] is False
    assert metadata["preprocessing"]["mixed_precision"] is False
    assert metadata["threshold"]["configured"] == 0.5
    # FR-018b: the record carries the evidence field, and here it is not positive
    # — this process never installed the weights.
    assert metadata["checkpoint_load_evidence"]["evidenced"] is False


def test_the_adapter_is_registered_under_its_config_name() -> None:
    from manga_text_seg.adapters import available

    assert "unetpp-efficientnetv2" in available()
    assert isinstance(create("unetpp-efficientnetv2"), UnetPlusPlusEfficientNetV2Adapter)
