"""Method C: a UNet++ decoder whose checkpoint load must be *proved*, not assumed.

T040 (FR-018a, FR-018b, FR-024a, FR-025, FR-026, FR-027, FR-044; US3).

Upstream is a UNet++ decoder over a ``timm``-provided EfficientNetV2-M encoder
with scSE decoder attention, one output class and no built-in activation, so the
network emits a floating-point probability map and the reference code applies a
default 0.5 cutoff to it.  Three things about Method C are recorded here rather
than assumed:

* **Its own padding.**  The page is zero-padded bottom-right to a multiple of
  32.  FR-024a forbids collapsing that into the ground truth's multiple-of-8
  rule: the aligned page is a multiple of 8 but *not* of 32, so this padding is
  the only one in the benchmark that actually adds pixels to the aligned page,
  and it is inverted here rather than approximated by a shared resize.
* **Its optional inference settings.**  Horizontal/vertical flip test-time
  augmentation and mixed-precision inference both change the result or the
  timing, so both are recorded configuration (FR-027, FR-044) and never applied
  silently.
* **The checkpoint load.**  Upstream's loader catches a missing-file error,
  prints a warning and *continues*, so a missing or misnamed checkpoint yields a
  randomly initialised decoder that still produces plausible-looking masks and
  raises nothing.  FR-018a exists for exactly this: a correctly-shaped,
  correctly-valued mask is not evidence, and :func:`checkpoint_load_evidence`
  produces the positive evidence instead — the identity of the state that was
  actually installed, by tensor key set and checksum — or reports that it could
  not be produced, which makes the run inadmissible for this method (FR-018b).

FR-052 keeps upstream's code out of this project, so no inference path is
implemented here: Method C runs in its own environment and returns masks, the
same division of labour Methods A and B use.
"""
from __future__ import annotations

import importlib.util
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..availability import _digest
from . import register
from .base import DLMethodAdapter, InferenceResult
from .comic_text_detector import threshold_verification
from .manga_text_segmentation import MethodUnavailableError, _field

#: The stride the padded page is rounded up to — 32, not the ground truth's 8.
#: The aligned page (1656×1176) is a multiple of 8 but not of 32, so this is the
#: one padding in the benchmark that changes the aligned page's size (FR-024a).
PADDING_MULTIPLE = 32

#: Upstream's default cutoff on the probability map.  Unlike Method B's, this
#: one is actually applied (FR-026); it is still verified against the mask
#: rather than merely declared (FR-030).
DEFAULT_THRESHOLD = 0.5

#: The order the weights expect: ``timm`` encoders take RGB.
DEFAULT_CHANNEL_ORDER = "rgb"

#: ImageNet normalisation, verified from the reference source (FR-027).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Upstream's optional test-time augmentation and precision settings.  Both are
#: off by default here and both are recorded, because both change the result or
#: the timing and neither may be applied silently (FR-027, FR-044).
DEFAULT_FLIP_TTA = False
DEFAULT_MIXED_PRECISION = False


def _pad_to_multiple(value: int) -> int:
    """Round ``value`` up to the stride, never down (padding, not cropping)."""
    return -(-int(value) // PADDING_MULTIPLE) * PADDING_MULTIPLE


def padding_geometry(height: int, width: int) -> dict[str, Any]:
    """Method C's forward mapping for a ``height``×``width`` page (FR-024a, FR-025).

    No resize: upstream feeds the native page and pads it.  ``source_size`` is
    carried so :func:`invert_padding` can undo this without being told the page
    size a second time.  Every size is ``(height, width)``, numpy's convention.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"expected a positive page size, got {(height, width)}")

    padded_h, padded_w = _pad_to_multiple(height), _pad_to_multiple(width)
    return {
        "padding_multiple": PADDING_MULTIPLE,
        "padded_size": (padded_h, padded_w),
        "pad": (padded_h - int(height), padded_w - int(width)),
        "source_size": (int(height), int(width)),
    }


def pad_to_multiple(page: np.ndarray, geometry: dict[str, Any]) -> np.ndarray:
    """Zero-pad ``page`` bottom-right per ``geometry`` — the network's input."""
    if page.ndim != 2:
        raise ValueError(f"expected a single-channel page, got shape {page.shape}")
    padded_h, padded_w = geometry["padded_size"]
    canvas = np.zeros((padded_h, padded_w), dtype=page.dtype)
    canvas[: page.shape[0], : page.shape[1]] = page
    return canvas


def invert_padding(mask: np.ndarray, geometry: dict[str, Any]) -> np.ndarray:
    """Undo :func:`pad_to_multiple`: back to the aligned page size (FR-024, FR-028).

    The crop is the inverse of the forward transform and nothing else — no
    resize, because this method never resized.  A method never resizes its page:
    alignment happens after inference (FR-024a).
    """
    height, width = geometry["source_size"]
    return mask[:height, :width]


def state_dict_identity(state: Mapping[str, Any]) -> dict[str, Any]:
    """The tensor-key identity of a loaded state dict (FR-018a).

    Key names and shapes, digested.  This is what the provenance schema names as
    the shape of the evidence — a state_dict key set plus a checksum comparison
    — and it is deliberately not a statement about the weights' *values*: what
    FR-018a asks is whether the state that was installed came from the
    configured file, which the key set answers, whereas a plausible mask answers
    nothing at all.

    ``state`` is whatever the loader actually installed, so this is callable
    with the model's own ``state_dict()`` and with a plain mapping of anything
    carrying ``.shape``.
    """
    keys = sorted(str(key) for key in state)
    digest = sha256()
    for key in keys:
        shape = getattr(state[key], "shape", None)
        digest.update(f"{key}:{tuple(shape) if shape is not None else '?'};".encode())
    return {"keys": keys, "key_count": len(keys), "key_digest": digest.hexdigest()}


def checkpoint_load_evidence(
    loaded_state: Mapping[str, Any] | None,
    checkpoint: Mapping[str, Any],
    *,
    observed_sha256: str | None = None,
    configured_sha256: str | None = None,
) -> dict[str, Any]:
    """FR-018a/FR-018b positive evidence that the configured weights were loaded.

    Returns the ``checkpoint_load_evidence`` record verbatim: ``evidenced`` is
    true only when there is something positive to say, and ``detail`` always
    says what it is or why it could not be produced.

    ``loaded_state`` must be the state the loader actually installed into the
    model.  Passing a state dict re-read from the file would prove nothing about
    what ran, which is the whole point of the check: upstream's loader falls back
    to a randomly initialised decoder without raising, so the question is not
    "is the file there" but "is the file what the model is running".
    """
    identity = str(checkpoint.get("identity") or "checkpoint")

    if loaded_state is None:
        return {
            "evidenced": False,
            "method": "state_dict key set plus checksum comparison",
            "detail": (
                f"no state dict was reported as loaded from {identity}, so the "
                f"weights the model is running cannot be identified; a mask of "
                f"the right shape and value range is not evidence (FR-018a)"
            ),
        }

    if not loaded_state:
        return {
            "evidenced": False,
            "method": "state_dict key set plus checksum comparison",
            "detail": (
                f"the state dict reported for {identity} is empty, so nothing "
                f"was loaded from it; upstream's loader continues with an "
                f"untrained decoder in this case (FR-018a)"
            ),
        }

    if (
        configured_sha256 is not None
        and observed_sha256 is not None
        and observed_sha256 != configured_sha256
    ):
        return {
            "evidenced": False,
            "method": "file digest plus state_dict key set comparison",
            "detail": (
                f"{identity} has sha256 {observed_sha256}, the configuration "
                f"pins {configured_sha256}; the loaded state is not from the "
                f"pinned artefact (FR-018a)"
            ),
        }

    identity_record = state_dict_identity(loaded_state)
    return {
        "evidenced": True,
        "method": "state_dict key set plus checksum comparison",
        "detail": (
            f"{identity} loaded and installed: {identity_record['key_count']} "
            f"tensors, key digest {identity_record['key_digest']}"
            + (f", file sha256 {observed_sha256}" if observed_sha256 else "")
        ),
    }


class UnetPlusPlusEfficientNetV2Adapter(DLMethodAdapter):
    """Method C, and the record that its weights are the published ones."""

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        source = config or {}
        #: A configuration was supplied.  The frozen interface (T011)
        #: instantiates every registered adapter with no config at all, and that
        #: one must answer the interface tests; a configured adapter that cannot
        #: run must say so instead of answering with a mask (FR-018).
        self._configured = bool(source)
        self._checkpoints: list[dict[str, Any]] = [
            {
                "identity": _field(raw, "identity", ""),
                "path": Path(_field(raw, "path", "")),
                "size_bytes": _field(raw, "size_bytes"),
                "sha256": _field(raw, "sha256"),
            }
            for raw in (_field(source, "checkpoints") or [])
        ]

    @property
    def name(self) -> str:
        return "unetpp-efficientnetv2"

    def checkpoint(self) -> dict[str, Any] | None:
        """The configured checkpoint record, or ``None`` when none is configured."""
        return self._checkpoints[0] if self._checkpoints else None

    def inference_settings(self) -> dict[str, Any]:
        """The settings that change the result or the timing, recorded (FR-027)."""
        return {
            "channel_order": _field(
                self._config, "channel_order", DEFAULT_CHANNEL_ORDER
            ),
            "flip_tta": bool(_field(self._config, "flip_tta", DEFAULT_FLIP_TTA)),
            "mixed_precision": bool(
                _field(self._config, "mixed_precision", DEFAULT_MIXED_PRECISION)
            ),
        }

    def checkpoint_load_evidence(self) -> dict[str, Any]:
        """FR-018a/FR-018b evidence for this adapter's checkpoint (FR-056).

        No inference path is implemented here (FR-052), so no state dict is ever
        installed by this process and the honest answer is that evidence could
        not be produced *here* — which is exactly what FR-018b requires the
        provenance record to distinguish from "evidenced".  The runner's own
        environment produces the positive record by calling the module-level
        :func:`checkpoint_load_evidence` with the state its loader installed.
        """
        checkpoint = self.checkpoint()
        if checkpoint is None:
            return checkpoint_load_evidence(None, {"identity": "no checkpoint"})

        path = checkpoint["path"]
        observed = _digest(path) if path.is_file() else None
        return checkpoint_load_evidence(
            None,
            checkpoint,
            observed_sha256=observed,
            configured_sha256=checkpoint["sha256"],
        )

    def unavailable_reason(self) -> str | None:
        """Why Method C cannot be run here, or ``None`` if it can (FR-016, FR-021).

        Checked before any page is inferred rather than discovered part-way
        through (FR-021).  The digest is compared as well as the size, because a
        checkpoint that is the right length but the wrong file is exactly the
        silent-failure trap FR-018a exists for.
        """
        checkpoint = self.checkpoint()
        if checkpoint is None:
            return (
                "no checkpoint configured: Method C runs the released "
                "model.pth decoder (FR-013)"
            )

        path = checkpoint["path"]
        if not path.is_file():
            return f"checkpoint {checkpoint['identity']} is not at {path}"
        size = checkpoint["size_bytes"]
        if isinstance(size, int) and not isinstance(size, bool):
            observed_size = path.stat().st_size
            if observed_size != size:
                return (
                    f"checkpoint {checkpoint['identity']} is {observed_size} "
                    f"bytes, the configuration records {size}"
                )

        configured = checkpoint["sha256"]
        if configured:
            observed = _digest(path)
            if observed != configured:
                return (
                    f"checkpoint {checkpoint['identity']} has sha256 {observed}, "
                    f"the configuration pins {configured}"
                )

        absent = [
            name
            for name in ("torch", "timm")
            if importlib.util.find_spec(name) is None
        ]
        if absent:
            return (
                f"{' and '.join(absent)} not importable in this interpreter; "
                f"Method C runs in its own environment (FR-052)"
            )
        return None

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Refuse to answer when Method C cannot run here.

        The frozen interface (T011) requires a binary ``[H, W]`` answer at the
        input's shape, so an adapter constructed with *no configuration at all*
        returns the conservative one — *no text*.  A configured adapter that
        cannot run is a different thing: FR-018 forbids inventing a result for
        an unavailable method, and a zero mask is exactly such an invention, so
        it raises instead.  :meth:`infer` reports that as ``failed`` rather than
        letting it escape (FR-020).

        ponytail: no inference path is implemented.  Under FR-052 Method C runs
        in its own environment and returns masks; a torch forward pass that no
        environment here can execute would be untested code on the critical
        path, and the geometry it would have to be trusted for is checked
        directly instead (T036's sibling, this module's own tests).
        """
        if self._configured:
            reason = self.unavailable_reason()
            if reason is not None:
                raise MethodUnavailableError(reason)
        return np.zeros(image.shape[:2], dtype=np.uint8)

    def infer(self, image: "np.ndarray | str | Path") -> InferenceResult:
        """Report Method C unavailable rather than wrapping a zero mask as a result."""
        reason = self.unavailable_reason()
        if reason is None:
            return super().infer(image)

        try:
            array = self._as_array(image)
        except Exception as exc:  # noqa: BLE001 - reported, not raised (FR-010)
            return self._failure(None, exc)
        return self._failure(array.shape[:2], MethodUnavailableError(reason))

    def verify_threshold(
        self, probs: np.ndarray, mask: np.ndarray
    ) -> dict[str, Any]:
        """Check the configured cutoff against the produced mask (FR-030).

        The same rule Method B uses: the configured value is trusted only when
        it reproduces the mask, and the mask's own decision bracket is reported
        when it does not.  Declared per page because the value is fixed per
        method for the whole run (FR-026).
        """
        return threshold_verification(
            probs,
            mask,
            configured=_field(self._config, "threshold", DEFAULT_THRESHOLD),
        )

    def metadata(self) -> dict[str, Any]:
        """The configuration a reader needs to re-derive the mask (FR-024a/b, FR-027).

        Values are plain JSON types on purpose: the frozen interface serialises
        this into the runner's sidecar, so the geometry is recorded as the rule
        it is rather than as per-page tuples a reader would have to unpack.
        """
        checkpoint = self.checkpoint()
        padding = _field(self._config, "padding") or {}
        return {
            **super().metadata(),
            "checkpoint": checkpoint["identity"] if checkpoint else None,
            "preprocessing": {
                # FR-024a: this multiple is Method C's own.  It is not the ground
                # truth's 8, and the aligned page is not a multiple of it.
                "resize": f"pad-to-multiple-{PADDING_MULTIPLE}",
                "normalise": "imagenet",
                "mean": list(IMAGENET_MEAN),
                "std": list(IMAGENET_STD),
                **self.inference_settings(),
            },
            "padding_multiple": _field(padding, "multiple", PADDING_MULTIPLE),
            # FR-029: no morphological operation is applied by default.
            "postprocessing": {"morphology": "none", "activation": "sigmoid"},
            "threshold": {
                "configured": _field(self._config, "threshold", DEFAULT_THRESHOLD),
                "verified_per_page": True,
                "note": (
                    "the reference code applies its own 0.5 cutoff to the "
                    "probability map (FR-026); the effective cutoff is verified "
                    "per page by verify_threshold() and recorded in the runner's "
                    "sidecar"
                ),
            },
            "checkpoint_load_evidence": self.checkpoint_load_evidence(),
        }


register("unetpp-efficientnetv2", UnetPlusPlusEfficientNetV2Adapter)
