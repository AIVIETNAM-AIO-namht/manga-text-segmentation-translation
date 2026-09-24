"""Method B: the refined segmentation head, letterboxed and threshold-verified.

T039 (FR-013, FR-024a, FR-025, FR-026, FR-030; US3).

Upstream's network is a YOLOv5s backbone with two heads.  One is a U-Net-style
transposed-convolution head whose sigmoid output is the pixel mask; the other is
a DB head emitting ICDAR-style text-line polygons.  This adapter consumes the
first and discards the second (FR-013) — the polygons are a different output
type, and nothing in this benchmark scores them.

Three things about Method B are recorded here rather than assumed:

* **Its own geometry.**  The page is letterboxed onto a 1024×1024 canvas and
  then zero-padded bottom-right to a multiple of 64.  FR-024a forbids treating
  that as the ground truth's multiple-of-8 rule: the aligned page is a multiple
  of 8 but not of 32, let alone 64, so the inverse is this method's own and is
  inverted here rather than approximated by a shared resize.
* **The cutoff that actually scored the mask.**  Upstream accepts a configurable
  mask threshold and never reads it, hardcoding its own cutoff on the sigmoid
  output.  FR-030 exists for exactly this, so the configured value is not
  written down as the effective one — :func:`threshold_verification` recovers
  the applied cutoff from the probability map and the produced mask, and a
  configuration that disagrees with the implementation is caught.
* **The channel order actually used.**  Upstream reads pages with OpenCV and
  never converts, so the array it feeds the network is BGR while the weights
  expect RGB.  That reversal degrades accuracy without raising anything, so it
  is recorded as the order actually used rather than silently corrected
  (FR-024a); the note travels with the metadata.

FR-052 keeps upstream's code out of this project, so no inference path is
implemented here: Method B runs in its own environment and returns masks, the
same division of labour Method A uses.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import register
from .base import DLMethodAdapter, InferenceResult
from .manga_text_segmentation import MethodUnavailableError, _field

#: The square canvas upstream letterboxes onto before padding to the stride.
CANVAS_SIZE = 1024

#: The stride the padded canvas is rounded up to — 64, not the ground truth's 8.
PADDING_MULTIPLE = 64

#: The cutoff upstream hardcodes on the sigmoid output (spec.md, known breakage
#: 4).  It is the *effective* value, not the configured one; FR-030 is why the
#: two are kept apart.
DEFAULT_THRESHOLD = 0.235

#: The order upstream actually feeds the network: it reads with OpenCV, which is
#: BGR, and does not convert.
DEFAULT_CHANNEL_ORDER = "bgr"

#: Interpolation for both directions.  The mask is binary and the alignment is
#: nearest-neighbour upstream; a smoothing kernel would invent values that were
#: never predicted.
_INTERPOLATION = cv2.INTER_NEAREST


def _pad_to_multiple(value: int) -> int:
    """Round ``value`` up to the stride, never down (padding, not cropping)."""
    return -(-int(value) // PADDING_MULTIPLE) * PADDING_MULTIPLE


def letterbox_geometry(height: int, width: int) -> dict[str, Any]:
    """Method B's forward mapping for a ``height``×``width`` page (FR-024a).

    The scale fits the long edge to the canvas, the short edge follows to keep
    the aspect ratio, and the result is zero-padded bottom-right up to the
    stride.  ``source_size`` is carried so :func:`invert_letterbox` can undo
    this mapping without being told the page size a second time — the inverse
    is a property of the transform, and a reader re-deriving it from the
    configuration alone would be re-deriving it by hand.

    Every size is ``(height, width)``, numpy's convention.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"expected a positive page size, got {(height, width)}")

    scale = CANVAS_SIZE / max(height, width)
    scaled_h = min(max(1, int(round(height * scale))), CANVAS_SIZE)
    scaled_w = min(max(1, int(round(width * scale))), CANVAS_SIZE)
    padded_h, padded_w = _pad_to_multiple(scaled_h), _pad_to_multiple(scaled_w)
    return {
        "canvas": (CANVAS_SIZE, CANVAS_SIZE),
        "padding_multiple": PADDING_MULTIPLE,
        "scale": scale,
        "scaled_size": (scaled_h, scaled_w),
        "padded_size": (padded_h, padded_w),
        "pad": (padded_h - scaled_h, padded_w - scaled_w),
        "source_size": (int(height), int(width)),
    }


def letterbox(mask: np.ndarray, geometry: dict[str, Any]) -> np.ndarray:
    """Place ``mask`` on the padded canvas per ``geometry`` (FR-024a)."""
    if mask.ndim != 2:
        raise ValueError(f"expected a single-channel mask, got shape {mask.shape}")
    scaled_h, scaled_w = geometry["scaled_size"]
    canvas = np.zeros(geometry["canvas"], dtype=np.uint8)
    canvas[:scaled_h, :scaled_w] = cv2.resize(
        mask, (scaled_w, scaled_h), interpolation=_INTERPOLATION
    )
    return canvas


def invert_letterbox(canvas_mask: np.ndarray, geometry: dict[str, Any]) -> np.ndarray:
    """Undo :func:`letterbox`: back to the aligned page, unresized (FR-024, FR-028).

    The crop is to ``scaled_size``, not ``padded_size``: the padding is zeros
    this method added, and resizing from the padded size would divide the page
    by the stride as well as by the scale.  The result is the input's exact
    shape — a method never resizes its page, because alignment happens after
    inference (FR-024a).
    """
    height, width = geometry["source_size"]
    scaled_h, scaled_w = geometry["scaled_size"]
    cropped = canvas_mask[:scaled_h, :scaled_w]
    if cropped.size == 0:
        return np.zeros((height, width), dtype=np.uint8)
    return cv2.resize(cropped, (width, height), interpolation=_INTERPOLATION)


def apply_threshold(probs: np.ndarray, cutoff: float) -> np.ndarray:
    """Binarise a sigmoid map at ``cutoff`` into the frozen convention (FR-028).

    ``>=`` rather than ``>``: the recovered cutoff is reported as the top of the
    bracket the mask itself decided, and that value has to reproduce the mask
    when fed back in.
    """
    return np.where(probs >= cutoff, 255, 0).astype(np.uint8)


def threshold_verification(
    probs: np.ndarray, mask: np.ndarray, *, configured: float
) -> dict[str, Any]:
    """Which cutoff explains ``mask``, and whether it is the configured one (FR-030).

    The configured value is trusted only when it reproduces the mask exactly.
    Otherwise the cutoff is recovered from the mask's own decision bracket —
    the highest probability left outside it and the lowest kept inside it — and
    reported as that bracket's midpoint.  The midpoint is what upstream's
    integer cutoff looks like once it has been compared against a float
    configuration: the exact constant is gone, but the interval it fell in is
    not, and that interval is evidence rather than a transcription of a number
    the code never read.

    A mask that is entirely text or entirely background carries no bracket; the
    cutoff is then unrecoverable, and this says so instead of guessing one.
    """
    configured = float(configured)
    if np.array_equal(apply_threshold(probs, configured), mask):
        return {
            "configured": configured,
            "applied": configured,
            "verified": True,
            "note": None,
        }

    inside = probs[mask > 0]
    outside = probs[mask == 0]
    if inside.size == 0 or outside.size == 0:
        return {
            "configured": configured,
            "applied": None,
            "verified": False,
            "note": (
                f"the mask is uniformly {'text' if inside.size else 'background'}, "
                f"so the cutoff that produced it cannot be recovered from it; the "
                f"configured {configured:g} is recorded as unverified (FR-030)"
            ),
        }

    applied = float(outside.max() + inside.min()) / 2.0
    return {
        "configured": configured,
        "applied": applied,
        "verified": False,
        "note": (
            f"the configured cutoff {configured:g} does not reproduce the "
            f"produced mask; the mask's own decision bracket is "
            f"({float(outside.max()):g}, {float(inside.min()):g}], so the cutoff "
            f"that actually scored it is about {applied:g} (FR-030)"
        ),
    }


class ComicTextDetectorAdapter(DLMethodAdapter):
    """Method B's segmentation head, and the record of how it was fed."""

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
            }
            for raw in (_field(source, "checkpoints") or [])
        ]

    @property
    def name(self) -> str:
        return "comic-text-detector"

    def channel_order(self) -> str:
        """The order the page is actually fed in, which is not always the right one."""
        return str(_field(self._config, "channel_order", DEFAULT_CHANNEL_ORDER))

    def unavailable_reason(self) -> str | None:
        """Why Method B cannot be run here, or ``None`` if it can (FR-016).

        Checked before any page is inferred rather than discovered part-way
        through (FR-021).
        """
        if not self._checkpoints:
            return (
                "no checkpoint configured: Method B runs the released "
                "comictextdetector.pt segmentation head (FR-013)"
            )

        checkpoint = self._checkpoints[0]
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

        if importlib.util.find_spec("torch") is None:
            return (
                "torch is not importable in this interpreter; Method B runs in "
                "its own environment (FR-052)"
            )
        return None

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Refuse to answer when Method B cannot run here.

        The frozen interface (T011) requires a binary ``[H, W]`` answer at the
        input's shape, so an adapter constructed with *no configuration at all*
        returns the conservative one — *no text*.  A configured adapter that
        cannot run is a different thing: FR-018 forbids inventing a result for
        an unavailable method, and a zero mask is exactly such an invention, so
        it raises instead.  :meth:`infer` reports that as ``failed`` rather than
        letting it escape (FR-020).

        ponytail: no inference path is implemented.  Under FR-052 Method B runs
        in its own environment and returns masks; a torch forward pass that no
        environment here can execute would be untested code on the critical
        path, and the geometry and threshold it would have to be trusted for
        are checked directly instead (T036).
        """
        if self._configured:
            reason = self.unavailable_reason()
            if reason is not None:
                raise MethodUnavailableError(reason)
        return np.zeros(image.shape[:2], dtype=np.uint8)

    def infer(self, image: "np.ndarray | str | Path") -> InferenceResult:
        """Report Method B unavailable rather than wrapping a zero mask as a result."""
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

        Values are plain JSON types on purpose: the frozen interface serialises
        this into the runner's sidecar, so the geometry is recorded as the rule
        it is rather than as per-page tuples a reader would have to unpack.
        """
        checkpoint = self._checkpoints[0]["identity"] if self._checkpoints else None
        notes = _field(self._config, "notes") or {}
        return {
            **super().metadata(),
            "checkpoint": checkpoint,
            "preprocessing": {
                "resize": (
                    f"letterbox-{CANVAS_SIZE}-pad-to-multiple-{PADDING_MULTIPLE}"
                ),
                # No ImageNet mean/std: upstream scales by 1/255 only.
                "normalise": "scale-1/255",
                "channel_order": self.channel_order(),
                "channel_order_note": _field(notes, "channel_order_note"),
            },
            # FR-013: the DB head's polygons are a different output type and
            # nothing in this benchmark scores them.
            "postprocessing": {"polygons": "discarded", "morphology": "none"},
            "threshold": {
                "configured": _field(self._config, "threshold", DEFAULT_THRESHOLD),
                "verified_per_page": True,
                "note": (
                    "upstream accepts this cutoff and never applies it, "
                    "hardcoding its own on the sigmoid output (FR-030); the "
                    "effective cutoff is recovered per page by "
                    "threshold_verification() and recorded in the runner's sidecar"
                ),
            },
        }


register("comic-text-detector", ComicTextDetectorAdapter)
