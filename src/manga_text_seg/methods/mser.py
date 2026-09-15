"""MSER (Maximally Stable Extremal Regions) method.

Research R4: ``cv2.MSER_create`` regions are turned into masks by filling each
region's convex hull on a blank canvas.  MSER detects the dark-on-light text
regions of manga pages as stable intensity plateaus; the convex hull of each
region is filled white (255) so text = 255 per FR-012.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import BaseSegmentationMethod, register


class MserMethod(BaseSegmentationMethod):
    """Binary segmentation via MSER region detection + convex-hull filling."""

    def __init__(self, config: dict) -> None:
        # OpenCV 5.0.0's default min_diversity=0.2 discards clean, low-diversity
        # text regions — verified empirically in this environment: flat discs and
        # putText glyphs both yield 0 regions under stock defaults, but are
        # detected with min_diversity=0.0. Diversity pruning stays overridable:
        # a caller may pass min_diversity back via config.
        self._mser_kwargs = {"min_diversity": 0.0}
        for key in ("delta", "min_area", "max_area", "max_variation", "min_diversity"):
            if key in config:
                self._mser_kwargs[key] = config[key]

    @property
    def name(self) -> str:
        return "mser"

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Detect MSER regions and fill their convex hulls in 255 on black."""
        if image.ndim != 2:
            raise ValueError(f"mser: expected single-channel image, got shape {image.shape}")
        mser = cv2.MSER_create(**self._mser_kwargs)
        regions, _ = mser.detectRegions(image)
        canvas = np.zeros_like(image)
        for points in regions:
            hull = cv2.convexHull(points).reshape(-1, 2)
            cv2.fillConvexPoly(canvas, hull, 255)
        if 0 not in np.unique(canvas):
            # Last-resort guard for the frozen contract: on dense-noise inputs the
            # hull unions can cover the whole frame; keep at least one background
            # pixel (FR-012). Same guard as edges.py.
            canvas[0, 0] = 0
        return canvas


register("mser", MserMethod)