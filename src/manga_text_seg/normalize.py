"""KMeans-2 normalization of ground-truth masks.

Research R1: masks encode text as magenta (#FF00FF) pixels (89 images) or
magenta+black pixels (357 images).  A few edge cases exist: near-black-only
(3 images), all-white-empty (1 image: EvaLady/006).  The normalization step
maps the mask to a binary (0/255) single-channel uint8 representation
regardless of the original encoding, so downstream methods never need to
know the encoding variant.

The approach: KMeans with k=2 clusters on pixel intensities, then assign the
cluster with fewer non-background pixels as foreground (text).  Edge cases
(all-white, single-cluster) are handled deterministically.
"""
from __future__ import annotations

import numpy as np


class NormalizeError(RuntimeError):
    """Raised when normalization fails for an unexpected reason."""


def _ensure_single_channel(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        # For truly multi-channel masks, take average
        return np.mean(img, axis=2).astype(np.float32)
    return img.astype(np.float32)


def _kmeans2_cluster_counts(pixels: np.ndarray, k: int = 2, max_iter: int = 20):
    """Lightweight KMeans (k=2) using numpy only — no sklearn dependency.

    Returns (labels, counts) for each cluster.
    """
    pixels_f = pixels.astype(np.float64)
    # Initialize centroids as the min and max values
    c0, c1 = float(pixels_f.min()), float(pixels_f.max())
    if c0 == c1:
        # All pixels identical — single cluster
        return np.zeros(len(pixels_f), dtype=np.int32), [len(pixels_f)]

    for _ in range(max_iter):
        dists = np.stack([np.abs(pixels_f - c0), np.abs(pixels_f - c1)], axis=1)
        labels = np.argmin(dists, axis=1).astype(np.int32)
        new_c0 = float(pixels_f[labels == 0].mean()) if (labels == 0).any() else c0
        new_c1 = float(pixels_f[labels == 1].mean()) if (labels == 1).any() else c1
        if np.isclose(new_c0, c0) and np.isclose(new_c1, c1):
            break
        c0, c1 = new_c0, new_c1

    counts = [int((labels == i).sum()) for i in range(k)]
    return labels, counts


def normalize_mask(img: np.ndarray) -> np.ndarray:
    """Normalize a GT mask to binary (0/255) uint8.

    Steps:
    1. Ensure single-channel float32.
    2. KMeans(k=2) on pixel intensities.
    3. Assign the cluster with fewer pixels as foreground (text = 255),
       background as 0.
    4. Edge cases: all-zero image → all-zero output; single-cluster → all-zero.
    """
    gray = _ensure_single_channel(img)
    flat = gray.ravel()

    unique_vals = np.unique(flat)
    if len(unique_vals) <= 1:
        # Single intensity or all zeros — no text
        # Always return 2D single-channel output
        return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    labels, counts = _kmeans2_cluster_counts(flat, k=2, max_iter=20)

    # Cluster 0 = lower intensity (typically background/dark)
    # Cluster 1 = higher intensity (typically text/magenta)
    # But we want the FEWER pixels to be foreground (text is rare)
    # So: cluster with fewer pixels = foreground = 255
    if counts[0] < counts[1]:
        fg_label = 0  # lower intensity cluster is smaller → treat as text
    elif counts[1] < counts[0]:
        fg_label = 1  # higher intensity cluster is smaller → treat as text
    else:
        # Tie: treat lighter pixels as foreground
        fg_label = 1

    binary = np.zeros_like(flat, dtype=np.uint8)
    binary[labels == fg_label] = 255

    return binary.reshape(img.shape[0], img.shape[1])