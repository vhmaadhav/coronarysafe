"""Uncertainty features from a probability map and a binary mask (no ground truth used).

Probabilities are assumed to be the foreground probability of one segmenter
(sigmoid output, or softmax foreground channel) already normalised per method.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..geometry import bbox_slices, dist_to
from ..graph import skeletonize

_EPS = 1e-7


def binary_entropy(p):
    """Binary entropy in nats."""
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def _dice(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def uncertainty_features(mask, prob, spacing) -> dict:
    """Entropy and boundary-uncertainty features.

    * mean_entropy_mask: mean binary entropy over mask voxels
    * mean_entropy_skel2mm: mean entropy within 2 mm of the mask skeleton
    * boundary_uncertain_frac: fraction of voxels within 3 mm of the mask
      boundary with p in [0.3, 0.7]
    * expected_hard_volume_ratio: sum(p) over the volume / mask voxel count
    """
    sp = tuple(float(s) for s in spacing)
    m = np.asarray(mask) > 0
    p = np.asarray(prob, dtype=float)
    out = {
        "mean_entropy_mask": np.nan,
        "mean_entropy_skel2mm": np.nan,
        "boundary_uncertain_frac": np.nan,
        "expected_hard_volume_ratio": np.nan,
    }
    if not m.any():
        return out
    H = binary_entropy(p)
    out["mean_entropy_mask"] = float(H[m].mean())
    out["expected_hard_volume_ratio"] = float(p.sum() / m.sum())

    sl = bbox_slices(m, sp, pad_mm=4.0)
    m_c = m[sl]
    H_c = H[sl]
    p_c = p[sl]

    skel = skeletonize(m_c)
    near_skel = dist_to(skel, sp) <= 2.0
    if near_skel.any():
        out["mean_entropy_skel2mm"] = float(H_c[near_skel].mean())

    boundary = m_c ^ ndimage.binary_erosion(m_c)
    if boundary.any():
        near_b = dist_to(boundary, sp) <= 3.0
        unc = near_b & (p_c >= 0.3) & (p_c <= 0.7)
        out["boundary_uncertain_frac"] = float(unc.sum() / near_b.sum())
    return out


def pairwise_dice(masks) -> float:
    """Mean pairwise Dice across a list of binary masks (Zenk-style agreement).

    Returns NaN when fewer than two masks are given.
    """
    ms = [np.asarray(x) > 0 for x in masks]
    if len(ms) < 2:
        return np.nan
    vals = [_dice(ms[i], ms[j]) for i in range(len(ms)) for j in range(i + 1, len(ms))]
    return float(np.mean(vals))


def tta_disagreement(masks) -> float:
    """Test-time-augmentation disagreement: 1 - mean pairwise Dice of the flipped predictions."""
    d = pairwise_dice(masks)
    return float(1.0 - d) if np.isfinite(d) else np.nan
