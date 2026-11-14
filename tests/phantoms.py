"""Synthetic coronary-like phantoms with anisotropic spacing (0.4, 0.4, 0.5) mm.

Coordinates are given in mm as (x, y, z), mapped to array axes (i, j, k) = (x, y, z).
Left tree: LM -> LAD (+ D1 side branch) and LCx. Right tree: RCA -> R-PDA.
"""

from __future__ import annotations

import numpy as np

SPACING = (0.4, 0.4, 0.5)
SHAPE = (100, 100, 120)
RADIUS_MM = 1.5

TUBES = {
    1: [((20, 20, 55), (20, 20, 45))],  # LM
    2: [((20, 20, 45), (20, 4, 45)), ((20, 4, 45), (20, 4, 5))],  # LAD (bends at corner)
    3: [((20, 20, 45), (36, 36, 40))],  # LCx
    4: [((20, 12, 45), (2, 12, 25))],  # D1 side branch off LAD
    9: [((32, 8, 58), (32, 8, 8))],  # RCA
    10: [((32, 8, 25), (36, 4, 12))],  # R-PDA
}


def _rasterize_segment(p0, p1, r):
    out = np.zeros(SHAPE, dtype=bool)
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    sp = np.asarray(SPACING)
    lo = np.minimum(p0, p1) - r - 1.0
    hi = np.maximum(p0, p1) + r + 1.0
    lo_i = np.maximum(np.floor(lo / sp).astype(int), 0)
    hi_i = np.minimum(np.ceil(hi / sp).astype(int) + 1, SHAPE)
    ii = [np.arange(lo_i[a], hi_i[a]) for a in range(3)]
    grids = np.meshgrid(*[ii[a] * sp[a] for a in range(3)], indexing="ij")
    grid = np.stack(grids, axis=-1)
    v = p1 - p0
    t = np.clip(((grid - p0) @ v) / (v @ v), 0.0, 1.0)
    closest = p0 + t[..., None] * v
    dist = np.linalg.norm(grid - closest, axis=-1)
    out[np.ix_(*ii)] = dist <= r
    return out


def tube_mask(p0, p1, r=RADIUS_MM):
    return _rasterize_segment(p0, p1, r)


def build_gt():
    """15-class GT volume (only classes 1-4, 9, 10 present)."""
    gt = np.zeros(SHAPE, dtype=np.int32)
    for lab, segs in TUBES.items():  # first-come: junction voxels keep the parent label
        for p0, p1 in segs:
            m = _rasterize_segment(p0, p1, RADIUS_MM)
            gt[m & (gt == 0)] = lab
    return gt


def _axis_z_mm(shape=SHAPE):
    k = np.arange(shape[2]) * SPACING[2]
    return np.broadcast_to(k[None, None, :], shape)


def lad_gap(pred, z_lo, z_hi):
    """Remove LAD voxels whose z (mm) lies in [z_lo, z_hi]."""
    gt = build_gt()
    z = _axis_z_mm()
    cut = (gt == 2) & (z >= z_lo) & (z <= z_hi)
    out = pred.copy()
    out[cut] = False
    return out
