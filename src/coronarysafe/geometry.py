"""Shared geometry helpers. No ground-truth access in this module."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def bbox_slices(mask, spacing, pad_mm: float = 0.0, pad_vox: int = 0):
    """Bounding-box slices of a boolean mask, padded by mm and/or voxels.

    Returns None when the mask is empty.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None
    found = ndimage.find_objects(mask.astype(np.uint8))[0]
    out = []
    for sl, n, sp in zip(found, mask.shape, spacing):
        pad = int(np.ceil(pad_mm / float(sp))) if pad_mm > 0 else 0
        pad += pad_vox
        out.append(slice(max(0, sl.start - pad), min(n, sl.stop + pad)))
    return tuple(out)


def dist_to(mask, spacing):
    """Euclidean distance (mm) from every voxel to the nearest True voxel of mask."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.full(mask.shape, np.inf)
    return ndimage.distance_transform_edt(~mask, sampling=tuple(float(s) for s in spacing))
