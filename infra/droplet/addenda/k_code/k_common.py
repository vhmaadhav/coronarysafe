"""Addendum K shared helpers (ROI, paths). Written 2026-11-15 for the controlled acquisition-shift test."""
import glob

import nibabel as nib
import numpy as np

DATA = "/root/data"
AORTA_DIR = f"{DATA}/m4/aorta"
MIN_SLICE_VOXELS = 20


def src_path(sid):
    hits = glob.glob(f"{DATA}/imagecas/volumes/**/{sid}.img.nii.gz", recursive=True)
    if len(hits) != 1:
        raise RuntimeError(f"scan {sid}: {len(hits)} source CT volumes")
    return hits[0]


def aorta_mask(sid, shape):
    """TotalSegmentator aorta mask from the m4 pipeline (same grid as the CT)."""
    arr = np.asarray(nib.load(f"{AORTA_DIR}/{sid}.nii.gz").dataobj) > 0.5
    if arr.shape != tuple(shape):
        raise RuntimeError(f"scan {sid}: aorta mask shape {arr.shape} != CT {shape}")
    return arr


def central_roi(mask):
    """Per axial slice: the central 50% of the cross-section (in-plane bounding box, the middle
    half of x and of y), restricted to aorta voxels. Slices with too few voxels are skipped.
    Returns (roi bool volume, list of slice indices used)."""
    roi = np.zeros(mask.shape, dtype=bool)
    zs = []
    ii = np.arange(mask.shape[0])[:, None]
    jj = np.arange(mask.shape[1])[None, :]
    for z in range(mask.shape[2]):
        m2 = mask[:, :, z]
        if m2.sum() < MIN_SLICE_VOXELS:
            continue
        xs, ys = np.nonzero(m2)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        w, h = x1 - x0, y1 - y0
        sel = m2 & (ii >= x0 + 0.25 * w) & (ii <= x1 - 0.25 * w) & (jj >= y0 + 0.25 * h) & (jj <= y1 - 0.25 * h)
        if sel.sum() < MIN_SLICE_VOXELS:
            continue
        roi[:, :, z] = sel
        zs.append(z)
    return roi, zs


def slice_sd(vol, roi, zs):
    return [float(np.std(vol[:, :, z][roi[:, :, z]])) for z in zs]
