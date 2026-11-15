"""Ostium / aorta features (M3).

Inputs: a predicted lumen mask and an aorta mask (TotalSegmentator `total` task, class aorta,
Apache-2.0). Ground truth is never used. Lengths are mm; arrays are in nibabel index order and
``spacing`` is the voxel size per axis.

Features
  n_ostia              predicted components >= min_component_mm that touch the aorta (dilated by contact_mm)
  ostia_deviation      |n_ostia - 2|
  left_attached        an attached component lies on the patient-left side of the aorta
  right_attached       an attached component lies on the patient-right side of the aorta
  attached_dist_mm     min/max distance (mm) from each large component to the aorta
  unattached_len_mm    skeleton length of every component that does not touch the aorta
  unattached_share     unattached_len_mm / total skeleton length
  aorta_quality_ok     aorta mask is one object, volume within bounds, and not clipped by the image edge

Side of a component is decided from the sign of its contact-point offset from the aorta centroid
along ``lr_axis``; ``lr_sign`` = +1 when increasing index on that axis is patient-left (caller
derives it from the affine). Components not touching the aorta are never assigned a side.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..graph import branch_graph

DEFAULTS = {
    "contact_mm": 3.0,          # a component touches the aorta if within this distance
    "min_component_mm": 10.0,   # components shorter than this do not count as ostia
    "prune_mm": 3.0,            # spur pruning for the skeleton length
    "aorta_vol_ml": (20.0, 300.0),
    "expected_ostia": 2,
}


def skeleton_length_mm(mask, spacing, prune_mm=3.0):
    """Total branch length (mm) of the skeleton of a boolean mask."""
    if not np.any(mask):
        return 0.0
    B = branch_graph(mask, spacing, prune_mm=prune_mm)
    return float(sum(d["length_mm"] for _, _, d in B.edges(data=True)))


def aorta_quality(aorta, spacing, vol_ml=DEFAULTS["aorta_vol_ml"]):
    """Return (ok, info). ok requires one dominant object, a plausible volume, and no clipping at the FOV edge."""
    aorta = np.asarray(aorta, dtype=bool)
    sp = np.asarray(spacing, dtype=float)
    n_vox = int(aorta.sum())
    info = {"aorta_voxels": n_vox, "aorta_volume_ml": n_vox * float(np.prod(sp)) / 1000.0}
    if n_vox == 0:
        info.update(n_objects=0, largest_fraction=0.0, touches_edge=False)
        return False, info
    lab, n = ndimage.label(aorta, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lab.ravel())[1:]
    largest = float(sizes.max()) / n_vox
    edge = bool(aorta[0].any() or aorta[-1].any() or aorta[:, 0].any() or aorta[:, -1].any()
                or aorta[:, :, 0].any() or aorta[:, :, -1].any())
    lo, hi = vol_ml
    ok = (n >= 1) and largest >= 0.99 and lo <= info["aorta_volume_ml"] <= hi and not edge
    info.update(n_objects=int(n), largest_fraction=largest, touches_edge=edge)
    return bool(ok), info


def ostium_features(pred_mask, aorta_mask, spacing, lr_axis=0, lr_sign=1, cfg=None):
    """Ostium/aorta feature dict for one predicted lumen mask. No ground truth involved."""
    c = dict(DEFAULTS)
    if cfg:
        c.update(cfg)
    pred = np.asarray(pred_mask, dtype=bool)
    aorta = np.asarray(aorta_mask, dtype=bool)
    sp = tuple(float(s) for s in spacing)

    quality_ok, qinfo = aorta_quality(aorta, sp, c["aorta_vol_ml"])
    feats = {f"aorta_{k}": v for k, v in qinfo.items()}
    feats["aorta_quality_ok"] = quality_ok

    if not pred.any():
        feats.update(n_ostia=0, ostia_deviation=c["expected_ostia"], left_attached=False,
                     right_attached=False, attached_dist_mm=[], unattached_len_mm=0.0,
                     unattached_share=0.0, total_len_mm=0.0, n_components=0)
        return feats

    # distance (mm) from every voxel to the aorta; inf when no aorta
    if aorta.any():
        d_ao = ndimage.distance_transform_edt(~aorta, sampling=sp)
        ao_coords = np.argwhere(aorta)
        ao_centroid = ao_coords.mean(axis=0)
    else:
        d_ao = np.full(pred.shape, np.inf)
        ao_centroid = None

    lab, n = ndimage.label(pred, structure=np.ones((3, 3, 3)))
    n_ostia = 0
    left = right = False
    attached_dist = []
    total_len = 0.0
    unattached_len = 0.0
    n_components = 0
    for i in range(1, n + 1):
        comp = lab == i
        length = skeleton_length_mm(comp, sp, c["prune_mm"])
        total_len += length
        n_components += 1
        dmin = float(d_ao[comp].min())
        touches = dmin <= c["contact_mm"]
        if touches:
            contact = comp & (d_ao <= c["contact_mm"])
            if length >= c["min_component_mm"]:
                n_ostia += 1
                attached_dist.append(dmin)
                if ao_centroid is not None:
                    off = np.argwhere(contact).mean(axis=0)[lr_axis] - ao_centroid[lr_axis]
                    if lr_sign * off > 0:
                        left = True
                    else:
                        right = True
        else:
            unattached_len += length
        if length >= c["min_component_mm"] and not touches:
            attached_dist.append(dmin)

    feats.update(
        n_components=n_components,
        n_ostia=n_ostia,
        ostia_deviation=abs(n_ostia - c["expected_ostia"]),
        left_attached=bool(left),
        right_attached=bool(right),
        attached_dist_mm=attached_dist,
        unattached_len_mm=unattached_len,
        unattached_share=(unattached_len / total_len) if total_len > 0 else 0.0,
        total_len_mm=total_len,
    )
    return feats
