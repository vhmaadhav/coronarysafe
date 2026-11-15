"""M4 synthetic corruption operators (TRAIN ground truth only).

Each operator takes a binary lumen mask and returns a corrupted copy. The corrupted mask is what
the M4 model sees. Labels are NOT assigned by the operator: they come from the frozen taxonomy v3
(labels.taxonomy.label_failures) against the original GT, so they match the endpoint.

Operators (names are used in the variant record):
  delete_run      remove a 5-40 mm run along a skeleton branch (cylinder cut)
  break_gap       1-6 mm gap across a vessel (thin slab cut)
  remove_distal   remove the terminal 10-30 mm of a branch (ball at an endpoint)
  fp_tube         add a detached false-positive tube (10-30 mm, r 1-2.5 mm), >= 5 mm from lumen
  fp_blob         add a detached false-positive sphere (r 3-6 mm), >= 5 mm from lumen
  leak_bulge      add a leak bulge (sphere r 2-5 mm) on a skeleton point
  bridge          add a tube (r 1.5 mm) joining the nearest left and right tree voxels
  erode_segment   erode a 5-20 mm stretch by one voxel (thinning)
Clean operators (mild, must NOT be labelled failures by design):
  clean_dilate, clean_erode (one voxel, whole mask), clean_truncate (<= 10 mm distal trim)

Coordinates: arrays are nibabel index order; ``sp`` is voxel size (mm) per axis.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from ..graph import skeletonize

FAILURE_OPS = [
    "delete_run", "break_gap", "remove_distal", "fp_tube", "fp_blob",
    "leak_bulge", "bridge", "erode_segment",
]
CLEAN_OPS = ["clean_dilate", "clean_erode", "clean_truncate"]
_NEIGH = np.ones((3, 3, 3), dtype=int)


def _grid(sl, shape):
    axes = [np.arange(s.start, s.stop) for s in sl]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def _window(shape, c, r_mm, sp):
    lo = [max(0, int(np.floor(ci - r_mm / s))) for ci, s in zip(c, sp)]
    hi = [min(n, int(np.ceil(ci + r_mm / s)) + 1) for ci, n, s in zip(c, shape, sp)]
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def _tangent(skel_coords, p, sp, rng, radius_mm=4.0):
    """Local tangent from PCA of skeleton voxels within radius_mm of p (random if too few)."""
    d = (skel_coords - p) * sp
    near = skel_coords[np.linalg.norm(d, axis=1) <= radius_mm]
    if len(near) < 3:
        v = rng.normal(size=3)
        return v / np.linalg.norm(v)
    x = (near - near.mean(0)) * sp
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return vt[0] / np.linalg.norm(vt[0])


def _cut_cylinder(mask, p, t, half_len_mm, radius_mm, sp):
    """Remove mask voxels inside a cylinder centred at p along unit axis t."""
    sl = _window(mask.shape, p, half_len_mm + radius_mm, sp)
    g = _grid(sl, mask.shape) - p
    x = g * sp
    proj = x @ t
    perp = np.linalg.norm(x - proj[..., None] * t, axis=-1)
    inside = (np.abs(proj) <= half_len_mm) & (perp <= radius_mm)
    sub = mask[sl]
    sub[inside] = False
    return mask


def _paint_segment(mask, a, b, r_mm, sp):
    """Paint a tube of radius r_mm between physical-index points a and b (voxel coords)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    lo = np.minimum(a, b) - r_mm / sp - 1
    hi = np.maximum(a, b) + r_mm / sp + 2
    lo = np.maximum(lo.astype(int), 0)
    hi = np.minimum(hi.astype(int), np.array(mask.shape))
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    g = _grid(sl, mask.shape).astype(float) * sp
    A, B = a * sp, b * sp
    d = B - A
    L2 = max(float(d @ d), 1e-9)
    t = np.clip(((g - A) @ d) / L2, 0.0, 1.0)
    dist = np.linalg.norm(g - (A + t[..., None] * d), axis=-1)
    mask[sl] |= dist <= r_mm
    return mask


def _paint_ball(mask, c, r_mm, sp):
    sl = _window(mask.shape, c, r_mm, sp)
    d = np.linalg.norm((_grid(sl, mask.shape) - c) * sp, axis=-1)
    mask[sl] |= d <= r_mm
    return mask


def _far_point(mask, sp, rng, min_mm=5.0, tries=200):
    """Random voxel at least min_mm from the mask and inside the mask's bounding box (+margin)."""
    coords = np.argwhere(mask)
    lo, hi = coords.min(0) - 10, coords.max(0) + 10
    dist = ndimage.distance_transform_edt(~mask, sampling=sp)
    for _ in range(tries):
        p = np.array([rng.integers(max(0, lo[i]), min(mask.shape[i], hi[i])) for i in range(3)])
        if dist[tuple(p)] >= min_mm:
            return p.astype(float)
    return None


def _endpoints(skel):
    nb = ndimage.convolve(skel.astype(int), _NEIGH, mode="constant") - skel.astype(int)
    return np.argwhere(skel & (nb == 1))


def apply_op(op, mask, gt_labels, sp, rng, skel=None, left_labels=(), right_labels=()):
    """Apply one operator to a copy of mask. Returns (mask, applied_bool)."""
    m = mask.copy()
    sp = np.asarray(sp, float)
    if skel is None:
        skel = skeletonize(m)
    sk_coords = np.argwhere(skel)
    if len(sk_coords) == 0:
        return m, False

    if op in ("delete_run", "break_gap", "erode_segment"):
        p = sk_coords[rng.integers(len(sk_coords))].astype(float)
        t = _tangent(sk_coords, p, sp, rng)
        if op == "delete_run":
            L = rng.uniform(5.0, 40.0)
            return _cut_cylinder(m, p, t, L / 2, 4.0, sp), True
        if op == "break_gap":
            g = rng.uniform(1.0, 6.0)
            return _cut_cylinder(m, p, t, g / 2, 4.0, sp), True
        # erode_segment: thin a stretch by one voxel inside a cylinder
        L = rng.uniform(5.0, 20.0)
        er = ndimage.binary_erosion(m)
        sl = _window(m.shape, p, L / 2 + 4.0, sp)
        g = _grid(sl, m.shape) - p
        x = g * sp
        proj = x @ t
        perp = np.linalg.norm(x - proj[..., None] * t, axis=-1)
        inside = (np.abs(proj) <= L / 2) & (perp <= 4.0)
        sub = m[sl]
        sub[inside] = er[sl][inside]
        return m, True

    if op == "remove_distal":
        ends = _endpoints(skel)
        if len(ends) == 0:
            return m, False
        e = ends[rng.integers(len(ends))].astype(float)
        r = rng.uniform(10.0, 30.0)
        sl = _window(m.shape, e, r, sp)
        d = np.linalg.norm((_grid(sl, m.shape) - e) * sp, axis=-1)
        m[sl][d <= r] = False
        return m, True

    if op == "fp_tube":
        p = _far_point(m, sp, rng)
        if p is None:
            return m, False
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v)
        L = rng.uniform(10.0, 30.0)
        r = rng.uniform(1.0, 2.5)
        q = p + v * (L / sp)
        q = np.clip(q, 0, np.array(m.shape) - 1)
        return _paint_segment(m, p, q, r, sp), True

    if op == "fp_blob":
        p = _far_point(m, sp, rng)
        if p is None:
            return m, False
        return _paint_ball(m, p, rng.uniform(3.0, 6.0), sp), True

    if op == "leak_bulge":
        inner = sk_coords[rng.integers(len(sk_coords))].astype(float)
        return _paint_ball(m, inner, rng.uniform(2.0, 5.0), sp), True

    if op == "bridge":
        lab_l = np.isin(gt_labels, list(left_labels))
        lab_r = np.isin(gt_labels, list(right_labels))
        if not lab_l.any() or not lab_r.any():
            return m, False
        cl = np.argwhere(lab_l)
        cr = np.argwhere(lab_r)
        cl_mm, cr_mm = cl * sp, cr * sp
        tree = cKDTree(cr_mm)
        i = rng.integers(len(cl))
        dist, j = tree.query(cl_mm[i])
        a = cl[i].astype(float)
        b = cr[j].astype(float)
        return _paint_segment(m, a, b, 1.5, sp), True

    if op == "clean_dilate":
        return ndimage.binary_dilation(m), True
    if op == "clean_erode":
        return ndimage.binary_erosion(m), True
    if op == "clean_truncate":
        ends = _endpoints(skel)
        if len(ends) == 0:
            return m, False
        e = ends[rng.integers(len(ends))].astype(float)
        r = rng.uniform(3.0, 10.0)
        sl = _window(m.shape, e, r, sp)
        d = np.linalg.norm((_grid(sl, m.shape) - e) * sp, axis=-1)
        m[sl][d <= r] = False
        return m, True

    raise ValueError(f"unknown operator {op}")


def corrupt(gt_labels, sp, rng, plan, left_labels=(), right_labels=()):
    """Apply a plan (list of op names, in order) to the GT lumen. Returns (mask, applied_ops)."""
    mask = np.asarray(gt_labels) > 0
    applied = []
    for op in plan:
        skel = skeletonize(mask) if op not in CLEAN_OPS else None
        mask, ok = apply_op(op, mask, gt_labels, sp, rng, skel=skel,
                            left_labels=left_labels, right_labels=right_labels)
        if ok:
            applied.append(op)
    return mask, applied


def sample_plan(rng, clean=False):
    """Clean plans: one or two clean ops. Failure plans: one or two failure ops, sometimes plus a clean op."""
    if clean:
        k = rng.integers(1, 3)
        return list(rng.choice(CLEAN_OPS, size=k, replace=True))
    k = rng.integers(1, 3)
    plan = list(rng.choice(FAILURE_OPS, size=k, replace=False))
    if rng.random() < 0.3:
        plan.append(str(rng.choice(CLEAN_OPS)))
    return plan
