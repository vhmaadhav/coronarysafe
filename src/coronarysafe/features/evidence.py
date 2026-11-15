"""M5 image-evidence features (Addendum B). Label-free: the ground truth is never read.

For a predicted tree the CT is asked whether vessel is present where the prediction has none.

  * Gap evidence. For each facing endpoint pair in different components (the n_gaps rule of
    graph_features), the minimal-cost path through the CT between the endpoints, with
    cost = 1 / (vesselness + eps), inside a box of +-30 mm around the pair. Features are taken over
    the path voxels that the prediction does not cover.
  * Tip evidence. From each tree tip, greedy tracking along the max-vesselness neighbour (within a
    60 degree cone of the outward direction) for up to 20 mm past the prediction.

Vesselness is a multi-scale Sato filter (sigmas 0.5-2 mm) on the intensity-normalised HU, computed
on a cropped ROI around the prediction. Intensities are normalised per scan by a reference HU: the
median inside the aorta mask when one is given, else the 95th percentile of HU within 3 mm of the
predicted mask (floored at 100 HU). Units: mm and HU.

Evidence per gap = contrast fraction x mean vesselness, where contrast fraction is the share of
uncovered path voxels with normalised HU >= 0.5. A bright unpredicted vessel (a likely missed
segment) gives high evidence; a dark gap gives low evidence.
"""

from __future__ import annotations

import numpy as np
from skimage.filters import sato
from skimage.graph import route_through_array

from ..geometry import bbox_slices
from ..graph import branch_graph, components
from .graph_features import _endpoint_direction

SIGMAS_MM = (0.5, 1.0, 1.5, 2.0)
ROI_PAD_MM = 12.0
REF_DILATE_MM = 3.0
REF_FLOOR_HU = 100.0
NHU_CLIP = (-1.0, 3.0)
CONTRAST_NHU = 0.5
COST_EPS = 1e-3
ROUTE_HALF_MM = 30.0
GAP_MAX_MM = 8.0
GAP_COS = 0.7
MIN_COMPONENT_MM = 2.0
PRUNE_MM = 3.0
MAJOR_RADIUS_MM = 1.25
TRACK_MAX_MM = 20.0
TRACK_MIN_MM = 5.0
TRACK_STOP_VESS = 0.1
TRACK_CONE_COS = 0.5
TRACK_MAX_STEPS = 400

_OFFSETS = np.array(
    [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)],
    dtype=np.int64,
)

# Scalar reference-intensity keys are reported but are not vessel features.
REFERENCE_KEYS = ("m5_ref_hu", "m5_ref_fallback")
FEATURE_KEYS = (
    "m5_n_gaps",
    "m5_gap_evidence_max",
    "m5_gap_contrast_max",
    "m5_gap_vess_max",
    "m5_gap_nhu_mean_max",
    "m5_gap_path_len_at_max_mm",
    "m5_gap_major_evidence_sum",
    "m5_n_tips",
    "m5_tip_evidence_max",
    "m5_tip_cont_len_max_mm",
    "m5_tip_major_evidence_max",
)


def _empty(ref=np.nan, fallback=np.nan):
    out = {k: 0.0 for k in FEATURE_KEYS}
    out["m5_ref_hu"] = float(ref)
    out["m5_ref_fallback"] = float(fallback)
    return out


def vesselness(nhu, spacing):
    """Multi-scale Sato ridge response, scaled to [0, 1] by its 99.5th percentile in the ROI."""
    mean_sp = float(np.mean(spacing))
    sig = [s / mean_sp for s in SIGMAS_MM]  # sigmas in voxels
    v = np.asarray(sato(nhu, sigmas=sig, black_ridges=False, mode="reflect"), dtype=float)
    scale = float(np.percentile(v, 99.5))
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros(v.shape)
    return np.clip(v / scale, 0.0, 1.0)


def reference_hu(hu, aorta, hu_roi, mask_roi, spacing):
    """(reference HU, fallback flag). Aorta median when available, else 95th percentile near the mask."""
    if aorta is not None and np.asarray(aorta).any():
        ref = float(np.median(np.asarray(hu)[np.asarray(aorta) > 0]))
        fallback = 0.0
    else:
        near = _near_mask(mask_roi, spacing)
        ref = float(np.percentile(hu_roi[near], 95))
        fallback = 1.0
    if not np.isfinite(ref) or ref < REF_FLOOR_HU:
        ref = REF_FLOOR_HU
    return ref, fallback


def _near_mask(mask, spacing):
    from scipy import ndimage

    return ndimage.distance_transform_edt(~mask, sampling=tuple(float(s) for s in spacing)) <= (
        REF_DILATE_MM
    )


def _gap_pairs(ends):
    pairs = []
    for i in range(len(ends)):
        ei = ends[i]
        for j in range(i + 1, len(ends)):
            ej = ends[j]
            if ei["cid"] == ej["cid"]:
                continue
            diff = ej["pos"] - ei["pos"]
            dist = float(np.linalg.norm(diff))
            if dist <= 0 or dist > GAP_MAX_MM:
                continue
            u = diff / dist
            if np.dot(ei["dir"], u) > GAP_COS and np.dot(ej["dir"], -u) > GAP_COS:
                pairs.append((i, j))
    return pairs


def _route_gap(a, b, nhu, vess, m_roi, sp):
    """Minimal-cost CT path between two ROI voxels and its features over uncovered voxels."""
    half = np.ceil(ROUTE_HALF_MM / np.asarray(sp)).astype(np.int64)
    shape = np.asarray(vess.shape, dtype=np.int64)
    lo = np.maximum(np.minimum(a, b) - half, 0)
    hi = np.minimum(np.maximum(a, b) + half + 1, shape)
    box = tuple(slice(int(lo[k]), int(hi[k])) for k in range(3))
    cost = 1.0 / (vess[box].astype(float) + COST_EPS)
    path, _ = route_through_array(
        cost,
        tuple(int(x) for x in (a - lo)),
        tuple(int(x) for x in (b - lo)),
        fully_connected=True,
        geometric=True,
    )
    p = np.asarray(path, dtype=np.int64) + lo
    length = float(np.linalg.norm(np.diff(p * np.asarray(sp), axis=0), axis=1).sum()) if len(p) > 1 else 0.0
    uncovered = p[~m_roi[p[:, 0], p[:, 1], p[:, 2]]]
    if len(uncovered) == 0:
        return {"path_len_mm": length, "nhu_mean": 0.0, "nhu_min": 0.0, "vess_mean": 0.0,
                "contrast": 0.0, "evidence": 0.0}
    nh = nhu[uncovered[:, 0], uncovered[:, 1], uncovered[:, 2]]
    vv = vess[uncovered[:, 0], uncovered[:, 1], uncovered[:, 2]]
    contrast = float((nh >= CONTRAST_NHU).mean())
    vmean = float(vv.mean())
    return {"path_len_mm": length, "nhu_mean": float(nh.mean()), "nhu_min": float(nh.min()),
            "vess_mean": vmean, "contrast": contrast, "evidence": contrast * vmean}


def _track(start, direction, nhu, vess, m_roi, sp):
    """Greedy max-vesselness tracking outward from a tip. Returns path voxels (ROI coords)."""
    spn = np.asarray(sp, dtype=float)
    offs_mm = _OFFSETS * spn
    step_len = np.linalg.norm(offs_mm, axis=1)
    units = offs_mm / step_len[:, None]
    shape = np.asarray(vess.shape, dtype=np.int64)
    pos = np.asarray(start, dtype=np.int64)
    d = np.asarray(direction, dtype=float)
    start_mm = pos * spn
    path, reach = [], 0.0  # reach = furthest uncovered distance (mm) from the tip
    while reach < TRACK_MAX_MM and len(path) < TRACK_MAX_STEPS:
        cone = np.flatnonzero(units @ d > TRACK_CONE_COS)
        if cone.size == 0:
            break
        q = pos + _OFFSETS[cone]
        inb = np.all((q >= 0) & (q < shape), axis=1)
        cone, q = cone[inb], q[inb]
        if cone.size == 0:
            break
        vals = vess[q[:, 0], q[:, 1], q[:, 2]]
        k = int(np.argmax(vals))
        if vals[k] < TRACK_STOP_VESS:
            break
        j = cone[k]
        covered = bool(m_roi[q[k, 0], q[k, 1], q[k, 2]])
        # Continuation must stay bright: stop at the first uncovered voxel below the contrast
        # threshold, so that a blunt tube end is not followed through background.
        if not covered and nhu[q[k, 0], q[k, 1], q[k, 2]] < CONTRAST_NHU:
            break
        # Covered voxels are passed through (the tip sits inside the prediction) but do not count
        # toward the 20 mm budget; only uncovered path length is reported as continuation.
        pos = q[k].copy()
        path.append(pos)
        d = units[j]
        if not covered:
            # Distance from the tip, not summed step length, so lateral zig-zags do not inflate it.
            reach = max(reach, float(np.linalg.norm(pos * spn - start_mm)))
    return np.asarray(path, dtype=np.int64).reshape(-1, 3), reach


def _track_features(path, length, nhu, vess, m_roi):
    """Continuation features over the uncovered part of a tracked path."""
    if len(path) == 0:
        return {"cont_len_mm": 0.0, "contrast": 0.0, "evidence": 0.0}
    unc = path[~m_roi[path[:, 0], path[:, 1], path[:, 2]]]
    if len(unc) == 0 or length < TRACK_MIN_MM:
        return {"cont_len_mm": length, "contrast": 0.0, "evidence": 0.0}
    nh = nhu[unc[:, 0], unc[:, 1], unc[:, 2]]
    vv = vess[unc[:, 0], unc[:, 1], unc[:, 2]]
    contrast = float((nh >= CONTRAST_NHU).mean())
    return {"cont_len_mm": length, "contrast": contrast, "evidence": contrast * float(vv.mean())}


def evidence_features(mask, hu, spacing, aorta=None):
    """Scan-level M5 features for one predicted mask and its CT.

    mask: boolean prediction; hu: CT in HU on the same grid; spacing: mm per axis;
    aorta: optional boolean aorta mask (TotalSegmentator) used for the reference HU.
    """
    sp = tuple(float(s) for s in spacing)
    m = np.asarray(mask) > 0
    hu = np.asarray(hu, dtype=np.float32)
    if hu.shape != m.shape:
        raise ValueError(f"CT shape {hu.shape} does not match mask shape {m.shape}")
    if not m.any():
        return _empty()

    sl = bbox_slices(m, sp, pad_mm=ROI_PAD_MM)
    origin = np.array([s.start for s in sl], dtype=np.int64)
    m_roi = m[sl]
    hu_roi = hu[sl]
    ref, fallback = reference_hu(hu, aorta, hu_roi, m_roi, sp)
    nhu = np.clip(hu_roi / ref, *NHU_CLIP)
    vess = vesselness(nhu, sp)
    out = _empty(ref, fallback)

    B = branch_graph(m, sp, prune_mm=PRUNE_MM)
    kept = [nodes for nodes, length in components(B) if length >= MIN_COMPONENT_MM]
    kept_of = {n: cid for cid, nodes in enumerate(kept) for n in nodes}

    ends = []
    for node, cid in kept_of.items():
        if B.degree(node) != 1:
            continue
        res = _endpoint_direction(B, node, sp)
        if res is None:
            continue
        direction, pos = res
        radius = float(next(iter(B.edges(node, data=True)))[2]["mean_radius"])
        ends.append({"node": node, "cid": cid, "dir": direction, "pos": pos, "radius": radius})

    gap_rows = []
    for i, j in _gap_pairs(ends):
        a = np.asarray(ends[i]["node"], dtype=np.int64) - origin
        b = np.asarray(ends[j]["node"], dtype=np.int64) - origin
        row = _route_gap(a, b, nhu, vess, m_roi, sp)
        row["major"] = max(ends[i]["radius"], ends[j]["radius"]) >= MAJOR_RADIUS_MM
        gap_rows.append(row)

    tip_rows = []
    for e in ends:
        start = np.asarray(e["node"], dtype=np.int64) - origin
        path, length = _track(start, e["dir"], nhu, vess, m_roi, sp)
        row = _track_features(path, length, nhu, vess, m_roi)
        row["major"] = e["radius"] >= MAJOR_RADIUS_MM
        tip_rows.append(row)

    out["m5_n_gaps"] = float(len(gap_rows))
    if gap_rows:
        ev = np.array([r["evidence"] for r in gap_rows])
        k = int(np.argmax(ev))
        out["m5_gap_evidence_max"] = float(ev[k])
        out["m5_gap_contrast_max"] = float(max(r["contrast"] for r in gap_rows))
        out["m5_gap_vess_max"] = float(max(r["vess_mean"] for r in gap_rows))
        out["m5_gap_nhu_mean_max"] = float(max(r["nhu_mean"] for r in gap_rows))
        out["m5_gap_path_len_at_max_mm"] = float(gap_rows[k]["path_len_mm"])
        out["m5_gap_major_evidence_sum"] = float(sum(r["evidence"] for r in gap_rows if r["major"]))

    out["m5_n_tips"] = float(len(tip_rows))
    if tip_rows:
        out["m5_tip_evidence_max"] = float(max(r["evidence"] for r in tip_rows))
        bright = [r["cont_len_mm"] for r in tip_rows if r["contrast"] >= CONTRAST_NHU and r["evidence"] > 0]
        out["m5_tip_cont_len_max_mm"] = float(max(bright)) if bright else 0.0
        major = [r["evidence"] for r in tip_rows if r["major"]]
        out["m5_tip_major_evidence_max"] = float(max(major)) if major else 0.0
    return out
