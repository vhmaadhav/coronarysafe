"""M6 proximal-connectivity features (Addendum F). Label-free, aorta-rooted, per tree. Never GT.

Per tree (left/right, from the ostium contact side as in features.ostium):
  attached            an ostium-attached component on this side (touches the dilated aorta)
  reach50_mm/reach80  skeleton length reachable from the ostium within 50 / 80 mm geodesic
  n_prox_gaps         gaps whose BOTH endpoints lie <= 60 mm (Euclidean) from this tree's ostium
  prox_gap_evid       max M5 gap evidence over those gaps (evidence.py _route_gap)
  radius_drop_60      max drop of the running-max smoothed radius over the first 60 mm,
                      among branches with mean radius >= 1.25 mm
  share_attached      length of this tree's attached components / total skeleton length
Scan level: worst tree per feature (min for reach/share, max for gaps/evidence/drop), plus the sum of
gap counts over both trees (m6_sum_prox_gaps).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy import ndimage

from ..geometry import bbox_slices
from ..graph import branch_graph, components
from .evidence import (
    MIN_COMPONENT_MM,
    MAJOR_RADIUS_MM,
    NHU_CLIP,
    PRUNE_MM,
    ROI_PAD_MM,
    _gap_pairs,
    _route_gap,
    reference_hu,
    vesselness,
)
from .graph_features import _endpoint_direction

REACH_MM = (50.0, 80.0)
GAP_ENDPOINT_MM = 60.0
RADIUS_WINDOW_MM = 60.0
CONTACT_MM = 3.0
PROX_ROI_MM = 80.0
TREE_KEYS = ("attached", "reach50_mm", "reach80_mm", "n_prox_gaps", "prox_gap_evid",
             "radius_drop_60", "share_attached")


def _empty(prefix_only=False):
    out = {}
    for t in ("L", "R"):
        out.update({f"m6_{t}_{k}": 0.0 for k in TREE_KEYS})
    out.update({f"m6_worst_{k}": 0.0 for k in TREE_KEYS})
    out["m6_sum_prox_gaps"] = 0.0
    return out


def proximal_features(mask, hu, spacing, aorta, lr_axis, lr_sign):
    """M6 features for one predicted mask. hu: CT on the same grid; aorta: boolean TotalSegmentator aorta."""
    sp = tuple(float(s) for s in spacing)
    m = np.asarray(mask) > 0
    ao = np.asarray(aorta) > 0 if aorta is not None else np.zeros(m.shape, bool)
    hu = np.asarray(hu, dtype=np.float32)
    if not m.any():
        return _empty()

    B = branch_graph(m, sp, prune_mm=PRUNE_MM)
    kept = [(nodes, length) for nodes, length in components(B) if length >= MIN_COMPONENT_MM]
    if not kept:
        return _empty()
    cid_of = {n: c for c, (nodes, _) in enumerate(kept) for n in nodes}
    comp_len = [length for _, length in kept]
    total_len = float(sum(comp_len))

    if ao.any():
        d_ao = ndimage.distance_transform_edt(~ao, sampling=sp)
        ao_c = np.argwhere(ao).mean(0)
    else:
        d_ao = np.full(m.shape, np.inf)
        ao_c = None

    G = nx.Graph()
    for u, v, d in B.edges(data=True):
        if u == v or u not in cid_of or v not in cid_of:
            continue
        w = d["length_mm"]
        if G.has_edge(u, v):
            G[u][v]["weight"] = min(G[u][v]["weight"], w)
        else:
            G.add_edge(u, v, weight=w)

    # attachment per tree side: contact edges (voxels within CONTACT_MM of the aorta)
    roots = {"L": [], "R": []}
    attached_cids = {"L": set(), "R": set()}
    for u, v, d in B.edges(data=True):
        if u not in cid_of or v not in cid_of or u == v:
            continue
        vox = np.asarray(d["voxels"], dtype=np.int64)
        dv = d_ao[vox[:, 0], vox[:, 1], vox[:, 2]]
        if not np.isfinite(dv).any() or dv.min() > CONTACT_MM or ao_c is None:
            continue
        contact = vox[dv <= CONTACT_MM]
        off = contact.mean(0)[lr_axis] - ao_c[lr_axis]
        side = "L" if lr_sign * off > 0 else "R"
        root = u if d_ao[tuple(np.asarray(u, dtype=np.int64))] <= d_ao[tuple(np.asarray(v, dtype=np.int64))] else v
        roots[side].append(root)
        attached_cids[side].add(cid_of[u])

    # ends and gap pairs (M5 geometry; evidence routed only for proximal pairs)
    ends = []
    for node, cid in cid_of.items():
        if B.degree(node) != 1:
            continue
        res = _endpoint_direction(B, node, sp)
        if res is None:
            continue
        direction, pos = res
        radius = float(next(iter(B.edges(node, data=True)))[2]["mean_radius"])
        ends.append({"node": node, "cid": cid, "dir": direction, "pos": pos, "radius": radius})
    pairs = _gap_pairs(ends)

    # M5 maps on the PROXIMAL ROI only (voxels <= PROX_ROI_MM from the aorta, padded). Speed: the full-volume
    # Sato step dominated runtime (~3 min/unit). M6 defines its own vesselness normalisation on this ROI.
    prox_mask = m & (d_ao <= PROX_ROI_MM) if ao.any() else m
    sl = bbox_slices(prox_mask, sp, pad_mm=ROI_PAD_MM)
    if sl is None:
        sl = bbox_slices(m, sp, pad_mm=ROI_PAD_MM)
    origin = np.array([s.start for s in sl], dtype=np.int64)
    m_roi = m[sl]
    hu_roi = hu[sl]
    ref, _ = reference_hu(hu, ao, hu_roi, m_roi, sp)
    nhu = np.clip(hu_roi / ref, *NHU_CLIP)
    vess = vesselness(nhu, sp)

    # each gap belongs to the attached tree whose ostium is nearest its midpoint (one tree per gap)
    pos_os_of = {}
    for t in ("L", "R"):
        if roots[t]:
            pos_os_of[t] = np.mean([np.asarray(B.nodes[r]["pos"], float) for r in roots[t]], axis=0)
    owner = []
    for i, j in pairs:
        mid = (np.asarray(ends[i]["pos"], float) + np.asarray(ends[j]["pos"], float)) / 2.0
        if not pos_os_of:
            owner.append(None)
            continue
        owner.append(min(pos_os_of, key=lambda t: np.linalg.norm(mid - pos_os_of[t])))

    out = {}
    per_tree = {}
    for t in ("L", "R"):
        rec = {k: 0.0 for k in TREE_KEYS}
        if not roots[t]:
            per_tree[t] = rec
            continue
        rec["attached"] = 1.0
        src = roots[t]
        pos_os = pos_os_of[t]
        dist = nx.multi_source_dijkstra_path_length(G, src, cutoff=max(REACH_MM) + 1.0, weight="weight")
        tree_cids = attached_cids[t]

        # reach within 50/80 mm geodesic: partial edge portions counted
        reach = {d: 0.0 for d in REACH_MM}
        drop = 0.0
        for u, v, d in B.edges(data=True):
            if u == v or cid_of.get(u) not in tree_cids:
                continue
            du, dv = dist.get(u, np.inf), dist.get(v, np.inf)
            if not (np.isfinite(du) or np.isfinite(dv)):
                continue
            L = float(d["length_mm"])
            lo, hi = min(du, dv), max(du, dv)
            for dd in REACH_MM:
                if hi <= dd:
                    reach[dd] += L
                elif lo < dd:
                    reach[dd] += min(L, dd - lo)
            if lo <= RADIUS_WINDOW_MM and d["mean_radius"] >= MAJOR_RADIUS_MM:
                rs = ndimage.uniform_filter1d(np.asarray(d["radii"], float), size=5, mode="nearest")
                drop = max(drop, float(np.max(np.maximum.accumulate(rs) - rs)))
        rec["reach50_mm"] = reach[50.0]
        rec["reach80_mm"] = reach[80.0]
        rec["radius_drop_60"] = drop
        rec["share_attached"] = (sum(comp_len[c] for c in tree_cids) / total_len) if total_len > 0 else 0.0

        # proximal gaps: both endpoints within 60 mm of this tree's ostium
        n_g, ev_max = 0, 0.0
        for (i, j), own in zip(pairs, owner):
            if own != t:
                continue
            a, b = ends[i], ends[j]
            da = np.linalg.norm(np.asarray(a["pos"], float) - pos_os)
            db = np.linalg.norm(np.asarray(b["pos"], float) - pos_os)
            if da > GAP_ENDPOINT_MM or db > GAP_ENDPOINT_MM:
                continue
            n_g += 1
            ai = np.asarray(a["node"], dtype=np.int64) - origin
            bi = np.asarray(b["node"], dtype=np.int64) - origin
            if np.any(ai < 0) or np.any(bi < 0) or np.any(ai >= m_roi.shape) or np.any(bi >= m_roi.shape):
                continue  # endpoint outside the proximal ROI: evidence not routed (counted as a gap, evidence 0)
            row = _route_gap(ai, bi, nhu, vess, m_roi, sp)
            ev_max = max(ev_max, float(row["evidence"]))
        rec["n_prox_gaps"] = float(n_g)
        rec["prox_gap_evid"] = ev_max
        per_tree[t] = rec

    for t in ("L", "R"):
        for k in TREE_KEYS:
            out[f"m6_{t}_{k}"] = float(per_tree[t][k])
    for k in TREE_KEYS:
        vals = [per_tree["L"][k], per_tree["R"][k]]
        out[f"m6_worst_{k}"] = float(min(vals) if k in ("attached", "reach50_mm", "reach80_mm", "share_attached")
                                     else max(vals))
    out["m6_sum_prox_gaps"] = float(per_tree["L"]["n_prox_gaps"] + per_tree["R"]["n_prox_gaps"])
    return out
