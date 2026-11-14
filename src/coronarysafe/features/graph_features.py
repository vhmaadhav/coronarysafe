"""Label-free topology and geometry features from a predicted binary mask."""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy import ndimage

from ..geometry import bbox_slices, dist_to
from ..graph import branch_graph, components

_TAPER_TOL = 0.2  # child branch may be 20% wider than its parent before it counts
_LEAK_RADIUS_MM = 3.0
_GAP_MAX_MM = 8.0
_GAP_COS = 0.7
_PROBE_MM = 3.0


def _nan_features(with_aorta: bool):
    out = {
        "n_components": 0.0,
        "n_components_ge_10mm": 0.0,
        "total_length_mm": 0.0,
        "len_comp1_mm": 0.0,
        "len_comp2_mm": 0.0,
        "share_top2": np.nan,
        "n_endpoints": 0.0,
        "n_junctions": 0.0,
        "max_branch_depth": 0.0,
        "radius_violations": 0.0,
        "mean_tortuosity": np.nan,
        "max_radius_mm": 0.0,
        "leak_fraction": np.nan,
        "n_gaps": 0.0,
    }
    if with_aorta:
        out["n_ostia"] = 0.0
        out["aorta_dist_mm"] = np.nan
    return out


def _endpoint_direction(B, node, sp):
    """Unit vector pointing out of the branch at an endpoint (index-mm), or None."""
    (_, _, d), = B.edges(node, data=True)
    vox = d["voxels"]
    if tuple(int(x) for x in vox[0]) != node:
        vox = vox[::-1]
    mm = vox * np.asarray(sp)
    steps = np.linalg.norm(np.diff(mm, axis=0), axis=1) if len(mm) > 1 else np.zeros(0)
    cum = np.concatenate([[0.0], np.cumsum(steps)])
    k = int(np.searchsorted(cum, _PROBE_MM))
    k = min(k, len(mm) - 1)
    if k == 0:
        return None
    v = mm[0] - mm[k]
    n = np.linalg.norm(v)
    return None if n < 1e-9 else v / n, mm[0]


def _count_gaps(B, kept_nodes_comp, sp):
    """Pairs of endpoints in different kept components, facing each other, <= 8 mm apart."""
    ends = []
    for node, cid in kept_nodes_comp.items():
        if B.degree(node) != 1:
            continue
        res = _endpoint_direction(B, node, sp)
        if res is None:
            continue
        direction, pos = res
        ends.append((cid, direction, pos))
    gaps = 0
    for i in range(len(ends)):
        ci, di, pi = ends[i]
        for j in range(i + 1, len(ends)):
            cj, dj, pj = ends[j]
            if ci == cj:
                continue
            diff = pj - pi
            dist = np.linalg.norm(diff)
            if dist <= 0 or dist > _GAP_MAX_MM:
                continue
            u = diff / dist
            if np.dot(di, u) > _GAP_COS and np.dot(dj, -u) > _GAP_COS:
                gaps += 1
    return gaps


def _tree_stats(B, nodes, tol):
    """Depth, radius-taper violations and root choice for the tree of one component."""
    ends = [n for n in nodes if B.degree(n) == 1]
    cand = ends or list(nodes)

    def best_radius(n):
        rs = [d["mean_radius"] for _, _, d in B.edges(n, data=True)]
        return max(rs) if rs else 0.0

    root = max(cand, key=best_radius)
    depth = {root: 0}
    parent_r = {root: None}
    queue = deque([root])
    violations = 0
    while queue:
        u = queue.popleft()
        for _, v, d in B.edges(u, data=True):
            if v == u or v in depth:
                continue
            depth[v] = depth[u] + 1
            if parent_r[u] is not None and d["mean_radius"] > (1.0 + tol) * parent_r[u]:
                violations += 1
            parent_r[v] = d["mean_radius"]
            queue.append(v)
    return max(depth.values()), violations


def compute_graph_features(
    mask,
    spacing,
    aorta_mask=None,
    *,
    prune_mm: float = 3.0,
    min_component_mm: float = 2.0,
) -> dict:
    """Topology / geometry features of a binary prediction (no ground truth used).

    Keys: n_components (length >= min_component_mm), n_components_ge_10mm,
    total_length_mm, len_comp1_mm, len_comp2_mm, share_top2, n_endpoints,
    n_junctions, max_branch_depth (hops from the largest-radius endpoint in the
    largest component), radius_violations (child branch > 1.2x parent radius,
    counted on the BFS tree), mean_tortuosity (path / chord per branch),
    max_radius_mm, leak_fraction (voxels with EDT > 3 mm / mask voxels), n_gaps
    (facing endpoint pairs in different components, <= 8 mm, cos > 0.7), and when
    aorta_mask is given: n_ostia and aorta_dist_mm (of the largest component).
    Ratios are NaN when undefined.
    """
    sp = tuple(float(s) for s in spacing)
    m = np.asarray(mask) > 0
    with_aorta = aorta_mask is not None
    if not m.any():
        return _nan_features(with_aorta)

    B = branch_graph(m, sp, prune_mm=prune_mm)
    comps = components(B)
    kept = [(nodes, length) for nodes, length in comps if length >= min_component_mm]
    lengths = [length for _, length in comps]
    total = float(sum(lengths))
    l1 = kept[0][1] if len(kept) > 0 else 0.0
    l2 = kept[1][1] if len(kept) > 1 else 0.0

    feats = {
        "n_components": float(len(kept)),
        "n_components_ge_10mm": float(sum(1 for _, length in kept if length >= 10.0)),
        "total_length_mm": total,
        "len_comp1_mm": float(l1),
        "len_comp2_mm": float(l2),
        "share_top2": float((l1 + l2) / total) if total > 0 else np.nan,
        "n_endpoints": float(sum(1 for n in B.nodes if B.degree(n) == 1)),
        "n_junctions": float(sum(1 for n in B.nodes if B.degree(n) >= 3)),
    }

    if kept:
        largest_nodes = kept[0][0]
        depth, viol = _tree_stats(B, largest_nodes, _TAPER_TOL)
    else:
        depth, viol = 0, 0
    feats["max_branch_depth"] = float(depth)
    feats["radius_violations"] = float(viol)

    tort = []
    for _, _, d in B.edges(data=True):
        vox = d["voxels"]
        chord = float(np.linalg.norm((vox[-1] - vox[0]) * np.asarray(sp)))
        if chord > 1e-6:
            tort.append(d["length_mm"] / chord)
    feats["mean_tortuosity"] = float(np.mean(tort)) if tort else np.nan

    sl = bbox_slices(m, sp, pad_mm=_LEAK_RADIUS_MM + 1.0)
    edt = ndimage.distance_transform_edt(m[sl], sampling=sp)
    feats["max_radius_mm"] = float(edt.max())
    feats["leak_fraction"] = float((edt > _LEAK_RADIUS_MM).sum() / m.sum())

    kept_comp_of = {}
    for cid, (nodes, _) in enumerate(kept):
        for n in nodes:
            kept_comp_of[n] = cid
    feats["n_gaps"] = float(_count_gaps(B, kept_comp_of, sp))

    if with_aorta:
        aorta = np.asarray(aorta_mask) > 0
        if aorta.any():
            union_sl = bbox_slices(m | aorta, sp, pad_mm=5.0)
            ao_dist = dist_to(aorta[union_sl], sp)
            origin = np.array([s.start for s in union_sl])

            def comp_min_dist(nodes):
                best = np.inf
                for n in nodes:
                    if B.degree(n) == 0:
                        p = np.asarray(n) - origin
                        best = min(best, float(ao_dist[p[0], p[1], p[2]]))
                    for _, _, d in B.edges(n, data=True):
                        v = d["voxels"] - origin
                        best = min(best, float(ao_dist[v[:, 0], v[:, 1], v[:, 2]].min()))
                return best

            touching = sum(1 for nodes, _ in kept if comp_min_dist(nodes) <= 2.0 + 1e-9)
            feats["n_ostia"] = float(touching)
            feats["aorta_dist_mm"] = float(comp_min_dist(largest_nodes)) if kept else np.nan
        else:
            feats["n_ostia"] = 0.0
            feats["aorta_dist_mm"] = np.nan
    return feats
