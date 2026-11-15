"""TEASAR centerline branch graph (kimimaro), Addendum E.

Output has the same layout as graph.branch_graph (MultiGraph; nodes are voxel tuples; edges carry
'voxels', 'radii', 'length_mm') plus 'pos_mm' (float centreline points, mm) used for the sub-voxel
cross-section radius. Spurs shorter than prune_mm at junctions are removed.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from ..graph import _prune_spurs

TEASAR = {"scale": 2.0, "const": 0.5}   # invalidation radius = scale * r + const (mm)


def teasar_branch_graph(mask, spacing, prune_mm=3.0, scale=None, const=None):
    import kimimaro

    sp = tuple(float(s) for s in spacing)
    B = nx.MultiGraph(spacing=sp)
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return B
    params = {"scale": TEASAR["scale"] if scale is None else scale,
              "const": TEASAR["const"] if const is None else const,
              "pdrf_scale": 100000, "pdrf_exponent": 4, "soma_acceptance_threshold": 3500,
              "soma_detection_threshold": 750, "soma_invalidation_const": 300,
              "soma_invalidation_scale": 2}
    res = kimimaro.skeletonize(m.astype(np.uint64), teasar_params=params, anisotropy=sp,
                               dust_threshold=0, progress=False, fix_branching=True,
                               fix_borders=True, parallel=1)
    if 1 not in res:
        return B
    S = res[1]
    V = np.asarray(S.vertices, dtype=float)      # mm, array axis order
    E = np.asarray(S.edges, dtype=np.int64)
    R = np.asarray(S.radii, dtype=float)
    if len(E) == 0:
        return B
    spa = np.asarray(sp)
    adj = [[] for _ in range(len(V))]
    for a, b in E:
        adj[a].append(b)
        adj[b].append(a)
    special = [i for i in range(len(V)) if len(adj[i]) != 2]
    key = {}
    for i in special:
        k = tuple(int(x) for x in np.rint(V[i] / spa))
        key.setdefault(k, i)
        B.add_node(k, pos=V[i].copy())
    node_of = {i: tuple(int(x) for x in np.rint(V[i] / spa)) for i in special}
    seen = set()
    for s0 in special:
        for n in adj[s0]:
            if (s0, n) in seen:
                continue
            path = [s0, n]
            prev, cur = s0, n
            while cur not in node_of:
                nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
                prev, cur = cur, nxt
                path.append(cur)
            seen.add((s0, path[1]))
            seen.add((path[-1], path[-2]))
            pos = V[path]
            seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
            u, v = node_of[s0], node_of[path[-1]]
            if u == v and len(path) < 3:
                continue
            B.add_edge(u, v, length_mm=float(seg.sum()), voxels=np.rint(pos / spa).astype(np.int64),
                       pos_mm=pos, radii=R[path], mean_radius=float(R[path].mean()),
                       min_radius=float(R[path].min()))
    _prune_spurs(B, prune_mm)
    return B
