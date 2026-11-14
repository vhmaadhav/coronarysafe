"""Skeleton graphs: 26-connected voxel graph and a collapsed branch graph.

Branch graph (networkx.MultiGraph)
  nodes: voxel tuples (global indices) of endpoints/junctions, attr ``pos`` (index-mm)
  edges: branches, attrs ``length_mm``, ``mean_radius``, ``min_radius`` (mm, from the
         Euclidean distance transform with the volume spacing), ``voxels`` (n,3) int
         ordered from one node to the other, ``radii`` (n,).
Graph attribute ``spacing`` holds the voxel spacing.
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize as _sk_skeletonize

from .geometry import bbox_slices

_LOG = logging.getLogger(__name__)

_HALF_26 = [
    (a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) > (0, 0, 0)
]


def skeletonize(mask):
    """3D thinning of a boolean mask (skimage).

    Guard: thinning can return an empty or very sparse skeleton for thick, even-width
    square-section structures. If the skeleton is empty or has fewer voxels than
    (mask voxels)^(1/3), the thinning is retried on a 1-voxel binary dilation of the mask
    and a warning is logged.
    """
    m = np.asarray(mask, dtype=bool)
    n = int(m.sum())
    if n == 0:
        return np.zeros(m.shape, dtype=bool)
    sk = np.asarray(_sk_skeletonize(m), dtype=bool)
    if sk.sum() == 0 or sk.sum() < n ** (1.0 / 3.0):
        _LOG.warning(
            "skeleton too sparse (%d voxels for a %d-voxel mask); retrying on 1-voxel dilated mask",
            int(sk.sum()),
            n,
        )
        dilated = ndimage.binary_dilation(m)
        sk = np.asarray(_sk_skeletonize(dilated), dtype=bool)
    return sk


def voxel_graph(skel, spacing):
    """26-connected graph over skeleton voxels; edge attr ``length`` in mm."""
    skel = np.asarray(skel, dtype=bool)
    sp = np.asarray(spacing, dtype=float)
    G = nx.Graph()
    coords = np.argwhere(skel)
    G.add_nodes_from(tuple(int(x) for x in c) for c in coords)
    shape = np.asarray(skel.shape)
    for off in _HALF_26:
        o = np.asarray(off)
        nb = coords + o
        ok = np.all((nb >= 0) & (nb < shape), axis=1)
        src = coords[ok]
        nb = nb[ok]
        hit = skel[nb[:, 0], nb[:, 1], nb[:, 2]]
        w = float(np.linalg.norm(o * sp))
        for a, b in zip(src[hit], nb[hit]):
            G.add_edge(tuple(int(x) for x in a), tuple(int(x) for x in b), length=w)
    return G


def _edge_attrs(vox, radii, sp):
    vox = np.asarray(vox, dtype=np.int64)
    radii = np.asarray(radii, dtype=float)
    length = 0.0
    if len(vox) > 1:
        length = float(np.linalg.norm(np.diff(vox * np.asarray(sp), axis=0), axis=1).sum())
    return {
        "length_mm": length,
        "mean_radius": float(radii.mean()),
        "min_radius": float(radii.min()),
        "voxels": vox,
        "radii": radii,
    }


def _orient(data, start):
    vox, rad = data["voxels"], data["radii"]
    if tuple(int(x) for x in vox[0]) == start:
        return vox, rad
    return vox[::-1], rad[::-1]


def _prune_spurs(B, min_mm):
    if min_mm <= 0:
        return
    changed = True
    while changed:
        changed = False
        for n in list(B.nodes):
            if n not in B or B.degree(n) != 1:
                continue
            u, v, _, d = next(iter(B.edges(n, keys=True, data=True)))
            other = v if u == n else u
            if other != n and B.degree(other) >= 3 and d["length_mm"] < min_mm:
                B.remove_node(n)
                changed = True


def _merge_degree2(B, sp):
    for n in list(B.nodes):
        if n not in B or B.degree(n) != 2:
            continue
        inc = list(B.edges(n, keys=True, data=True))
        if len(inc) != 2:
            continue
        (u1, v1, _, d1), (u2, v2, _, d2) = inc
        if u1 == v1 or u2 == v2:
            continue
        a = v1 if u1 == n else u1
        b = v2 if u2 == n else u2
        vox1, rad1 = _orient(d1, a)
        vox2, rad2 = _orient(d2, n)
        vox = np.vstack([vox1, vox2[1:]])
        rad = np.concatenate([rad1, rad2[1:]])
        B.remove_node(n)
        B.add_edge(a, b, **_edge_attrs(vox, rad, sp))


def branch_graph(mask, spacing, prune_mm: float = 0.0):
    """Collapse the skeleton of ``mask`` into a branch graph.

    Spurs (terminal branches shorter than ``prune_mm`` that end at a junction) are
    removed iteratively, then degree-2 nodes are merged.
    """
    sp = tuple(float(s) for s in spacing)
    B = nx.MultiGraph(spacing=sp)
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return B
    sl = bbox_slices(m, sp, pad_vox=2)
    sub = m[sl]
    origin = np.array([s.start for s in sl], dtype=np.int64)
    skel = skeletonize(sub)
    radius = ndimage.distance_transform_edt(sub, sampling=sp)
    G = voxel_graph(skel, sp)

    special = {n for n in G.nodes if G.degree(n) != 2}
    visited = set()

    def walk(s, n):
        path = [s, n]
        prev, cur = s, n
        while cur not in special:
            visited.add(cur)
            nxt = [q for q in G.neighbors(cur) if q != prev][0]
            prev, cur = cur, nxt
            path.append(cur)
        return path

    paths = []
    direct = set()
    for s in list(special):
        for n in G.neighbors(s):
            if n in special:
                key = frozenset((s, n))
                if key not in direct:
                    direct.add(key)
                    paths.append([s, n])
            elif n not in visited:
                paths.append(walk(s, n))

    # pure cycles without any endpoint or junction
    for c in list(G.nodes):
        if c in visited or c in special or G.degree(c) != 2:
            continue
        special.add(c)
        for n in G.neighbors(c):
            if n not in visited:
                paths.append(walk(c, n))
                break

    def glob(t):
        return tuple(int(x) for x in (np.asarray(t, dtype=np.int64) + origin))

    for n in special:
        g = glob(n)
        B.add_node(g, pos=np.asarray(g, dtype=float) * np.asarray(sp))
    for p in paths:
        arr = np.asarray(p, dtype=np.int64)
        radii = radius[arr[:, 0], arr[:, 1], arr[:, 2]]
        B.add_edge(glob(p[0]), glob(p[-1]), **_edge_attrs(arr + origin, radii, sp))

    _prune_spurs(B, prune_mm)
    _merge_degree2(B, sp)
    return B


def components(B):
    """List of (node_set, length_mm) for the branch graph, longest first."""
    out = []
    for nodes in nx.connected_components(B):
        length = 0.0
        for _, _, d in B.subgraph(nodes).edges(data=True):
            length += d["length_mm"]
        out.append((frozenset(nodes), length))
    out.sort(key=lambda t: -t[1])
    return out
