"""Reduced-order steady hyperaemic 1D model on a rooted coronary tree (Addendum C, fallback solver).

Descriptive, not a validation against invasive FFR. Pressure-flow coupled (standard reduced-order
FFR-CT form):

* Tree: skeleton branch graph rooted at the ostia; each branch is resampled to 0.5 mm elements.
* Element resistance: Poiseuille 8 mu L / (pi r^4) with the smoothed EDT radius, in series with a
  Young-Tsai stenosis loss (viscous + expansion, quadratic in Q) at each detected lesion.
* Lesion: smoothed radius r_s < 0.70 x reference r0, where r0 is the 90th percentile of r_s in a
  +-10 mm window; the run must be >= 2 mm; points within 1.5 x local radius of a branch end or
  junction are excluded from detection. Ls = run length with r_s < 0.90 r0.
* Radius smoothing: median (3 mm window) then Gaussian (sigma 1 mm) along each branch.
* Outlets: each leaf is a microvascular resistance R_out = (P_ao - P_v)/Q_ref with P_v = 0. Q_ref is
  the stenosis-free Murray-split flow from the total allometric flow Q = K V^0.75 (V in mL, Q in mL/s).
* Solve: the full nonlinear network (flows and nodal pressures) by Newton iteration with a sparse
  Jacobian. Root inlet pressure P_ao = 93 mmHg. Flow therefore falls as a stenosis tightens.
* FFR_sim = P_distal / P_ao, clipped to [0, 1].

SI units inside the solver (m, Pa, m^3/s); mm and mL outside.
"""

from __future__ import annotations

from collections import deque

import networkx as nx
import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from ..geometry import bbox_slices
from ..graph import branch_graph, components
from .harm import MATCH_MM, PAD_MM, SPUR_MM, nearest_gt_labels

MU = 3.5e-3                       # Pa s
RHO = 1060.0                      # kg/m^3
P_AORTA = 93.0 * 133.322368       # Pa (93 mmHg)
P_VEIN = 0.0                      # Pa (outlet pressure for R_out)
KT = 1.52
STENOSIS_FRAC = 0.70              # lesion if smoothed r < 0.70 r0
NARROW_FRAC = 0.90                # Ls: run with r < 0.90 r0
REF_HALF_MM = 10.0                # reference window (+-mm)
END_MARGIN = 1.5                  # ignore detection within 1.5 r of ends
MIN_LESION_MM = 2.0
MED_MM = 3.0                      # median filter window (mm)
SUBVOX_HALF_MM = 2.5              # cross-section disc half-width (mm)
SUBVOX_STEP_MM = 0.1              # cross-section sampling step (mm)
CEN_SIGMA_MM = 1.0                # centreline smoothing (mm)
GAUSS_MM = 1.0                    # gaussian sigma (mm)
FINE_MM = 0.25                    # profile sampling (mm)
H_MM = 0.5                        # element length (mm), fixed
MIN_R_MM = 0.1
CONTACT_MM = 3.0                  # ostium contact distance to the aorta (mm)
MIN_VESSEL_MM = 10.0              # FFR reported only for vessels with GT length >= 10 mm
DECISION = 0.80
FLOW_VESSELS = {"LAD": 2, "LCx": 3, "RCA": 9}
# Allometric flow constant (mL/s per mL^0.75). See K_FLOW below.
K_FLOW = 6.20  # starting point from pass-4 calibration; recalibrated for the TEASAR graph (Addendum E)


def _mode_nonzero(vals):
    nz = vals[vals > 0]
    if nz.size == 0:
        return 0
    return int(np.bincount(nz).argmax())


def crop_inputs(gt_labels, pred, aorta, spacing):
    """Crop GT labels, prediction and aorta (may be None) to one bounding box with PAD_MM margin."""
    m = (gt_labels > 0) | pred
    if aorta is not None:
        m = m | aorta
    sl = bbox_slices(m, spacing, pad_mm=PAD_MM)
    ao = None if aorta is None else np.asarray(aorta[sl], dtype=bool)
    return np.asarray(gt_labels)[sl], np.asarray(pred, dtype=bool)[sl], ao, sl


def pick_roots(B, aorta_dist=None):
    """Root nodes. Aorta-touching nodes when an aorta is available, else the largest-radius endpoint
    of each of the two largest components (fallback). Returns (roots, method)."""
    if aorta_dist is not None:
        roots = [n for n in B.nodes if aorta_dist[n] <= CONTACT_MM]
        return roots, "aorta" if roots else "aorta_no_contact"
    roots = []
    for nodes, _ in components(B)[:2]:
        best, best_r = None, -1.0
        for n in nodes:
            if B.degree(n) == 1:
                r = float(next(iter(B.edges(n, data=True)))[2]["radii"].max())
                if r > best_r:
                    best, best_r = n, r
        if best is None:
            best = max(nodes, key=lambda n: max(d["radii"].max() for _, _, d in B.edges(n, data=True)))
        roots.append(best)
    return roots, "fallback_largest_radius_endpoint"


def young_tsai(r0, rs, Ls):
    """Young-Tsai coefficients (SI): dP = a Q + b Q^2 for reference radius r0, minimum rs, length Ls."""
    A0 = np.pi * r0 ** 2
    As = np.pi * rs ** 2
    ratio = A0 / As
    Kv = 32.0 * Ls / (2.0 * r0) * ratio ** 2
    a = Kv * MU / (2.0 * np.pi * r0 ** 3)
    b = KT * RHO / (2.0 * A0 ** 2) * (ratio - 1.0) ** 2
    return a, b


def fine_grid(total_mm):
    n = max(2, int(np.ceil(total_mm / FINE_MM)) + 1)
    return np.linspace(0.0, total_mm, n)


def smooth_profile(sf, rf_raw):
    """Median (MED_MM) then Gaussian (GAUSS_MM) smoothing of a radius profile on a uniform grid (mm)."""
    ds = sf[1] - sf[0]
    k = max(1, int(round(MED_MM / ds)))
    rf = ndimage.median_filter(rf_raw, size=2 * k + 1, mode="nearest")
    rf = ndimage.gaussian_filter1d(rf, GAUSS_MM / ds, mode="nearest")
    return np.maximum(rf, MIN_R_MM)


def subvoxel_radius(mask_f, sp, centers, tangents):
    """Radius sqrt(A/pi) from the lumen cross-section on planes orthogonal to the centreline.

    mask_f: float mask (0/1) sampled with linear interpolation; lumen = iso-0.5. The area is the
    connected component containing the centre of each disc (discs touching their border are rejected).
    NaN where undefined; the caller falls back to the EDT radius.
    """
    n_side = int(round(2 * SUBVOX_HALF_MM / SUBVOX_STEP_MM)) + 1
    g = np.linspace(-SUBVOX_HALF_MM, SUBVOX_HALF_MM, n_side)
    X, Y = np.meshgrid(g, g, indexing="ij")
    cen = n_side // 2
    spa = np.asarray(sp, dtype=float)
    out = np.full(len(centers), np.nan)
    for i, (c, t) in enumerate(zip(centers, tangents)):
        ref = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(t, ref)
        nu = np.linalg.norm(u)
        if nu < 1e-9:
            continue
        u = u / nu
        v = np.cross(t, u)
        P = c[None, None, :] + X[..., None] * u + Y[..., None] * v
        idx = (P / spa).reshape(-1, 3).T
        val = ndimage.map_coordinates(mask_f, idx, order=1, mode="constant", cval=0.0).reshape(n_side, n_side)
        lab, _ = ndimage.label(val > 0.5)
        cl = lab[cen, cen]
        if cl == 0:
            continue
        comp = lab == cl
        if comp[0].any() or comp[-1].any() or comp[:, 0].any() or comp[:, -1].any():
            continue
        out[i] = np.sqrt(comp.sum() * SUBVOX_STEP_MM ** 2 / np.pi)
    return out


def branch_profile(sf, s, pos, rad_mm, mask_f, sp, radius="subvoxel"):
    """Raw radius (mm) on the fine grid sf: sub-voxel cross-section radius with EDT fallback."""
    r_edt = np.interp(sf, s, rad_mm)
    if radius == "edt" or mask_f is None:
        return r_edt
    ds = sf[1] - sf[0]
    cen = np.column_stack([np.interp(sf, s, pos[:, i]) for i in range(3)])
    cen = ndimage.gaussian_filter1d(cen, CEN_SIGMA_MM / ds, axis=0, mode="nearest")
    tan = np.gradient(cen, axis=0)
    nrm = np.linalg.norm(tan, axis=1, keepdims=True)
    tan = tan / np.maximum(nrm, 1e-12)
    r_sub = subvoxel_radius(mask_f, sp, cen, tan)
    return np.where(np.isfinite(r_sub), r_sub, r_edt)


def detect_lesions(sf, rf):
    """Lesions on a smoothed profile (mm arc, mm radius). Returns dicts with s, r0, rs, Ls (all mm)."""
    ds = sf[1] - sf[0]
    W = int(round(REF_HALF_MM / ds))
    ref = ndimage.percentile_filter(rf, 90, size=2 * W + 1, mode="nearest")
    margin = END_MARGIN * rf
    interior = (sf > margin) & (sf < sf[-1] - margin)
    cand = (rf < STENOSIS_FRAC * ref) & interior
    out = []
    n = len(rf)
    i = 0
    while i < n:
        if not cand[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and cand[j + 1]:
            j += 1
        if sf[j] - sf[i] + ds >= MIN_LESION_MM:
            m = i + int(np.argmin(rf[i:j + 1]))
            thr = NARROW_FRAC * ref[m]
            lo = hi = m
            while lo > 0 and rf[lo - 1] < thr:
                lo -= 1
            while hi < n - 1 and rf[hi + 1] < thr:
                hi += 1
            out.append({"s": float(sf[m]), "r0": float(ref[m]), "rs": float(rf[m]),
                        "Ls": float(sf[hi] - sf[lo] + ds)})
        i = j + 1
    return out


def _lesion_terms(sf, rf, total_mm, n):
    """Per-element Young-Tsai coefficients (SI) for an element split of a branch of length total_mm."""
    a_el = np.zeros(n)
    b_el = np.zeros(n)
    for les in detect_lesions(sf, rf):
        k = min(int(les["s"] / (total_mm / n)), n - 1)
        a, b = young_tsai(les["r0"] / 1000.0, les["rs"] / 1000.0, les["Ls"] / 1000.0)
        a_el[k] += a
        b_el[k] += b
    return a_el, b_el


def _add_edge(vox, rad_mm, sp, lab_vol, pin, h_mm, rec, mask_f=None, radius="subvoxel", pos_mm=None):
    vox = np.asarray(vox, dtype=np.int64)
    rad_mm = np.asarray(rad_mm, dtype=float)
    pos = vox * np.asarray(sp, dtype=float) if pos_mm is None else np.asarray(pos_mm, dtype=float)
    seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= 1e-9:
        return None
    n = max(1, int(round(total / h_mm)))
    bounds = np.linspace(0.0, total, n + 1)
    sf = fine_grid(total)
    rf = smooth_profile(sf, branch_profile(sf, s, pos, rad_mm, mask_f, sp, radius))
    a_el, b_el = _lesion_terms(sf, rf, total, n)
    labs = lab_vol[vox[:, 0], vox[:, 1], vox[:, 2]]
    prev = pin
    for k in range(n):
        a, b = bounds[k], bounds[k + 1]
        mid = 0.5 * (a + b)
        sel = (s >= a) & ((s < b) if k < n - 1 else (s <= b))
        if not sel.any():
            sel[int(np.argmin(np.abs(s - mid)))] = True
        rec["parent"].append(prev)
        rec["L"].append((b - a) / 1000.0)
        rec["R"].append(max(float(np.interp(mid, sf, rf)) / 1000.0, MIN_R_MM / 1000.0))
        rec["a"].append(a_el[k])
        rec["b"].append(b_el[k])
        rec["label"].append(_mode_nonzero(labs[sel].astype(np.int64)))
        rec["mid"].append([float(np.interp(mid, s, pos[:, i])) for i in range(3)])
        rec["k"].append(k)
        rec["n"].append(n)
        prev = len(rec["parent"]) - 1
    return prev


_REC_KEYS = ("parent", "L", "R", "a", "b", "label", "mid", "k", "n")


def build_tree(B, roots, sp, lab_vol, h_mm=H_MM, mask_f=None, radius="subvoxel"):
    """Spanning tree from the roots (BFS). Returns (net or None, visited node set)."""
    rec = {key: [] for key in _REC_KEYS}
    visited = set(roots)
    q = deque((r, -1) for r in roots)
    while q:
        u, pin = q.popleft()
        for _, v, _, d in B.edges(u, keys=True, data=True):
            if v == u or v in visited:
                continue
            vox, rad = d["voxels"], d["radii"]
            pos = d.get("pos_mm")
            if tuple(int(x) for x in vox[0]) != u:
                vox, rad = vox[::-1], rad[::-1]
                pos = None if pos is None else pos[::-1]
            visited.add(v)
            last = _add_edge(vox, rad, sp, lab_vol, pin, h_mm, rec, mask_f, radius, pos)
            if last is not None:
                q.append((v, last))
    if not rec["parent"]:
        return None, visited
    return _finalize(rec), visited


def _finalize(rec):
    parent = np.asarray(rec["parent"], dtype=int)
    N = len(parent)
    L = np.asarray(rec["L"], dtype=float)
    R = np.asarray(rec["R"], dtype=float)
    w = np.zeros(N)
    for p in np.unique(parent):
        idx = np.where(parent == p)[0]
        r3 = R[idx] ** 3
        w[idx] = r3 / r3.sum()
    depth = np.zeros(N)
    for e in range(N):
        depth[e] = (depth[parent[e]] if parent[e] >= 0 else 0.0) + L[e]
    has_child = np.zeros(N, dtype=bool)
    has_child[parent[parent >= 0]] = True
    a_arr = np.asarray(rec["a"], dtype=float)
    b_arr = np.asarray(rec["b"], dtype=float)
    return {"parent": parent, "L": L, "R": R, "w": w, "depth": depth,
            "a": a_arr, "b": b_arr, "n_stenoses": int(np.sum((a_arr > 0) | (b_arr > 0))),
            "label": np.asarray(rec["label"], dtype=int), "mid": np.asarray(rec["mid"], dtype=float),
            "leaf": ~has_child, "Rout": np.full(N, np.inf), "Q_ref": np.zeros(N)}


def chain_net(radii_m, lengths_m):
    """A single straight chain of equal elements (analytic and stenosis checks)."""
    radii_m = np.asarray(radii_m, dtype=float)
    lengths_m = np.asarray(lengths_m, dtype=float)
    N = len(radii_m)
    total = float(lengths_m.sum()) * 1000.0
    sf = (np.arange(2 * N) + 0.5) * (total / (2 * N))
    idx = np.minimum((sf / (total / N)).astype(int), N - 1)
    rf = radii_m[idx] * 1000.0
    a_el, b_el = _lesion_terms(sf, rf, total, N)
    rec = {"parent": [i - 1 for i in range(N)], "L": list(lengths_m), "R": list(radii_m),
           "a": list(a_el), "b": list(b_el), "k": list(range(N)), "n": [N] * N,
           "label": [0] * N, "mid": [[0.0, 0.0, 0.0]] * N}
    return _finalize(rec)


def volume_ml(net):
    return float(np.sum(np.pi * net["R"] ** 2 * net["L"]) * 1e6)


def assign_outlets(net, q_tot):
    """Stenosis-free Murray reference flows and leaf microvascular resistances (m^3/s input)."""
    parent, w = net["parent"], net["w"]
    N = len(parent)
    Qr = np.zeros(N)
    for e in range(N):
        p = parent[e]
        Qr[e] = (q_tot if p < 0 else Qr[p]) * w[e]
    net["Q_ref"] = Qr
    Rout = np.full(N, np.inf)
    Rout[net["leaf"]] = (P_AORTA - P_VEIN) / Qr[net["leaf"]]
    net["Rout"] = Rout
    return net


def solve(net, max_iter=60):
    """Newton solve of the nonlinear network. Returns (Q, x), x = outlet pressure per element (Pa)."""
    par, R, L, a, b, Rout = net["parent"], net["R"], net["L"], net["a"], net["b"], net["Rout"]
    N = len(par)
    has = par >= 0
    pc = par[has]
    cidx = np.where(has)[0]
    leaf = net["leaf"]
    inv_out = np.where(leaf, 1.0 / np.where(np.isfinite(Rout), Rout, 1.0), 0.0)
    scale = max(float(np.sum(net["Q_ref"][par < 0])), 1e-12)
    x = np.full(N, P_AORTA * 0.99)
    c = 8.0 * MU * L / (np.pi * R ** 4) + a
    for _ in range(max_iter):
        p_in = np.full(N, P_AORTA)
        p_in[has] = x[par[has]]
        dP = p_in - x
        disc = c * c + 4.0 * b * dP
        Qb = (-c + np.sqrt(np.maximum(disc, 0.0))) / np.where(b > 0, 2.0 * b, 1.0)
        Q = np.where(b > 0, Qb, dP / c)
        g = 1.0 / (c + 2.0 * b * Q)
        sumQc = np.bincount(pc, weights=Q[cidx], minlength=N)
        F = Q - sumQc - x * inv_out
        if np.max(np.abs(F)) < 1e-10 * scale:
            break
        diag = -g - np.bincount(pc, weights=g[cidx], minlength=N) - inv_out
        rows = np.concatenate([np.arange(N), pc, cidx])
        cols = np.concatenate([np.arange(N), cidx, pc])
        vals = np.concatenate([diag, g[cidx], g[cidx]])
        J = csr_matrix((vals, (rows, cols)), shape=(N, N))
        dx = spsolve(J, -F)
        step = np.max(np.abs(dx))
        if step > 0.2 * P_AORTA:
            dx *= 0.2 * P_AORTA / step
        x = x + dx
    p_in = np.full(N, P_AORTA)
    p_in[has] = x[par[has]]
    dP = p_in - x
    disc = c * c + 4.0 * b * dP
    Qb = (-c + np.sqrt(np.maximum(disc, 0.0))) / np.where(b > 0, 2.0 * b, 1.0)
    Q = np.where(b > 0, Qb, dP / c)
    return Q, x


def vessel_metrics(net, Q, x):
    out = {}
    for name, lab in FLOW_VESSELS.items():
        idx = np.where(net["label"] == lab)[0]
        if idx.size == 0:
            out[name] = {"length_mm": 0.0, "ffr_distal": np.nan, "ffr_min": np.nan,
                         "distal_mm": None, "proximal_flow_ml_s": np.nan,
                         "proximal_ref_flow_ml_s": np.nan}
            continue
        distal = idx[np.argmax(net["depth"][idx])]
        prox = idx[np.argmin(net["depth"][idx])]
        ffr = np.clip(x / P_AORTA, 0.0, 1.0)
        out[name] = {
            "length_mm": float(net["L"][idx].sum() * 1000.0),
            "ffr_distal": float(ffr[distal]),
            "ffr_min": float(ffr[idx].min()),
            "distal_mm": [float(v) for v in net["mid"][distal]],
            "proximal_flow_ml_s": float(Q[prox] * 1e6),
            "proximal_ref_flow_ml_s": float(net["Q_ref"][prox] * 1e6),
        }
    return out


DEFAULT_GRAPH = "teasar"  # Addendum E: kimimaro TEASAR centrelines (voxel skeleton was unstable)


def analyse_mask(mask, sp, lab_vol, aorta=None, h_mm=H_MM, k_flow=None, radius="subvoxel", graph=None):
    graph = DEFAULT_GRAPH if graph is None else graph
    """Build the rooted tree of one lumen mask (cropped frame) and solve it.

    lab_vol: int label volume in the same frame (nearest GT label, harm.nearest_gt_labels).
    Returns per-vessel metrics, rooting method, volumes, and a voxel cloud (mm, rooted) for the
    reachability test. An empty mask or no rooted tree gives NaN metrics.
    """
    sp = tuple(float(s) for s in sp)
    k_flow = K_FLOW if k_flow is None else k_flow
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        B = nx.MultiGraph()
    elif graph == "teasar":
        from .centerline import teasar_branch_graph
        B = teasar_branch_graph(mask, sp, prune_mm=SPUR_MM)
    else:
        B = branch_graph(mask, sp, prune_mm=SPUR_MM)
    nan_vessels = {name: {"length_mm": 0.0, "ffr_distal": np.nan, "ffr_min": np.nan,
                          "distal_mm": None, "proximal_flow_ml_s": np.nan,
                          "proximal_ref_flow_ml_s": np.nan} for name in FLOW_VESSELS}
    base = {"method": None, "n_roots": 0, "volume_ml": 0.0, "target_flow_ml_s": 0.0,
            "flow_ml_s": 0.0, "n_elements": 0, "n_stenoses": 0, "vessels": nan_vessels,
            "cloud_xyz": np.zeros((0, 3)), "cloud_rooted": np.zeros(0, dtype=bool)}
    if B.number_of_edges() == 0:
        return base
    ad = ndimage.distance_transform_edt(~aorta, sampling=sp) if (aorta is not None and aorta.any()) else None
    roots, method = pick_roots(B, ad)
    net, visited = build_tree(B, roots, sp, lab_vol, h_mm, mask.astype(float), radius)
    pts, flags = [], []
    for u, v, d in B.edges(data=True):
        pts.append(d["pos_mm"] if "pos_mm" in d else d["voxels"] * np.asarray(sp))
        flags.append(np.full(len(pts[-1]), u in visited))
    if pts:
        base["cloud_xyz"] = np.vstack(pts)
        base["cloud_rooted"] = np.concatenate(flags)
    base.update(method=method, n_roots=len(roots))
    if net is None:
        return base
    V = volume_ml(net)
    target = k_flow * V ** 0.75
    assign_outlets(net, target * 1e-6)
    Q, x = solve(net)
    base.update(volume_ml=V, target_flow_ml_s=target, flow_ml_s=float(Q[net["parent"] < 0].sum() * 1e6),
                n_elements=int(len(net["parent"])), n_stenoses=net["n_stenoses"],
                vessels=vessel_metrics(net, Q, x))
    return base


def compare_units(gt_res, pr_res, match_mm=MATCH_MM):
    """Per-vessel GT vs prediction. Only vessels with GT length >= 10 mm are eligible."""
    out = {}
    tree = cKDTree(pr_res["cloud_xyz"]) if len(pr_res["cloud_xyz"]) else None
    for name in FLOW_VESSELS:
        g = gt_res["vessels"][name]
        p = pr_res["vessels"][name]
        row = {"eligible": bool(g["length_mm"] >= MIN_VESSEL_MM and np.isfinite(g["ffr_distal"])),
               "gt_length_mm": g["length_mm"], "pred_length_mm": p["length_mm"],
               "gt_ffr_distal": g["ffr_distal"], "gt_ffr_min": g["ffr_min"],
               "pred_ffr_distal": p["ffr_distal"], "pred_ffr_min": p["ffr_min"],
               "unmeasurable": False, "missing": p["length_mm"] <= 0}
        if row["eligible"]:
            near_rooted = False
            if tree is not None and g["distal_mm"] is not None:
                dist, j = tree.query(np.asarray(g["distal_mm"]), k=1)
                near_rooted = bool(dist <= match_mm and pr_res["cloud_rooted"][j])
            row["unmeasurable"] = bool((not near_rooted) or p["length_mm"] <= 0)
            if row["unmeasurable"] or not np.isfinite(p["ffr_distal"]):
                row.update(d_ffr_distal=np.nan, d_ffr_min=np.nan, flip=None)
            else:
                row["d_ffr_distal"] = abs(p["ffr_distal"] - g["ffr_distal"])
                row["d_ffr_min"] = abs(p["ffr_min"] - g["ffr_min"])
                row["flip"] = bool((g["ffr_distal"] < DECISION) != (p["ffr_distal"] < DECISION))
        out[name] = row
    return out
