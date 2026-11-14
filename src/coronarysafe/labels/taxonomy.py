"""Failure taxonomy (taxonomy_v1). Uses ground truth: labels/evaluation only.

Flags
  F1  missed/broken segment: a contiguous run >= min_run_mm of GT centerline points that are
      UNCOVERED (nearest predicted lumen voxel farther than max(tolerance_mm, local GT radius)).
      Runs are followed through junctions along the GT branch graph, so a gap straddling a
      junction is one run. Severity: severe on a major segment, minor otherwise.
  F1b degraded (reported only, never counted in severe/minor): a contiguous run >= min_run_mm
      along one GT branch where the 8 mm cube local Dice is below the threshold.
  F2  disconnection: a GT tree whose points are covered (within cover_mm) by >= 2 predicted
      components, each covering >= min_run_mm.
  F3  leak: >= min_run_mm of predicted skeleton farther than far_mm from GT lumen (subtype
      leak), or one predicted component covering both trees (subtype bridge).
Severity of F1/F2/F3 events: severe if the involved segment is major, otherwise minor.
The segment of a run is the label at its midpoint along the longest path of the run.

Frame: centerline points are voxel indices of the GT array (nibabel order). Use
``io.world_to_index`` for VTK centerlines in world mm.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import yaml
from scipy import ndimage
from scipy.spatial import cKDTree

from ..geometry import bbox_slices, dist_to
from ..graph import branch_graph, skeletonize, voxel_graph

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "taxonomy_v1.yaml"
_TREE_NAMES = {0: "none", 1: "left", 2: "right"}
_EPS = 1e-9


def load_taxonomy_config(path=None) -> dict:
    with open(Path(path) if path else DEFAULT_CONFIG) as fh:
        return yaml.safe_load(fh)


def label_failures(
    gt_multilabel,
    gt_centerline_points,
    pred_mask,
    spacing,
    cfg=None,
    dominance=None,
) -> dict:
    """Label failures of one prediction against the GT.

    gt_multilabel: 15-class GT array (0 background, 1..14 segments).
    gt_centerline_points: None (derive from the GT skeleton), a single (N,3) array,
        or a sequence of (N_i,3) polylines in voxel-index coordinates.
    pred_mask: binary prediction array on the same grid.
    spacing: voxel size in mm per axis.
    dominance: optional 'left' or 'right'; inferred from GT PDA sample counts when None.
    """
    cfg = load_taxonomy_config() if cfg is None else cfg
    sp = np.asarray(spacing, dtype=float)
    gt = np.asarray(gt_multilabel).astype(np.int32)
    pred = np.asarray(pred_mask) > 0
    lumen = gt > 0
    if not lumen.any():
        raise ValueError("ground-truth mask has no labelled voxels")
    sl = bbox_slices(lumen | pred, sp, pad_mm=cfg["local_dice"]["cube_mm"] + 4.0)
    origin = np.array([s.start for s in sl], dtype=float)
    raw = _collect(gt[sl], pred[sl], sp, origin, gt_centerline_points, cfg)
    return _decide_all(raw, cfg, dominance)


# ---------------------------------------------------------------- collection


ROOT_LABEL = {1: 1, 2: 9}  # tree side -> root label: left tree from LM (1), right from RCA (9)


def _geodesic_from_ostia(pts, poly, links, sp, labels, sides, radius, linked):
    """Geodesic distance (mm) along the GT centerline graph from each tree's root.

    Root = a terminal sample of the tree labelled LM (left) or RCA (right), else the widest
    terminal of that tree. Returns (geo per sample, {side: root sample}).
    """
    n = len(pts)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n - 1):
        if poly[i] == poly[i + 1]:
            G.add_edge(i, i + 1, weight=float(np.linalg.norm((pts[i + 1] - pts[i]) * sp)))
    for a, b in links:
        a, b = int(a), int(b)
        G.add_edge(a, b, weight=float(np.linalg.norm((pts[a] - pts[b]) * sp)))
    geo = np.full(n, np.nan)
    roots = {}
    for side, root_lab in ROOT_LABEL.items():
        term = [i for i in range(n) if i not in linked and sides[i] == side]
        if not term:
            continue
        pref = [i for i in term if labels[i] == root_lab] or term
        root = max(pref, key=lambda i: radius[i])
        roots[side] = root
        for k, dist in nx.single_source_dijkstra_path_length(G, root, weight="weight").items():
            if sides[k] == side:
                geo[k] = dist
    return geo, roots


def _anchor(raw, idx):
    """Event anchor: the sample in idx closest (geodesically) to the ostium.

    Returns (sample index, dist_from_ostium_mm or None, gt radius mm at that sample).
    """
    idx = np.asarray(sorted(set(int(i) for i in idx)), dtype=np.int64)
    g = raw["geo"][idx]
    if not np.isfinite(g).any():
        j = int(idx[0])
        return j, None, float(raw["radius"][j])
    g = np.where(np.isfinite(g), g, np.inf)
    j = int(idx[int(np.argmin(g))])
    return j, float(raw["geo"][j]), float(raw["radius"][j])


def _nearest_anchor(raw, centroid_local):
    """Anchor for a predicted run: nearest GT centerline sample to its centroid."""
    _, j = raw["pts_tree"].query(np.asarray(centroid_local) * raw["sp"])
    j = int(j)
    g = raw["geo"][j]
    return j, (float(g) if np.isfinite(g) else None), float(raw["radius"][j])



def _collect(gt_c, pred_c, sp, origin, points_in, cfg):
    cc = cfg["centerline"]
    step = float(cc["step_mm"])
    cube = float(cfg["local_dice"]["cube_mm"])
    lumen_c = gt_c > 0
    labelled_xyz = np.argwhere(lumen_c)
    tree_lab = cKDTree(labelled_xyz * sp)
    side_vox = _side_of_labelled(gt_c, labelled_xyz, cfg)
    edt_lumen = ndimage.distance_transform_edt(lumen_c, sampling=tuple(sp))

    polys, source = _polylines(gt_c, points_in, origin, sp, cc)
    pts_list, pid_list = [], []
    first, last = {}, {}
    offset = 0
    for pid, p in enumerate(polys):
        r = _resample(p, sp, step)
        if len(r) == 0:
            continue
        pts_list.append(r)
        pid_list.append(np.full(len(r), pid))
        first[pid] = offset
        last[pid] = offset + len(r) - 1
        offset += len(r)
    if not pts_list:
        raise ValueError("no centerline points available")
    pts = np.vstack(pts_list)
    poly = np.concatenate(pid_list)
    links = _junction_links(polys, first, last)

    _, j = tree_lab.query(pts * sp)
    labels = gt_c[tuple(labelled_xyz[j].T)]
    sides = side_vox[j]
    dice = _local_dice(gt_c, pred_c, pts, sp, cube)
    idx_r = np.clip(np.rint(pts).astype(int), 0, np.array(gt_c.shape) - 1)
    radius = edt_lumen[tuple(idx_r.T)]

    pred_lab, _ = ndimage.label(pred_c, structure=np.ones((3, 3, 3)))
    if pred_c.any():
        pv = np.argwhere(pred_c)
        d, jp = cKDTree(pv * sp).query(pts * sp)
        comp = pred_lab[tuple(pv[jp].T)]
    else:
        d = np.full(len(pts), np.inf)
        comp = np.zeros(len(pts), dtype=np.int64)

    linked = {int(i) for pair in links for i in pair}
    term_idx = []
    for pid in first:
        for idx in (first[pid], last[pid]):
            if idx not in linked:
                term_idx.append(idx)
    term_mm = pts[term_idx] * sp if term_idx else np.zeros((0, 3))
    geo, roots = _geodesic_from_ostia(pts, poly, links, sp, labels, sides, radius, set(linked))
    far_runs = _far_runs(
        pred_c, lumen_c, sp, cfg["f3"]["far_mm"], tree_lab, labelled_xyz, gt_c, term_mm
    )
    return {
        "pts": pts,
        "poly": poly,
        "links": links,
        "labels": labels,
        "sides": sides,
        "dice": dice,
        "radius": radius,
        "comp": comp,
        "cover_dist": d,
        "far_runs": far_runs,
        "term_mm": term_mm,
        "geo": geo,
        "roots": roots,
        "pts_tree": cKDTree(pts * sp),
        "step": step,
        "sp": sp,
        "source": source,
    }


def _polylines(gt_c, points_in, origin, sp, cc):
    if points_in is None:
        B = branch_graph(gt_c > 0, sp, prune_mm=cc["gt_skeleton_prune_mm"])
        polys = [np.asarray(d["voxels"], dtype=float) for _, _, d in B.edges(data=True)]
        return polys, "gt_skeleton"
    if isinstance(points_in, np.ndarray) and points_in.ndim == 2:
        seq = [points_in]
    else:
        seq = list(points_in)
    polys = [np.asarray(p, dtype=float) - origin for p in seq if len(p) > 0]
    return polys, "provided"


def _junction_links(polys, first, last):
    """Pairs of sample indices at polyline ends that share a coordinate (junctions)."""
    by_key = {}
    for pid in first:
        ends = ((polys[pid][0], first[pid]), (polys[pid][-1], last[pid]))
        for raw, idx in ends:
            key = tuple(np.round(np.asarray(raw, dtype=float), 3))
            by_key.setdefault(key, []).append(idx)
    return [(a, b) for v in by_key.values() for a, b in combinations(v, 2) if a != b]


def _resample(poly, sp, step):
    """Resample a polyline every ``step`` mm of arc length, always keeping both end points."""
    poly = np.asarray(poly, dtype=float)
    if len(poly) < 2:
        return poly.reshape(-1, 3)
    seg = np.linalg.norm(np.diff(poly * sp, axis=0), axis=1)
    keep = np.concatenate([[True], seg > _EPS])
    poly = poly[keep]
    if len(poly) < 2:
        return poly[:1]
    seg = np.linalg.norm(np.diff(poly * sp, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    targets = np.append(np.arange(0.0, total - _EPS, step), total)
    return np.column_stack([np.interp(targets, s, poly[:, a]) for a in range(3)])


def _side_of_labelled(gt_c, labelled_xyz, cfg):
    """Side per labelled voxel: 1 left, 2 right, 0 none. 'Other' components go to the tree
    whose labels touch them most."""
    left = list(cfg["left_labels"])
    right = list(cfg["right_labels"])
    other = int(cfg["other_label"])
    lab = gt_c[tuple(labelled_xyz.T)]
    side = np.zeros(len(lab), dtype=np.int64)
    side[np.isin(lab, left)] = 1
    side[np.isin(lab, right)] = 2
    other_mask = gt_c == other
    if other_mask.any():
        comp, n = ndimage.label(other_mask, structure=np.ones((3, 3, 3)))
        comp_side = np.zeros(n + 1, dtype=np.int64)
        struct = np.ones((3, 3, 3), dtype=bool)
        for c in range(1, n + 1):
            m = comp == c
            shell = ndimage.binary_dilation(m, structure=struct) & ~m
            touched = gt_c[shell]
            nl = int(np.isin(touched, left).sum())
            nr = int(np.isin(touched, right).sum())
            if nl > nr:
                comp_side[c] = 1
            elif nr > nl:
                comp_side[c] = 2
        sel = lab == other
        side[sel] = comp_side[comp[tuple(labelled_xyz[sel].T)]]
    return side


def _local_dice(gt_c, pred_c, pts, sp, cube_mm):
    """Dice of GT lumen vs prediction inside a cube of side cube_mm around each point."""
    half = [max(1, int(round((cube_mm / 2.0) / s))) for s in sp]
    shape = gt_c.shape
    out = np.empty(len(pts))
    lumen = gt_c > 0
    for i, p in enumerate(np.rint(pts).astype(int)):
        sl = []
        for a in range(3):
            c = min(max(p[a], 0), shape[a] - 1)
            sl.append(slice(max(0, c - half[a]), min(shape[a], c + half[a] + 1)))
        sl = tuple(sl)
        g = lumen[sl]
        q = pred_c[sl]
        denom = g.sum() + q.sum()
        out[i] = 1.0 if denom == 0 else 2.0 * np.logical_and(g, q).sum() / denom
    return out


def _far_runs(pred_c, lumen_c, sp, far_mm, tree_lab, labelled_xyz, gt_c, term_mm):
    """Connected runs of predicted skeleton farther than far_mm from the GT lumen.

    Each run records: length_mm, label (nearest GT segment), centroid, radius_mm (mean EDT
    of the predicted mask on the run), connected (touches predicted skeleton within far_mm),
    term_dist_mm (nearest GT terminal point), dist_gt_min_mm (closest approach to GT lumen).
    """
    if not pred_c.any() or not lumen_c.any():
        return []
    skel = skeletonize(pred_c)
    dl = dist_to(lumen_c, sp)
    far = skel & (dl > far_mm)
    if not far.any():
        return []
    near = skel & ~far
    near_dil = ndimage.binary_dilation(near, structure=np.ones((3, 3, 3)))
    edt_p = ndimage.distance_transform_edt(pred_c, sampling=tuple(sp))
    term_tree = cKDTree(term_mm) if len(term_mm) else None
    G = voxel_graph(far, sp)
    runs = []
    for nodes in nx.connected_components(G):
        sub = G.subgraph(nodes)
        length = float(sum(d["length"] for _, _, d in sub.edges(data=True)))
        v = np.asarray(list(nodes), dtype=np.int64)
        coords = v.astype(float)
        centroid = coords.mean(axis=0)
        _, j = tree_lab.query(centroid * sp)
        term_d = float(term_tree.query(coords * sp)[0].min()) if term_tree is not None else np.inf
        runs.append(
            {
                "length_mm": length,
                "label": int(gt_c[tuple(labelled_xyz[j])]),
                "centroid": centroid,
                "radius_mm": float(edt_p[v[:, 0], v[:, 1], v[:, 2]].mean()),
                "connected": bool(near_dil[v[:, 0], v[:, 1], v[:, 2]].any()),
                "term_dist_mm": term_d,
                "dist_gt_min_mm": float(dl[v[:, 0], v[:, 1], v[:, 2]].min()),
            }
        )
    return runs


# ------------------------------------------------------------------ decisions


def _infer_dominance(labels, cfg):
    n_r = int(np.sum(labels == cfg["pda_labels"]["right"]))
    n_l = int(np.sum(labels == cfg["pda_labels"]["left"]))
    if n_r > n_l:
        return "right"
    if n_l > n_r:
        return "left"
    return None


def _uncovered_runs(uncov, raw, min_run):
    """Connected runs of uncovered samples along GT paths (through junction links).

    Returns (length_mm, path) per run whose longest path is >= min_run; path is the list
    of sample indices of that longest path (the diameter of the run's tree).
    """
    pts, poly, sp = raw["pts"], raw["poly"], raw["sp"]
    n = len(uncov)
    G = nx.Graph()
    idx = np.flatnonzero(uncov)
    G.add_nodes_from(int(i) for i in idx)

    def mm(i, j):
        return float(np.linalg.norm((pts[i] - pts[j]) * sp))

    for i in idx:
        i = int(i)
        j = i + 1
        if j < n and uncov[j] and poly[j] == poly[i]:
            G.add_edge(i, j, weight=mm(i, j))
    for a, b in raw["links"]:
        if uncov[a] and uncov[b]:
            G.add_edge(int(a), int(b), weight=mm(a, b))

    runs = []
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        start = next(iter(comp))
        d1 = nx.single_source_dijkstra_path_length(sub, start, weight="weight")
        a = max(d1, key=d1.get)
        d2, paths = nx.single_source_dijkstra(sub, a, weight="weight")
        b = max(d2, key=d2.get)
        if d2[b] >= min_run - _EPS:
            runs.append((float(d2[b]), paths[b]))
    return runs


def _f3_v2_events(raw, cfg, min_run, names):
    """taxonomy_v2 far-skeleton rules (see configs/taxonomy_v2.yaml for the rationale)."""
    f3 = cfg["f3"]
    out = []
    for run in raw["far_runs"]:
        if run["length_mm"] < min_run - _EPS:
            continue
        lab = run["label"]
        _, dist_o, rad_o = _nearest_anchor(raw, run["centroid"])
        base = {
            "label": lab,
            "label_ids": [lab],
            "segment": names.get(lab, str(lab)),
            "segments": [names.get(lab, str(lab))],
            "dist_from_ostium_mm": dist_o,
            "gt_radius_mm": rad_o,
            "mean_gt_radius_mm": rad_o,
            "length_mm": float(run["length_mm"]),
            "radius_mm": float(run["radius_mm"]),
            "connected": bool(run["connected"]),
            "term_dist_mm": float(run["term_dist_mm"]),
            "dist_gt_min_mm": float(run["dist_gt_min_mm"]),
            "centroid_idx": run["centroid"].tolist(),
        }
        thin = run["radius_mm"] < f3["thin_radius_mm"]
        thick = run["radius_mm"] >= f3["thick_radius_mm"]
        if not run["connected"]:
            big = run["length_mm"] >= f3["detached_severe_length_mm"] or thick
            out.append(
                dict(base, flag="F3b", subtype="detached_fp", severity="severe" if big else "minor")
            )
        elif thin and run["term_dist_mm"] <= f3["terminal_mm"]:
            out.append(dict(base, flag="F3a", subtype="over_extension", severity="reported"))
        elif thin and run["dist_gt_min_mm"] <= f3["halo_mm"]:
            out.append(dict(base, flag="F3h", subtype="halo", severity="reported"))
        elif thick:
            out.append(dict(base, flag="F3c", subtype="leak_thick", severity="severe"))
        else:
            out.append(dict(base, flag="F3d", subtype="thin_spur", severity="minor"))
    return out


TIER_DEFAULTS = {"distance_mm": 50.0, "radius_mm": 1.25, "detached_radius_mm": 2.0}


def _tier(events, base_major, tier):
    """Critical tier (taxonomy_v3). Returns (critical, reasons).

    (a) F1 or F2 on a base major segment (LM, LAD, LCx, RCA; not PDA/PLA/side branches) whose
        start is within distance_mm of the ostium or whose GT radius is >= radius_mm;
    (b) any F3c (leak or bridge);
    (c) F3b detached false positive with predicted radius >= detached_radius_mm.
    """
    reasons = []
    for e in events:
        f = e["flag"]
        if f in ("F1", "F2"):
            if set(e.get("label_ids", [])) & base_major:
                d = e.get("dist_from_ostium_mm")
                r = e.get("gt_radius_mm")
                near = d is not None and d <= tier["distance_mm"]
                thick = r is not None and np.isfinite(r) and r >= tier["radius_mm"]
                if near or thick:
                    reasons.append(f"{f}:{e.get('segment', '+'.join(e.get('segments', [])))}")
        elif f == "F3c":
            reasons.append(f"F3c:{e.get('subtype')}")
        elif f == "F3b" and e.get("radius_mm", 0.0) >= tier["detached_radius_mm"]:
            reasons.append("F3b:detached_thick")
    return bool(reasons), reasons


def _burden(events, major):
    """Burden = sum over major F1 runs of length*r^2 (r = mean GT radius on the run) plus
    sum over major F2 breaks of 10 mm * r^2 (r at the break). missed_major_mm = major F1 length."""
    burden, missed = 0.0, 0.0
    for e in events:
        if e["flag"] == "F1" and e["label"] in major:
            burden += e["length_mm"] * (e["mean_gt_radius_mm"] or 0.0) ** 2
            missed += e["length_mm"]
        elif e["flag"] == "F2" and set(e.get("label_ids", [])) & major:
            burden += 10.0 * (e["gt_radius_mm"] or 0.0) ** 2
    return float(burden), float(missed)


def _decide(raw, cfg, min_run, dice_thr, dominance, tol, tier=None):
    step = raw["step"]
    labels, sides, dice = raw["labels"], raw["sides"], raw["dice"]
    comp, cover = raw["comp"], raw["cover_dist"]
    cover_mm = float(cfg["f2"]["cover_mm"])
    names = {int(k): str(v) for k, v in cfg["label_names"].items()}

    dom = dominance if dominance is not None else _infer_dominance(labels, cfg)
    pda = cfg["pda_labels"].get(dom) if dom is not None else None
    major = set(int(x) for x in cfg["major_base_labels"])
    if pda is not None:
        major.add(int(pda))

    def severity(labs):
        return "severe" if any(int(x) in major for x in labs) else "minor"

    events = []

    # F1: uncovered-point runs along GT paths (junction-spanning).
    uncov = cover > np.maximum(tol, raw["radius"])
    for length, path in _uncovered_runs(uncov, raw, min_run):
        mid = path[len(path) // 2]
        lab = int(labels[mid])
        anc, dist_o, rad_o = _anchor(raw, path)
        labs_path = sorted({int(labels[p]) for p in path})
        events.append(
            {
                "flag": "F1",
                "subtype": "uncovered_run",
                "label": lab,
                "label_ids": labs_path,
                "segment": names.get(lab, str(lab)),
                "segments": [names.get(x, str(x)) for x in labs_path],
                "tree": _TREE_NAMES[int(sides[mid])],
                "severity": severity([lab]),
                "length_mm": length,
                "labels_on_path": labs_path,
                "centroid_idx": raw["pts"][mid].tolist(),
                "dist_from_ostium_mm": dist_o,
                "gt_radius_mm": rad_o,
                "mean_gt_radius_mm": float(np.mean(raw["radius"][path])),
            }
        )

    # F1b: degraded (local Dice) contiguous runs along one branch. Reported, not scored.
    fail = dice < dice_thr
    poly = raw["poly"]
    n = len(labels)
    i = 0
    while i < n:
        if not fail[i]:
            i += 1
            continue
        j = i
        while (
            j + 1 < n
            and fail[j + 1]
            and poly[j + 1] == poly[i]
            and labels[j + 1] == labels[i]
        ):
            j += 1
        length = (j - i + 1) * step
        if length >= min_run - _EPS:
            lab = int(labels[(i + j) // 2])
            events.append(
                {
                    "flag": "F1b",
                    "subtype": "degraded_local_dice",
                    "label": lab,
                    "segment": names.get(lab, str(lab)),
                    "tree": _TREE_NAMES[int(sides[(i + j) // 2])],
                    "severity": "degraded",
                    "length_mm": float(length),
                    "min_dice": float(dice[i : j + 1].min()),
                }
            )
        i = j + 1

    # F2: a tree whose GT points are covered by >= 2 predicted components (each >= min_run).
    for side_val in (1, 2):
        in_tree = sides == side_val
        if not in_tree.any():
            continue
        covered = in_tree & (cover <= cover_mm)
        ids, counts = np.unique(comp[covered], return_counts=True)
        big = [
            (int(c), float(k) * step)
            for c, k in zip(ids, counts)
            if c > 0 and float(k) * step >= min_run - _EPS
        ]
        if len(big) < 2:
            continue
        main_c = max(big, key=lambda t: t[1])[0]
        others = [c for c, _ in big if c != main_c]
        sel = covered & np.isin(comp, others)
        labs = sorted(set(int(x) for x in labels[sel]))
        anc, dist_o, rad_o = _anchor(raw, np.flatnonzero(sel))
        events.append(
            {
                "flag": "F2",
                "subtype": "disconnection",
                "tree": _TREE_NAMES[side_val],
                "labels": labs,
                "label_ids": labs,
                "segments": [names.get(x, str(x)) for x in labs],
                "severity": severity(labs),
                "n_components": len(big),
                "length_mm": float(sel.sum()) * step,
                "dist_from_ostium_mm": dist_o,
                "gt_radius_mm": rad_o,
                "mean_gt_radius_mm": rad_o,
            }
        )

    v2 = cfg.get("f3_rules") == "v2"
    if v2:
        events += _f3_v2_events(raw, cfg, min_run, names)
    # F3 leak (v1): predicted skeleton far from GT lumen.
    for run in ([] if v2 else raw["far_runs"]):
        if run["length_mm"] >= min_run - _EPS:
            lab = run["label"]
            _, dist_o, rad_o = _nearest_anchor(raw, run["centroid"])
            events.append(
                {
                    "flag": "F3",
                    "subtype": "leak",
                    "label": lab,
                    "label_ids": [lab],
                    "segment": names.get(lab, str(lab)),
                    "segments": [names.get(lab, str(lab))],
                    "severity": severity([lab]),
                    "length_mm": float(run["length_mm"]),
                    "centroid_idx": run["centroid"].tolist(),
                    "dist_from_ostium_mm": dist_o,
                    "gt_radius_mm": rad_o,
                    "mean_gt_radius_mm": rad_o,
                    "radius_mm": float(run.get("radius_mm", np.nan)),
                }
            )

    # F3 bridge: one predicted component covers both trees.
    bmin = float(cfg["f3"]["bridge_min_coverage_mm"])
    for c in np.unique(comp[comp > 0]):
        cov = (comp == c) & (cover <= cover_mm)
        l_mm = float((cov & (sides == 1)).sum()) * step
        r_mm = float((cov & (sides == 2)).sum()) * step
        if l_mm >= bmin and r_mm >= bmin:
            labs = sorted(set(int(x) for x in labels[cov & np.isin(sides, [1, 2])]))
            anc_idx = np.flatnonzero(cov & np.isin(sides, [1, 2]))
            _, dist_o, rad_o = _anchor(raw, anc_idx)
            events.append(
                {
                    "flag": "F3c" if v2 else "F3",
                    "subtype": "bridge",
                    "labels": labs,
                    "label_ids": labs,
                    "segments": [names.get(x, str(x)) for x in labs],
                    "dist_from_ostium_mm": dist_o,
                    "gt_radius_mm": rad_o,
                    "mean_gt_radius_mm": rad_o,
                    "severity": "severe" if v2 else severity(labs),
                    "left_covered_mm": l_mm,
                    "right_covered_mm": r_mm,
                }
            )

    flags = sorted({e["flag"] for e in events})
    tier = {**TIER_DEFAULTS, **(cfg.get("tier") or {}), **(tier or {})}
    base_major = set(int(x) for x in cfg["major_base_labels"])
    critical, reasons = _tier(events, base_major, tier)
    burden, missed = _burden(events, major)
    return {
        "events": events,
        "critical": critical,
        "critical_reasons": reasons,
        "burden": burden,
        "missed_major_mm": missed,
        "tier": tier,

        "flags": flags,
        "severe": any(e["severity"] == "severe" for e in events),
        "minor": any(e["severity"] == "minor" for e in events),
        "dominance": dom,
        "major_labels": sorted(major),
        "min_run_mm": float(min_run),
        "dice_threshold": float(dice_thr),
        "uncovered_tolerance_mm": float(tol),
    }


def _decide_all(raw, cfg, dominance=None) -> dict:
    main = _decide(
        raw,
        cfg,
        float(cfg["min_run_mm"]),
        float(cfg["local_dice"]["threshold"]),
        dominance,
        float(cfg["f1"]["tolerance_mm"]),
    )
    names = {int(k): str(v) for k, v in cfg["label_names"].items()}
    segments = {}
    cover_mm = float(cfg["f2"]["cover_mm"])
    for lab in np.unique(raw["labels"]):
        m = raw["labels"] == lab
        segments[names.get(int(lab), str(lab))] = {
            "label": int(lab),
            "n_points": int(m.sum()),
            "median_dice": float(np.median(raw["dice"][m])),
            "min_dice": float(raw["dice"][m].min()),
            "covered_fraction": float((m & (raw["cover_dist"] <= cover_mm)).sum() / m.sum()),
        }
    sensitivity = {}
    for v in cfg.get("sensitivity") or []:
        tier_override = {}
        if "critical_distance_mm" in v:
            tier_override["distance_mm"] = float(v["critical_distance_mm"])
        if "critical_radius_mm" in v:
            tier_override["radius_mm"] = float(v["critical_radius_mm"])
        r = _decide(
            raw,
            cfg,
            float(v.get("min_run_mm", cfg["min_run_mm"])),
            float(v.get("dice_threshold", cfg["local_dice"]["threshold"])),
            dominance,
            float(v.get("tolerance_mm", cfg["f1"]["tolerance_mm"])),
            tier=tier_override,
        )
        sensitivity[v["name"]] = {
            "flags": r["flags"],
            "severe": r["severe"],
            "minor": r["minor"],
            "critical": r["critical"],
        }
    main["segments"] = segments
    main["sensitivity"] = sensitivity
    main["centerline_source"] = raw["source"]
    main["n_centerline_points"] = int(len(raw["labels"]))
    main["config_version"] = cfg["version"]
    return main
