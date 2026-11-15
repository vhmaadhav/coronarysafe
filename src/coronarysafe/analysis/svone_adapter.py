"""svOneDSolver cross-check adapter (Addendum C/D).

Exports the element tree of hemo1d (after assign_outlets) to a legacy svOneDSolver input deck:
  * maximal single-child element chains become SEGMENTs (inlet/outlet area from the chain end radii);
  * each junction becomes a JOINT; two root branches are joined to a 0.01 cm trunk (segment 0);
  * inlet = PRESSURE (93 mmHg) on the trunk; each leaf = RESISTANCE with R_out from hemo1d;
  * units are CGS (cm, g, s) as in the svOneDSolver examples: dyn/cm^2, cm^3/s.
Parses the per-segment pressure output (last time step) into FFR at element outlets.
Known approximations (reported with the cross-check): linear area taper inside a chain,
area-based joint continuity instead of Murray split, and the output file naming assumed below.
"""

from __future__ import annotations

import glob
import os

import numpy as np

from .hemo1d import P_AORTA

CGS_PER_MMHG = 1333.22
CGS_PER_PA = 10.0
R_CGS_PER_SI = 1.0e-5           # Pa s / m^3  ->  dyn s / cm^5
MU_CGS = 0.035                  # poise (3.5 mPa s)
RHO_CGS = 1.06                  # g / cm^3


def chains(net):
    """Maximal chains of elements: a chain continues while the parent has exactly one child."""
    par = net["parent"]
    N = len(par)
    nch = np.bincount(par[par >= 0], minlength=N)
    chain_id = np.full(N, -1, dtype=int)
    out = []
    for e in range(N):
        p = par[e]
        if p >= 0 and nch[p] == 1:
            chain_id[e] = chain_id[p]
            out[chain_id[e]].append(e)
        else:
            chain_id[e] = len(out)
            out.append([e])
    return out, chain_id


def build_deck(net, name="model"):
    """Return (deck_text, meta). meta maps chain index -> segment id and element index lists."""
    par, R, L = net["parent"], net["R"], net["L"]
    Rout = net["Rout"]
    ch, chain_id = chains(net)
    C = len(ch)
    roots = [c for c in range(C) if par[ch[c][0]] < 0]
    if not roots:
        raise ValueError("tree has no roots")
    if len(roots) > 2:
        raise ValueError(f"{len(roots)} roots: the adapter handles one or two ostia only")
    # segment ids: trunk = 0 (added when two roots), chains follow
    trunk = len(roots) == 2
    seg_of = {c: (c + 1 if trunk else c) for c in range(C)}
    n_seg = C + (1 if trunk else 0)
    # nodes: 0 inlet; 1 trunk outlet (two roots) or the single root's inlet node; chain outlets follow
    out_node = {c: 2 + c for c in range(C)} if trunk else {c: 1 + c for c in range(C)}
    in_node = {}
    for c in range(C):
        p = par[ch[c][0]]
        if p < 0:
            in_node[c] = 1 if trunk else 0
        else:
            in_node[c] = out_node[chain_id[p]]
    n_nodes = (2 + C) if trunk else (1 + C)
    lines = [f"MODEL {name}", ""]
    coords = {}
    for c in range(C):
        last = ch[c][-1]
        coords[out_node[c]] = net["mid"][last] / 10.0  # mm -> cm
    coords.setdefault(0, np.zeros(3))
    coords.setdefault(1, np.zeros(3))
    for nid in range(n_nodes):
        x, y, z = coords.get(nid, np.zeros(3))
        lines.append(f"NODE {nid} {x:.6f} {y:.6f} {z:.6f}")
    lines.append("")
    tables = []
    if trunk:
        r_trunk = max(R[ch[c][0]] for c in roots) * 100.0
        area = np.pi * r_trunk ** 2
        lines.append(f"SEGMENT trunk 0 0.01 1 0 1 {area:.8f} {area:.8f} 0.0 MAT1 NONE 0.0 0 0 PRESSURE INLETDATA")
        joint_rows = [(1, 0, [seg_of[c] for c in roots])]
    else:
        joint_rows = []
    for c in range(C):
        first, last = ch[c][0], ch[c][-1]
        length = float(np.sum(L[ch[c]])) * 100.0
        a_in = np.pi * (R[first] * 100.0) ** 2
        a_out = np.pi * (R[last] * 100.0) ** 2
        if np.isfinite(Rout[last]) and not np.any(par == last):
            tname = f"R_{seg_of[c]}"
            tables.append((tname, [(0.0, Rout[last] * R_CGS_PER_SI), (10.0, Rout[last] * R_CGS_PER_SI)]))
            bc = f"RESISTANCE {tname}"
        else:
            bc = "NOBOUND NONE"
        lines.append(f"SEGMENT seg{seg_of[c]} {seg_of[c]} {length:.6f} {len(ch[c])} {in_node[c]} "
                     f"{out_node[c]} {a_in:.8f} {a_out:.8f} 0.0 MAT1 NONE 0.0 0 0 {bc}")
    # joints at nodes with two or more children
    child_map = {}
    for c in range(C):
        child_map.setdefault(in_node[c], []).append(c)
    for node, kids in child_map.items():
        if node == 0:
            continue
        if len(kids) < 2:
            continue
        inlet_seg = None
        for c in range(C):
            if out_node[c] == node:
                inlet_seg = seg_of[c]
        if inlet_seg is None:
            continue
        joint_rows.append((node, inlet_seg, [seg_of[c] for c in kids]))
    for k, (node, inlet_seg, outs) in enumerate(joint_rows):
        lines.append(f"JOINT J{k} {node} INSEGS{k} OUTSEGS{k}")
        lines.append(f"JOINTINLET INSEGS{k} 1 {inlet_seg}")
        lines.append(f"JOINTOUTLET OUTSEGS{k} {len(outs)} " + " ".join(str(o) for o in outs))
    lines.append("")
    lines.append("DATATABLE INLETDATA LIST")
    lines.append(f"0.0 {P_AORTA * CGS_PER_PA:.6f}")
    lines.append(f"10.0 {P_AORTA * CGS_PER_PA:.6f}")
    lines.append("ENDDATATABLE")
    for tname, rows in tables:
        lines.append(f"DATATABLE {tname} LIST")
        lines += [f"{t} {v:.6e}" for t, v in rows]
        lines.append("ENDDATATABLE")
    lines.append("")
    lines.append("SOLVEROPTIONS 0.001 100 1000 2 INLETDATA PRESSURE 1.0e-8 1 1")
    lines.append(f"MATERIAL MAT1 OLUFSEN {RHO_CGS} {MU_CGS} 0 2.0 1.0e15 -20 1e9")
    lines.append("OUTPUT TEXT")
    meta = {"chains": ch, "chain_id": chain_id, "seg_of": seg_of, "trunk": trunk, "n_segments": n_seg}
    return "\n".join(lines) + "\n", meta


def read_pressures(out_dir, seg_ids):
    """Last time-step pressure values (dyn/cm^2) for each segment id, read from text outputs.

    Output file naming is not documented: any file under out_dir whose name contains 'pressure' and
    the segment token (seg<id>, or 'trunk' for segment 0 when named so) is used.
    """
    vals = {}
    for sid in seg_ids:
        tokens = [f"seg{sid}", "trunk"] if sid == 0 else [f"seg{sid}"]
        cands = [f for f in glob.glob(os.path.join(out_dir, "**", "*"), recursive=True)
                 if os.path.isfile(f) and "pressure" in os.path.basename(f).lower()
                 and any(t in os.path.basename(f).lower() for t in tokens)]
        if not cands:
            raise FileNotFoundError(f"no pressure output for segment {sid} under {out_dir}")
        with open(cands[0]) as fh:
            rows = [ln for ln in fh.read().splitlines() if ln.strip()]
        vals[sid] = np.array([float(x) for x in rows[-1].split()])
    return vals


def ffr_by_element(net, meta, pressures):
    """FFR (P_outlet_of_element / P_ao) for every element, from segment node pressures.

    Element j (0-based) of a chain with n elements: node j+1 of the segment (node 0 = inlet).
    Returns (n_elements,) array, NaN where the output is missing.
    """
    N = len(net["parent"])
    out = np.full(N, np.nan)
    for c, elems in enumerate(meta["chains"]):
        sid = meta["seg_of"][c]
        if sid not in pressures:
            continue
        pv = pressures[sid]
        for j, e in enumerate(elems):
            idx = min(j + 1, len(pv) - 1)
            out[e] = pv[idx] / (P_AORTA * CGS_PER_PA)
    return np.clip(out, 0.0, 1.0)
