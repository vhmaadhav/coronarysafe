"""Addendum C verification of the 1D hyperaemic model (fallback solver).

Checks, all on the droplet (cs-infer):
  1. straight tube: Poiseuille dP within 1% of analytic;
  2. stenosis series 0-80% diameter: FFR decreases monotonically;
  3. skeleton step-size convergence (element length 0.5 / 1 / 2 mm) on train GT trees;
  4. GT test-retest: GT mask resampled to 0.45 mm and back, |dFFR| < 0.01 for >= 90% of vessels (20 train scans);
  5. plausibility: FFR_sim distribution on 50 train GT trees;
  calibration: K_FLOW such that the median LAD proximal flow over the 50 train GT trees is 3 mL/s.

Writes /root/data/cfd/verification.json and verification.md.
Usage: python pipeline/cfd_verify.py --root /root/data --out /root/data/cfd --calibrate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cfd_run import load_scan  # noqa: E402
from coronarysafe.analysis import hemo1d as h  # noqa: E402
from coronarysafe.analysis.harm import nearest_gt_labels  # noqa: E402

Q_CHECK = 3.0e-6          # m^3/s (3 mL/s)
TARGET_LAD_FLOW = 3.0     # mL/s
RESOLUTION_MM = 0.45
N_PLAUSIBILITY = 50
N_TEST_RETEST = 20
N_CONVERGENCE = 5


def check_tube():
    """Straight tube in series with a leaf resistance: Q = P_ao/(R_tube + R_out), dP_tube = R_tube Q."""
    L_mm, r_mm, N = 40.0, 1.5, 40
    r = np.full(N, r_mm / 1000.0)
    Ls = np.full(N, (L_mm / N) / 1000.0)
    R_tube = 8.0 * h.MU * (L_mm / 1000.0) / (np.pi * (r_mm / 1000.0) ** 4)
    R_out = 2.0 * R_tube
    net = h.chain_net(r, Ls)
    net["Rout"][-1] = R_out
    net["leaf"][:] = False
    net["leaf"][-1] = True
    Q, _ = h.solve(net)
    q_ana = h.P_AORTA / (R_tube + R_out)
    dp_ana = R_tube * q_ana
    dp_num = R_tube * Q[-1]
    rel_q = abs(Q[-1] - q_ana) / q_ana
    rel_dp = abs(dp_num - dp_ana) / dp_ana
    return {"flow_numeric_m3s": float(Q[-1]), "flow_analytic_m3s": float(q_ana),
            "dp_tube_numeric_pa": float(dp_num), "dp_tube_analytic_pa": float(dp_ana),
            "rel_error_flow": float(rel_q), "rel_error_dp": float(rel_dp),
            "pass": bool(max(rel_q, rel_dp) < 0.01)}


def check_stenosis_series():
    """Coupled solve on a 60 mm chain with a cosine lesion of 0-80% diameter reduction."""
    n, r0 = 120, 1.5e-3  # 0.5 mm elements, 60 mm
    x = (np.arange(n) + 0.5) * 0.5  # mm
    centre, half = 30.0, 5.0
    series = []
    for pct in range(0, 81, 10):
        frac = pct / 100.0
        rmin = r0 * (1.0 - frac)
        r = np.full(n, r0)
        inside = np.abs(x - centre) < half
        r[inside] = rmin + (r0 - rmin) * (1 - np.cos(np.pi * (x[inside] - centre) / half)) / 2.0
        net = h.chain_net(r, np.full(n, 0.5e-3))
        h.assign_outlets(net, Q_CHECK)
        Q, xp = h.solve(net)
        ffr = float(np.clip(xp[-1] / h.P_AORTA, 0.0, 1.0))
        series.append({"stenosis_pct": pct, "ffr": ffr, "flow_ml_s": float(Q[-1] * 1e6),
                       "n_lesions": int(net["n_stenoses"])})
    vals = [q["ffr"] for q in series]
    flows = [q["flow_ml_s"] for q in series]
    mono = all(b_ < a_ for a_, b_ in zip(vals[:-1], vals[1:]))
    mono_flow = all(b_ < a_ for a_, b_ in zip(flows[:-1], flows[1:]))
    in_range = all(0.0 < v <= 1.0 for v in vals)
    return {"series": series, "strictly_decreasing_ffr": bool(mono),
            "strictly_decreasing_flow": bool(mono_flow), "ffr_in_0_1": bool(in_range),
            "pass": bool(mono and mono_flow and in_range)}


def roundtrip(arr, sp, new_mm=RESOLUTION_MM):
    """Resample to new_mm (nearest neighbour) and back to the original grid."""
    arr = np.asarray(arr)
    shape = np.array(arr.shape)
    newshape = np.maximum(1, np.round(shape * np.asarray(sp) / new_mm)).astype(int)
    z = ndimage.zoom(arr, newshape / shape, order=0, grid_mode=False)
    back = ndimage.zoom(z, shape / np.array(z.shape), order=0)
    if back.shape != arr.shape:
        out = np.zeros(arr.shape, dtype=back.dtype)
        sl = tuple(slice(0, min(a, b)) for a, b in zip(arr.shape, back.shape))
        out[sl] = back[sl]
        back = out
    return back


def _scan_list(root, n, scan_file="m4/train_scans.txt", need_aorta=False):
    ids = [s.strip() for s in (Path(root) / scan_file).read_text().split() if s.strip()]
    out = []
    for s in ids:
        if need_aorta and not (Path(root) / "m4/aorta" / f"{s}.nii.gz").exists():
            continue
        if (Path(root) / "imagecas_x/ImageCAS-X_dataset_v3/segmentations" / f"{s}.coronary.nii.gz").exists():
            out.append(s)
        if len(out) >= n:
            break
    return out


def _gt_crop(root, scan):
    gt, sp, aorta = load_scan(root, scan)
    lumen = gt > 0
    from coronarysafe.analysis.hemo1d import crop_inputs
    g, _, ao, _ = crop_inputs(gt, np.zeros(gt.shape, dtype=bool), aorta, sp)
    return g, sp, ao


def check_convergence(root, scans):
    """Element length 0.25 mm vs the fixed 0.5 mm model (primary), plus 1.0 mm for information."""
    d_fine, d_coarse = [], []
    for s_ in scans:
        g, sp, ao = _gt_crop(root, s_)
        lab = nearest_gt_labels(g, sp)
        res = {hh: h.analyse_mask(g > 0, sp, lab, ao, h_mm=hh) for hh in (0.25, 0.5, 1.0)}
        for v in h.FLOW_VESSELS:
            a_, b_, c_ = (res[x]["vessels"][v]["ffr_distal"] for x in (0.25, 0.5, 1.0))
            if np.isfinite(a_) and np.isfinite(b_):
                d_fine.append(abs(a_ - b_))
            if np.isfinite(b_) and np.isfinite(c_):
                d_coarse.append(abs(b_ - c_))
    df, dc = np.asarray(d_fine), np.asarray(d_coarse)
    ok = bool(df.size and np.max(df) < 0.01)
    return {"scans": len(scans), "n_vessels": int(df.size),
            "max_abs_dffr_0.25_vs_0.5mm": float(df.max()) if df.size else None,
            "p95_abs_dffr_0.25_vs_0.5mm": float(np.percentile(df, 95)) if df.size else None,
            "median_abs_dffr_0.25_vs_0.5mm": float(np.median(df)) if df.size else None,
            "max_abs_dffr_0.5_vs_1mm_info": float(dc.max()) if dc.size else None,
            "pass": ok}


def _lin_roundtrip(mask, sp, new_mm=RESOLUTION_MM):
    """Linear-interpolation resample of the float mask to new_mm and back, thresholded at 0.5."""
    f = np.asarray(mask, dtype=float)
    shape = np.array(f.shape)
    newshape = np.maximum(1, np.round(shape * np.asarray(sp) / new_mm)).astype(int)
    z = ndimage.zoom(f, newshape / shape, order=1, grid_mode=False)
    back = ndimage.zoom(z, shape / np.array(z.shape), order=1)
    if back.shape != f.shape:
        out = np.zeros(f.shape)
        sl = tuple(slice(0, min(a, b)) for a, b in zip(f.shape, back.shape))
        out[sl] = back[sl]
        back = out
    return back > 0.5


def _shift_mask(mask, axis, sign, voxels=0.5):
    """Rigid sub-voxel shift along one axis (linear interpolation), thresholded at 0.5."""
    f = np.asarray(mask, dtype=float)
    grids = np.indices(f.shape).astype(float)
    grids[axis] = grids[axis] + sign * voxels
    return ndimage.map_coordinates(f, grids, order=1, mode="constant", cval=0.0) > 0.5


def _retest_delta(root_gt, sp, ao, g, mask2):
    """|dFFR| (distal) between the GT tree and a perturbed tree, eligible vessels only."""
    lab = nearest_gt_labels(g, sp)
    base = h.analyse_mask(g > 0, sp, lab, ao)
    g2 = np.where(mask2, nearest_gt_labels(g, sp), 0)
    rt = h.analyse_mask(mask2, sp, nearest_gt_labels(g2, sp), ao)
    per = h.compare_units(base, rt)
    return [r["d_ffr_distal"] for r in per.values() if r["eligible"] and not r["unmeasurable"]]


def _v2_scan(args):
    root, s_ = args
    g, sp, ao = _gt_crop(root, s_)
    m = g > 0
    da_ = _retest_delta(root, sp, ao, g, _lin_roundtrip(m, sp))
    db_ = []
    for axis in range(3):
        for sign in (-1, 1):
            db_ += _retest_delta(root, sp, ao, g, _shift_mask(m, axis, sign))
    return da_, db_


def check_test_retest_v2(root, scans, workers=8):
    """Geometry-preserving perturbations: (a) linear 0.45 mm round trip, (b) +-0.5 voxel shifts per axis."""
    from concurrent.futures import ProcessPoolExecutor
    da, db = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for a_, b_ in ex.map(_v2_scan, [(root, s_) for s_ in scans]):
            da += a_
            db += b_
    A, B = np.asarray(da), np.asarray(db)
    P = np.concatenate([A, B])

    def summ(x):
        return {"n": int(x.size), "frac_below_0.01": float(np.mean(x < 0.01)) if x.size else None,
                "median": float(np.median(x)) if x.size else None,
                "p90": float(np.percentile(x, 90)) if x.size else None,
                "p95": float(np.percentile(x, 95)) if x.size else None,
                "max": float(x.max()) if x.size else None}

    res = {"scans": len(scans), "a_linear_roundtrip_0.45mm": summ(A),
           "b_subvoxel_shift_0.5vox": summ(B), "pooled": summ(P)}
    res["noise_floor_p95"] = res["pooled"]["p95"]
    res["pass"] = bool(P.size and res["pooled"]["frac_below_0.01"] >= 0.90)
    return res


def check_test_retest(root, scans):
    deltas = []
    n_elig = 0
    for s in scans:
        g, sp, ao = _gt_crop(root, s)
        lab = nearest_gt_labels(g, sp)
        base = h.analyse_mask(g > 0, sp, lab, ao)
        g2 = roundtrip(g, sp)
        lab2 = nearest_gt_labels(g2, sp)
        rt = h.analyse_mask(g2 > 0, sp, lab2, ao)
        per = h.compare_units(base, rt)
        for v, r in per.items():
            if r["eligible"] and not r["unmeasurable"]:
                n_elig += 1
                deltas.append(r["d_ffr_distal"])
    d = np.asarray(deltas, dtype=float)
    frac = float(np.mean(d < 0.01)) if d.size else float("nan")
    return {"scans": len(scans), "n_vessels": int(n_elig), "frac_below_0.01": frac,
            "median_abs_dffr": float(np.median(d)) if d.size else None,
            "p95_abs_dffr": float(np.percentile(d, 95)) if d.size else None,
            "max_abs_dffr": float(d.max()) if d.size else None,
            "pass": bool(d.size and frac >= 0.90)}


def gt_distribution(root, scans):
    """FFR_sim of GT trees: distal and minimum FFR, plus rooting and aorta availability."""
    rows = {v: [] for v in h.FLOW_VESSELS}
    rowsmin = {v: [] for v in h.FLOW_VESSELS}
    flow_lad = []  # stenosis-free reference flow, K=1
    methods = {}
    ao_n = 0
    for s in scans:
        g, sp, ao = _gt_crop(root, s)
        ao_n += int(ao is not None)
        lab = nearest_gt_labels(g, sp)
        res = h.analyse_mask(g > 0, sp, lab, ao, k_flow=1.0)
        methods[res["method"]] = methods.get(res["method"], 0) + 1
        for v in h.FLOW_VESSELS:
            m = res["vessels"][v]
            if m["length_mm"] >= h.MIN_VESSEL_MM and np.isfinite(m["ffr_distal"]):
                rows[v].append(m["ffr_distal"])
                rowsmin[v].append(m["ffr_min"])
        if np.isfinite(res["vessels"]["LAD"]["proximal_ref_flow_ml_s"]):
            flow_lad.append(res["vessels"]["LAD"]["proximal_ref_flow_ml_s"])

    def summ(a):
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            return {"n": 0}
        p25, med, p75 = np.percentile(a, [25, 50, 75])
        return {"n": int(a.size), "median": float(med), "p25": float(p25), "p75": float(p75),
                "frac_gt_0.80": float(np.mean(a > 0.80)), "frac_lt_0.80": float(np.mean(a < 0.80))}

    return {"scans": len(scans), "aorta_available": ao_n, "rooting_methods": methods,
            "distal": {v: summ(rows[v]) for v in h.FLOW_VESSELS},
            "minimum": {v: summ(rowsmin[v]) for v in h.FLOW_VESSELS},
            "lad_reference_flow_ml_s_at_K1": float(np.median(flow_lad)) if flow_lad else None}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/data")
    ap.add_argument("--out", default="/root/data/cfd")
    ap.add_argument("--scan-file", default="m4/train_scans.txt")
    ap.add_argument("--need-aorta", action="store_true")
    args = ap.parse_args(argv)
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scans50 = _scan_list(root, N_PLAUSIBILITY, args.scan_file, args.need_aorta)
    scans20 = scans50[:N_TEST_RETEST]
    scans5 = scans50[:N_CONVERGENCE]
    res = {"solver": "fallback_lumped_network", "k_flow_ml_s_per_ml075": h.K_FLOW,
           "checks": {}}
    res["checks"]["tube_poiseuille"] = check_tube()
    res["checks"]["stenosis_series"] = check_stenosis_series()
    res["checks"]["step_size_convergence"] = check_convergence(root, scans5)
    res["checks"]["gt_test_retest_0.45mm"] = check_test_retest(root, scans20)
    res["checks"]["gt_test_retest_v2"] = check_test_retest_v2(root, scans20)
    res["plausibility_gt_trees_train"] = gt_distribution(root, scans50)
    lad = res["plausibility_gt_trees_train"]["lad_reference_flow_ml_s_at_K1"]
    if lad:
        res["calibration"] = {"lad_median_reference_flow_ml_s_at_K1": lad,
                              "k_flow_needed_for_3_ml_s": TARGET_LAD_FLOW / lad,
                              "scans": len(scans50)}
    res["scan_ids"] = {"plausibility_and_calibration": scans50,
                       "test_retest": scans20, "convergence": scans5}
    res["all_checks_pass"] = all(v["pass"] for v in res["checks"].values())
    (out / "verification.json").write_text(json.dumps(res, indent=1))
    (out / "verification.md").write_text(to_markdown(res))
    print(json.dumps({k: v for k, v in res.items() if k != "scan_ids"}, indent=1))


def to_markdown(res):
    c = res["checks"]
    lines = ["# Addendum C 1D model verification (fallback solver)", "",
             f"Solver: {res['solver']}; K_FLOW = {res['k_flow_ml_s_per_ml075']} mL/s per mL^0.75", "",
             "| check | result | pass |", "|---|---|---|",
             f"| straight tube dP vs analytic | rel. error (flow, dP) {c['tube_poiseuille']['rel_error_flow']:.1e}, {c['tube_poiseuille']['rel_error_dp']:.1e} | "
             f"{c['tube_poiseuille']['pass']} |"]
    s = c["stenosis_series"]
    lines.append("| stenosis series 0-80% | " + ", ".join(
        f"{x['stenosis_pct']}%: {x['ffr']:.3f}" for x in s["series"]) + f" | {s['pass']} |")
    cv = c["step_size_convergence"]
    lines.append(f"| element-length convergence | max abs dFFR 0.25 vs 0.5 mm = {cv['max_abs_dffr_0.25_vs_0.5mm']:.4f}, "
                 f"p95 = {cv['p95_abs_dffr_0.25_vs_0.5mm']:.4f} ({cv['n_vessels']} vessels) | {cv['pass']} |")
    tr = c["gt_test_retest_0.45mm"]
    lines.append(f"| GT test-retest 0.45 mm | frac |dFFR|<0.01 = {tr['frac_below_0.01']:.3f}, median {tr['median_abs_dffr']:.4f}, p95 {tr['p95_abs_dffr']:.4f}, max {tr['max_abs_dffr']:.4f} "
                 f"({tr['n_vessels']} vessels) | {tr['pass']} |")
    v2 = c["gt_test_retest_v2"]
    for key, label in (("a_linear_roundtrip_0.45mm", "v2a linear 0.45 mm round trip"),
                       ("b_subvoxel_shift_0.5vox", "v2b +-0.5 voxel shifts"), ("pooled", "v2 pooled")):
        x = v2[key]
        lines.append(f"| {label} | frac |dFFR|<0.01 = {x['frac_below_0.01']:.3f}, median {x['median']:.4f}, "
                     f"p90 {x['p90']:.4f}, p95 {x['p95']:.4f} (n={x['n']}) | {v2['pass'] if key == 'pooled' else '-'} |")
    lines.append(f"Noise floor (v2 pooled p95): {v2['noise_floor_p95']:.4f}")
    p = res["plausibility_gt_trees_train"]
    lines += ["", f"Plausibility on {p['scans']} train GT trees (aorta available for {p['aorta_available']}; "
              f"rooting {p['rooting_methods']}):", "",
              "| vessel | n | median distal FFR [IQR] | frac >0.80 | frac <0.80 |", "|---|---|---|---|---|"]
    for v, d in p["distal"].items():
        if d["n"]:
            lines.append(f"| {v} | {d['n']} | {d['median']:.3f} [{d['p25']:.3f}, {d['p75']:.3f}] | "
                         f"{d['frac_gt_0.80']:.2f} | {d['frac_lt_0.80']:.2f} |")
        else:
            lines.append(f"| {v} | 0 | n/a | n/a | n/a |")
    lines += ["", f"Calibration: {res.get('calibration')}", "",
              f"All checks pass: {res['all_checks_pass']}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
