"""Clinical measures that replace FFR as the headline (Addendum G). Uses ground truth; not a ranker.

Per major vessel (LM, LAD, LCx, RCA), for the GT and the predicted reconstruction:
  (a) minimum lumen diameter (MLD, mm): 2 x minimum sub-voxel radius over branch points, after
      excluding TRIM_RADII x local radius at each branch end (tips and junctions);
  (b) %-diameter stenosis and CAD-RADS band (0, 1-24, 25-49, 50-69, 70-99, 100). The reference
      diameter is the median radius over the first and last REF_WINDOW_MM of every branch (the
      healthy proximal and distal stretches). Stenosis = (1 - min r / ref r) x 100. A predicted
      vessel with no lumen is occluded (100). Band flip = GT band != predicted band (secondary);
  (c) SCCT segment coverage: for every GT skeleton voxel of the segment, covered when the
      predicted lumen lies within max(COVER_TOL_MM, local GT radius); coverage is the covered
      fraction. Reported for the four major vessels and for every GT segment label.
  Decision endpoints (primary descriptive): obstructive flip (GT and prediction on opposite sides
  of 50% stenosis, CAD-RADS >= 3 boundary) and severe flip (opposite sides of 70%, CAD-RADS 4).
  A flip counts only if the stenosed run at the threshold is >= MIN_FLIP_LEN_MM long in the GT or
  the prediction. An occluded predicted vessel has an infinite run (it is counted).

Reuses the vessel mapping and the sub-voxel radius helper from analysis.harm.
Caveat: the reference is taken from branch ends, which approximates the healthy proximal and
distal segments of each vessel.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..geometry import bbox_slices
from ..graph import branch_graph, skeletonize
from .harm import (
    MAJOR_LABELS,
    PAD_MM,
    SPUR_MM,
    SUBVOXEL_RADIUS,
    TRIM_RADII,
    nearest_gt_labels,
)

REF_WINDOW_MM = 10.0
COVER_TOL_MM = 1.5
OBSTRUCTIVE_PCT = 50.0
SEVERE_PCT = 70.0
MIN_FLIP_LEN_MM = 2.0
MLD_GATE_STENOSIS_PCT = 25.0
OCCLUDED_LEN_MM = 1.0e4  # sentinel run length for an absent (occluded) predicted vessel
BAND_ORDER = ("0", "1-24", "25-49", "50-69", "70-99", "100")


def cadrads_band(stenosis_pct):
    """%-diameter-stenosis CAD-RADS band. None when undefined."""
    s = float(stenosis_pct)
    if not np.isfinite(s):
        return None
    if s >= 100.0:
        return "100"
    if s < 1.0:
        return "0"
    if s < 25.0:
        return "1-24"
    if s < 50.0:
        return "25-49"
    if s < 70.0:
        return "50-69"
    return "70-99"


def _edt(mask, sp):
    return ndimage.distance_transform_edt(mask, sampling=sp)


def _max_run_mm(mask, ks):
    """Longest arc-length run (mm) of consecutive True points along one branch."""
    best = 0.0
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            best = max(best, float(ks[j] - ks[i]))
            i = j + 1
        else:
            i += 1
    return best


def _vessel_profile(skel, lumen_edt, sp, lumen_f):
    """MLD, reference diameter, stenosis, band and stenosed-run lengths of one vessel skeleton."""
    spn = np.asarray(sp, dtype=float)
    B = branch_graph(skel, sp, prune_mm=SPUR_MM)
    branches = [d["voxels"] for _, _, d in B.edges(data=True)]
    if not branches:
        branches = [np.argwhere(skel)]
    per_branch = []  # (kept radii, kept arc positions, branch length, reference window mask)
    for vox in branches:
        r_edt = lumen_edt[vox[:, 0], vox[:, 1], vox[:, 2]]
        if len(vox) > 1:
            s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(vox * spn, axis=0), axis=1))])
        else:
            s = np.zeros(1)
        total = float(s[-1])
        keep = (s >= TRIM_RADII * r_edt[0]) & (s <= total - TRIM_RADII * r_edt[-1])
        if not keep.any():
            keep = np.zeros(len(vox), dtype=bool)
            keep[len(vox) // 2] = True
        kv, kr, ks = vox[keep], r_edt[keep].astype(float), s[keep]
        if SUBVOXEL_RADIUS is not None and lumen_f is not None and len(kv) > 0:
            centers = kv * spn
            tang = np.gradient(centers, axis=0) if len(centers) > 1 else np.array([[0.0, 0.0, 1.0]])
            norm = np.linalg.norm(tang, axis=1, keepdims=True)
            tang = tang / np.where(norm > 0, norm, 1.0)
            sv = np.asarray(SUBVOXEL_RADIUS(lumen_f, spn, centers, tang), dtype=float)
            kr = np.where(np.isfinite(sv), sv, kr)
        win = (ks <= REF_WINDOW_MM) | (ks >= total - REF_WINDOW_MM)
        per_branch.append((kr, ks, win))

    all_r = np.concatenate([kr for kr, _, _ in per_branch])
    ref_pool = np.concatenate([kr[win] for kr, _, win in per_branch])
    min_r = float(all_r.min())
    ref_r = float(np.median(ref_pool)) if ref_pool.size else float(np.median(all_r))
    if ref_r > 0:
        stenosis = float(np.clip((1.0 - min_r / ref_r) * 100.0, 0.0, 100.0))
    else:
        stenosis = np.nan

    def run_len(threshold):
        best = 0.0
        for kr, ks, _ in per_branch:
            pct = (1.0 - kr / ref_r) * 100.0 if ref_r > 0 else np.zeros(len(kr))
            best = max(best, _max_run_mm(pct >= threshold, ks))
        return best

    return {
        "present": True,
        "mld_mm": 2.0 * min_r,
        "ref_diam_mm": 2.0 * ref_r,
        "stenosis_pct": stenosis,
        "band": cadrads_band(stenosis),
        "len_obstructive_mm": run_len(OBSTRUCTIVE_PCT),
        "len_severe_mm": run_len(SEVERE_PCT),
    }


def _absent_profile():
    return {"present": False, "mld_mm": 0.0, "ref_diam_mm": np.nan, "stenosis_pct": 100.0,
            "band": "100", "len_obstructive_mm": OCCLUDED_LEN_MM, "len_severe_mm": OCCLUDED_LEN_MM}


def _segment_coverage(gt_skel, gt_edt, pred_dist):
    """Covered fraction of a GT centerline segment within max(COVER_TOL_MM, local GT radius)."""
    if not gt_skel.any():
        return np.nan
    r = gt_edt[gt_skel]
    d = pred_dist[gt_skel]
    return float((d <= np.maximum(COVER_TOL_MM, r)).mean())


def clinical_unit(gt_labels, pred_mask, spacing):
    """Clinical measures for one (scan, segmenter) unit.

    Returns {"vessels": {name: {"gt": profile, "pred": profile, "coverage": float}},
             "segments": {label: coverage}}.
    """
    sp = tuple(float(s) for s in spacing)
    gt = np.asarray(gt_labels).astype(np.int32)
    pred = np.asarray(pred_mask) > 0
    lumen = gt > 0
    if not lumen.any() or not pred.any():
        raise ValueError("unit needs both a non-empty GT lumen and a non-empty prediction")
    sl = bbox_slices(lumen | pred, sp, pad_mm=PAD_MM)
    g = gt[sl]
    p = pred[sl]
    lum = g > 0

    gt_edt = _edt(lum, sp)
    pred_edt = _edt(p, sp)
    pred_dist = _edt(~p, sp)  # distance from every voxel to the predicted lumen
    skel_gt = skeletonize(lum)
    skel_pred = skeletonize(p)
    lab_pred = np.where(skel_pred, nearest_gt_labels(g, sp), 0).astype(np.int32)

    vessels = {}
    for name, L in MAJOR_LABELS.items():
        gt_sk = skel_gt & (g == L)
        pr_sk = skel_pred & (lab_pred == L)
        gp = _vessel_profile(gt_sk, gt_edt, sp, lum.astype(float)) if gt_sk.any() else None
        pp = _vessel_profile(pr_sk, pred_edt, sp, p.astype(float)) if pr_sk.any() else _absent_profile()
        vessels[name] = {"gt": gp, "pred": pp, "coverage": _segment_coverage(gt_sk, gt_edt, pred_dist)}

    segments = {}
    for L in range(1, 15):
        gt_sk = skel_gt & (g == L)
        if gt_sk.any():
            segments[str(L)] = _segment_coverage(gt_sk, gt_edt, pred_dist)
    return {"vessels": vessels, "segments": segments}


def clinical_rows(result):
    """Flat per-vessel rows for the GT-present vessels, with the decision endpoints."""
    rows = []
    for name, res in result["vessels"].items():
        gp, pp = res["gt"], res["pred"]
        if gp is None:
            continue
        s_gt, s_pr = gp["stenosis_pct"], pp["stenosis_pct"]
        long_obs = max(gp["len_obstructive_mm"], pp["len_obstructive_mm"]) >= MIN_FLIP_LEN_MM
        long_sev = max(gp["len_severe_mm"], pp["len_severe_mm"]) >= MIN_FLIP_LEN_MM
        obs_flip = ((s_gt >= OBSTRUCTIVE_PCT) != (s_pr >= OBSTRUCTIVE_PCT)) and long_obs
        sev_flip = ((s_gt >= SEVERE_PCT) != (s_pr >= SEVERE_PCT)) and long_sev
        rows.append({
            "vessel": name,
            "mld_gt_mm": gp["mld_mm"],
            "mld_pred_mm": pp["mld_mm"],
            "mld_err_mm": pp["mld_mm"] - gp["mld_mm"],
            "stenosis_gt_pct": s_gt,
            "stenosis_pred_pct": s_pr,
            "abs_stenosis_err_pct": abs(s_pr - s_gt),
            "band_gt": gp["band"],
            "band_pred": pp["band"],
            "band_flip": bool(gp["band"] != pp["band"]),
            "obstructive_flip": bool(obs_flip),
            "severe_flip": bool(sev_flip),
            "coverage": res["coverage"],
            "pred_present": bool(pp["present"]),
        })
    return rows


def _iqr(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "median": None, "p25": None, "p75": None}
    p25, med, p75 = np.percentile(v, [25, 50, 75])
    return {"n": int(v.size), "median": float(med), "p25": float(p25), "p75": float(p75)}


def _cluster_rate(values, patients, n_boot=2000, seed=14):
    """Rate with a patient-cluster bootstrap 95% CI (resamples patients with replacement)."""
    v = np.asarray(values, dtype=float)
    keep = np.isfinite(v)
    v, p = v[keep], np.asarray(patients)[keep]
    if v.size == 0:
        return None
    uniq, inv = np.unique(p, return_inverse=True)
    sums = np.bincount(inv, weights=v, minlength=len(uniq))
    cnt = np.bincount(inv, minlength=len(uniq))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot = sums[draws].sum(axis=1) / cnt[draws].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"estimate": float(v.mean()), "lo": float(lo), "hi": float(hi),
            "n": int(v.size), "patients": int(len(uniq))}


def summarise_clinical(rows, n_boot=2000):
    """Summaries by vessel and group with decision flip rates and patient-bootstrap CIs.

    rows carry vessel, scan_id, method, critical, dice plus clinical_rows fields.
    """
    import pandas as pd

    df = pd.DataFrame(rows)
    df["critical"] = df["critical"].astype(bool)
    df["unit"] = df["method"].astype(str) + "/" + df["scan_id"].astype(str)
    groups = {
        "all": df,
        "v3_critical": df[df.critical],
        "v3_noncritical": df[~df.critical],
        "dice_ge_0.90_all": df[df.dice >= 0.90],
        "dice_ge_0.90_critical": df[(df.dice >= 0.90) & df.critical],
        "dice_ge_0.90_noncritical": df[(df.dice >= 0.90) & ~df.critical],
    }
    out = {}
    for gname, sub in groups.items():
        per = {}
        for vessel in list(MAJOR_LABELS) + ["ALL_VESSELS"]:
            vs = sub if vessel == "ALL_VESSELS" else sub[sub.vessel == vessel]
            if len(vs) == 0:
                per[vessel] = {"n": 0}
                continue
            gated = vs[vs.stenosis_gt_pct >= MLD_GATE_STENOSIS_PCT]
            per[vessel] = {
                "n": int(len(vs)),
                "obstructive_flip_rate": _cluster_rate(vs.obstructive_flip, vs.scan_id, n_boot),
                "severe_flip_rate": _cluster_rate(vs.severe_flip, vs.scan_id, n_boot),
                "band_flip_rate_secondary": _cluster_rate(vs.band_flip, vs.scan_id, n_boot),
                "mld_err_mm": _iqr(vs.mld_err_mm),
                "mld_err_gt_stenosis_ge25_mm": _iqr(gated.mld_err_mm),
                "n_gt_stenosis_ge25": int(len(gated)),
                "abs_stenosis_err_pct": _iqr(vs.abs_stenosis_err_pct),
                "pred_occluded_rate": float((~vs.pred_present).mean()),
                "coverage": _iqr(vs.coverage),
                "coverage_below_0.9_rate": float((vs.coverage < 0.9).mean()),
            }
        # Per-unit endpoints: a unit flips if any major vessel flips.
        units = sub.groupby("unit").agg(
            scan_id=("scan_id", "first"),
            obs=("obstructive_flip", "max"),
            sev=("severe_flip", "max"),
            band=("band_flip", "max"),
        )
        per["PER_UNIT_any_major_vessel"] = {
            "n": int(len(units)),
            "obstructive_flip_rate": _cluster_rate(units.obs, units.scan_id, n_boot),
            "severe_flip_rate": _cluster_rate(units.sev, units.scan_id, n_boot),
            "band_flip_rate_secondary": _cluster_rate(units.band, units.scan_id, n_boot),
        }
        out[gname] = per
    return out


def _fmt_rate(r):
    if r is None:
        return "n/a"
    return f"{r['estimate']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"


def _fmt_iqr(s, d=2):
    if s["median"] is None:
        return "n/a"
    return f"{s['median']:.{d}f} [{s['p25']:.{d}f}, {s['p75']:.{d}f}] (n={s['n']})"


def markdown(summary, headline):
    lines = ["# Clinical measures, val (descriptive; replaces FFR as headline)", ""]
    lines += [f"- {k}: {v}" for k, v in headline.items()]
    lines += ["", "Rates are estimates with 95% patient-cluster bootstrap CIs.", "",
              "| group | vessel | n | obstructive flip (50%) | severe flip (70%) | "
              "band flip (secondary) | MLD err mm, GT stenosis >=25% | pred occluded | coverage median [IQR] |",
              "|---|---|---|---|---|---|---|---|---|"]
    for gname, per in summary.items():
        for vessel, s in per.items():
            if s.get("n", 0) == 0:
                continue
            if vessel == "PER_UNIT_any_major_vessel":
                lines.append(
                    f"| {gname} | per unit (any major vessel) | {s['n']} | "
                    f"{_fmt_rate(s['obstructive_flip_rate'])} | {_fmt_rate(s['severe_flip_rate'])} | "
                    f"{_fmt_rate(s['band_flip_rate_secondary'])} | n/a | n/a | n/a |")
                continue
            occ = s.get("pred_occluded_rate")
            lines.append(
                f"| {gname} | {vessel} | {s['n']} | {_fmt_rate(s['obstructive_flip_rate'])} | "
                f"{_fmt_rate(s['severe_flip_rate'])} | {_fmt_rate(s['band_flip_rate_secondary'])} | "
                f"{_fmt_iqr(s['mld_err_gt_stenosis_ge25_mm'])} | "
                f"{'n/a' if occ is None else f'{occ:.3f}'} | {_fmt_iqr(s['coverage'], 3)} |")
    return "\n".join(lines) + "\n"
