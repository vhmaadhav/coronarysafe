"""Downstream-harm descriptive analysis (Addendum B). Uses ground truth; not a ranker.

For LM, LAD, LCx and RCA the GT and predicted reconstructions are compared on
  * centerline length (mm), from the branch graph of the vessel skeleton;
  * minimum lumen diameter (mm): 2 x Euclidean distance transform along the vessel skeleton;
  * a Poiseuille resistance proxy, sum over branches of 8 mu L / (pi r^4), mu = 3.5 mPa s, with L and
    r in mm (so the unit is mPa s / mm^3). This is a proxy, not FFR.

Predicted skeleton voxels take the GT segment label of their nearest GT lumen voxel when that is
within 3 mm; otherwise they are unassigned. A predicted vessel with no assigned skeleton is
recorded as missing.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..geometry import bbox_slices
from ..graph import branch_graph, skeletonize

MAJOR_LABELS = {"LM": 1, "LAD": 2, "LCx": 3, "RCA": 9}
MU_MPA_S = 3.5
MATCH_MM = 3.0
MIN_RADIUS_MM = 0.2
SPUR_MM = 3.0
PAD_MM = 5.0


def nearest_gt_labels(gt_labels, spacing):
    """GT label of the nearest GT lumen voxel for every voxel; 0 beyond MATCH_MM (3 mm)."""
    g = np.asarray(gt_labels).astype(np.int32)
    lum = g > 0
    out = np.zeros(g.shape, dtype=np.int32)
    if not lum.any():
        return out
    d_gt, idx = ndimage.distance_transform_edt(~lum, sampling=tuple(float(s) for s in spacing),
                                               return_indices=True)
    near = d_gt <= MATCH_MM
    out[near] = g[idx[0][near], idx[1][near], idx[2][near]]
    return out


def _load_subvoxel_radius():
    """sqrt(A/pi) sub-voxel radius from the CFD agent's hemo1d module, or None if unavailable.

    Loaded by file path (CORONARYSAFE_CFD_SRC/coronarysafe/analysis/hemo1d.py) so that the two
    source trees do not shadow each other. Falls back to the EDT radius when missing.
    """
    import importlib.util
    import os
    from pathlib import Path

    src = os.environ.get("CORONARYSAFE_CFD_SRC")
    if not src:
        return None
    path = Path(src) / "coronarysafe" / "analysis" / "hemo1d.py"
    try:
        spec = importlib.util.spec_from_file_location("hemo1d_cfd", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.subvoxel_radius
    except Exception:  # noqa: BLE001 - any failure falls back to the EDT radius
        return None


SUBVOXEL_RADIUS = _load_subvoxel_radius()
RADIUS_SOURCE = "subvoxel_sqrtA_pi" if SUBVOXEL_RADIUS is not None else "edt"
DIAMETER_PERCENTILE = 5.0
TRIM_RADII = 2.0  # exclude 2 x local radius at each branch end (endpoints and junctions)


def _vessel_stats(skel, lumen_edt, sp, lumen_f=None):
    """Length (mm), diameter (mm) and resistance proxy of one vessel skeleton.

    diameter = 2 x the 5th percentile of the radius over branch points, after trimming
    TRIM_RADII x local radius from each branch end (endpoints and junctions). The radius is the
    sub-voxel sqrt(A/pi) when the CFD helper is available (lumen_f given), else the EDT radius.
    The resistance proxy keeps the EDT radius and the untrimmed branches, as before.
    """
    nan = {"length_mm": 0.0, "min_diam_mm": np.nan, "resistance": np.nan}
    if not skel.any():
        return nan
    spn = np.asarray(sp, dtype=float)
    B = branch_graph(skel, sp, prune_mm=SPUR_MM)
    length, resistance, radii = 0.0, 0.0, []
    for _, _, d in B.edges(data=True):
        vox = d["voxels"]
        r_edt = lumen_edt[vox[:, 0], vox[:, 1], vox[:, 2]]
        L = float(d["length_mm"])
        r = max(float(r_edt.mean()), MIN_RADIUS_MM)
        length += L
        resistance += 8.0 * MU_MPA_S * L / (np.pi * r**4)

        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(vox * spn, axis=0), axis=1))])
        keep = (s >= TRIM_RADII * r_edt[0]) & (s <= s[-1] - TRIM_RADII * r_edt[-1])
        if not keep.any():
            keep = np.zeros(len(vox), dtype=bool)
            keep[len(vox) // 2] = True
        kv = vox[keep]
        kr = r_edt[keep].astype(float)
        if SUBVOXEL_RADIUS is not None and lumen_f is not None and len(kv) > 0:
            centers = kv * spn
            if len(centers) > 1:
                tang = np.gradient(centers, axis=0)
            else:
                tang = np.tile(np.array([0.0, 0.0, 1.0]), (1, 1))
            norm = np.linalg.norm(tang, axis=1, keepdims=True)
            tang = tang / np.where(norm > 0, norm, 1.0)
            sv = np.asarray(SUBVOXEL_RADIUS(lumen_f, spn, centers, tang), dtype=float)
            kr = np.where(np.isfinite(sv), sv, kr)  # NaN discs fall back to the EDT radius
        radii.append(kr)
    if radii:
        pool = np.concatenate(radii)
    else:
        pool = lumen_edt[skel].astype(float)
    p5 = float(np.percentile(pool, DIAMETER_PERCENTILE))
    return {"length_mm": length, "min_diam_mm": 2.0 * p5, "resistance": resistance}


def analyse_unit(gt_labels, pred_mask, spacing):
    """Per-vessel GT and predicted stats for one (scan, segmenter) unit.

    gt_labels: 15-class GT array (0 background, labels 1..14); pred_mask: binary prediction.
    Returns {vessel: {"gt": stats, "pred": stats, "pred_assigned_vox": int}}.
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

    edt_gt = ndimage.distance_transform_edt(lum, sampling=sp)
    edt_pred = ndimage.distance_transform_edt(p, sampling=sp)
    skel_gt = skeletonize(lum)
    skel_pred = skeletonize(p)
    lab_pred = np.where(skel_pred, nearest_gt_labels(g, sp), 0).astype(np.int32)

    out = {}
    for name, L in MAJOR_LABELS.items():
        gt_stats = _vessel_stats(skel_gt & (g == L), edt_gt, sp, lumen_f=lum.astype(float))
        pr_stats = _vessel_stats((lab_pred == L) & skel_pred, edt_pred, sp, lumen_f=p.astype(float))
        out[name] = {"gt": gt_stats, "pred": pr_stats,
                     "pred_assigned_vox": int(((lab_pred == L) & skel_pred).sum())}
    return out


def unit_errors(per_vessel):
    """Flat per-vessel error rows: relative length error, diameter error (mm), log resistance ratio."""
    rows = []
    for name, res in per_vessel.items():
        gt, pr = res["gt"], res["pred"]
        gt_present = gt["length_mm"] > 0
        missing = pr["length_mm"] <= 0
        rel_len = np.nan
        diam_err = np.nan
        log_r = np.nan
        if gt_present:
            rel_len = (pr["length_mm"] - gt["length_mm"]) / gt["length_mm"]
            if not missing:
                diam_err = pr["min_diam_mm"] - gt["min_diam_mm"]
                if gt["resistance"] > 0 and pr["resistance"] > 0:
                    log_r = float(np.log(pr["resistance"] / gt["resistance"]))
        rows.append({
            "vessel": name,
            "gt_present": bool(gt_present),
            "missing": bool(missing),
            "gt_length_mm": gt["length_mm"],
            "pred_length_mm": pr["length_mm"],
            "rel_length_err": float(rel_len),
            "gt_min_diam_mm": gt["min_diam_mm"],
            "pred_min_diam_mm": pr["min_diam_mm"],
            "diam_err_mm": float(diam_err),
            "log_resistance_ratio": float(log_r),
        })
    return rows


def _iqr_summary(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "median": None, "p25": None, "p75": None}
    p25, med, p75 = np.percentile(v, [25, 50, 75])
    return {"n": int(v.size), "median": float(med), "p25": float(p25), "p75": float(p75)}


def summarise(rows):
    """Summaries per vessel and group. rows: dicts with vessel, critical, dice and the error fields."""
    import pandas as pd

    df = pd.DataFrame(rows)
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
        per_vessel = {}
        for vessel in list(MAJOR_LABELS) + ["ALL_VESSELS"]:
            vs = sub if vessel == "ALL_VESSELS" else sub[sub.vessel == vessel]
            present = vs[vs.gt_present]
            per_vessel[vessel] = {
                "units_with_gt_vessel": int(len(present)),
                "missing_pred_fraction": float(present.missing.mean()) if len(present) else None,
                "rel_length_err": _iqr_summary(present.rel_length_err),
                "diam_err_mm": _iqr_summary(present[~present.missing].diam_err_mm),
                "resistance_ratio_pred_over_gt": _iqr_summary(
                    np.exp(present[~present.missing].log_resistance_ratio)
                ),
            }
        out[gname] = per_vessel
    return out


def markdown(summary, headline):
    """Compact Markdown table of the summary (medians with IQR)."""
    lines = ["# Downstream-harm descriptive analysis (proxy, not FFR)", ""]
    lines += [f"- {k}: {v}" for k, v in headline.items()]
    lines += ["", "| group | vessel | n | missing | rel. length err median [IQR] | "
              "diameter err mm median [IQR] | resistance pred/GT median [IQR] |",
              "|---|---|---|---|---|---|---|"]

    def fmt(s, digits=3):
        if s["median"] is None:
            return "n/a"
        return f"{s['median']:.{digits}f} [{s['p25']:.{digits}f}, {s['p75']:.{digits}f}]"

    for gname, per_vessel in summary.items():
        for vessel, s in per_vessel.items():
            miss = s["missing_pred_fraction"]
            miss_s = "n/a" if miss is None else f"{miss:.3f}"
            lines.append(
                f"| {gname} | {vessel} | {s['units_with_gt_vessel']} | {miss_s} | "
                f"{fmt(s['rel_length_err'])} | {fmt(s['diam_err_mm'])} | "
                f"{fmt(s['resistance_ratio_pred_over_gt'])} |"
            )
    return "\n".join(lines) + "\n"
