"""
Independent-lumen burden — feature-disjoint label-robustness
(ANALYSIS_PLAN_v1 Addendum L, FROZEN 2026-11-15 with ERRATA 7 implementation pins).

Primary label (frozen):
  B_alt = 1 - |orig_mj & pred| / |orig_mj|
  where orig_mj = original ImageCAS binary lumen voxels inside the Euclidean ball of PAD mm
  (EDT distance, physical mm) around the ImageCAS-X GT major-segment gate G = {1 LM, 2 LAD, 3 LCx, 9 RCA}.

Pure voxel set operations: no centerline, no F-class rules, no radius or ostium-distance thresholds,
no segment identity beyond the GT gate, no PDA. The frozen model never sees the original lumen.
Sensitivity label B_alt5 uses PAD = 5.0 mm. Units with |orig_mj| < MIN_ORIG_MJ (including 0) are
excluded from statistics and counted.

Prediction and GT must match in shape and affine (allclose, atol 0.01 mm); otherwise the unit is an error.

With --scores (CSV: method, scan_id, score), per segmenter and pooled:
  1) Spearman rho(frozen score, B_alt) + patient-cluster bootstrap CI (2000, seed 0);
  2) AUROC of the frozen score for high burden {B_alt >= 0.10} + CI;
  3) paired mean B_alt, frozen top-floor(0.2 n) flagged vs the rest (fractional ties) + CI;
  C1  Spearman(burden_v3, B_alt) and agreement(critical_v3, B_alt >= 0.10) + CIs;
  C2  Spearman(frozen score, residual of B_alt after OLS on (burden_v3, critical_v3)) + CI,
      with the OLS refitted inside every bootstrap resample.
Decision rule (Errata 7, complete):
  (i)   rho CI lower > 0 AND C2 CI lower > 0;
  (ii)  rho CI lower > 0 AND C2 CI lower <= 0;
  (iii) otherwise (transfer not shown; reported as a limitation).

  python -m coronarysafe.analysis.burden_indep --split val --methods cas_net swin_unetr --out DIR \
      [--scores scores.csv] [--workers 8]

TEST split: coordinator only. Requires --scores AND --canonical-read MANIFEST (the ONE canonical
sealed read), refuses a non-empty --out, and writes provenance.json with sha256 of the inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to
from scipy import ndimage
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from ..eval.bootstrap import cluster_bootstrap
from ..io import load_nifti
from .label_robust import (
    ICX_GT_DIR,
    N_BOOT,
    ROOT,
    SEED,
    dice,
    index_original_masks,
    read_split_ids,
)

ORIG_DIR = ROOT / "imagecas" / "orig_labels"
MAJOR = (1, 2, 3, 9)  # LM, LAD, LCx, RCA (base major, no PDA)
PAD_MM = 2.0
PAD_MM_SENS = 5.0
MIN_ORIG_MJ = 50
HIGH_BURDEN = 0.10
BUDGET = 0.2
AFFINE_ATOL = 0.01
SCORE_DTYPES = {"method": str, "scan_id": str}


def load_scores(path) -> pd.DataFrame:
    """Score CSV reader: identifiers stay strings (leading zeros survive)."""
    return pd.read_csv(path, dtype=SCORE_DTYPES)


def dilate_mm(mask: np.ndarray, spacing, mm: float) -> np.ndarray:
    """Euclidean ball of radius ``mm`` (physical mm) around the mask: voxel in gate iff EDT <= mm."""
    m = np.asarray(mask) > 0
    if not m.any():
        return np.zeros(m.shape, dtype=bool)
    sp = tuple(float(s) for s in spacing)
    dist = ndimage.distance_transform_edt(~m, sampling=sp)
    return dist <= mm + 1e-9


def compute_balt(
    orig: np.ndarray, gt: np.ndarray, pred: np.ndarray, spacing, pad_mm: float
) -> dict:
    """One unit's independent-lumen burden. orig/pred are binary on the GT grid (pred: any value > 0)."""
    orig = np.asarray(orig) > 0
    pred = np.asarray(pred) > 0
    gate = np.isin(np.asarray(gt), MAJOR)
    omj = orig & dilate_mm(gate, spacing, pad_mm)
    n = int(omj.sum())
    if n == 0:
        return {"b_alt": float("nan"), "n_orig_mj": 0}
    covered = int((omj & pred).sum())
    return {"b_alt": float(1.0 - covered / n), "n_orig_mj": n}


def _load_original(path: Path, ref_shape, ref_affine):
    """Addendum I grid rule for the original lumen: use the file as-is when identical, else nearest resample."""
    img = nib.load(str(path))
    if tuple(img.shape) == tuple(ref_shape) and np.allclose(img.affine, ref_affine, atol=AFFINE_ATOL):
        return np.asanyarray(img.dataobj) > 0
    res = resample_from_to(img, (tuple(ref_shape), ref_affine), order=0)
    return np.asanyarray(res.dataobj) > 0


def score_unit(task) -> dict:
    """One (method, scan) unit. Errors are recorded, never raised."""
    method, sid, root, orig_path = task
    root = Path(root)
    rec = {
        "method": method, "scan_id": sid, "status": "ok", "error": "",
        "burden_v3": float("nan"), "critical_v3": float("nan"),
        "b_alt": float("nan"), "n_orig_mj": 0,
        "b_alt5": float("nan"), "n_orig_mj5": 0,
        "icx_orig_dice": float("nan"),
    }
    try:
        qa = json.loads((root / "qa" / "v3" / method / f"{sid}.json").read_text())
        rec["burden_v3"] = float(qa["burden"])
        rec["critical_v3"] = float(bool(qa["critical"]))
        gt, sp, aff = load_nifti(root / ICX_GT_DIR / "segmentations" / f"{sid}.coronary.nii.gz")
        gt = np.asarray(gt)
        pred_arr, _, pred_aff = load_nifti(root / "preds" / method / sid / "mask.nii.gz")
        pred = np.asarray(pred_arr) > 0
        if pred.shape != gt.shape:
            raise ValueError(f"prediction shape {pred.shape} != GT {gt.shape}")
        if not np.allclose(np.asarray(pred_aff, float), np.asarray(aff, float), atol=AFFINE_ATOL):
            raise ValueError("prediction affine differs from GT affine (allclose, atol 0.01 mm)")
        orig = _load_original(orig_path, gt.shape, aff)
        rec["icx_orig_dice"] = dice(orig, gt > 0)
        for pad, bkey, nkey in (
            (PAD_MM, "b_alt", "n_orig_mj"),
            (PAD_MM_SENS, "b_alt5", "n_orig_mj5"),
        ):
            r = compute_balt(orig, gt, pred, sp, pad)
            rec[bkey] = r["b_alt"]
            rec[nkey] = r["n_orig_mj"]
        # exclusion decision uses the PRIMARY label (PAD = 2.0 mm) support; zero support is excluded too
        if rec["n_orig_mj"] < MIN_ORIG_MJ:
            rec["b_alt"] = float("nan")
            rec["status"] = "excluded_low_support"
    except Exception as exc:  # recorded per unit, the run continues
        rec["status"] = "error"
        rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return rec


# ------------------------------------------------------------------ statistics


def nr_weights(score, budget: float = BUDGET):
    """Fractional-tie top-set weights under the NR@20 convention (metrics.py): k = floor(budget * n);
    tied scores at the cut-off share the remaining slots equally. Returns (weights summing to k, k),
    or (None, k) when k is 0 or k >= n."""
    s = np.asarray(score, dtype=float)
    n = len(s)
    k = math.floor(budget * n + 1e-12)
    if k <= 0 or k >= n:
        return None, k
    cutoff = np.sort(s)[::-1][k - 1]
    above = s > cutoff
    tied = s == cutoff
    w = above.astype(float)
    w[tied] = (k - above.sum()) / tied.sum()
    return w, k


def decision_branch(rho_lo: float, c2_lo: float) -> str:
    """Errata 7 decision rule on the lower CI bounds."""
    rho_pos = bool(np.isfinite(rho_lo) and rho_lo > 0)
    if rho_pos and np.isfinite(c2_lo) and c2_lo > 0:
        return "(i)"
    if rho_pos and np.isfinite(c2_lo) and c2_lo <= 0:
        return "(ii)"
    return "(iii)"


VERDICTS = {
    "(i)": "(i) transfer to the independent label source; circularity concern not supported",
    "(ii)": "(ii) transfer present but mediated / v3-adjacent information",
    "(iii)": "(iii) transfer not shown; reported as a limitation",
}


def _rho_ci(frozen, y, clusters, n_boot=N_BOOT, seed=SEED):
    est = (
        spearmanr(frozen, y)[0]
        if len(np.unique(frozen)) > 1 and len(np.unique(y)) > 1
        else float("nan")
    )
    res = cluster_bootstrap(
        clusters,
        lambda rows: (
            spearmanr(frozen[rows], y[rows])[0]
            if len(set(frozen[rows])) > 1 and len(set(y[rows])) > 1
            else float("nan")
        ),
        n_boot=n_boot,
        seed=seed,
    )
    return {"estimate": float(est), "lo": res["lo"], "hi": res["hi"]}


def _auroc_ci(y, frozen, clusters, n_boot=N_BOOT, seed=SEED):
    if not 0 < y.sum() < len(y):
        return {
            "estimate": float("nan"),
            "lo": float("nan"),
            "hi": float("nan"),
            "n_pos": int(y.sum()),
        }
    est = float(roc_auc_score(y, frozen))
    res = cluster_bootstrap(
        clusters,
        lambda rows: (
            float(roc_auc_score(y[rows], frozen[rows]))
            if 0 < y[rows].sum() < len(y[rows])
            else float("nan")
        ),
        n_boot=n_boot,
        seed=seed,
    )
    return {"estimate": est, "lo": res["lo"], "hi": res["hi"], "n_pos": int(y.sum())}


def _flagged_diff(frozen, y, clusters, budget=BUDGET, n_boot=N_BOOT, seed=SEED):
    """Paired mean B_alt in the per-resample NR top set minus the rest (fractional ties)."""

    def stat(rows):
        w, k = nr_weights(frozen[rows], budget)
        if w is None:
            return float("nan")
        yy = y[rows]
        return float((w * yy).sum() / k - ((1.0 - w) * yy).sum() / (len(rows) - k))

    res = cluster_bootstrap(clusters, stat, n_boot=n_boot, seed=seed)
    return {"estimate": res["estimate"], "lo": res["lo"], "hi": res["hi"], "flag_fraction": budget}


def _c2_residual(b_alt, burden, crit) -> np.ndarray:
    """Residual of OLS B_alt ~ 1 + burden_v3 + critical_v3 (C2 control)."""
    x = np.column_stack([np.ones(len(b_alt)), burden, crit])
    beta = np.linalg.lstsq(x, b_alt, rcond=None)[0]
    return b_alt - x @ beta


def _c2_stat(rows, frozen, b, b_v3, c_v3) -> float:
    """C2 statistic on a resample: the OLS residual is refitted on the resampled rows."""
    resid = _c2_residual(b[rows], b_v3[rows], c_v3[rows])
    if len(np.unique(resid)) <= 1 or len(np.unique(frozen[rows])) <= 1:
        return float("nan")
    return float(spearmanr(frozen[rows], resid)[0])


def _fmt(x, nd=3):
    return "nan" if x is None or not np.isfinite(x) else f"{float(x):.{nd}f}"


def stats_block(units: pd.DataFrame, scores: pd.DataFrame) -> dict:
    """Per-segmenter and pooled Addendum L statistics (pre-specified list only)."""
    ok = units[(units["status"] == "ok") & units["b_alt"].notna()].reset_index(drop=True)
    if ok.empty:
        raise ValueError("no usable units (all excluded or errored)")
    if ok.duplicated(["method", "scan_id"]).any():
        raise ValueError("more than one scan per patient within a segmenter (ImageCAS: one scan per patient)")
    out = {}
    sc = scores[["method", "scan_id", "score"]].drop_duplicates(["method", "scan_id"])
    for key, d in [("pooled", ok)] + sorted((m, d) for m, d in ok.groupby("method")):
        if key != "pooled":
            d = d[d["method"] == key]
        d = d.merge(sc, on=["method", "scan_id"], how="inner").reset_index(drop=True)
        if len(d) < 10:
            out[key] = {"n": int(len(d)), "note": "too few units for statistics"}
            continue
        frozen = d["score"].to_numpy(float)
        b = d["b_alt"].to_numpy(float)
        clusters = d["scan_id"].to_numpy(str)  # patient = scan (one scan per patient)
        b_v3 = d["burden_v3"].to_numpy(float)
        c_v3 = d["critical_v3"].to_numpy(float) > 0.5
        c_v3_f = c_v3.astype(float)
        high = (b >= HIGH_BURDEN).astype(int)
        residual = _c2_residual(b, b_v3, c_v3_f)
        rho_c2 = spearmanr(frozen, residual)[0] if len(np.unique(residual)) > 1 else float("nan")
        c2_ci = cluster_bootstrap(
            clusters,
            lambda rows: _c2_stat(rows, frozen, b, b_v3, c_v3_f),
            n_boot=N_BOOT,
            seed=SEED,
        )
        agree = float(np.mean(c_v3 == (high > 0)))
        agree_ci = cluster_bootstrap(
            clusters,
            lambda rows: float(np.mean(c_v3[rows] == (high[rows] > 0))),
            n_boot=N_BOOT,
            seed=SEED,
        )
        rho_c1 = spearmanr(b_v3, b)[0] if len(np.unique(b_v3)) > 1 else float("nan")
        c1_ci = cluster_bootstrap(
            clusters,
            lambda rows: (
                spearmanr(b_v3[rows], b[rows])[0] if len(set(b_v3[rows])) > 1 else float("nan")
            ),
            n_boot=N_BOOT,
            seed=SEED,
        )
        entry = {
            "n": int(len(d)),
            "n_patients": int(len(set(clusters))),
            "n_critical_v3": int(c_v3.sum()),
            "prev_high_balt": float(high.mean()),
            "mean_balt": float(np.nanmean(b)),
            "spearman_score_balt": _rho_ci(frozen, b, clusters),
            "auroc_score_high_balt": _auroc_ci(high > 0, frozen, clusters),
            "flagged_top20_minus_rest_balt": _flagged_diff(frozen, b, clusters),
            "c1_spearman_burdenv3_balt": {
                "estimate": float(rho_c1),
                "lo": c1_ci["lo"],
                "hi": c1_ci["hi"],
            },
            "c1_agreement_critv3_highbalt": {
                "estimate": agree,
                "lo": agree_ci["lo"],
                "hi": agree_ci["hi"],
            },
            "c2_spearman_score_resid": {
                "estimate": float(rho_c2),
                "lo": c2_ci["lo"],
                "hi": c2_ci["hi"],
            },
        }
        branch = decision_branch(
            entry["spearman_score_balt"]["lo"], entry["c2_spearman_score_resid"]["lo"]
        )
        entry["decision_branch"] = branch
        entry["verdict"] = VERDICTS[branch]
        out[key] = entry
    return out


def write_report(out: Path, args, units: pd.DataFrame, stats: dict | None, n_requested: int) -> None:
    ok = units[units["b_alt"].notna()]
    n_excl = int((units["status"] == "excluded_low_support").sum())
    lines = [
        f"# Independent-lumen burden (Addendum L, FROZEN 2026-11-15), split={args.split}",
        "",
        "Label: B_alt = 1 - |orig_mj ∩ pred| / |orig_mj|; "
        "orig_mj = original ImageCAS lumen ∩ EDT ball(2 mm) of the ImageCAS-X GT "
        "major gate {LM, LAD, LCx, RCA}; B_alt5 at 5 mm; pure voxel ops, "
        "no v3 machinery; "
        f"units with |orig_mj| < {MIN_ORIG_MJ} (including 0) excluded.",
        "",
        "The frozen model never sees the original lumen (label-free). "
        "Secondary, robustness analysis; does not re-test S1–S3.",
        "",
        f"Units: {n_requested} requested, {len(ok)} with B_alt, "
        f"{n_excl} excluded (low or zero support), "
        f"{int((units['status'] == 'error').sum())} errors.",
        "",
    ]
    if ok.empty:
        lines.append("No usable units.")
        (out / "BURDEN_INDEP.md").write_text("\n".join(lines))
        return
    per = (
        ok.groupby("method")
        .agg(
            n=("b_alt", "size"),
            mean_balt=("b_alt", "mean"),
            mean_balt5=("b_alt5", "mean"),
            mean_burden_v3=("burden_v3", "mean"),
            n_critical_v3=("critical_v3", "sum"),
        )
        .reset_index()
    )
    tbl = [
        "| method | n | mean B_alt(2mm) | mean B_alt(5mm) | mean burden_v3 | n critical v3 |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in per.iterrows():
        tbl.append(
            f"| {row['method']} | {row['n']} | {row['mean_balt']:.3f} | {row['mean_balt5']:.3f} "
            f"| {row['mean_burden_v3']:.3f} | {int(row['n_critical_v3'])} |"
        )
    lines += ["## Per segmenter (label statistics only)", ""] + tbl + [""]
    if stats is not None:
        lines += ["## Frozen score vs B_alt (pre-specified statistics)", ""]
        for key, e in stats.items():
            if "note" in e:
                lines += [f"### {key}: {e['note']}", ""]
                continue
            lines += [
                f"### {key} (n={e['n']}, patients={e['n_patients']}, n_critical_v3={e['n_critical_v3']},"
                f" prev(B_alt≥{HIGH_BURDEN})={e['prev_high_balt']:.3f},"
                f" mean B_alt={e['mean_balt']:.3f})",
                "",
                f"- Spearman(score, B_alt) = {_fmt(e['spearman_score_balt']['estimate'])} "
                f"[{_fmt(e['spearman_score_balt']['lo'])}, {_fmt(e['spearman_score_balt']['hi'])}]",
                f"- AUROC(score, B_alt≥{HIGH_BURDEN}) = "
                f"{_fmt(e['auroc_score_high_balt']['estimate'])} "
                f"[{_fmt(e['auroc_score_high_balt']['lo'])}, "
                f"{_fmt(e['auroc_score_high_balt']['hi'])}] "
                f"({e['auroc_score_high_balt']['n_pos']} positive)",
                f"- top-20%-flagged − rest, mean B_alt = "
                f"{_fmt(e['flagged_top20_minus_rest_balt']['estimate'])} "
                f"[{_fmt(e['flagged_top20_minus_rest_balt']['lo'])}, "
                f"{_fmt(e['flagged_top20_minus_rest_balt']['hi'])}]",
                f"- C1 Spearman(burden_v3, B_alt) = "
                f"{_fmt(e['c1_spearman_burdenv3_balt']['estimate'])} "
                f"[{_fmt(e['c1_spearman_burdenv3_balt']['lo'])}, "
                f"{_fmt(e['c1_spearman_burdenv3_balt']['hi'])}]; "
                f"agreement(critical_v3, high B_alt) = "
                f"{_fmt(e['c1_agreement_critv3_highbalt']['estimate'])} "
                f"[{_fmt(e['c1_agreement_critv3_highbalt']['lo'])}, "
                f"{_fmt(e['c1_agreement_critv3_highbalt']['hi'])}]",
                f"- C2 Spearman(score, residual‖v3) = "
                f"{_fmt(e['c2_spearman_score_resid']['estimate'])} "
                f"[{_fmt(e['c2_spearman_score_resid']['lo'])}, "
                f"{_fmt(e['c2_spearman_score_resid']['hi'])}]",
                f"- Decision branch: {e['decision_branch']}",
                f"- Verdict: {e['verdict']}",
                "",
            ]
    (out / "BURDEN_INDEP.md").write_text("\n".join(lines))
    units.to_csv(out / "units.csv", index=False)
    if stats is not None:
        (out / "stats.json").write_text(json.dumps(stats, indent=1, default=float))


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scores", default=None, help="CSV with columns method, scan_id, score")
    ap.add_argument(
        "--canonical-read",
        default=None,
        help="MANIFEST of the ONE canonical sealed read (required for --split test)",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.split == "test":
        # TEST gate (Addendum L / D4): coordinator only, one canonical sealed read.
        if not a.scores:
            raise SystemExit("test split requires --scores from the ONE canonical sealed read")
        if not a.canonical_read or not Path(a.canonical_read).is_file():
            raise SystemExit("test split requires --canonical-read MANIFEST (existing file)")
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"refusing non-empty --out {out} for the test split")
    out.mkdir(parents=True, exist_ok=True)
    orig = index_original_masks(ORIG_DIR)
    scans = read_split_ids(Path(a.root), a.split)
    jobs = [(m, s, a.root, str(orig[s])) for m in a.methods for s in scans if s in orig]
    n_requested = len(jobs)
    with Pool(a.workers) as pool:
        recs = list(pool.imap_unordered(score_unit, jobs))
    units = pd.DataFrame(recs).sort_values(["method", "scan_id"]).reset_index(drop=True)
    stats = None
    if a.scores:
        stats = stats_block(units, load_scores(a.scores))
    write_report(out, a, units, stats, n_requested)
    if a.split == "test":
        prov = {
            "split": "test",
            "scores_csv": str(a.scores),
            "scores_sha256": _sha256(a.scores),
            "canonical_read_manifest": str(a.canonical_read),
            "canonical_read_manifest_sha256": _sha256(a.canonical_read),
            "methods": a.methods,
            "n_requested": n_requested,
        }
        (out / "provenance.json").write_text(json.dumps(prov, indent=1))
    print(
        f"units {n_requested} requested, ok {int((units['status'] == 'ok').sum())}, "
        f"excluded {int((units['status'] == 'excluded_low_support').sum())}, "
        f"errors {int((units['status'] == 'error').sum())}"
    )
    if stats:
        print(json.dumps({k: v.get("verdict", v.get("note")) for k, v in stats.items()}, indent=1))


if __name__ == "__main__":
    main()
