"""M5 driver for the droplet (CPU only, at most 6 processes).

  features --split {val,test}   evidence features per (segmenter, scan) -> <out>
  harm                          GT vs prediction vessel analysis, val + test, cas_net and swin_unetr
  dev                           dev-only evaluation on val x {cas_net, swin_unetr}: M1, M5, M1+M5

Layout under --root (default /root/data): imagecas/volumes (CT), m4/aorta (optional TotalSegmentator
aorta masks), preds/<segmenter>/<scan>/mask.nii.gz, imagecas_x/ImageCAS-X_dataset_v3 (GT and
filelists), qa/v3/table_<split>.parquet. Ground truth is read only by `harm` and `dev`-free paths
(the features step never opens GT). The test split is never read by `dev`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

METHODS = ("cas_net", "swin_unetr")
GT_DIR = "imagecas_x/ImageCAS-X_dataset_v3"
WORKERS = 6  # coordinator cap: at most 6 processes, run under nice 10
SEED = 14
BUDGET = 0.2
DELTA = 0.1
ALPHAS = (0.10, 0.05)
REFERENCE_KEYS = ("m5_ref_hu", "m5_ref_fallback")


# ----------------------------------------------------------------------------- paths


def read_split(root, split):
    ids, seen = [], set()
    for line in (root / GT_DIR / "filelist" / f"{split}.txt").read_text().splitlines():
        line = line.strip()
        if line and str(int(line)) not in seen:
            seen.add(str(int(line)))
            ids.append(str(int(line)))
    return ids


def find_one(folder, scan):
    if not folder.is_dir():
        return None
    n = int(scan)
    for name in (f"{scan}.nii.gz", f"{scan}.nii", f"{n:03d}.nii.gz", f"{n:04d}.nii.gz"):
        p = folder / name
        if p.exists():
            return p
    return None


def ct_path(root, scan):
    """CT volumes are staged as imagecas/volumes/<group>/<scan>.img.nii.gz (verified on droplet)."""
    hits = sorted((root / "imagecas" / "volumes").glob(f"*/{int(scan)}.img.nii.gz"))
    if len(hits) > 1:
        raise ValueError(f"ambiguous CT for scan {scan}: {hits}")
    return hits[0] if hits else None


def aorta_path(root, scan):
    return find_one(root / "m4" / "aorta", scan)


def gt_path(root, scan):
    return root / GT_DIR / "segmentations" / f"{scan}.coronary.nii.gz"


def pred_path(root, method, scan):
    return root / "preds" / method / str(int(scan)) / "mask.nii.gz"


def load_table(path):
    df = pd.read_parquet(path)
    if "patient_group" in df and "scan_id" not in df:
        df = df.rename(columns={"patient_group": "scan_id"})
    df["scan_id"] = df["scan_id"].astype("string").str.strip()
    if "status" in df:
        df = df[df.status.eq("ok")]
    return df[df.method.isin(METHODS)].reset_index(drop=True)


def label_table(root, split, methods):
    """Per-unit labels from qa/v3/<method>/<scan>.json (covers ffr_unet and the single-segmenter
    tables, which the nnU-Net table_<split>.parquet does not contain)."""
    rows = []
    for method in methods:
        for scan in read_split(root, split):
            rec = json.loads((root / "qa" / "v3" / method / f"{scan}.json").read_text())
            rows.append({"method": method, "scan_id": scan, "status": "ok",
                         "critical": bool(rec["critical"]), "dice": float(rec["dice"])})
    return pd.DataFrame(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- features


def _feature_job(job):
    from coronarysafe.features.evidence import evidence_features
    from coronarysafe.io import load_nifti

    method, scan, ct, mask, aorta = job
    row = {"scan_id": scan, "method": method, "status": "ok", "aorta_ref": aorta is not None}
    try:
        hu, sp, _ = load_nifti(ct)
        m, _, _ = load_nifti(mask)
        ao = load_nifti(aorta)[0] > 0 if aorta else None
        row.update(evidence_features(np.asarray(m) > 0, hu, sp, ao))
    except Exception as exc:  # recorded, never silently dropped
        row["status"] = f"error:{type(exc).__name__}:{str(exc)[:200]}"
    return row


def cmd_features(a):
    root = Path(a.root)
    scans = read_split(root, a.split)
    jobs, missing = [], []
    for method in METHODS:
        for scan in scans:
            ct = ct_path(root, scan)
            mask = pred_path(root, method, scan)
            if ct is None or not mask.exists():
                missing.append((method, scan, str(ct), str(mask)))
                continue
            ao = aorta_path(root, scan)
            jobs.append((method, scan, str(ct), str(mask), str(ao) if ao else None))
    if missing:
        raise SystemExit(f"{len(missing)} units lack CT or prediction; first: {missing[:3]}")
    t0 = time.time()
    with Pool(WORKERS) as pool:
        rows = pool.map(_feature_job, jobs, chunksize=1)
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(json.dumps({
        "split": a.split, "units": len(df), "ok": int((df.status == "ok").sum()),
        "aorta_reference": int(df.aorta_ref.sum()), "seconds": round(time.time() - t0, 1),
    }))


# ----------------------------------------------------------------------------- harm


def _harm_job(job):
    from coronarysafe.analysis.harm import analyse_unit, unit_errors
    from coronarysafe.io import load_nifti

    root, split, method, scan, critical, dice = job
    gt, sp, _ = load_nifti(gt_path(Path(root), scan))
    pred, _, _ = load_nifti(pred_path(Path(root), method, scan))
    rows = unit_errors(analyse_unit(gt, np.asarray(pred) > 0, sp))
    for r in rows:
        r.update({"split": split, "method": method, "scan_id": scan,
                  "critical": bool(critical), "dice": float(dice)})
    return rows


def cmd_harm(a):
    from coronarysafe.analysis.harm import markdown, summarise

    root = Path(a.root)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    counts = {}
    for split in a.splits:
        tab = label_table(root, split, a.methods)
        for method in a.methods:
            sub = tab[tab.method == method]
            counts[f"{split}/{method}"] = {
                "units": int(len(sub)), "critical": int(sub.critical.astype(bool).sum()),
                "dice_ge_0.90": int((sub.dice >= 0.90).sum()),
            }
            for r in sub.itertuples(index=False):
                jobs.append((str(root), split, method, r.scan_id, bool(r.critical), float(r.dice)))
    t0 = time.time()
    with Pool(WORKERS) as pool:
        parts = pool.map(_harm_job, jobs, chunksize=2)
    rows = [r for part in parts for r in part]
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "harm_units.parquet", index=False)
    summary = summarise(rows)
    headline = {"units_by_split_method": counts, "vessel_rows": len(df),
                "seconds": round(time.time() - t0, 1)}
    (out_dir / "harm_summary.json").write_text(
        json.dumps({"headline": headline, "proxy_note": "Poiseuille proxy, not FFR",
                    "summary": summary}, indent=2, allow_nan=False) + "\n")
    (out_dir / "harm_summary.md").write_text(markdown(summary, headline))
    print(json.dumps(headline))


# ----------------------------------------------------------------------------- dev


def _interval(result):
    return {k: (float(result[k]) if np.isfinite(result[k]) else None) for k in ("estimate", "lo", "hi")}


def cmd_dev(a):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from coronarysafe.eval.bootstrap import cluster_bootstrap
    from coronarysafe.eval.conformal import apply_threshold, calibrate_review_threshold
    from coronarysafe.eval.metrics import (
        auroc, normalised_recall, oracle_recall_at_budget, recall_at_budget,
    )
    from coronarysafe.eval.triage import GRAPH

    root = Path(a.root)
    table_path = root / "qa" / "v3" / "table_val.parquet"
    tab = load_table(table_path)
    feats = pd.read_parquet(a.features)
    feats = feats[feats.status.eq("ok")].copy()
    feats["scan_id"] = feats["scan_id"].astype("string").str.strip()
    m5 = [c for c in feats.columns if c.startswith("m5_") and c not in REFERENCE_KEYS]
    df = tab.merge(feats, on=["scan_id", "method"], how="inner", validate="one_to_one")
    if len(df) != len(tab):
        raise SystemExit(f"feature/table mismatch: {len(df)} merged of {len(tab)}")
    df = df.sort_values(["scan_id", "method"]).reset_index(drop=True)
    for c in GRAPH + m5:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    y = df.critical.astype(bool).to_numpy()
    groups = df.scan_id.to_numpy()
    n = len(df)

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = np.zeros(n, dtype=int)
    for k, (_, te) in enumerate(skf.split(np.zeros(n), y, groups)):
        folds[te] = k

    def oof(cols):
        scores = np.full(n, np.nan)
        for k in range(5):
            tr, te = folds != k, folds == k
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  LogisticRegression(C=0.1, max_iter=1000))
            model.fit(df.loc[tr, cols], y[tr])
            scores[te] = model.predict_proba(df.loc[te, cols])[:, 1]
        return scores

    def crossfit(s, alpha):
        misses = nf = flagged = total = 0
        thresholds, folds_out = [], []
        for k in range(5):
            cal, te = folds != k, folds == k
            cal_res = calibrate_review_threshold(s[cal], y[cal], groups[cal], alpha, DELTA)
            ap = apply_threshold(s[te], y[te], cal_res["threshold"])
            misses += ap["n_missed"]
            nf += ap["n_nonflagged"]
            flagged += ap["review_fraction"] * ap["n"]
            total += ap["n"]
            thresholds.append(cal_res["threshold"])
            folds_out.append({"fold": k, "dev_cal_miss_upper_bound": cal_res["dev_miss_upper_bound"],
                              "cal_review_fraction": cal_res["review_fraction"], **ap})
        insample = calibrate_review_threshold(s, y, groups, alpha, DELTA)
        return {
            "alpha": alpha, "delta": DELTA,
            "pooled_miss_rate_nonflagged": float(misses / nf) if nf else 0.0,
            "pooled_review_fraction": float(flagged / total),
            "n_missed": int(misses), "n_nonflagged": int(nf),
            "fold_thresholds": thresholds,
            "folds": folds_out,
            "insample_threshold_on_all_dev": insample,
        }

    def metrics(s):
        def stat(rows):
            return normalised_recall(s[rows], y[rows], BUDGET)

        res = {
            "normalised_recall_at_20pct": _interval(cluster_bootstrap(groups, stat, a.bootstrap, SEED)),
            "recall_at_20pct": float(recall_at_budget(s, y, BUDGET)),
            "oracle_recall_at_20pct": float(oracle_recall_at_budget(y, BUDGET)),
            "auroc": float(auroc(s, y)),
        }
        for alpha in ALPHAS:
            res[f"conformal_alpha_{alpha:.2f}"] = crossfit(s, alpha)
        return res

    scores = {
        "M1_graph": oof(GRAPH),
        "M5_evidence": oof(m5),
        "M1_plus_M5": oof(GRAPH + m5),
    }
    results = {
        "description": "Dev-only evaluation: ImageCAS-X val x {cas_net, swin_unetr}. "
                       "5-fold patient-grouped CV (StratifiedGroupKFold, seed 14), L2 logistic (C=0.1). "
                       "Conformal thresholds are cross-fitted (calibrated on the other folds).",
        "units": n,
        "patients": int(len(np.unique(groups))),
        "critical": int(y.sum()),
        "prevalence": float(y.mean()),
        "budget_review_units": int(np.floor(BUDGET * n)),
        "seed": SEED,
        "bootstrap_resamples": a.bootstrap,
        "features": {"M1": GRAPH, "M5": m5},
        "models": {name: metrics(s) for name, s in scores.items()},
        "input_sha256": {"table_val": sha(table_path), "features_val": sha(a.features)},
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: results["models"][k]["normalised_recall_at_20pct"] for k in results["models"]}))


CLINICAL_WORKERS = 6  # coordinator cap for the clinical run (nice 10)


def _clinical_job(job):
    from coronarysafe.analysis.clinical import clinical_rows, clinical_unit
    from coronarysafe.io import load_nifti

    root, method, scan, critical, dice = job
    gt, sp, _ = load_nifti(gt_path(Path(root), scan))
    pred, _, _ = load_nifti(pred_path(Path(root), method, scan))
    res = clinical_unit(gt, np.asarray(pred) > 0, sp)
    rows = clinical_rows(res)
    for r in rows:
        r.update({"method": method, "scan_id": scan, "critical": bool(critical), "dice": float(dice)})
    segs = [{"method": method, "scan_id": scan, "segment": int(L), "coverage": float(c),
             "critical": bool(critical)} for L, c in res["segments"].items()]
    return rows, segs


def cmd_clinical(a):
    from coronarysafe.analysis.clinical import markdown, summarise_clinical

    root = Path(a.root)
    tab = label_table(root, a.split, a.methods)
    jobs = [(str(root), r.method, r.scan_id, bool(r.critical), float(r.dice))
            for r in tab.itertuples(index=False)]
    t0 = time.time()
    with Pool(CLINICAL_WORKERS) as pool:
        parts = pool.map(_clinical_job, jobs, chunksize=2)
    rows = [r for part, _ in parts for r in part]
    segs = [s for _, part in parts for s in part]
    prefix = Path(a.out)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(prefix.with_name(prefix.name + "_vessel_rows.parquet"), index=False)
    seg_df = pd.DataFrame(segs)
    seg_summary = {
        str(L): {"median_coverage_all": float(g.coverage.median()),
                 "median_coverage_critical": float(g[g.critical].coverage.median())
                 if g.critical.any() else None,
                 "n": int(len(g))}
        for L, g in seg_df.groupby("segment")
    }
    summary = summarise_clinical(rows)
    headline = {"units": len(jobs), "vessel_rows": len(rows),
                "critical_units": int(sum(1 for j in jobs if j[3])),
                "dice_ge_0.90_units": int(sum(1 for j in jobs if j[4] >= 0.90)),
                "seconds": round(time.time() - t0, 1),
                "descriptive_note": "Replaces FFR as headline; FFR reported as negative methods finding"}
    prefix.with_suffix(".json").write_text(json.dumps(
        {"headline": headline, "summary": summary, "segment_coverage": seg_summary},
        indent=2, allow_nan=False, default=float) + "\n")
    prefix.with_suffix(".md").write_text(markdown(summary, headline))
    print(json.dumps(headline))


RULE_SIGNS = {
    # Fixed signs chosen before looking at dev outcomes: fragmentation and gap contrast raise risk.
    "rule_neg_share_top2": ("g_share_top2", -1.0),
    "rule_n_components": ("g_n_components", 1.0),
    "rule_n_components_ge_10mm": ("g_n_components_ge_10mm", 1.0),
    "rule_m5_gap_evidence_max": ("m5_gap_evidence_max", 1.0),
}
L1_C_GRID = np.logspace(-2, 0.5, 10)
L1_MAX_FEATURES = 3


def cmd_dev2(a):
    """Small-positive candidates on dev only: fixed-sign rules, nested L1, nested rank-average."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from coronarysafe.eval.bootstrap import cluster_bootstrap
    from coronarysafe.eval.metrics import (
        auroc, normalised_recall, oracle_recall_at_budget, recall_at_budget,
    )
    from coronarysafe.eval.triage import GRAPH

    root = Path(a.root)
    table_path = root / "qa" / "v3" / "table_val.parquet"
    tab = load_table(table_path)
    feats = pd.read_parquet(a.features)
    feats = feats[feats.status.eq("ok")].copy()
    feats["scan_id"] = feats["scan_id"].astype("string").str.strip()
    m5 = [c for c in feats.columns if c.startswith("m5_") and c not in REFERENCE_KEYS]
    df = tab.merge(feats, on=["scan_id", "method"], how="inner", validate="one_to_one")
    if len(df) != len(tab):
        raise SystemExit(f"feature/table mismatch: {len(df)} merged of {len(tab)}")
    df = df.sort_values(["scan_id", "method"]).reset_index(drop=True)
    for c in GRAPH + m5:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    y = df.critical.astype(bool).to_numpy()
    groups = df.scan_id.to_numpy()
    n = len(df)
    l1_cols = GRAPH + m5

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = np.zeros(n, dtype=int)
    for k, (_, te) in enumerate(skf.split(np.zeros(n), y, groups)):
        folds[te] = k

    # Single fixed-sign rules: no fitting, so the out-of-fold score is the full-dev score.
    rule_raw, rule_fill = {}, {}
    for name, (col, sign) in RULE_SIGNS.items():
        x = df[col].astype(float).to_numpy()
        med = float(np.nanmedian(x))  # label-free dev-only fill for undefined values
        rule_fill[name] = med
        rule_raw[name] = sign * np.where(np.isnan(x), med, x)
    rule_names = list(RULE_SIGNS)

    # Nested rank-average: top-2 single rules chosen on each outer training fold by training NR,
    # ranked by the empirical CDF on that training fold so fold-level scores are comparable.
    rank_oof = np.full(n, np.nan)
    rank_choice = []
    for k in range(5):
        tr, te = folds != k, folds == k
        nr_train = {nm: normalised_recall(rule_raw[nm][tr], y[tr], BUDGET) for nm in rule_names}
        order = sorted(rule_names, key=lambda nm: -(nr_train[nm] if np.isfinite(nr_train[nm]) else -1.0))
        top2 = order[:2]
        ranks = []
        for nm in top2:
            train_vals = np.sort(rule_raw[nm][tr])
            ranks.append(np.searchsorted(train_vals, rule_raw[nm][te], side="right") / len(train_vals))
        rank_oof[te] = np.mean(ranks, axis=0)
        rank_choice.append({"fold": k, "top2": top2,
                            "train_nr": {nm: nr_train[nm] for nm in top2}})

    # Nested L1-logistic: C chosen by inner grouped CV log-loss among fits with 1..3 features.
    def fit_l1(X, yy, C):
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000))
        model.fit(X, yy)
        return model

    l1_oof = np.full(n, np.nan)
    l1_choice = []
    for k in range(5):
        tr, te = folds != k, folds == k
        Xtr = df.loc[tr, l1_cols].to_numpy(dtype=float)
        Xte = df.loc[te, l1_cols].to_numpy(dtype=float)
        ytr, gtr = y[tr], groups[tr]
        inner = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
                     .split(Xtr, ytr, gtr))
        best = None
        for C in L1_C_GRID:
            full = fit_l1(Xtr, ytr, C)
            nnz = int(np.count_nonzero(full[-1].coef_))
            if nnz == 0 or nnz > L1_MAX_FEATURES:
                continue
            losses = []
            for itr, iva in inner:
                m = fit_l1(Xtr[itr], ytr[itr], C)
                losses.append(log_loss(ytr[iva], m.predict_proba(Xtr[iva])[:, 1], labels=[0, 1]))
            score = float(np.mean(losses))
            if best is None or score < best[0]:
                best = (score, float(C))
        C = best[1] if best else float(L1_C_GRID[0])
        model = fit_l1(Xtr, ytr, C)
        chosen = [l1_cols[i] for i in np.flatnonzero(model[-1].coef_.ravel())]
        l1_oof[te] = model.predict_proba(Xte)[:, 1]
        l1_choice.append({"fold": k, "C": C, "features": chosen, "n_features": len(chosen)})

    def metrics(s):
        def stat(rows):
            return normalised_recall(s[rows], y[rows], BUDGET)

        return {
            "normalised_recall_at_20pct": _interval(cluster_bootstrap(groups, stat, a.bootstrap, SEED)),
            "recall_at_20pct": float(recall_at_budget(s, y, BUDGET)),
            "oracle_recall_at_20pct": float(oracle_recall_at_budget(y, BUDGET)),
            "auroc": float(auroc(s, y)),
        }

    candidates = {nm: metrics(rule_raw[nm]) | {"selection": "fixed sign, no fitting"}
                  for nm in rule_names}
    candidates["nested_L1_logistic_max3"] = metrics(l1_oof) | {
        "selection": "inner 3-fold grouped log-loss over C, nnz <= 3", "per_fold": l1_choice}
    candidates["nested_rank_average_top2_rules"] = metrics(rank_oof) | {
        "selection": "top-2 single rules by training-fold NR, empirical-CDF rank average",
        "per_fold": rank_choice}
    results = {
        "description": "Dev-only (ImageCAS-X val, cas_net and swin_unetr). 5-fold patient-grouped folds "
                       "(StratifiedGroupKFold, seed 14). Rules use fixed signs; L1 and rank-average are "
                       "nested. No test data.",
        "units": n,
        "patients": int(len(np.unique(groups))),
        "critical": int(y.sum()),
        "budget_review_units": int(np.floor(BUDGET * n)),
        "bootstrap_resamples": a.bootstrap,
        "rule_signs": {k: {"feature": v[0], "sign": v[1], "dev_median_fill": rule_fill[k]}
                       for k, v in RULE_SIGNS.items()},
        "candidates": candidates,
        "input_sha256": {"table_val": sha(table_path), "features_val": sha(a.features)},
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: round(v["normalised_recall_at_20pct"]["estimate"], 3) for k, v in candidates.items()}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="/root/data")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("features")
    f.add_argument("--split", choices=("val", "test"), required=True)
    f.add_argument("--out", required=True)
    f.set_defaults(fn=cmd_features)

    h = sub.add_parser("harm")
    h.add_argument("--out-dir", required=True)
    h.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    h.add_argument("--methods", nargs="+", default=list(METHODS))
    h.set_defaults(fn=cmd_harm)

    d = sub.add_parser("dev")
    d.add_argument("--features", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--bootstrap", type=int, default=2000)
    d.set_defaults(fn=cmd_dev)

    d2 = sub.add_parser("dev2")
    d2.add_argument("--features", required=True)
    d2.add_argument("--out", required=True)
    d2.add_argument("--bootstrap", type=int, default=2000)
    d2.set_defaults(fn=cmd_dev2)

    c = sub.add_parser("clinical")
    c.add_argument("--out", required=True, help="output prefix; writes <prefix>.json and <prefix>.md")
    c.add_argument("--split", choices=("val", "test"), default="val")
    c.add_argument("--methods", nargs="+", default=list(METHODS))
    c.set_defaults(fn=cmd_clinical)

    a = p.parse_args()
    a.root = Path(a.root)
    a.fn(a)


if __name__ == "__main__":
    main()
