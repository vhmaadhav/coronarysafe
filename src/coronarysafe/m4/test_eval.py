"""Sealed test evaluation of the frozen model M1+M3+M4 (Addendum H). Single-pass CLI.

  python -m coronarysafe.m4.test_eval --split test --out /root/data/test_eval
  python -m coronarysafe.m4.test_eval --split val  --out /root/data/test_eval_dryrun   (dry run, in-sample)

Startup verifies the frozen bundle sha256 against Addendum H and aborts on mismatch.
Per segmenter (ffr_unet primary; cas_net, swin_unetr secondary; nnunet_mv for S3 on test):
  frozen model (fixed coefficients from dev), pre-specified rule n_components >= 10 mm,
  exploratory g_share_top2 (negated), M1+M3 sensitivity (refit on dev, L2 logistic, C=0.1).
S1 NR >= 0.75; S2 paired patient bootstrap (2000, seed 0) of NR(model) - NR(rule) >= +0.10 with CI lo > 0;
S3 frozen on ffr minus best ensemble comparator on nnunet_mv >= -0.05 (independent bootstrap, units differ);
S4 recall among critical units with Dice >= 0.90 in the 20% set; S5 not implemented (scan-level model);
LOSO: refit on dev units of the other two segmenters, evaluate on each held-out segmenter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from ..eval.bootstrap import paired_bootstrap_difference  # noqa: E402
from ..eval.metrics import aurc, normalised_recall, recall_at_budget  # noqa: E402
from ..eval.triage import GRAPH  # noqa: E402

FROZEN_DIR = Path("/root/data/m4/frozen")
EXPECTED_SHA = {  # Addendum H, sha256 of the frozen bundle
    "frozen_model.json": "56a666ef39ed652b85e65fafb18101841dc8963cecff4b27df5ef18de0f7aa5d",
    "m4_model.pt": "baf841cfba4d0f08b21cbead0239e930d1f8db19a744720733b94afbfb30a266",
    "lr_convention.json": "53ea24bfdfc8978c6ec02b979789bda1a068dad527457cc1fad7b0d182a3f6db",
}
OST = ["n_ostia", "ostia_deviation", "left_attached", "right_attached",
       "unattached_len_mm", "unattached_share", "aorta_quality_ok"]
SEGMENTERS = ["ffr_unet", "cas_net", "swin_unetr"]
BUDGET = 0.2
N_BOOT = 2000
SEED = 0


def verify_frozen():
    for name, want in EXPECTED_SHA.items():
        got = hashlib.sha256((FROZEN_DIR / name).read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"ABORT: {name} sha256 {got} != Addendum H {want}")
    return json.loads((FROZEN_DIR / "frozen_model.json").read_text()), \
        json.loads((FROZEN_DIR / "lr_convention.json").read_text())["summary"]["left_world_unit"]


def unit_features(root, method, sid, aorta_dir, left_unit, net, dev):
    from pipeline.label_core import _gt_seg  # noqa: F401  (import check only)
    from coronarysafe.io import load_nifti
    from coronarysafe.features.ostium import ostium_features
    from coronarysafe.m4.lr_convention import lr_from_affine
    from coronarysafe.m4.score import m4_score

    qa = json.loads((root / "qa" / "v3" / method / f"{sid}.json").read_text())
    rec = {"method": method, "scan_id": sid, "critical": int(bool(qa["critical"])),
           "burden": float(qa["burden"]), "dice": float(qa["dice"]), "severe": bool(qa["severe"])}
    for k, v in (qa.get("graph") or {}).items():
        rec[f"g_{k}"] = float(v)
    mask_arr, sp, paff = load_nifti(root / "preds" / method / sid / "mask.nii.gz")
    mask = np.asarray(mask_arr) > 0
    aorta = np.asarray(load_nifti(Path(aorta_dir) / f"{sid}.nii.gz")[0]) > 0
    ax, sg = lr_from_affine(paff, left_unit)
    ost = ostium_features(mask, aorta, sp, lr_axis=ax, lr_sign=sg)
    for k in OST:
        rec[f"m3_{k}"] = float(ost[k])
    p = float(m4_score(net, mask, sp, dev))
    rec["m4_prob"] = p
    rec["m4_logit"] = float(np.log(max(p, 1e-6) / max(1 - p, 1e-6)))
    return rec


def build_table(root, split, aorta_dir, left_unit, net, dev):
    from pipeline.label_core import read_split
    scans = read_split(str(root), split)
    methods = list(SEGMENTERS)
    if split == "test" and (root / "preds" / "nnunet_mv").exists():
        methods.append("nnunet_mv")
    rows = []
    for m in methods:
        for s in scans:
            if not (root / "preds" / m / s / "mask.nii.gz").exists() or not (root / "qa" / "v3" / m / f"{s}.json").exists():
                continue
            rows.append(unit_features(root, m, s, aorta_dir, left_unit, net, dev))
    return pd.DataFrame(rows)


def apply_frozen(df, frozen):
    cols = frozen["features"]
    pre = frozen["preprocessing"]
    x = np.column_stack([df[c].to_numpy(float) for c in cols])
    med = np.array([pre["impute_median"][c] for c in cols])
    x = np.where(np.isnan(x), med, x)
    mu = np.array([pre["scale_mean"][c] for c in cols])
    sd = np.array([pre["scale_std"][c] for c in cols])
    z = frozen["logistic"]["intercept"] + ((x - mu) / sd) @ np.array([frozen["logistic"]["coef"][c] for c in cols])
    return 1.0 / (1.0 + np.exp(-z))


def fit_spec(dfs, feats):
    tr = pd.concat(dfs)
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(C=0.1, max_iter=1000))
    m.fit(tr[feats], tr["critical"].astype(int))
    return m


def metrics(score, df):
    y = df["critical"].to_numpy(bool)
    b = df["burden"].to_numpy(float)
    n = len(y)
    k = int(np.floor(BUDGET * n + 1e-12))
    npos = int(y.sum())
    rec = float(recall_at_budget(score, y, BUDGET))
    oracle = min(k, npos) / npos if npos else float("nan")
    return {
        "n": n, "n_critical": npos, "prevalence": npos / n, "k": k,
        "recall_at_20": rec, "oracle_at_20": oracle,
        "normalised_recall": float(normalised_recall(score, y, BUDGET)),
        "auroc": float(roc_auc_score(y, score)) if 0 < npos < n else float("nan"),
        "aurc_burden": float(aurc(score, b)),
    }


def nr_on(score, df):
    return float(normalised_recall(score, df["critical"].to_numpy(bool), BUDGET))


def paired(df, score_a, score_b):
    """Paired patient bootstrap of NR(a) - NR(b) on the same units (one unit per patient per segmenter)."""
    y = df["critical"].to_numpy(bool)
    res = paired_bootstrap_difference(
        df["scan_id"].to_numpy(),
        lambda rows: normalised_recall(score_a[rows], y[rows], BUDGET),
        lambda rows: normalised_recall(score_b[rows], y[rows], BUDGET),
        n_boot=N_BOOT, seed=SEED,
    )
    return {"estimate": float(res["estimate"]), "lo": float(res["lo"]), "hi": float(res["hi"])}


def s4_dice90(df, score):
    y = df["critical"].to_numpy(bool)
    top = np.zeros(len(df), bool)
    top[np.argsort(-score, kind="stable")[: int(np.floor(BUDGET * len(df) + 1e-12))]] = True
    sel = y & (df["dice"].to_numpy(float) >= 0.90)
    if sel.sum() == 0:
        return {"n_critical_dice90": 0, "recall": float("nan")}
    return {"n_critical_dice90": int(sel.sum()), "recall": float((top & sel).sum() / sel.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default="/root/data")
    ap.add_argument("--aorta-dir", default="/root/data/m4/aorta")
    ap.add_argument("--ensemble-csv", default="/root/data/test_eval/ensemble_comparators.csv")
    a = ap.parse_args()

    frozen, left_unit = verify_frozen()
    import torch
    from coronarysafe.m4.score import load_model
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_model(str(FROZEN_DIR / "m4_model.pt"), dev)
    root = Path(a.root)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # features for the split, and the dev units (val x {cas_net, swin_unetr}) used for refits
    df = build_table(root, a.split, a.aorta_dir, left_unit, net, dev)
    if a.split == "test":
        val = build_table(root, "val", a.aorta_dir, left_unit, net, dev)
    else:
        val = df[df["method"].isin(SEGMENTERS)]
    df.to_csv(out / "features.csv", index=False)
    dev_main = val[val["method"].isin(["cas_net", "swin_unetr"])]

    feats_frozen = frozen["features"]
    # consistency check: refit on dev reproduces the frozen coefficients (sanity, not a result)
    refit = fit_spec([dev_main], feats_frozen)
    coef_refit = refit[-1].coef_[0]
    coef_frozen = np.array([frozen["logistic"]["coef"][c] for c in feats_frozen])
    refit_diff = float(np.max(np.abs(coef_refit - coef_frozen)))

    feats_m1m3 = GRAPH + [f"m3_{k}" for k in OST]
    m1m3 = fit_spec([dev_main], feats_m1m3)

    results = {"split": a.split, "frozen_sha256_verified": True,
               "frozen_refit_max_coef_diff": refit_diff, "per_segmenter": {}, "S": {}}
    flags = []
    segs_present = [m for m in ["ffr_unet", "cas_net", "swin_unetr", "nnunet_mv"] if m in set(df["method"])]
    for m in segs_present:
        d = df[df["method"] == m].reset_index(drop=True)
        s_frozen = apply_frozen(d, frozen)
        s_rule = d["g_n_components_ge_10mm"].fillna(0).to_numpy(float)
        s_share = -d["g_share_top2"].fillna(d["g_share_top2"].median()).to_numpy(float)
        s_m1m3 = m1m3.predict_proba(d[feats_m1m3])[:, 1]
        entry = {
            "frozen": metrics(s_frozen, d),
            "rule_components_ge_10mm": metrics(s_rule, d),
            "exploratory_g_share_top2": metrics(s_share, d),
            "m1m3_sensitivity": metrics(s_m1m3, d),
            "S4_dice90_recall": {"frozen": s4_dice90(d, s_frozen), "rule": s4_dice90(d, s_rule)},
        }
        entry["paired_frozen_minus_rule"] = paired(d, s_frozen, s_rule)
        entry["paired_m1m3_minus_rule"] = paired(d, s_m1m3, s_rule)
        results["per_segmenter"][m] = entry
        k = entry["frozen"]["k"]
        top = np.zeros(len(d), bool)
        top[np.argsort(-s_frozen, kind="stable")[:k]] = True
        for i in range(len(d)):
            flags.append({"method": m, "scan_id": d.loc[i, "scan_id"], "score": float(s_frozen[i]),
                          "flagged_top20": int(top[i])})
    pd.DataFrame(flags).to_csv(out / "flags.csv", index=False)

    # LOSO: refit the same spec on dev units of the other two segmenters, evaluate on the held-out one
    loso = {}
    if a.split == "test":
        for h in ["ffr_unet", "cas_net", "swin_unetr"]:
            others = [val[val["method"] == o] for o in SEGMENTERS if o != h]
            mdl = fit_spec(others, feats_frozen)
            d = df[df["method"] == h].reset_index(drop=True)
            s = mdl.predict_proba(d[feats_frozen])[:, 1]
            rule = d["g_n_components_ge_10mm"].fillna(0).to_numpy(float)
            loso[h] = {"model": metrics(s, d), "rule": metrics(rule, d),
                       "paired_model_minus_rule": paired(d, s, rule)}
    results["LOSO"] = loso

    # S3: ensemble comparators on nnunet_mv (test only), frozen on nnunet_mv, independent bootstrap for the difference
    s3 = {"available": False}
    if a.split == "test" and "nnunet_mv" in segs_present and Path(a.ensemble_csv).exists():
        ens = pd.read_csv(a.ensemble_csv, dtype={"scan_id": str})
        mv = df[df["method"] == "nnunet_mv"].merge(ens, on="scan_id", how="left")
        comps = {"pairwise_dice": -mv["mean_pair_dice"].to_numpy(float),
                 "pairwise_cldice": -mv["mean_pair_cldice"].to_numpy(float),
                 "pairwise_betti0": mv["mean_pair_b0err"].to_numpy(float)}
        comp_nr = {c: nr_on(s, mv) for c, s in comps.items()}
        best = max(comp_nr, key=comp_nr.get)
        frozen_mv = apply_frozen(mv, frozen)
        ffr = df[df["method"] == "ffr_unet"].reset_index(drop=True)
        frozen_ffr = apply_frozen(ffr, frozen)
        nr_ffr = nr_on(frozen_ffr, ffr)
        rng = np.random.default_rng(SEED)
        y1, y2 = ffr["critical"].to_numpy(bool), mv["critical"].to_numpy(bool)
        s_ffr, s_best = frozen_ffr, comps[best]
        diffs = []
        for _ in range(N_BOOT):
            i1 = rng.integers(0, len(y1), len(y1))
            i2 = rng.integers(0, len(y2), len(y2))
            a_ = normalised_recall(s_ffr[i1], y1[i1], BUDGET)
            b_ = normalised_recall(s_best[i2], y2[i2], BUDGET)
            diffs.append(a_ - b_)
        diffs = np.asarray(diffs, float)
        s3 = {"available": True, "ensemble_comparator_nr": comp_nr, "best_comparator": best,
              "frozen_on_nnunet_mv": metrics(frozen_mv, mv),
              "frozen_on_ffr_nr": nr_ffr,
              "difference_frozen_ffr_minus_best_comparator": float(nr_ffr - comp_nr[best]),
              "ci95": [float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))],
              "units_differ": "ffr_unet units vs nnU-Net majority-vote units (same 160 patients)"}
    results["S3"] = s3

    prim = results["per_segmenter"].get("ffr_unet", {})
    nr = prim.get("frozen", {}).get("normalised_recall", float("nan"))
    p = prim.get("paired_frozen_minus_rule", {"estimate": float("nan"), "lo": float("nan")})
    S = {
        "S1_frozen_ffr_NR_ge_0.75": {"value": nr, "pass": bool(nr >= 0.75)},
        "S2_frozen_minus_rule_ge_0.10_CI_gt_0": {"estimate": p["estimate"], "ci_lo": p["lo"], "ci_hi": p.get("hi"),
                                                "pass": bool(p["estimate"] >= 0.10 and p["lo"] > 0)},
        "S3_noninferior_vs_best_ensemble_ge_-0.05": {
            "difference": s3.get("difference_frozen_ffr_minus_best_comparator"),
            "ci95": s3.get("ci95"), "pass": bool(s3.get("difference_frozen_ffr_minus_best_comparator", -9) >= -0.05)
            if s3.get("available") else None},
        "S4_dice90_critical_recall_frozen_ffr": prim.get("S4_dice90_recall", {}).get("frozen"),
        "S5": "not implemented (scan-level model, no segment-level score)",
    }
    results["S"] = S
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))

    lines = [f"# Sealed test evaluation (split={a.split}), frozen M1+M3+M4", "",
             f"Frozen bundle sha256 verified. Refit-vs-frozen max coefficient diff: {refit_diff:.2e}.", "",
             "| check | value | CI / note | pass |", "|---|---|---|---|",
             f"| S1 NR ffr_unet frozen (>=0.75) | {nr:.3f} | | {'PASS' if S['S1_frozen_ffr_NR_ge_0.75']['pass'] else 'FAIL'} |",
             f"| S2 frozen - rule(components>=10mm) | {p['estimate']:.3f} | [{p['lo']:.3f}, {p.get('hi', float('nan')):.3f}] | "
             f"{'PASS' if S['S2_frozen_minus_rule_ge_0.10_CI_gt_0']['pass'] else 'FAIL'} |",
             f"| S3 frozen(ffr) - best ensemble ({s3.get('best_comparator', 'n/a')}) | "
             f"{s3.get('difference_frozen_ffr_minus_best_comparator', float('nan')):.3f} | "
             f"{s3.get('ci95', 'n/a')} | {S['S3_noninferior_vs_best_ensemble_ge_-0.05']['pass']} |",
             "| S4 Dice>=0.90 critical recall (ffr frozen) | "
             f"{S['S4_dice90_critical_recall_frozen_ffr']['recall']:.3f} | n={S['S4_dice90_critical_recall_frozen_ffr']['n_critical_dice90']} | - |",
             "| S5 | not implemented | scan-level model | - |", "",
             "| segmenter | model NR | rule NR | model AUROC | prevalence |", "|---|---|---|---|---|"]
    for m, e in results["per_segmenter"].items():
        lines.append(f"| {m} | {e['frozen']['normalised_recall']:.3f} | {e['rule_components_ge_10mm']['normalised_recall']:.3f} | "
                     f"{e['frozen']['auroc']:.3f} | {e['frozen']['prevalence']:.3f} |")
    if loso:
        lines += ["", "LOSO (refit on the other two segmenters' dev units):", "", "| held-out | model NR | rule NR |", "|---|---|---|"]
        for h, e in loso.items():
            lines.append(f"| {h} | {e['model']['normalised_recall']:.3f} | {e['rule']['normalised_recall']:.3f} |")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(S, default=float, indent=1))
    print("TEST_EVAL_CLI_DONE", flush=True)


if __name__ == "__main__":
    main()
