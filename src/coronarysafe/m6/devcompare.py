"""M6 dev comparison, Addendum G (VAL x {cas_net, swin_unetr}; never test, never FFR-UNet).

Candidates (fixed-sign monotone logistic, <=3 features, 5-fold patient-grouped CV, seed 14):
  every 3-subset of the four justified M6 features, signs fixed a priori:
    worst reachable length within 50 mm (-), worst proximal gap count (+), worst proximal radius drop (+),
    worst tree attachment (-)  [tree unattached -> riskier]
  best single justified feature with its fixed sign (no fitting)
Comparators: M1 (L2 logistic on graph features, CV), fragmentation rule (components >= 10 mm),
  proximal-only fragmentation rule (<= 60 mm from the aorta), g_share_top2 (negated).
Sharpening (descriptive): Dice bins, fraction of critical caught at 20%; fragmentation AUROC for critical
  vs severe-minor-only units; proximal-only rule.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from ..eval.bootstrap import cluster_bootstrap
from ..eval.metrics import aurc, normalised_recall, recall_at_budget
from ..eval.triage import GRAPH

SEED = 14
BUDGET = 0.2
JUSTIFIED = {  # feature -> fixed a-priori sign (+1 riskier when higher)
    "m6_worst_reach50_mm": -1,
    "m6_worst_n_prox_gaps": +1,
    "m6_worst_radius_drop_60": +1,
    "m6_worst_attached": -1,
}


def load(qa_csv, units_dir, sharp_dir):
    t = pd.read_csv(qa_csv, dtype={"scan_id": str})
    rows = [json.loads(p.read_text()) for p in Path(units_dir).glob("*__val__*.json")]
    m = pd.DataFrame(rows)
    m["scan_id"] = m["scan"].astype(str)
    m6cols = [c for c in m.columns if c.startswith("m6_")]
    m = m[["scan_id", "method"] + m6cols].drop_duplicates(["scan_id", "method"])
    s = pd.DataFrame([json.loads(p.read_text()) for p in Path(sharp_dir).glob("*.json")])
    s["scan_id"] = s["scan"].astype(str)
    d = t.merge(m, on=["scan_id", "method"], how="inner", validate="one_to_one")
    d = d.merge(s[["scan_id", "method", "prox_n_components_ge_10mm"]], on=["scan_id", "method"], how="inner",
                validate="one_to_one")
    d = d.sort_values(["scan_id", "method"]).reset_index(drop=True)
    d["critical"] = d["critical"].astype(bool)
    return d


def _mono_fit(X, y, signs, lam=1e-2):
    """Logistic with non-negative weights on sign-oriented features (monotone by construction)."""
    Xs = X * np.asarray(signs, float)
    n, k = Xs.shape

    def f(theta):
        b0, w = theta[0], theta[1:]
        z = b0 + Xs @ w
        p = 1.0 / (1.0 + np.exp(-z))
        loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)) + lam * np.sum(w ** 2)
        r = (p - y) / n
        g = np.concatenate([[r.sum()], Xs.T @ r + 2 * lam * w])
        return loss, g

    res = minimize(f, np.zeros(k + 1), jac=True, method="L-BFGS-B", bounds=[(None, None)] + [(0, None)] * k)
    return res.x


def _prep(Xtr, Xte):
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    Xte = np.where(np.isnan(Xte), med, Xte)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def cv_monotone(d, feats):
    signs = [JUSTIFIED[f] for f in feats]
    X = d[feats].to_numpy(float)
    y = d["critical"].to_numpy(float)
    score = np.full(len(d), np.nan)
    coefs = []
    split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, te in split.split(X, y, d["scan_id"]):
        Xtr, Xte = _prep(X[tr], X[te])
        th = _mono_fit(Xtr, y[tr], signs)
        score[te] = 1.0 / (1.0 + np.exp(-(th[0] + (Xte * np.asarray(signs)) @ th[1:])))
        coefs.append(th[1:].tolist())
    Xall, _ = _prep(X, X)
    th = _mono_fit(Xall, y, signs)
    return score, {"features": feats, "signs": signs, "final_weights_nonneg": dict(zip(feats, map(float, th[1:]))),
                   "fold_weights": coefs}


def cv_m1(d):
    X = d[GRAPH].to_numpy(float)
    y = d["critical"].to_numpy(int)
    score = np.full(len(d), np.nan)
    split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, te in split.split(X, y, d["scan_id"]):
        Xtr, Xte = _prep(X[tr], X[te])
        m = LogisticRegression(C=0.1, max_iter=2000).fit(Xtr, y[tr])
        score[te] = m.predict_proba(Xte)[:, 1]
    return score


def _metrics(score, d, n_boot=2000):
    y = d["critical"].to_numpy(bool)
    b = d["burden"].to_numpy(float)
    ci = cluster_bootstrap(d["scan_id"].to_numpy(), lambda rows: normalised_recall(score[rows], y[rows], BUDGET),
                           n_boot, SEED)
    return {
        "normalised_recall_at_20": float(normalised_recall(score, y, BUDGET)),
        "ci95": [float(ci["lo"]), float(ci["hi"])],
        "recall_at_20": float(recall_at_budget(score, y, BUDGET)),
        "auroc": float(roc_auc_score(y, score)),
        "aurc_burden": float(aurc(score, b)),
    }


def _topk_mask(score, k):
    idx = np.argsort(-score, kind="stable")[:k]
    m = np.zeros(len(score), bool)
    m[idx] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-table", required=True)
    ap.add_argument("--units-dir", required=True)
    ap.add_argument("--sharp-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = load(a.qa_table, a.units_dir, a.sharp_dir)
    n = len(d)
    k = int(np.floor(BUDGET * n + 1e-12))

    rankers = {}
    # fixed-sign monotone, every 3-subset of the four justified features
    mono = {}
    for sub in itertools.combinations(list(JUSTIFIED), 3):
        s, info = cv_monotone(d, list(sub))
        mono[",".join(sub)] = (s, info)
        rankers["M6 monotone: " + ",".join(sub)] = s
    # best single justified feature with its fixed sign (no fitting)
    singles = {f: sg * d[f].to_numpy(float) for f, sg in JUSTIFIED.items()}
    single_res = {f: _metrics(singles[f], d) for f in singles}
    best_single = max(single_res, key=lambda f: (single_res[f]["normalised_recall_at_20"], single_res[f]["auroc"]))
    rankers[f"M6 single (fixed sign): {best_single}"] = singles[best_single]
    # comparators
    rankers["M1 (graph L2 logistic)"] = cv_m1(d)
    rankers["fragmentation rule (components >=10 mm)"] = d["g_n_components_ge_10mm"].fillna(0).to_numpy(float)
    rankers["proximal-only fragmentation rule (<=60 mm)"] = d["prox_n_components_ge_10mm"].fillna(0).to_numpy(float)
    rankers["g_share_top2 (negated)"] = -d["g_share_top2"].fillna(d["g_share_top2"].median()).to_numpy(float)

    res = {"n_units": n, "n_patients": int(d.scan_id.nunique()), "n_critical": int(d.critical.sum()),
           "budget_k": k, "rankers": {}, "single_features": single_res, "monotone_detail": {
               name: info for name, (_, info) in mono.items()}}
    for name, s in rankers.items():
        res["rankers"][name] = _metrics(s, d)

    # selection (dev): among the M6 candidates, highest normalised recall, tie -> lower AURC
    m6_names = [n_ for n_ in res["rankers"] if n_.startswith("M6")]
    chosen = sorted(m6_names, key=lambda n_: (-res["rankers"][n_]["normalised_recall_at_20"],
                                              res["rankers"][n_]["aurc_burden"]))[0]
    res["chosen_m6_candidate"] = chosen

    # sharpening (i): Dice bins
    bins = {"<0.85": d.dice < 0.85, "0.85-0.90": (d.dice >= 0.85) & (d.dice < 0.90), ">=0.90": d.dice >= 0.90}
    crit = d.critical.to_numpy(bool)
    dice_res = {}
    for bname, bm in bins.items():
        bm = bm.to_numpy(bool)
        entry = {"units": int(bm.sum()), "critical": int((bm & crit).sum()), "caught_at_20pct": {}}
        for rname, s in rankers.items():
            top = _topk_mask(s, k)
            c = bm & crit
            entry["caught_at_20pct"][rname] = float((top & c).sum() / c.sum()) if c.sum() else float("nan")
        dice_res[bname] = entry
    res["dice_bins"] = dice_res

    # sharpening (ii): fragmentation AUROC, critical vs severe-minor-only, and proximal-only rule
    sev_only = (d.severe.astype(bool) & ~d.critical).to_numpy(bool)
    sub = crit | sev_only
    frag = d["g_n_components_ge_10mm"].fillna(0).to_numpy(float)
    prox = d["prox_n_components_ge_10mm"].fillna(0).to_numpy(float)
    res["fragmentation_auroc"] = {
        "all_units_critical_vs_rest": float(roc_auc_score(crit, frag)),
        "critical_vs_severe_minor_only": float(roc_auc_score(crit[sub], frag[sub])) if sub.sum() and crit[sub].any() and (~crit[sub]).any() else float("nan"),
        "n_severe_minor_only": int(sev_only.sum()),
        "proximal_critical_vs_rest": float(roc_auc_score(crit, prox)),
        "proximal_critical_vs_severe_minor_only": float(roc_auc_score(crit[sub], prox[sub])) if sub.sum() and crit[sub].any() and (~crit[sub]).any() else float("nan"),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, default=float))

    lines = ["# M6 dev comparison, Addendum G (VAL x {cas_net, swin_unetr}, patient-grouped CV)", "",
             f"Units {n}, patients {res['n_patients']}, critical {res['n_critical']}, budget k={k}. "
             "Oracle at 20% is trivially 1 (criticals < budget), so normalised recall = recall.", "",
             "| ranker | norm. recall@20 (95% CI) | AUROC | AURC(burden) |", "|---|---|---|---|"]
    for name, r in res["rankers"].items():
        lines.append(f"| {name} | {r['normalised_recall_at_20']:.3f} ({r['ci95'][0]:.2f}–{r['ci95'][1]:.2f}) | "
                     f"{r['auroc']:.3f} | {r['aurc_burden']:.4f} |")
    lines += ["", f"Chosen M6 candidate (dev rule): {chosen}", "",
              "| Dice bin | units | critical | " + " | ".join(
                  ["caught@20%: " + r for r in ["fragmentation", "prox-frag", "M1", chosen]]) + " |",
              "|---|---|---|---|---|---|---|"]
    for bname, e in dice_res.items():
        c = e["caught_at_20pct"]
        cells = [c["fragmentation rule (components >=10 mm)"], c["proximal-only fragmentation rule (<=60 mm)"],
                 c["M1 (graph L2 logistic)"], c[chosen]]
        lines.append(f"| {bname} | {e['units']} | {e['critical']} | " + " | ".join(f"{x:.2f}" for x in cells) + " |")
    fa = res["fragmentation_auroc"]
    lines += ["", f"Fragmentation AUROC, critical vs all: {fa['all_units_critical_vs_rest']:.3f}; "
                  f"critical vs severe-minor-only (n={fa['n_severe_minor_only']}): {fa['critical_vs_severe_minor_only']:.3f}; "
                  f"proximal-only rule: vs rest {fa['proximal_critical_vs_rest']:.3f}, vs severe-minor-only {fa['proximal_critical_vs_severe_minor_only']:.3f}."]
    Path(a.out).with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"chosen": chosen, "n": n, "critical": res["n_critical"]}, indent=1))


if __name__ == "__main__":
    main()
