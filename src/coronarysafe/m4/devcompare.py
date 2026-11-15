"""Dev-only comparison (ImageCAS-X VAL x {cas_net, swin_unetr}) and freezing of ONE model.

Never reads test or FFR-UNet rows. Inputs:
  --table   v3 QA table of VAL units (parquet, from the queue's label stage; graph + uncertainty columns)
  --units   output of coronarysafe.m4.score (per unit: critical, burden, m4_prob, m3 dict)
Candidates (5-fold patient-grouped CV on dev, one fit per fold, all rows of a patient kept together):
  M1        graph features (triage.GRAPH)
  M1+M3     graph + ostium features
  M1+M3+M4  graph + ostium + M4 logit (stacked)
  M4        M4 probability alone
Choice rule (dev only): highest normalised recall@20%; ties broken by lower AURC(burden).
The frozen model is refit on all dev rows and written with its feature list, preprocessing and sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..eval.bootstrap import cluster_bootstrap
from ..eval.metrics import aurc, auroc, normalised_recall, oracle_recall_at_budget, recall_at_budget
from ..eval.triage import GRAPH, UNCERTAINTY, rule_scores, strongest_rule

OST = ["n_ostia", "ostia_deviation", "left_attached", "right_attached",
       "unattached_len_mm", "unattached_share", "aorta_quality_ok"]
SEED = 14
BUDGET = 0.2
CANDIDATES = {
    "M1": ["graph"],
    "M1+M3": ["graph", "ost"],
    "M1+M3+M4": ["graph", "ost", "m4"],
    "M4": ["m4"],
}


def _cols(groups):
    cols = []
    for g in groups:
        cols += {"graph": GRAPH, "ost": [f"m3_{c}" for c in OST], "m4": ["m4_logit"]}[g]
    return cols


def build_frame(table_path, units_path):
    t = pd.read_csv(table_path) if str(table_path).endswith(".csv") else pd.read_parquet(table_path)
    if "patient_group" in t and "scan_id" not in t:
        t = t.rename(columns={"patient_group": "scan_id"})
    t["scan_id"] = t["scan_id"].astype(str)
    u = json.loads(Path(units_path).read_text())
    rows = []
    for x in u:
        if x.get("status") != "ok":
            continue
        r = {"scan_id": str(x["scan"]), "method": x["method"], "m4_prob": x["m4_prob"]}
        r["m4_logit"] = float(np.log(max(x["m4_prob"], 1e-6) / max(1 - x["m4_prob"], 1e-6)))
        for k in OST:
            v = x["m3"].get(k)
            r[f"m3_{k}"] = np.nan if v is None else float(v)
        rows.append(r)
    m = pd.DataFrame(rows)
    d = t.merge(m, on=["scan_id", "method"], how="inner", validate="one_to_one")
    d = d.sort_values(["scan_id", "method"]).reset_index(drop=True)
    d["critical"] = d["critical"].astype(bool)
    return d


def _model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=1000))


def cv_scores(d, cols, seed=SEED):
    split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    out = np.full(len(d), np.nan)
    for tr, te in split.split(d, d.critical, d.scan_id):
        m = _model()
        m.fit(d.iloc[tr][cols], d.iloc[tr].critical)
        out[te] = m.predict_proba(d.iloc[te][cols])[:, 1]
    return out


def metrics(score, d, n_boot=2000):
    y = d.critical.to_numpy().astype(bool)
    b = d.burden.to_numpy()
    ci = cluster_bootstrap(d.scan_id.to_numpy(),
                           lambda rows: normalised_recall(score[rows], y[rows], BUDGET),
                           n_boot, SEED)
    return {
        "normalised_recall_at_20": float(normalised_recall(score, y, BUDGET)),
        "recall_at_20": float(recall_at_budget(score, y, BUDGET)),
        "oracle_recall_at_20": float(oracle_recall_at_budget(y, BUDGET)),
        "auroc_critical": float(auroc(score, y)),
        "aurc_burden": float(aurc(score, b)),
        "n_units": int(len(d)),
        "n_critical": int(y.sum()),
        "ci95_normalised_recall": [float(ci["lo"]), float(ci["hi"])],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--units", required=True)
    ap.add_argument("--m4-weights", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fetch-dir", default="")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    d = build_frame(a.table, a.units)
    if d.scan_id.nunique() < 2 or not d.critical.any() or d.critical.all():
        raise SystemExit("dev frame lacks both labels or patients")

    results = {"n_units": int(len(d)), "n_patients": int(d.scan_id.nunique()), "candidates": {}}
    for name, groups in CANDIDATES.items():
        cols = _cols(groups)
        s = cv_scores(d, cols)
        results["candidates"][name] = metrics(s, d)
    # simple rules, selected on dev (same rule set as the plan)
    rule_res = {}
    for method, td in d.groupby("method"):
        for rname, rs in rule_scores(td).items():
            rule_res.setdefault(rname, []).append(float(normalised_recall(rs, td.critical.to_numpy(), BUDGET)))
    rule_means = {k: float(np.nanmean(v)) for k, v in rule_res.items()}
    results["simple_rules_normalised_recall_mean"] = rule_means
    results["best_simple_rule"] = max(rule_means, key=rule_means.get) if rule_means else None

    # choose ONE by dev rule
    ranked = sorted(results["candidates"].items(),
                    key=lambda kv: (-kv[1]["normalised_recall_at_20"], kv[1]["aurc_burden"]))
    chosen = ranked[0][0]
    results["chosen"] = chosen
    results["choice_rule"] = "highest normalised recall@20 (dev, patient-grouped CV); tie -> lower AURC(burden)"

    # refit the chosen model on all dev rows and freeze
    cols = _cols(CANDIDATES[chosen])
    final = _model()
    final.fit(d[cols], d.critical)
    imp = final.named_steps["simpleimputer"]
    sc = final.named_steps["standardscaler"]
    lr = final.named_steps["logisticregression"]
    frozen = {
        "name": chosen,
        "features": cols,
        "preprocessing": {
            "impute_median": dict(zip(cols, map(float, imp.statistics_))),
            "scale_mean": dict(zip(cols, map(float, sc.mean_))),
            "scale_std": dict(zip(cols, map(float, sc.scale_))),
        },
        "logistic": {"coef": dict(zip(cols, map(float, lr.coef_[0]))), "intercept": float(lr.intercept_[0]),
                     "C": 0.1},
        "m4_weights": a.m4_weights,
        "m4_weights_sha256": hashlib.sha256(Path(a.m4_weights).read_bytes()).hexdigest(),
        "m4_input": "binary corrupted lumen, 96^3, 2 mm (coronarysafe.m4.build.to_grid)",
        "m3_orientation": "coronarysafe.m4.lr_convention (train-GT fitted, per-scan affine)",
        "trained_on": "VAL dev units only (ImageCAS-X val x cas_net, swin_unetr); TEST never read",
    }
    frozen_path = out / "frozen_model.json"
    frozen_path.write_text(json.dumps(frozen, indent=2))
    results["frozen_model"] = {"path": str(frozen_path),
                               "sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest()}
    (out / "dev_results.json").write_text(json.dumps(results, indent=2, default=float))
    lines = [
        "# M4 / M1 dev comparison (ImageCAS-X VAL x {cas_net, swin_unetr})",
        "",
        f"Units: {results['n_units']} over {results['n_patients']} patients. Patient-grouped 5-fold CV.",
        "",
        "| model | normalised recall@20 | recall@20 | oracle@20 | AUROC (critical) | AURC (burden) |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in results["candidates"].items():
        lines.append(f"| {name} | {r['normalised_recall_at_20']:.3f} | {r['recall_at_20']:.3f} | "
                     f"{r['oracle_recall_at_20']:.3f} | {r['auroc_critical']:.3f} | {r['aurc_burden']:.4f} |")
    lines += ["", f"Best simple rule (dev mean normalised recall): {results['best_simple_rule']}",
              f"Chosen (frozen): {chosen}. Weights sha256: {frozen['m4_weights_sha256']}"]
    (out / "dev_results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"chosen": chosen, "candidates": {k: v["normalised_recall_at_20"] for k, v in results["candidates"].items()}}, indent=1))


if __name__ == "__main__":
    main()
