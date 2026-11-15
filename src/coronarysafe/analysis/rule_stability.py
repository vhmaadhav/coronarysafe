"""Addendum N (FROZEN 2026-11-15): rule stability (N1) and frozen-model feature ablation (N2). Val/dev only, CPU.

Subcommands
  dev-scores  frozen-model scores of the 160 dev units (cas_net, swin_unetr on val) -> scores CSV
              (method, scan_id, score). Input to burden_indep Phase 1. No fitting: frozen coefficients only.
  n           N1 rule stability on the val units, and N2 in-sample feature ablation on the 160 dev units.

N1: for each pre-specified simple rule (components, gaps, total-length z, max-radius z) and the EXPLORATORY
    g_share_top2 (negated):
    - ranking form (the frozen comparator, unchanged): NR@20 and AUROC on the feature ranking;
    - flag form at c in {0.8, 1.0, 1.2}: a unit is flagged when the rule's numeric cut-off is passed.
      Cut-offs: count rules flag count >= c * 1; z rules flag z >= c * 1 (z from the frozen scaler's
      mean and std); g_share_top2 flags share <= c * median(share) (exploratory).
    DEVIATION (reported): the 10 mm component-length cut and the 8 mm gap-distance cut are hard-coded
    inside the graph extractor and only their counts are stored in qa/v3, so they are not re-derived;
    the perturbation is applied to the stored feature cut-off instead.
N2: for each of the frozen model's 22 inputs, refit the identical L2 logistic (C=0.1, median impute,
    z-score) on the 21 remaining inputs using the 160 dev units. IN-SAMPLE ablation, a post-hoc
    diagnostic on the fitting set only; it is not evidence about test.

  python -m coronarysafe.analysis.rule_stability dev-scores --root /root/data --dev-units DEV.json \
      --frozen-dir /root/data/m4/frozen --out scores_dev.csv
  python -m coronarysafe.analysis.rule_stability n --root /root/data --dev-units DEV.json \
      --unit-dir UNIT_JSONS --frozen-dir /root/data/m4/frozen --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..eval.metrics import aurc, normalised_recall
from ..m4.valscore import apply_frozen_lr, assemble_features, verify_frozen_dir
from .burden_indep import BUDGET
from .label_robust import read_split_ids

CUTS = (0.8, 1.0, 1.2)
DEV_METHODS = ("cas_net", "swin_unetr")


def _val_and_test_ids(root: Path):
    val = read_split_ids(root, "val")
    test = set(read_split_ids(root, "test"))
    if set(val) & test:
        raise SystemExit("ABORT: val scan ids overlap filelist/test.txt")
    return val


def _load_dev_units(path) -> dict:
    out = {}
    for u in json.loads(Path(path).read_text()):
        if u.get("status") == "ok" and "m4_prob" in u:
            out[(u["method"], str(u["scan"]))] = u
    return out


def build_table(root: Path, frozen: dict, methods, dev_units: dict, unit_dir: Path | None):
    """One row per (method, val scan): critical, burden (v3), and the frozen 22 inputs."""
    val = _val_and_test_ids(root)
    rows, skipped = [], {}
    for m in methods:
        for s in val:
            qa_path = root / "qa" / "v3" / m / f"{s}.json"
            unit = dev_units.get((m, s))
            if unit is None and unit_dir is not None:
                jp = unit_dir / f"{m}__{s}.json"
                if jp.exists():
                    cand = json.loads(jp.read_text())
                    if cand.get("status") in ("ok", "cached") and "m4_prob" in cand:
                        unit = cand
            if unit is None or not qa_path.exists():
                skipped[m] = skipped.get(m, 0) + 1
                continue
            qa = json.loads(qa_path.read_text())
            values, missing = assemble_features(qa, unit, frozen["features"])
            if missing:
                skipped[m] = skipped.get(m, 0) + 1
                continue
            rows.append({
                "method": m, "scan_id": s, "critical": bool(qa["critical"]),
                "burden": float(qa["burden"]), **values,
                "origin": "dev" if (m, s) in dev_units else "unit_json",
            })
    return pd.DataFrame(rows), skipped


def frozen_scores(df: pd.DataFrame, frozen: dict) -> np.ndarray:
    return np.array([apply_frozen_lr({k: r[k] for k in frozen["features"]}, frozen)
                     for _, r in df.iterrows()])


# ------------------------------------------------------------------ N1


def _rule_specs(frozen: dict, df: pd.DataFrame):
    """(name, feature, direction, base cut-off, ranking score) for each pre-specified rule."""
    pre = frozen["preprocessing"]
    z = lambda f: (df[f] - pre["scale_mean"][f]) / pre["scale_std"][f]  # noqa: E731
    share_med = float(df["g_share_top2"].median())
    return {
        "rule_n_components_ge_10mm": dict(kind="count", feat="g_n_components_ge_10mm", base=1.0,
                                          rank=df["g_n_components_ge_10mm"].fillna(0.0).to_numpy(float),
                                          flag=lambda c: df["g_n_components_ge_10mm"].fillna(0.0).to_numpy(float) >= c * 1.0),
        "rule_n_gaps": dict(kind="count", feat="g_n_gaps", base=1.0,
                            rank=df["g_n_gaps"].fillna(0.0).to_numpy(float),
                            flag=lambda c: df["g_n_gaps"].fillna(0.0).to_numpy(float) >= c * 1.0),
        "rule_total_length_z": dict(kind="z", feat="g_total_length_mm", base=1.0,
                                    rank=z("g_total_length_mm").fillna(z("g_total_length_mm").min()).to_numpy(float),
                                    flag=lambda c: z("g_total_length_mm").fillna(-np.inf).to_numpy(float) >= c * 1.0),
        "rule_max_radius_z": dict(kind="z", feat="g_max_radius_mm", base=1.0,
                                  rank=z("g_max_radius_mm").fillna(z("g_max_radius_mm").min()).to_numpy(float),
                                  flag=lambda c: z("g_max_radius_mm").fillna(-np.inf).to_numpy(float) >= c * 1.0),
        "exploratory_neg_share_top2": dict(kind="share", feat="g_share_top2", base=share_med,
                                           rank=-df["g_share_top2"].fillna(share_med).to_numpy(float),
                                           flag=lambda c: df["g_share_top2"].fillna(share_med).to_numpy(float) <= c * share_med),
    }


def _metrics(score, y, burden=None):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return {"nr20": float("nan"), "auroc": float("nan")}
    return {
        "nr20": float(normalised_recall(score, y, BUDGET)),
        "auroc": float(roc_auc_score(y, score)),
    }


def n1_table(df: pd.DataFrame, frozen: dict) -> pd.DataFrame:
    rules = _rule_specs(frozen, df)
    groups = [("pooled", df)] + [(m, df[df["method"] == m]) for m in sorted(df["method"].unique())]
    out = []
    for gname, g in groups:
        idx = g.index.to_numpy()  # df has a RangeIndex, so labels are positions
        for name, spec in rules.items():
            rank = np.asarray(spec["rank"], float)[idx]
            rec = {"group": gname, "n": len(g), "rule": name, "form": "ranking",
                   "cut": "", **_metrics(rank, g["critical"].to_numpy(bool))}
            out.append(rec)
            for c in CUTS:
                flag = np.asarray(spec["flag"](c), float)[idx]
                out.append({"group": gname, "n": len(g), "rule": name, "form": "flag",
                            "cut": f"x{c:.1f}", "flagged_fraction": float(flag.mean()),
                            **_metrics(flag, g["critical"].to_numpy(bool))})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ N2


def _fit(feats, dev: pd.DataFrame):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(C=0.1, max_iter=1000))
    m.fit(dev[feats], dev["critical"].astype(int))
    return m


def n2_table(dev: pd.DataFrame, frozen: dict) -> pd.DataFrame:
    feats_all = list(frozen["features"])
    y = dev["critical"].to_numpy(bool)
    burden = dev["burden"].to_numpy(float)
    full = frozen_scores(dev, frozen)
    rows = [{"ablated": "none (frozen 22-input model)", "n_inputs": len(feats_all),
             **_metrics(full, y), "aurc": float(aurc(full, burden))}]
    for f in feats_all:
        feats = [x for x in feats_all if x != f]
        s = _fit(feats, dev).predict_proba(dev[feats])[:, 1]
        rows.append({"ablated": f, "n_inputs": len(feats), **_metrics(s, y), "aurc": float(aurc(s, burden))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ CLI


def _dev_df(root, frozen, dev_units):
    dev = {k: v for k, v in dev_units.items() if k[0] in DEV_METHODS}
    df, skipped = build_table(root, frozen, DEV_METHODS, dev, None)
    if len(df) != 160:
        raise SystemExit(f"expected 160 dev units, built {len(df)} (skipped {skipped})")
    return df.reset_index(drop=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("dev-scores", "n"):
        p = sub.add_parser(name)
        p.add_argument("--root", default="/root/data")
        p.add_argument("--dev-units", required=True)
        p.add_argument("--frozen-dir", required=True)
        p.add_argument("--out", required=True)
        if name == "n":
            p.add_argument("--unit-dir", required=True)
    a = ap.parse_args(argv)
    root = Path(a.root)
    frozen_dir = Path(a.frozen_dir)
    verify_frozen_dir(frozen_dir)
    frozen = json.loads((frozen_dir / "frozen_model.json").read_text())
    dev_units = _load_dev_units(a.dev_units)

    if a.cmd == "dev-scores":
        df = _dev_df(root, frozen, dev_units)
        out = pd.DataFrame({"method": df["method"], "scan_id": df["scan_id"].astype(str),
                            "score": frozen_scores(df, frozen)})
        out.to_csv(a.out, index=False)
        print(f"dev scores: {len(out)} rows -> {a.out}")
        return

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = list(DEV_METHODS) + ["ffr_unet"] + [f"nnunet_f{i}" for i in range(5)]
    df, skipped = build_table(root, frozen, methods, dev_units, Path(a.unit_dir))
    df = df.reset_index(drop=True)
    dev_df = df[df["method"].isin(DEV_METHODS) & (df["origin"] == "dev")].reset_index(drop=True)
    n1 = n1_table(df, frozen)
    n2 = n2_table(dev_df, frozen)
    n1.to_csv(out_dir / "N1_rule_stability.csv", index=False)
    n2.to_csv(out_dir / "N2_feature_ablation_IN_SAMPLE.csv", index=False)
    summary = {
        "n_val_units": int(len(df)), "n_dev_units": int(len(dev_df)),
        "methods": sorted(df["method"].unique().tolist()), "skipped_by_method": skipped,
        "n1_note": "flag cut-offs perturbed on stored feature values (see module docstring deviation)",
        "n2_note": "IN-SAMPLE: refit and evaluated on the same 160 dev units; post-hoc, not test evidence",
    }
    (out_dir / "N_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
