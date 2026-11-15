"""Post-freeze descriptive analyses on the sealed-evaluation outputs. Descriptive only: no refitting,
no ranker selection, no new thresholds. Run on the droplet (see droplet/run_posthoc.sh).

Inputs (--in DIR): features.csv (per unit: critical, dice, burden, severe, g_* features),
flags.csv (frozen score, flagged_top20), optional ensemble_comparators.csv. Also read:
--qa (qa/v3 per-unit JSON: events, critical_reasons), --clinical (clinical vessel-row parquet),
--desc (Descriptors.xlsx).

Six analyses run in parallel (ProcessPoolExecutor, <= 8 workers):
 1 risk-coverage curves with AURC, E-AURC and AUGRC (Traub et al., NeurIPS 2024)
 2 decision-curve net benefit for review triage (Vickers and Elkin 2006)
 3 patient-level paired bootstrap CIs for pairwise ranking differences (NR@20 and AUROC)
 4 failure atlas: critical rate and failure subtypes by segment, and the Dice-hides-it view
 5 calibration of the frozen probabilities (reliability, Brier, ECE) and subgroup NR/AUROC
 6 clinical flagged-vs-unflagged split (flip rates, coverage, stenosis error)

Notes: the M1+M3 ranker is not recomputed here (it needs a refit on dev; the sealed results carry
its numbers). Ensemble comparators are scored as 1 - mean pairwise Dice where present.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

try:  # pandas.to_markdown needs the optional tabulate package
    import tabulate  # noqa: F401
except ImportError:  # minimal pipe-table fallback

    def _to_markdown_fallback(self, index=False, **_):
        cols = [str(c) for c in self.columns]
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for row in self.itertuples(index=False):
            out.append("| " + " | ".join(str(v) for v in row) + " |")
        return "\n".join(out)

    pd.DataFrame.to_markdown = _to_markdown_fallback

BUDGET = 0.2
SEED = 0
NBOOT = 2000
PT_GRID = np.linspace(0.05, 0.50, 46)
SEGMENTERS = ("ffr_unet", "cas_net", "swin_unetr")
RANKERS = {
    "frozen": "risk_frozen",
    "rule_components_ge_10mm": "risk_rule_comp10",
    "exploratory_share_top2": "risk_share_top2",
}


# ----------------------------------------------------------------------------- loading


def load_units(in_dir, qa, desc):
    feats = pd.read_csv(in_dir / "features.csv")
    flags = pd.read_csv(in_dir / "flags.csv")
    df = feats.merge(flags[["method", "scan_id", "score", "flagged_top20"]],
                     on=["method", "scan_id"], how="inner", validate="one_to_one")
    df["scan_id"] = df["scan_id"].astype(str)
    df["critical"] = df["critical"].astype(bool)
    df["risk_frozen"] = df["score"].astype(float)
    df["risk_rule_comp10"] = df["g_n_components_ge_10mm"].astype(float)
    df["risk_share_top2"] = 1.0 - df["g_share_top2"].astype(float).fillna(0.0)  # less share = riskier
    df["flagged"] = df["flagged_top20"].astype(bool)
    d = pd.read_excel(desc).rename(columns={"Scan ID": "scan_id"})
    d["scan_id"] = d["scan_id"].astype(str)
    df = df.merge(d[["scan_id", "Image Quality", "Dominance", "Disease"]], on="scan_id", how="left")

    events, reasons = [], []
    for r in df.itertuples(index=False):
        rec = json.loads((qa / r.method / f"{r.scan_id}.json").read_text())
        reasons.append({"method": r.method, "scan_id": r.scan_id,
                        "critical_reasons": ";".join(rec.get("critical_reasons", []))})
        for ev in rec.get("events", []):
            events.append({"method": r.method, "scan_id": r.scan_id, "flag": ev.get("flag"),
                           "subtype": ev.get("subtype"), "segment": ev.get("segment"),
                           "severity": ev.get("severity")})
    return df, pd.DataFrame(events), pd.DataFrame(reasons)


def _ensemble_units(in_dir, qa):
    path = in_dir / "ensemble_comparators.csv"
    if not path.exists():
        return None
    ens = pd.read_csv(path)
    ens["scan_id"] = ens["scan_id"].astype(str)
    rows = []
    for r in ens.itertuples(index=False):
        p = qa / "nnunet_mv" / f"{r.scan_id}.json"
        if not p.exists():
            continue
        rec = json.loads(p.read_text())
        rows.append({"scan_id": r.scan_id, "critical": bool(rec["critical"]),
                     "dice": float(rec["dice"]), "risk": 1.0 - float(r.mean_pair_dice)})
    return pd.DataFrame(rows) if rows else None


def _per_seg(df, seg):
    return df[df.method == seg].reset_index(drop=True)


def _patient_index(groups):
    uniq, inv = np.unique(groups, return_inverse=True)
    return [np.flatnonzero(inv == k) for k in range(len(uniq))]


def _boot_draws(n_patients, nboot, seed):
    return np.random.default_rng(seed).integers(0, n_patients, size=(nboot, n_patients))


def _rank_top(scores, budget=BUDGET):
    n = len(scores)
    k = int(math.floor(budget * n + 1e-12))
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    flag = np.zeros(n, dtype=bool)
    flag[order[:k]] = True
    return flag


def _nr(scores, y):
    from coronarysafe.eval.metrics import normalised_recall

    return float(normalised_recall(np.asarray(scores), np.asarray(y).astype(bool), BUDGET))


def _auroc(scores, y):
    from coronarysafe.eval.metrics import auroc

    return float(auroc(np.asarray(scores), np.asarray(y).astype(bool)))


def _ci(est, boots):
    b = np.asarray(boots, dtype=float)
    b = b[np.isfinite(b)]
    if b.size == 0:
        return (np.nan, np.nan, np.nan)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return (float(est), float(lo), float(hi))


# ----------------------------------------------------------------------------- analysis 1


def _risk_coverage(scores, loss):
    order = np.argsort(np.asarray(scores, dtype=float), kind="stable")  # lowest risk kept first
    l = np.asarray(loss, dtype=float)[order]
    n = len(l)
    k = np.arange(1, n + 1)
    cum = np.cumsum(l)
    sel = cum / k
    cov = k / n
    aurc = float(sel.mean())
    augrc = float((cum / n).mean())
    l_opt = np.sort(np.asarray(loss, dtype=float))
    cum_o = np.cumsum(l_opt)
    aurc_opt = float((cum_o / k).mean())
    augrc_opt = float((cum_o / n).mean())
    return cov, sel, {"AURC": aurc, "E-AURC": aurc - aurc_opt, "AUGRC": augrc,
                      "E-AUGRC": augrc - augrc_opt}


def job_risk_coverage(in_dir, out, df, ens):
    import matplotlib.pyplot as plt

    rows, summ = [], []
    fig, axes = plt.subplots(1, len(SEGMENTERS) + (ens is not None), figsize=(4 * (len(SEGMENTERS) + 1), 3.6))
    axes = np.atleast_1d(axes)
    for ax, seg in zip(axes, SEGMENTERS):
        sub = _per_seg(df, seg)
        for rk, col in RANKERS.items():
            cov, sel, m = _risk_coverage(sub[col].to_numpy(), sub.critical.to_numpy())
            ax.plot(cov, sel, label=rk)
            summ.append({"segmenter": seg, "ranker": rk, **m, "n": len(sub),
                         "critical": int(sub.critical.sum())})
            rows += [{"segmenter": seg, "ranker": rk, "coverage": c, "selective_risk": s}
                     for c, s in zip(cov[::5], sel[::5])]
        ax.set_title(seg)
        ax.set_xlabel("coverage (fraction reviewed by confidence)")
        ax.set_ylabel("selective critical risk")
    if ens is not None:
        ax = axes[-1]
        cov, sel, m = _risk_coverage(ens.risk.to_numpy(), ens.critical.to_numpy())
        ax.plot(cov, sel, label="ensemble_1_minus_dice")
        summ.append({"segmenter": "nnunet_mv", "ranker": "ensemble_1_minus_dice", **m,
                     "n": len(ens), "critical": int(ens.critical.sum())})
        ax.set_title("nnunet_mv ensemble")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "risk_coverage.png", dpi=130)
    plt.close(fig)
    pd.DataFrame(summ).to_csv(out / "risk_coverage_metrics.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "risk_coverage_curves.csv", index=False)
    tbl = pd.DataFrame(summ)[["segmenter", "ranker", "AURC", "E-AURC", "AUGRC", "E-AUGRC"]]
    return {"name": "1 Risk-coverage (AURC, E-AURC, AUGRC)", "table": tbl.round(4).to_markdown(index=False)}


# ----------------------------------------------------------------------------- analysis 2


def job_decision_curve(in_dir, out, df, ens):
    import matplotlib.pyplot as plt

    rows = []
    fig, axes = plt.subplots(1, len(SEGMENTERS), figsize=(4 * len(SEGMENTERS), 3.6), sharey=True)
    for ax, seg in zip(np.atleast_1d(axes), SEGMENTERS):
        sub = _per_seg(df, seg)
        y = sub.critical.to_numpy()
        n = len(y)
        prev = y.mean()
        review_all = prev - (1 - prev) * PT_GRID / (1 - PT_GRID)
        ax.plot(PT_GRID, review_all, "k--", label="review all")
        ax.plot(PT_GRID, np.zeros_like(PT_GRID), "k:", label="review none")
        for rk, col in RANKERS.items():
            flag = _rank_top(sub[col].to_numpy())
            tp = (flag & y).sum() / n
            fp = (flag & ~y).sum() / n
            nb = tp - fp * PT_GRID / (1 - PT_GRID)
            ax.plot(PT_GRID, nb, label=rk)
            rows += [{"segmenter": seg, "ranker": rk, "threshold": t, "net_benefit": v}
                     for t, v in zip(PT_GRID, nb)]
        rows += [{"segmenter": seg, "ranker": "review_all", "threshold": t, "net_benefit": v}
                 for t, v in zip(PT_GRID, review_all)]
        ax.set_title(seg)
        ax.set_xlabel("threshold probability")
    np.atleast_1d(axes)[0].set_ylabel("net benefit")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "decision_curves.png", dpi=130)
    plt.close(fig)
    dc = pd.DataFrame(rows)
    dc.to_csv(out / "decision_curves.csv", index=False)
    pick = dc[np.isclose(dc.threshold, 0.10)].round(4)
    return {"name": "2 Decision curves (net benefit at threshold 0.10)",
            "table": pick.to_markdown(index=False)}


# ----------------------------------------------------------------------------- analysis 3


def job_paired_bootstrap(in_dir, out, df, ens):
    import itertools

    rng_draws = {}
    rows = []
    for seg in SEGMENTERS:
        sub = _per_seg(df, seg).reset_index(drop=True)
        y = sub.critical.to_numpy()
        groups = sub.scan_id.to_numpy()
        pidx = _patient_index(groups)
        draws = rng_draws.setdefault(len(pidx), _boot_draws(len(pidx), NBOOT, SEED))
        for a, b in itertools.combinations(RANKERS, 2):
            sa = sub[RANKERS[a]].to_numpy()
            sb = sub[RANKERS[b]].to_numpy()
            for metric, fn in (("NR@20", _nr), ("AUROC", _auroc)):
                est = fn(sa, y) - fn(sb, y)
                boots = []
                for d in draws:
                    idx = np.concatenate([pidx[k] for k in d])
                    yy = y[idx]
                    if yy.all() or (~yy).all():
                        boots.append(np.nan)
                        continue
                    boots.append(fn(sa[idx], yy) - fn(sb[idx], yy))
                e, lo, hi = _ci(est, boots)
                rows.append({"segmenter": seg, "comparison": f"{a} minus {b}", "metric": metric,
                             "estimate": e, "ci_lo": lo, "ci_hi": hi,
                             "excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0))})
    res = pd.DataFrame(rows)
    res.to_csv(out / "paired_bootstrap.csv", index=False)
    return {"name": "3 Paired patient bootstrap (2000, seed 0)",
            "table": res.round(3).to_markdown(index=False)}


# ----------------------------------------------------------------------------- analysis 4


def job_failure_atlas(in_dir, out, df, ens, events, reasons):
    import matplotlib.pyplot as plt

    m = df.merge(reasons, on=["method", "scan_id"], how="left")
    rows = []
    for seg in SEGMENTERS:
        sub = m[m.method == seg]
        ev = events[(events.method == seg) & events.flag.isin(["F1", "F2", "F3"])]
        for (flag, sub_t, segname), g in ev.groupby(["flag", "subtype", "segment"], dropna=False):
            rows.append({"segmenter": seg, "flag": flag, "subtype": sub_t, "segment": segname,
                         "units_with_event": int(g.scan_id.nunique()),
                         "units_critical_reason_here": int(sub.critical_reasons.fillna("").str.contains(
                             f":{segname}").sum()) if isinstance(segname, str) else 0})
    atlas = pd.DataFrame(rows)
    atlas.to_csv(out / "failure_atlas.csv", index=False)

    fig, axes = plt.subplots(1, len(SEGMENTERS), figsize=(4 * len(SEGMENTERS), 3.6), sharey=True)
    hid = []
    for ax, seg in zip(np.atleast_1d(axes), SEGMENTERS):
        sub = _per_seg(df, seg)
        jit = np.random.default_rng(SEED).uniform(-0.04, 0.04, len(sub))
        ax.scatter(sub.dice, sub.critical.astype(float) + jit, s=12, c="grey", alpha=0.6, label="units")
        hi = sub[(sub.dice >= 0.90) & sub.critical]
        ax.scatter(hi.dice, hi.critical.astype(float) + 0.0, s=28, c="red", label="critical, Dice>=0.90")
        ax.axvline(0.90, ls="--", c="k", lw=0.8)
        ax.set_title(seg)
        ax.set_xlabel("Dice")
        for rk, col in RANKERS.items():
            flag = _rank_top(sub[col].to_numpy())
            hid_mask = (sub.dice >= 0.90).to_numpy() & sub.critical.to_numpy()
            hid.append({"segmenter": seg, "ranker": rk, "n_dice_ge90_critical": int(hid_mask.sum()),
                        "fraction_flagged": float(flag[hid_mask].mean()) if hid_mask.any() else np.nan})
    np.atleast_1d(axes)[0].set_ylabel("critical (0/1)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "dice_hides_it.png", dpi=130)
    plt.close(fig)
    hid_df = pd.DataFrame(hid)
    hid_df.to_csv(out / "dice_hides_it.csv", index=False)
    return {"name": "4 Failure atlas and Dice-hides-it",
            "table": hid_df.round(3).to_markdown(index=False)}


# ----------------------------------------------------------------------------- analysis 5


def _ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def job_calibration(in_dir, out, df, ens):
    import matplotlib.pyplot as plt

    rows, cal_rows = [], []
    fig, axes = plt.subplots(1, len(SEGMENTERS), figsize=(4 * len(SEGMENTERS), 3.6), sharey=True)
    for ax, seg in zip(np.atleast_1d(axes), SEGMENTERS):
        sub = _per_seg(df, seg)
        p = np.clip(sub.risk_frozen.to_numpy(), 0, 1)
        y = sub.critical.to_numpy().astype(float)
        brier = float(np.mean((p - y) ** 2))
        ece = _ece(p, y)
        edges = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
        xs, ys = [], []
        for b in range(10):
            m = idx == b
            if m.any():
                xs.append(p[m].mean())
                ys.append(y[m].mean())
                cal_rows.append({"segmenter": seg, "bin": b, "mean_pred": p[m].mean(),
                                 "observed": y[m].mean(), "n": int(m.sum())})
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.plot(xs, ys, "o-")
        ax.set_title(f"{seg}  Brier {brier:.3f}  ECE {ece:.3f}", fontsize=8)
        ax.set_xlabel("frozen probability")
        rows.append({"segmenter": seg, "brier": brier, "ece": ece, "n": len(sub)})
    np.atleast_1d(axes)[0].set_ylabel("observed critical rate")
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=130)
    plt.close(fig)
    pd.DataFrame(cal_rows).to_csv(out / "reliability_bins.csv", index=False)

    sub_rows = []
    for seg in SEGMENTERS:
        sub = _per_seg(df, seg).reset_index(drop=True)
        pidx = _patient_index(sub.scan_id.to_numpy())
        draws = _boot_draws(len(pidx), NBOOT, SEED)
        for col, levels in (("Image Quality", sorted(sub["Image Quality"].dropna().unique())),
                            ("Disease", sorted(sub["Disease"].dropna().unique())),
                            ("Dominance", sorted(sub["Dominance"].dropna().unique()))):
            for lev in levels:
                mask = (sub[col] == lev).to_numpy()
                y = sub.critical.to_numpy()[mask]
                s = sub.risk_frozen.to_numpy()[mask]
                if len(y) < 4 or y.sum() < 2 or (~y).sum() < 2:
                    sub_rows.append({"segmenter": seg, "subgroup": f"{col}={lev}", "n": int(len(y)),
                                     "critical": int(y.sum()), "NR": np.nan, "AUROC": np.nan})
                    continue
                nr_e = _nr(s, y)
                au_e = _auroc(s, y)
                # Bootstrap over patients restricted to the subgroup.
                gsub = sub.scan_id.to_numpy()[mask]
                gp = _patient_index(gsub)
                bdraws = _boot_draws(len(gp), NBOOT, SEED)
                bn, ba = [], []
                for d in bdraws:
                    idx = np.concatenate([gp[k] for k in d])
                    yy = y[idx]
                    if yy.all() or (~yy).all():
                        continue
                    bn.append(_nr(s[idx], yy))
                    ba.append(_auroc(s[idx], yy))
                _, nlo, nhi = _ci(nr_e, bn)
                _, alo, ahi = _ci(au_e, ba)
                sub_rows.append({"segmenter": seg, "subgroup": f"{col}={lev}", "n": int(len(y)),
                                 "critical": int(y.sum()), "NR": nr_e, "NR_lo": nlo, "NR_hi": nhi,
                                 "AUROC": au_e, "AUROC_lo": alo, "AUROC_hi": ahi})
    subs = pd.DataFrame(sub_rows)
    subs.to_csv(out / "subgroups.csv", index=False)
    cal = pd.DataFrame(rows)
    return {"name": "5 Calibration and subgroups",
            "table": cal.round(3).to_markdown(index=False) + "\n\n" + subs.round(3).head(12).to_markdown(index=False)}


# ----------------------------------------------------------------------------- analysis 6


def job_clinical_split(in_dir, out, df, ens, clinical):
    cl = pd.read_parquet(clinical)
    cl["scan_id"] = cl["scan_id"].astype(str)
    cl = cl.merge(df[["method", "scan_id", "flagged"]], on=["method", "scan_id"], how="inner")
    rows = []
    for grp_name, mask in (("flagged", cl.flagged), ("unflagged", ~cl.flagged)):
        g = cl[mask]
        if len(g) == 0:
            continue
        pidx = _patient_index(g.scan_id.to_numpy())
        draws = _boot_draws(len(pidx), NBOOT, SEED)
        metrics = {
            "obstructive_flip_rate": g.obstructive_flip.astype(float).to_numpy(),
            "severe_flip_rate": g.severe_flip.astype(float).to_numpy(),
            "coverage_below_0.9_rate": (g.coverage < 0.9).astype(float).to_numpy(),
            "abs_stenosis_err_pct_median": g.abs_stenosis_err_pct.to_numpy(),
        }
        for name, vals in metrics.items():
            est = float(np.nanmedian(vals)) if name.endswith("median") else float(np.nanmean(vals))
            boots = []
            for d in draws:
                idx = np.concatenate([pidx[k] for k in d])
                v = vals[idx]
                v = v[np.isfinite(v)]
                if v.size:
                    boots.append(float(np.median(v)) if name.endswith("median") else float(v.mean()))
            e, lo, hi = _ci(est, boots)
            rows.append({"group": grp_name, "metric": name, "estimate": e, "ci_lo": lo, "ci_hi": hi,
                         "vessel_rows": int(len(g)), "patients": int(len(pidx))})
    res = pd.DataFrame(rows)
    res.to_csv(out / "clinical_flagged_split.csv", index=False)
    return {"name": "6 Clinical flagged vs unflagged",
            "table": res.round(3).to_markdown(index=False)}


# ----------------------------------------------------------------------------- driver


def _run(job, args):
    """Top-level worker: every job receives paths and recomputes its own inputs (picklable)."""
    name = job
    in_dir, out, qa, clinical, desc, ensemble = args
    df, events, reasons = load_units(in_dir, qa, desc)
    ens = _ensemble_units(in_dir, qa)
    if name == "1":
        return job_risk_coverage(in_dir, out, df, ens)
    if name == "2":
        return job_decision_curve(in_dir, out, df, ens)
    if name == "3":
        return job_paired_bootstrap(in_dir, out, df, ens)
    if name == "4":
        return job_failure_atlas(in_dir, out, df, ens, events, reasons)
    if name == "5":
        return job_calibration(in_dir, out, df, ens)
    if name == "6":
        return job_clinical_split(in_dir, out, df, ens, clinical)
    raise ValueError(name)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--qa", type=Path, default=Path("/root/data/qa/v3"))
    p.add_argument("--clinical", type=Path, required=True)
    p.add_argument("--desc", type=Path,
                   default=Path("/root/data/imagecas_x/ImageCAS-X_dataset_v3/Descriptors.xlsx"))
    p.add_argument("--ensemble", type=Path, default=None, help="unused; read from --in if present")
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    args = (a.in_dir, a.out, a.qa, a.clinical, a.desc, a.ensemble)
    jobs = ["1", "2", "3", "4", "5", "6"]
    results = {}
    with ProcessPoolExecutor(max_workers=min(a.workers, 8)) as ex:
        futs = {j: ex.submit(_run, j, args) for j in jobs}
        for j in jobs:
            results[j] = futs[j].result()
    lines = ["# Post-freeze descriptive analyses", "",
             "Descriptive only. Frozen model, rules and thresholds are as sealed.", ""]
    for j in jobs:
        lines += [f"## {results[j]['name']}", "", results[j]["table"], ""]
    (a.out / "POSTHOC.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({j: results[j]["name"] for j in jobs}))


if __name__ == "__main__":
    main()
