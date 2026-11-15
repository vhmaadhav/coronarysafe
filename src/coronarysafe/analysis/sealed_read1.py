"""Sealed Read 1 (ADDENDA_FROZEN_2026-11-15, Errata 8): Addendum M (reduced, Errata 2) + Addendum J.

One-shot read, launched ONCE by the coordinator. Test mode requires --manifest and refuses a
non-empty --out. Val mode is the dry run (same code path, val scans only, no test file is read).

Inputs (frozen, never re-derived):
  * frozen M1+M3+M4 bundle (sha256 per Addendum H, verified by m4/valscore.verify_frozen_dir)
  * frozen NR@20 (eval/metrics.py: floor k, fractional ties) and the frozen M4 scoring helpers
    (m4/valscore.py: unit job, clamp 1e-6, frozen 22-feature LR, median imputation only for NaN)
  * FFR-UNet TEST scores: READ from the sealed outputs under <root>/test_eval (features.csv and
    flags.csv). Never recomputed.
  * nnU-Net f0-f4 units (test and val): COMPUTED with the frozen pipeline from preds, qa/v3 and aorta.
  * FFR-UNet VAL units (dry run only): COMPUTED with the same frozen pipeline.

Outputs (--out DIR): MANIFEST.json (written first), results.json, READ1.md, provenance.json, units/.

  python -m coronarysafe.analysis.sealed_read1 --split val --out /root/data/addenda/read1_val
  python -m coronarysafe.analysis.sealed_read1 --split test --out DIR --manifest PATH   (coordinator only)

Estimators (pre-declared):
  M  m0 realised 2x2 among critical units (fractional ties); m1 paired patient bootstrap, 20,000
     resamples, seed 0, same function as the sealed S2 (plus a 2,000-resample check against the posted S2 CI); m2 exact McNemar;
     m4 AURC and AUROC differences, patient bootstrap 2,000, seed 0; m5 power simulation (optimistic).
  J  per-fold D_f = NR@20(model) - NR@20(rule); D_A = mean_f D_f; patient-block bootstrap 2,000, seed 0,
     one patient draw shared by all five folds; success = D_A >= 0.10 and CI lower > 0.
     g_share_top2 is EXPLORATORY and reported side by side only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
for _p in (REPO / "src", REPO / "pipeline", REPO):
    sys.path.insert(0, str(_p))

from coronarysafe.eval import metrics as MET  # noqa: E402
from coronarysafe.m4 import valscore as VS  # noqa: E402  frozen helpers (sha check, clamp, LR, unit job)

CODE_VERSION = "sealed-read1-v1"
BUDGET = 0.20
BAR_PP = 0.10
FFR = "ffr_unet"
FOLDS = ["nnunet_f0", "nnunet_f1", "nnunet_f2", "nnunet_f3", "nnunet_f4"]
SEED_S2 = 0  # matches m4/test_eval.py SEED (the sealed S2); see Errata 10
SEED_M4 = 0
SEED_J = 0
N_BOOT_S2 = 20000
N_BOOT_S2_CHECK = 2000
N_BOOT_M4 = 2000
N_BOOT_J = 2000
POSTED_S2 = {"point": 0.125, "lo": -0.003, "hi": 0.239}
POWER_N = (160, 240, 320, 480)
POWER_OUTER = 200
POWER_INNER = 500
POWER_WORKERS = 4  # <= 8 workers in total on the droplet
UNIT_WORKERS_DEFAULT = 8
GT_DIR = "imagecas_x/ImageCAS-X_dataset_v3"


# ----------------------------------------------------------------------------- basics


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def nr(scores, y, budget=BUDGET) -> float:
    """Frozen NR@20 (eval/metrics.py)."""
    return float(MET.normalised_recall(scores, y, budget))


def flag_weights(scores, budget=BUDGET) -> np.ndarray:
    """Fractional membership in the flagged set. Ties at the cut-off share the remaining slots
    equally, so sum(w * y) reproduces MET.recall_at_budget exactly."""
    s = np.asarray(scores, dtype=float).ravel()
    n = len(s)
    k = MET._review_count(n, budget)
    w = np.zeros(n)
    if k == 0:
        return w
    cutoff = np.partition(s, n - k)[n - k]
    above = s > cutoff
    tied = s == cutoff
    w[above] = 1.0
    w[tied] = (k - above.sum()) / tied.sum()
    return w


def check_weights(scores, y, budget=BUDGET) -> None:
    w = flag_weights(scores, budget)
    rec = float(np.sum(w * (np.asarray(y) == 1))) / float(np.sum(y))
    ref = float(MET.recall_at_budget(scores, y, budget))
    if abs(rec - ref) > 1e-9:
        raise AssertionError(f"flag weights disagree with metrics.recall_at_budget: {rec} vs {ref}")


def ci_pct(x):
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    lo, hi = np.percentile(x[ok], [2.5, 97.5]) if ok.any() else (math.nan, math.nan)
    return float(lo), float(hi), int((~ok).sum())


def mcnemar(b: float, c: float) -> dict:
    """Exact two-sided McNemar: binomial on b of b+c with p = 0.5. Fractional counts (ties) are
    reported as the range over floor/ceil roundings as well as the rounded primary value."""

    def p_of(bb, cc):
        bb, cc = int(bb), int(cc)
        if bb + cc == 0:
            return 1.0
        return float(stats.binomtest(bb, bb + cc, 0.5, alternative="two-sided").pvalue)

    primary = p_of(round(b), round(c))
    res = {"b": b, "c": c, "p_two_sided_exact": primary}
    if abs(b - round(b)) > 1e-9 or abs(c - round(c)) > 1e-9:
        combos = [p_of(fb(b), fc(c)) for fb in (math.floor, math.ceil) for fc in (math.floor, math.ceil)]
        res["p_range_over_floor_ceil"] = [min(combos), max(combos)]
    return res


# ----------------------------------------------------------------------------- M (FFR-UNet)


def load_sealed_ffr(root: Path, test_ids) -> pd.DataFrame:
    """TEST only: READ the sealed FFR-UNet outputs. Nothing is recomputed."""
    te = root / "test_eval"
    feats = pd.read_csv(te / "features.csv", dtype={"scan_id": str})
    flags = pd.read_csv(te / "flags.csv", dtype={"scan_id": str})
    feats = feats[feats["method"] == FFR]
    flags = flags[flags["method"] == FFR]
    df = feats.merge(flags[["scan_id", "score", "flagged_top20"]], on="scan_id", validate="one_to_one")
    if len(df) != len(test_ids) or set(df["scan_id"]) != set(test_ids):
        raise SystemExit("sealed FFR-UNet rows do not match filelist/test.txt")
    out = pd.DataFrame(
        {
            "scan_id": df["scan_id"].values,
            "y": df["critical"].astype(int).values,
            "m": df["score"].astype(float).values,
            "r": df["g_n_components_ge_10mm"].astype(float).values,
            "s": 1.0 - df["g_share_top2"].astype(float).fillna(0.0).values,
            "flag_sealed": df["flagged_top20"].astype(bool).values,
        }
    )
    return out.sort_values("scan_id").reset_index(drop=True)


def run_M(ffr: pd.DataFrame, log) -> dict:
    y = ffr["y"].to_numpy(int)
    m = ffr["m"].to_numpy(float)
    r = ffr["r"].to_numpy(float)
    n = len(y)
    P = int(y.sum())
    K = MET._review_count(n, BUDGET)
    check_weights(m, y)
    check_weights(r, y)
    wm, wr = flag_weights(m), flag_weights(r)
    crit = y == 1
    a = float(np.sum(wm[crit] * wr[crit]))
    b = float(np.sum(wm[crit] * (1 - wr[crit])))
    c = float(np.sum((1 - wm[crit]) * wr[crit]))
    d = float(np.sum((1 - wm[crit]) * (1 - wr[crit])))
    C_m, C_r = a + b, a + c
    NR_m, NR_r = nr(m, y), nr(r, y)
    oracle = min(K, P) / P
    m0 = {
        "n": n, "P": P, "K": K, "oracle_recall": oracle,
        "a_both": a, "b_model_only": b, "c_rule_only": c, "n_neither": d,
        "C_model": C_m, "C_rule": C_r,
        "NR_model": NR_m, "NR_rule": NR_r, "delta_NR": NR_m - NR_r,
        "delta_NR_from_b_minus_c": (b - c) / min(K, P),
        "tied_units_model": int(np.sum((wm > 0) & (wm < 1))),
        "tied_units_rule": int(np.sum((wr > 0) & (wr < 1))),
        "note": "a = sum(wm*wr) over critical units; exact when no ties, otherwise assumes independent tie-breaking",
    }
    log(f"m0: P={P} K={K} C_model={C_m:.2f} C_rule={C_r:.2f} a={a:.2f} b={b:.2f} c={c:.2f} n={d:.2f}")

    # m1: identical frozen estimator (paired patient bootstrap, percentile)
    # Uses the SAME function and seed as the sealed S2 (m4/test_eval.py: paired_bootstrap_difference,
    # clusters = scan_id, seed 0). The "seed 14" in the frozen addenda doc was wrong; see Errata 10.
    from coronarysafe.eval.bootstrap import paired_bootstrap_difference
    yb = y.astype(bool)
    clusters = ffr["scan_id"].to_numpy()

    def s2_boot(B, seed):
        res = paired_bootstrap_difference(
            clusters,
            lambda rows: MET.normalised_recall(m[rows], yb[rows], BUDGET),
            lambda rows: MET.normalised_recall(r[rows], yb[rows], BUDGET),
            n_boot=B, seed=seed,
        )
        return float(res["lo"]), float(res["hi"]), 0

    lo_c, hi_c, nan_c = s2_boot(N_BOOT_S2_CHECK, SEED_S2)
    repro = (
        abs(lo_c - POSTED_S2["lo"]) <= 0.002 and abs(hi_c - POSTED_S2["hi"]) <= 0.002
        and abs((NR_m - NR_r) - POSTED_S2["point"]) <= 0.0005
    )
    m1_check = {
        "resamples": N_BOOT_S2_CHECK, "seed": SEED_S2, "point": NR_m - NR_r,
        "ci_lo": lo_c, "ci_hi": hi_c, "nan_draws": nan_c,
        "posted": POSTED_S2,
        "reproduces_posted_within_0.002": bool(repro),
    }
    log(f"m1 check (2000, seed {SEED_S2}): point {NR_m - NR_r:+.4f} CI [{lo_c:+.4f}, {hi_c:+.4f}] "
        f"posted {POSTED_S2} reproduces={repro}")
    lo, hi, nan_n = s2_boot(N_BOOT_S2, SEED_S2)
    m1 = {"resamples": N_BOOT_S2, "seed": SEED_S2, "point": NR_m - NR_r, "ci_lo": lo, "ci_hi": hi,
          "nan_draws": nan_n, "check_2000": m1_check,
          "bar_met": bool((NR_m - NR_r) >= BAR_PP and lo > 0)}
    log(f"m1 (20000, seed {SEED_S2}): CI [{lo:+.4f}, {hi:+.4f}]")

    # m2: exact McNemar on the discordant pairs (fractional counts reported as a range)
    m2 = mcnemar(b, c)
    log(f"m2: McNemar p={m2['p_two_sided_exact']:.4g}")

    # m4: AURC and AUROC differences, patient bootstrap 2000 seed 0
    aurc_m, aurc_r = float(MET.aurc(m, y)), float(MET.aurc(r, y))
    auc_m, auc_r = float(MET.auroc(m, y)), float(MET.auroc(r, y))
    rng = np.random.default_rng(SEED_M4)
    da = np.empty(N_BOOT_M4)
    dz = np.empty(N_BOOT_M4)
    for i in range(N_BOOT_M4):
        idx = rng.integers(0, n, n)
        da[i] = MET.aurc(m[idx], y[idx]) - MET.aurc(r[idx], y[idx])
        dz[i] = MET.auroc(m[idx], y[idx]) - MET.auroc(r[idx], y[idx])
    alo, ahi, _ = ci_pct(da)
    zlo, zhi, _ = ci_pct(dz)
    m4 = {
        "AURC_model": aurc_m, "AURC_rule": aurc_r, "dAURC_model_minus_rule": aurc_m - aurc_r,
        "dAURC_ci": [alo, ahi], "AUROC_model": auc_m, "AUROC_rule": auc_r,
        "dAUROC_model_minus_rule": auc_m - auc_r, "dAUROC_ci": [zlo, zhi],
        "resamples": N_BOOT_M4, "seed": SEED_M4,
        "loss": "critical indicator (binary failure)",
    }
    log(f"m4: dAURC {aurc_m - aurc_r:+.4f} [{alo:+.4f},{ahi:+.4f}]  dAUROC {auc_m - auc_r:+.4f} [{zlo:+.4f},{zhi:+.4f}]")

    # m5: planning-only power (optimistic: the observed effect and discordance structure are used as truth)
    jobs = [(N, y, m, r, 1000 + N) for N in POWER_N]
    m5 = {}
    with ProcessPoolExecutor(max_workers=POWER_WORKERS) as ex:
        for N, pw, mean_d in ex.map(_power_cell, jobs):
            m5[str(N)] = {"N": N, "power": pw, "mean_delta_NR_sim": mean_d,
                          "outer": POWER_OUTER, "inner": POWER_INNER, "label": "optimistic"}
    log("m5: " + ", ".join(f"N={k}: {v['power']:.3f}" for k, v in m5.items()))

    return {"m0": m0, "m1": m1, "m2": m2, "m4": m4, "m5": m5}


def _power_cell(args):
    N, y, m, r, seed = args
    rng = np.random.default_rng(seed)
    n = len(y)
    hits, deltas = 0, []
    for _ in range(POWER_OUTER):
        idx = rng.integers(0, n, N)
        yS, mS, rS = y[idx], m[idx], r[idx]
        d = nr(mS, yS) - nr(rS, yS)
        deltas.append(d)
        boot = np.empty(POWER_INNER)
        for j in range(POWER_INNER):
            jj = rng.integers(0, N, N)
            boot[j] = nr(mS[jj], yS[jj]) - nr(rS[jj], yS[jj])
        lo = float(np.nanpercentile(boot, 2.5))
        if np.isfinite(d) and d >= BAR_PP and lo > 0:
            hits += 1
    return N, hits / POWER_OUTER, float(np.nanmean(deltas))


# ----------------------------------------------------------------------------- J (nnU-Net folds)


def fold_row(name, y, m, r, s) -> dict:
    n = len(y)
    P = int(y.sum())
    K = MET._review_count(n, BUDGET)
    check_weights(m, y)
    check_weights(r, y)
    wm, wr = flag_weights(m), flag_weights(r)
    crit = y == 1
    Cm = float(np.sum(wm[crit]))
    Cr = float(np.sum(wr[crit]))
    NRm, NRr, NRs = nr(m, y), nr(r, y), nr(s, y)
    return {
        "fold": name, "n": n, "P": P, "K": K,
        "C_model": Cm, "C_rule": Cr,
        "precision_model": Cm / K, "precision_rule": Cr / K,
        "recall_model": Cm / P, "recall_rule": Cr / P,
        "NR_model": NRm, "NR_rule": NRr, "D_f": NRm - NRr,
        "AUROC_model": float(MET.auroc(m, y)), "AUROC_rule": float(MET.auroc(r, y)),
        "dAUROC_model_minus_rule": float(MET.auroc(m, y) - MET.auroc(r, y)),
        "EXPLORATORY_NR_share_top2": NRs, "EXPLORATORY_D_share_top2": NRm - NRs,
    }


def run_J(folds: dict, scans: list, log) -> dict:
    # Critical labels are per segmenter (the v3 labeler runs on each reconstruction vs GT), so each
    # fold keeps its own y. The patient draw is shared across folds.
    names = list(folds)
    arr = {}
    for nm in names:
        df = folds[nm].set_index("scan_id").loc[scans]
        arr[nm] = (df["y"].to_numpy(int), df["m"].to_numpy(float), df["r"].to_numpy(float), df["s"].to_numpy(float))
    rows = [fold_row(nm, arr[nm][0], *arr[nm][1:]) for nm in names]
    D_point = np.array([row["D_f"] for row in rows])
    DA = float(D_point.mean())

    n = len(scans)
    rng = np.random.default_rng(SEED_J)
    D = np.empty((N_BOOT_J, len(names)))
    for i in range(N_BOOT_J):
        idx = rng.integers(0, n, n)  # one patient draw, shared by all folds
        for j, nm in enumerate(names):
            yf, m, r, _ = arr[nm]
            D[i, j] = nr(m[idx], yf[idx]) - nr(r[idx], yf[idx])
    DA_star = np.nanmean(D, axis=1)
    lo, hi, nan_n = ci_pct(DA_star)
    success = bool(DA >= BAR_PP and lo > 0)
    log(f"J: D_A = {DA:+.4f}, CI [{lo:+.4f}, {hi:+.4f}] (nan draws {nan_n}); success={success}")
    return {
        "folds": rows, "D_A": DA, "D_A_ci": [lo, hi], "bootstrap_resamples": N_BOOT_J,
        "seed": SEED_J, "nan_draws": nan_n, "n_scans": n, "success": success,
        "bar": {"D_A_min": BAR_PP, "ci_lo_min_exclusive": 0.0},
        "families": {"A_nnunet_f0_f4": "included as one family (five folds averaged)",
                     "B_cascade": "EXCLUDED (decided before outcome: checkpoint absent, split unverified)"},
        "per_fold_D_point": dict(zip(names, D_point.tolist())),
    }


# ----------------------------------------------------------------------------- unit scoring (frozen pipeline)


def compute_units(methods, scans, root: Path, aorta_dir: Path, frozen_dir: Path, frozen: dict,
                  left_unit, provenance: dict, out: Path, workers: int, device: str, log):
    import torch

    from coronarysafe.m4.score import load_model

    unit_dir = out / "units"
    unit_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(m, s) for m in methods for s in scans]
    args = [(m, s, str(root), str(aorta_dir), left_unit, str(unit_dir), provenance) for m, s in jobs]
    t0 = time.time()
    status_counts: dict = {}
    with Pool(workers) as pool:
        for k, (m, s, status) in enumerate(pool.imap_unordered(VS._safe_unit_job, args), 1):
            status_counts[status] = status_counts.get(status, 0) + 1
            if k % 40 == 0 or status not in ("ok", "cached"):
                log(f"  unit {k}/{len(jobs)} {m} {s} {status}")
    log(f"  unit jobs done in {time.time() - t0:.0f}s: {status_counts}")

    torch.set_num_threads(8)
    net = load_model(frozen_dir / "m4_model.pt", torch.device(device))
    t1 = time.time()
    VS._m4_pass(device, net, unit_dir, jobs)
    log(f"  M4 pass done in {time.time() - t1:.0f}s on {device}")

    rows, excluded = [], {}
    for m, s in jobs:
        jp = unit_dir / f"{m}__{s}.json"
        rec = json.loads(jp.read_text()) if jp.exists() else {"status": "missing_unit"}
        if rec.get("status") not in ("ok", "cached") or "m4_prob" not in rec:
            key = f"{m}:{rec.get('status')}"
            excluded[key] = excluded.get(key, 0) + 1
            continue
        qa = json.loads((root / "qa" / "v3" / m / f"{s}.json").read_text())
        values, missing = VS.assemble_features(qa, rec, frozen["features"])
        if missing:
            key = f"{m}:missing_feature"
            excluded[key] = excluded.get(key, 0) + 1
            continue
        score = VS.apply_frozen_lr(values, frozen)
        g = qa.get("graph") or {}
        share = g.get("share_top2")
        rows.append({
            "method": m, "scan_id": str(s), "y": int(rec["critical"]), "m": float(score),
            "r": float(g["n_components_ge_10mm"]),
            "s": 1.0 - (float(share) if share is not None else 0.0),
        })
    df = pd.DataFrame(rows)
    return df, excluded, len(jobs)


# ----------------------------------------------------------------------------- manifest


def build_manifest(split, root: Path, aorta_dir: Path, frozen_dir: Path, shas, split_ids, methods, sealed: bool):
    from pipeline.label_core import _gt_seg

    split_file = root / GT_DIR / "filelist" / f"{split}.txt"
    man = {
        "code_version": CODE_VERSION,
        "split": split,
        "n_scans": len(split_ids),
        "split_file": str(split_file),
        "split_file_sha256": sha256_file(split_file),
        "scan_ids_sha256": hashlib.sha256(" ".join(sorted(split_ids)).encode()).hexdigest(),
        "frozen_dir": str(frozen_dir),
        "frozen_sha256": shas,
        "sealed_inputs": {},
        "methods": {},
    }
    if sealed:
        for name in ("features.csv", "flags.csv"):
            p = root / "test_eval" / name
            with open(p, "rb") as f:
                n_lines = sum(1 for _ in f) - 1
            man["sealed_inputs"][str(p)] = {"sha256": sha256_file(p), "data_rows": n_lines}
    for m in methods:
        cats = {"mask": [], "qa": [], "aorta": [], "gt": []}
        missing = {k: 0 for k in cats}
        for s in split_ids:
            paths = {
                "mask": root / "preds" / m / s / "mask.nii.gz",
                "qa": root / "qa" / "v3" / m / f"{s}.json",
                "aorta": aorta_dir / f"{s}.nii.gz",
                "gt": _gt_seg(root, s),
            }
            for k, p in paths.items():
                if p.exists():
                    cats[k].append(str(p))
                else:
                    missing[k] += 1
        man["methods"][m] = {
            "counts": {k: len(v) for k, v in cats.items()},
            "missing": missing,
            "files": cats,
        }
    return man


# ----------------------------------------------------------------------------- reporting


def write_report(out: Path, split: str, res: dict, meta: dict):
    M = res["M"]
    J = res["J"]
    m0, m1, m2, m4, m5 = M["m0"], M["m1"], M["m2"], M["m4"], M["m5"]
    L = []
    L.append(f"# Sealed Read 1 ({split}) : Addendum M (reduced) + Addendum J\n")
    L.append(f"Code {CODE_VERSION}; frozen M1+M3+M4; rule g_n_components_ge_10mm; NR@20 floor k; seed/resamples per addendum.\n")
    L.append("## M, FFR-UNet units (disclosure only; S2 verdict FAIL is unchanged)\n")
    L.append("### m0 realised 2x2 among critical units\n")
    L.append("| n | P | K | oracle recall | a both | b model only | c rule only | n neither | C_model | C_rule | NR model | NR rule | dNR |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    L.append(f"| {m0['n']} | {m0['P']} | {m0['K']} | {m0['oracle_recall']:.3f} | {m0['a_both']:.2f} | {m0['b_model_only']:.2f} | "
             f"{m0['c_rule_only']:.2f} | {m0['n_neither']:.2f} | {m0['C_model']:.2f} | {m0['C_rule']:.2f} | "
             f"{m0['NR_model']:.3f} | {m0['NR_rule']:.3f} | {m0['delta_NR']:+.3f} |\n")
    L.append(f"Tied units (fractional weight): model {m0['tied_units_model']}, rule {m0['tied_units_rule']}.\n")
    L.append("### m1 paired patient bootstrap (percentile)\n")
    L.append(f"- 20,000 resamples, seed {SEED_S2}: dNR {m1['point']:+.4f}, CI [{m1['ci_lo']:+.4f}, {m1['ci_hi']:+.4f}]\n")
    c2 = m1["check_2000"]
    L.append(f"- Check at 2,000 resamples, seed {SEED_S2}: CI [{c2['ci_lo']:+.4f}, {c2['ci_hi']:+.4f}] vs posted "
             f"+0.125 [-0.003, +0.239]; reproduces within 0.002: **{c2['reproduces_posted_within_0.002']}**\n")
    L.append("### m2 exact McNemar\n")
    L.append(f"- b = {m2['b']:.2f}, c = {m2['c']:.2f}, two-sided exact p = {m2['p_two_sided_exact']:.4g}"
             + (f", range over floor/ceil {m2['p_range_over_floor_ceil'][0]:.4g} to {m2['p_range_over_floor_ceil'][1]:.4g}" if "p_range_over_floor_ceil" in m2 else "") + "\n")
    L.append("### m4 AURC and AUROC (model minus rule), patient bootstrap 2,000, seed 0\n")
    L.append("| metric | model | rule | difference | 95% CI |\n|---|---|---|---|---|")
    L.append(f"| AURC (critical loss) | {m4['AURC_model']:.4f} | {m4['AURC_rule']:.4f} | {m4['dAURC_model_minus_rule']:+.4f} | [{m4['dAURC_ci'][0]:+.4f}, {m4['dAURC_ci'][1]:+.4f}] |")
    L.append(f"| AUROC | {m4['AUROC_model']:.4f} | {m4['AUROC_rule']:.4f} | {m4['dAUROC_model_minus_rule']:+.4f} | [{m4['dAUROC_ci'][0]:+.4f}, {m4['dAUROC_ci'][1]:+.4f}] |\n")
    L.append("### m5 power (planning only, OPTIMISTIC: observed effect and discordance used as truth)\n")
    L.append("| N | power P(dNR >= 0.10 and CI lo > 0) | mean simulated dNR |\n|---|---|---|")
    for k, v in m5.items():
        L.append(f"| {v['N']} | {v['power']:.3f} | {v['mean_delta_NR_sim']:+.3f} |")
    L.append("\n## J, nnU-Net f0-f4 (Family A; Family B excluded)\n")
    L.append("| fold | n | P | K | C_model | C_rule | precision model/rule | recall model/rule | NR model | NR rule | D_f | dAUROC | EXPL: NR share_top2 | EXPL: D share |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in J["folds"]:
        L.append(f"| {row['fold']} | {row['n']} | {row['P']} | {row['K']} | {row['C_model']:.2f} | {row['C_rule']:.2f} | "
                 f"{row['precision_model']:.3f} / {row['precision_rule']:.3f} | {row['recall_model']:.3f} / {row['recall_rule']:.3f} | "
                 f"{row['NR_model']:.3f} | {row['NR_rule']:.3f} | {row['D_f']:+.3f} | {row['dAUROC_model_minus_rule']:+.3f} | "
                 f"{row['EXPLORATORY_NR_share_top2']:.3f} | {row['EXPLORATORY_D_share_top2']:+.3f} |")
    L.append(f"\n- D_A (mean of folds) = **{J['D_A']:+.4f}**, patient-block bootstrap 2,000 seed 0: CI [{J['D_A_ci'][0]:+.4f}, {J['D_A_ci'][1]:+.4f}], nan draws {J['nan_draws']}")
    L.append(f"- Success (D_A >= +0.10 and CI lower > 0): **{J['success']}**")
    L.append("- g_share_top2 columns are EXPLORATORY and not part of the verdict.\n")
    L.append("## Exclusions and inputs\n")
    L.append("```\n" + json.dumps(meta["excluded"], indent=1) + "\n```")
    L.append("\nAssumptions: see results.json field `assumptions`.\n")
    (out / "READ1.md").write_text("\n".join(L))


# ----------------------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=None, help="required for test: pre-launch manifest path")
    ap.add_argument("--root", default="/root/data")
    ap.add_argument("--aorta-dir", default="/root/data/m4/aorta")
    ap.add_argument("--frozen-dir", default="/root/data/m4/frozen")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=UNIT_WORKERS_DEFAULT)
    a = ap.parse_args(argv)
    if a.workers > 8:
        raise SystemExit("at most 8 workers")

    t_start = time.time()
    split = a.split
    root, out = Path(a.root), Path(a.out)
    frozen_dir, aorta_dir = Path(a.frozen_dir), Path(a.aorta_dir)
    if split == "test":
        if not a.manifest:
            raise SystemExit("test mode requires --manifest")
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"refusing non-empty --out {out}")
    out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"[{time.time() - t_start:7.1f}s] {msg}", flush=True)

    from pipeline.label_core import read_split

    shas = VS.verify_frozen_dir(frozen_dir)  # aborts unless all three frozen files match Addendum H
    frozen = json.loads((frozen_dir / "frozen_model.json").read_text())
    left_unit = json.loads((frozen_dir / "lr_convention.json").read_text())["summary"]["left_world_unit"]
    provenance = {"split": split, "frozen_sha256": shas, "code": CODE_VERSION}

    split_ids = read_split(root, split)
    if split == "val":
        overlap = set(split_ids) & set(read_split(root, "test"))
        if overlap:
            raise SystemExit(f"ABORT: {len(overlap)} val ids overlap the test filelist")
    if split == "test" and len(split_ids) != 160:
        raise SystemExit(f"test split has {len(split_ids)} ids, expected 160")

    methods = FOLDS if split == "test" else [FFR] + FOLDS
    man = build_manifest(split, root, aorta_dir, frozen_dir, shas, split_ids, methods, sealed=(split == "test"))
    man_text = json.dumps(man, indent=1, sort_keys=True)
    (out / "MANIFEST.json").write_text(man_text)
    man_sha = hashlib.sha256(man_text.encode()).hexdigest()
    log(f"MANIFEST.json written (sha256 {man_sha[:16]}), {len(split_ids)} scans")
    if a.manifest:
        mp = Path(a.manifest)
        if mp.exists():
            if json.loads(mp.read_text()) != json.loads(man_text):
                raise SystemExit(f"ABORT: computed manifest differs from --manifest {mp}")
            log(f"manifest matches --manifest {mp}")
        else:
            mp.parent.mkdir(parents=True, exist_ok=True)
            mp.write_text(man_text)
            log(f"wrote --manifest {mp}")

    excluded = {}
    if split == "test":
        ffr = load_sealed_ffr(root, split_ids)
        log(f"FFR-UNet sealed outputs read: {len(ffr)} units (not recomputed)")
        wm_chk = flag_weights(ffr["m"].to_numpy(float)) > 0
        excluded["sealed_flag_mismatch_vs_recomputed_top20"] = int(np.sum(wm_chk != ffr["flag_sealed"].to_numpy(bool)))
        ffr = ffr[["scan_id", "y", "m", "r", "s"]]
    else:
        ffr_df, ex, n_jobs = compute_units([FFR], split_ids, root, aorta_dir, frozen_dir, frozen, left_unit,
                                           provenance, out, a.workers, a.device, log)
        excluded.update({f"ffr_val:{k}": v for k, v in ex.items()})
        ffr = ffr_df[ffr_df["method"] == FFR].sort_values("scan_id").reset_index(drop=True)
        log(f"FFR-UNet val units scored: {len(ffr)} of {n_jobs}")
        ffr = ffr[["scan_id", "y", "m", "r", "s"]]

    fold_df, ex, n_jobs = compute_units(FOLDS, split_ids, root, aorta_dir, frozen_dir, frozen, left_unit,
                                        provenance, out, a.workers, a.device, log)
    excluded.update({f"folds:{k}": v for k, v in ex.items()})
    log(f"nnU-Net unit rows scored: {len(fold_df)} of {n_jobs}")

    folds, scans_common = {}, None
    for nm in FOLDS:
        d = fold_df[fold_df["method"] == nm][["scan_id", "y", "m", "r", "s"]].copy()
        folds[nm] = d
        ids = set(d["scan_id"])
        scans_common = ids if scans_common is None else scans_common & ids
    scans_common = sorted(scans_common)
    excluded["J_common_scans"] = len(scans_common)
    excluded["FFR_scored"] = len(ffr)

    t_m = time.time()
    M_res = run_M(ffr, log)
    log(f"M done in {time.time() - t_m:.0f}s")
    J_res = run_J(folds, scans_common, log)

    assumptions = [
        "S2 bootstrap reuses eval.bootstrap.paired_bootstrap_difference with seed 0, exactly as m4/test_eval.py; the 2,000-resample check must reproduce the posted CI.",
        "FFR-UNet test scores come from the sealed flags.csv `score` and the features.csv rule/critical columns (read, not recomputed).",
        "nnU-Net f0-f4 scores use m4/valscore helpers (sha256 check, clamp 1e-6, frozen 22-feature LR, NaN-only median imputation) and m4/score._unit_job; critical labels come from the unit record.",
        "AURC uses the critical indicator as the loss.",
        "m0 a-cell uses the product of fractional memberships; exact when no ties.",
        "Family B is excluded (decided before outcome); the five folds are averaged as one family.",
        "Units with a missing feature or unit error are excluded and counted; J uses scans present in all five folds.",
    ]
    res = {
        "split": split, "code_version": CODE_VERSION, "n_scans": len(split_ids),
        "budget_k": MET._review_count(len(split_ids), BUDGET),
        "M": M_res, "J": J_res, "excluded": excluded, "assumptions": assumptions,
        "manifest_sha256": man_sha,
    }
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    write_report(out, split, res, {"excluded": excluded})

    prov = {
        "split": split, "code_version": CODE_VERSION,
        "code_file_sha256": sha256_file(Path(__file__)),
        "manifest_sha256": man_sha,
        "frozen_sha256": shas,
        "results_json_sha256": sha256_file(out / "results.json"),
        "READ1_md_sha256": sha256_file(out / "READ1.md"),
        "argv": sys.argv,
        "python": platform.python_version(),
        "runtime_s": round(time.time() - t_start, 1),
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=1))
    log(f"DONE in {time.time() - t_start:.0f}s -> {out}")


if __name__ == "__main__":
    main()
