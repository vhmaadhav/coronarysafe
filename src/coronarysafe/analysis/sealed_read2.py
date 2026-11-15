"""Sealed Read 2 (ADDENDA_FROZEN_2026-11-15, Errata 8): Addendum L-test + Addendum K-eval.

One-shot read, launched ONCE by the coordinator. Test mode requires --manifest and refuses a
non-empty --out. Val mode is the dry run: the same code path on val scans only; no test file is
read. In val mode Part K runs on plain cas_net val units as a labelled "dry-run substitute" (there
are no K perturbations on val).

Inputs (frozen, never re-derived):
  * frozen M1+M3+M4 bundle (sha256 per Addendum H, verified by m4/valscore.verify_frozen_dir)
  * frozen NR@20 (eval/metrics.py) and the frozen M4 scoring helpers via analysis/sealed_read1.py
    (compute_units: unit job, clamp 1e-6, frozen 22-feature LR, NaN-only median imputation)
  * Part L label + statistics: analysis/burden_indep.py (score_unit, stats_block, write_report);
    B_alt is NOT reimplemented here.
  * Part L test scores for cas_net / swin_unetr / ffr_unet: READ from the sealed outputs under
    <root>/test_eval (features.csv + flags.csv, method rows). Only a method with no sealed rows is
    computed, once, by the same frozen code path (deterministic; output hash logged).
  * Part K test units: cas_net@K_{noise,contrast,blur} preds + qa/v3 labels, scored by the frozen
    pipeline. Units that are missing or failed are counted, never silently dropped.

Outputs (--out DIR): MANIFEST.json (written first), results.json, READ2.md, provenance.json,
BURDEN_INDEP.md + units.csv + stats.json (burden_indep.write_report), scores_L.csv, units/.

  python -m coronarysafe.analysis.sealed_read2 --split val --out /root/data/addenda/read2_val
  python -m coronarysafe.analysis.sealed_read2 --split test --out DIR --manifest PATH   (coordinator only)

Estimators (pre-declared):
  K  per perturbation p: D_p = NR@20(model) - NR@20(rule); patient-block (scan_id) percentile
     bootstrap, 2,000 resamples, seed 0. Also P, K, C_model, C_rule, precision, recall, AUROC
     (model and rule), missing/failed counts. A perturbation with < 150 labelled units is
     reported as INCOMPLETE with its count. No pooling across perturbations.
  L  per segmenter and pooled: burden_indep.stats_block (Spearman, AUROC, flagged-vs-rest,
     C1, C2, decision branch (i)/(ii)/(iii)); bootstrap 2,000, seed 0.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
for _p in (REPO / "src", REPO / "pipeline", REPO):
    sys.path.insert(0, str(_p))

from coronarysafe.analysis import burden_indep as BI  # noqa: E402  Part L label + stats (reused)
from coronarysafe.analysis import sealed_read1 as R1  # noqa: E402  frozen scoring + manifest helpers
from coronarysafe.eval import metrics as MET  # noqa: E402
from coronarysafe.eval.bootstrap import paired_bootstrap_difference  # noqa: E402
from coronarysafe.m4 import valscore as VS  # noqa: E402

CODE_VERSION = "sealed-read2-v1"
BUDGET = 0.20
N_BOOT = 2000
SEED = 0
MIN_LABELLED = 150
K_PERTS = ["noise", "contrast", "blur"]
K_SEGMENTER = "cas_net"
L_METHODS = ["cas_net", "swin_unetr", "ffr_unet"]
MAX_WORKERS = 16  # raised by coordinator: CPU is free once K and Read 1 finish


def k_method(p: str) -> str:
    return f"{K_SEGMENTER}@K_{p}"


def nr(scores, y) -> float:
    return float(MET.normalised_recall(scores, y, BUDGET))


def safe_auroc(scores, y) -> float:
    try:
        return float(MET.auroc(scores, y))
    except Exception:
        return float("nan")


# ----------------------------------------------------------------------------- sealed reads (TEST only)


def load_sealed_method(root: Path, method: str, test_ids) -> pd.DataFrame | None:
    """TEST only: READ one segmenter's sealed per-unit outputs. Returns None when test_eval has no
    rows for this method (then the caller computes it with the frozen pipeline)."""
    te = root / "test_eval"
    feats = pd.read_csv(te / "features.csv", dtype={"scan_id": str})
    flags = pd.read_csv(te / "flags.csv", dtype={"scan_id": str})
    feats = feats[feats["method"] == method]
    flags = flags[flags["method"] == method]
    if len(feats) == 0 or len(flags) == 0:
        return None
    df = feats.merge(flags[["scan_id", "score", "flagged_top20"]], on="scan_id", validate="one_to_one")
    if len(df) != len(test_ids) or set(df["scan_id"]) != set(test_ids):
        raise SystemExit(f"sealed {method} rows do not match filelist/test.txt")
    out = pd.DataFrame(
        {
            "method": method,
            "scan_id": df["scan_id"].values,
            "y": df["critical"].astype(int).values,
            "m": df["score"].astype(float).values,
            "r": df["g_n_components_ge_10mm"].astype(float).values,
            "s": 1.0 - df["g_share_top2"].astype(float).fillna(0.0).values,
        }
    )
    wm_chk = R1.flag_weights(out["m"].to_numpy(float)) > 0
    if int(np.sum(wm_chk != df["flagged_top20"].astype(bool).values)) != 0:
        raise SystemExit(f"sealed {method} flags disagree with recomputed top-20% set")
    return out.sort_values("scan_id").reset_index(drop=True)


def reference_unperturbed(root: Path) -> dict:
    """TEST only: the unperturbed cas_net test numbers, READ from the existing sealed summary.
    Stored verbatim (the lines of RESULTS.md that mention cas_net); nothing is recomputed."""
    rp = root / "test_eval" / "RESULTS.md"
    lines = [ln for ln in rp.read_text().splitlines() if "cas_net" in ln]
    return {"source": str(rp), "source_sha256": R1.sha256_file(rp), "cas_net_lines": lines}


# ----------------------------------------------------------------------------- Part K


def k_stats(d: pd.DataFrame, method: str, label: str, expected: int, excluded: dict) -> dict:
    d = d.sort_values("scan_id").reset_index(drop=True)
    n = len(d)
    ex = {k.split(":", 1)[1]: v for k, v in excluded.items() if k.startswith(method + ":")}
    out = {
        "method": method,
        "label": label,
        "expected_units": expected,
        "scored_labelled_units": n,
        "missing_or_failed_units": expected - n,
        "missing_or_failed_breakdown": ex,
        "min_labelled": MIN_LABELLED,
        "status": "COMPLETE" if n >= MIN_LABELLED else "INCOMPLETE",
    }
    if n == 0:
        out["note"] = "no scored units"
        return out
    y = d["y"].to_numpy(int)
    m = d["m"].to_numpy(float)
    r = d["r"].to_numpy(float)
    P = int(y.sum())
    K = MET._review_count(n, BUDGET)
    out.update({"P": P, "K": K})
    if P == 0:
        out["note"] = "no critical units; NR undefined"
        return out
    R1.check_weights(m, y)
    R1.check_weights(r, y)
    wm, wr = R1.flag_weights(m), R1.flag_weights(r)
    crit = y == 1
    C_m = float(np.sum(wm[crit]))
    C_r = float(np.sum(wr[crit]))
    NR_m, NR_r = nr(m, y), nr(r, y)
    boot = paired_bootstrap_difference(
        d["scan_id"].to_numpy(str),
        lambda rows: nr(m[rows], y[rows]),
        lambda rows: nr(r[rows], y[rows]),
        n_boot=N_BOOT,
        seed=SEED,
    )
    out.update(
        {
            "C_model": C_m,
            "C_rule": C_r,
            "precision_model": C_m / K,
            "precision_rule": C_r / K,
            "recall_model": C_m / P,
            "recall_rule": C_r / P,
            "NR_model": NR_m,
            "NR_rule": NR_r,
            "D": NR_m - NR_r,
            "D_ci95_percentile": [float(boot["lo"]), float(boot["hi"])],
            "D_bootstrap_nan_draws": int(np.isnan(boot["samples"]).sum()),
            "AUROC_model": safe_auroc(m, y),
            "AUROC_rule": safe_auroc(r, y),
            "bootstrap": {"resamples": N_BOOT, "seed": SEED, "cluster": "patient (scan_id)"},
        }
    )
    return out


# ----------------------------------------------------------------------------- Part L


def l_block(units: pd.DataFrame, scores: pd.DataFrame, n_requested: int) -> dict:
    stats = None
    try:
        stats = BI.stats_block(units, scores)
    except ValueError as exc:
        return {"error": str(exc), "n_requested": n_requested}
    return {
        "n_requested": n_requested,
        "n_ok": int(((units["status"] == "ok") & units["b_alt"].notna()).sum()),
        "n_excluded_low_support": int((units["status"] == "excluded_low_support").sum()),
        "n_errors": int((units["status"] == "error").sum()),
        "stats": stats,
    }


def write_read2_md(out: Path, split: str, res: dict) -> None:
    K, L = res["K"], res["L"]
    lines = [f"# Sealed Read 2 ({split}): Addendum L-test + Addendum K-eval", ""]
    lines.append(f"Code {CODE_VERSION}; frozen M1+M3+M4; rule g_n_components_ge_10mm; NR@20 floor k; "
                 f"patient-block bootstrap {N_BOOT}, seed {SEED}. Units: {res['units_note']}.")
    lines.append("")
    lines.append("## K: controlled acquisition shift (per perturbation, NO pooling)")
    lines.append("")
    lines.append("| perturbation | status | labelled | missing/failed | P | K | C_model | C_rule | "
                 "prec m/r | recall m/r | NR m | NR r | D (m-r) | 95% CI | AUROC m/r |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, k in K.items():
        if "D" not in k:
            lines.append(f"| {key} ({k.get('label')}) | {k['status']} | {k['scored_labelled_units']} | "
                         f"{k['missing_or_failed_units']} | - | - | - | - | - | - | - | - | - | - | - |")
            continue
        ci = k["D_ci95_percentile"]
        lines.append(
            f"| {key} ({k['label']}) | {k['status']} | {k['scored_labelled_units']} | {k['missing_or_failed_units']} | "
            f"{k['P']} | {k['K']} | {k['C_model']:.2f} | {k['C_rule']:.2f} | "
            f"{k['precision_model']:.3f}/{k['precision_rule']:.3f} | {k['recall_model']:.3f}/{k['recall_rule']:.3f} | "
            f"{k['NR_model']:.3f} | {k['NR_rule']:.3f} | {k['D']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
            f"{k['AUROC_model']:.3f}/{k['AUROC_rule']:.3f} |"
        )
    lines.append("")
    for key, k in K.items():
        if k.get("missing_or_failed_breakdown"):
            lines.append(f"- {key} missing/failed breakdown: {k['missing_or_failed_breakdown']}")
    lines.append("")
    lines.append("## L: independent-lumen burden (B_alt), per segmenter and pooled")
    lines.append("")
    if "error" in L:
        lines.append(f"L stats not computed: {L['error']}")
    else:
        lines.append(f"Requested {L['n_requested']}, ok {L['n_ok']}, excluded low support "
                     f"{L['n_excluded_low_support']}, errors {L['n_errors']}.")
        lines.append("")
        lines.append("| key | n | patients | n critical v3 | Spearman(score,B_alt) [CI] | AUROC high B_alt [CI] | "
                     "top-20 minus rest B_alt [CI] | C1 Spearman(v3,B_alt) | C2 Spearman(score, resid) [CI] | branch |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for key, e in (L["stats"] or {}).items():
            if "note" in e:
                lines.append(f"| {key} | {e['n']} | - | - | {e['note']} | | | | | |")
                continue
            f = BI._fmt
            sp, au, fl = e["spearman_score_balt"], e["auroc_score_high_balt"], e["flagged_top20_minus_rest_balt"]
            c1, c2 = e["c1_spearman_burdenv3_balt"], e["c2_spearman_score_resid"]
            lines.append(
                f"| {key} | {e['n']} | {e['n_patients']} | {e['n_critical_v3']} | "
                f"{f(sp['estimate'])} [{f(sp['lo'])}, {f(sp['hi'])}] | "
                f"{f(au['estimate'])} [{f(au['lo'])}, {f(au['hi'])}] | "
                f"{f(fl['estimate'])} [{f(fl['lo'])}, {f(fl['hi'])}] | {f(c1['estimate'])} | "
                f"{f(c2['estimate'])} [{f(c2['lo'])}, {f(c2['hi'])}] | {e['decision_branch']} |"
            )
    lines.append("")
    if res.get("reference_unperturbed"):
        lines.append("## Reference: unperturbed cas_net test numbers (READ from sealed summary, not recomputed)")
        lines.append("")
        for ln in res["reference_unperturbed"]["cas_net_lines"]:
            lines.append(f"    {ln}")
        lines.append("")
    lines.append("## Assumptions")
    lines.append("")
    for a in res["assumptions"]:
        lines.append(f"- {a}")
    lines.append("")
    (out / "READ2.md").write_text("\n".join(lines))


# ----------------------------------------------------------------------------- main


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=None, help="required for test: pre-launch manifest path")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--root", default="/root/data")
    ap.add_argument("--aorta-dir", default="/root/data/m4/aorta")
    ap.add_argument("--frozen-dir", default="/root/data/m4/frozen")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    if a.workers > MAX_WORKERS:
        raise SystemExit(f"at most {MAX_WORKERS} workers")

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

    ids = read_split(root, split)
    if split == "val":
        overlap = set(ids) & set(read_split(root, "test"))
        if overlap:
            raise SystemExit(f"ABORT: {len(overlap)} val ids overlap the test filelist")
    if split == "test" and len(ids) != 160:
        raise SystemExit(f"test split has {len(ids)} ids, expected 160")

    K_methods = [k_method(p) for p in K_PERTS] if split == "test" else [K_SEGMENTER]
    methods_all = list(dict.fromkeys(K_methods + L_METHODS))
    man = R1.build_manifest(split, root, aorta_dir, frozen_dir, shas, ids, methods_all,
                            sealed=(split == "test"))
    man_text = json.dumps(man, indent=1, sort_keys=True)
    (out / "MANIFEST.json").write_text(man_text)
    man_sha = hashlib_sha(man_text)
    log(f"MANIFEST.json written (sha256 {man_sha[:16]}), {len(ids)} scans")
    mp = Path(a.manifest) if a.manifest else None
    if mp is not None:
        if mp.exists():
            if json.loads(mp.read_text()) != json.loads(man_text):
                raise SystemExit(f"ABORT: computed manifest differs from --manifest {mp}")
            log(f"manifest matches --manifest {mp}")
        else:
            mp.parent.mkdir(parents=True, exist_ok=True)
            mp.write_text(man_text)
            log(f"wrote --manifest {mp}")

    excluded: dict = {}
    hashes: dict = {}

    # ---- L scores: sealed read for test (cas/swin/ffr), frozen pipeline otherwise
    if split == "test":
        sealed = {m: load_sealed_method(root, m, ids) for m in L_METHODS}
        log("L sealed reads: " + ", ".join(f"{m}={'read' if sealed[m] is not None else 'missing'}"
                                           for m in L_METHODS))
        need = [m for m in L_METHODS if sealed[m] is None]
        L_frames = [sealed[m] for m in L_METHODS if sealed[m] is not None]
        if need:
            df_c, ex_c, _ = R1.compute_units(need, ids, root, aorta_dir, frozen_dir, frozen, left_unit,
                                             provenance, out, a.workers, a.device, log)
            excluded.update({f"L:{k}": v for k, v in ex_c.items()})
            L_frames.append(df_c)
        L_scores = pd.concat(L_frames, ignore_index=True)
        K_df, ex_k, _ = R1.compute_units(K_methods, ids, root, aorta_dir, frozen_dir, frozen, left_unit,
                                         provenance, out, a.workers, a.device, log)
        excluded.update({f"K:{k}": v for k, v in ex_k.items()})
        unperturbed = reference_unperturbed(root)
        units_note = "test, 160 sealed scans; K perturbations scored by the frozen pipeline"
    else:
        all_df, ex_all, _ = R1.compute_units(L_METHODS, ids, root, aorta_dir, frozen_dir, frozen, left_unit,
                                             provenance, out, a.workers, a.device, log)
        excluded.update({f"val:{k}": v for k, v in ex_all.items()})
        L_scores = all_df
        K_df = all_df[all_df["method"] == K_SEGMENTER]
        unperturbed = None
        units_note = "VAL DRY RUN (not sealed): K = dry-run substitute on plain cas_net val units"

    L_scores = L_scores[["method", "scan_id", "score"]] if "score" in L_scores else \
        L_scores.assign(score=L_scores["m"])[["method", "scan_id", "score"]]
    L_scores = L_scores.assign(scan_id=L_scores["scan_id"].astype(str))
    sc_path = out / "scores_L.csv"
    L_scores.to_csv(sc_path, index=False)
    hashes["scores_L_csv_sha256"] = R1.sha256_file(sc_path)
    log(f"L scores: {len(L_scores)} rows, sha256 {hashes['scores_L_csv_sha256'][:16]}")

    # ---- Part K
    K_res: dict = {}
    if split == "test":
        for p in K_PERTS:
            meth = k_method(p)
            K_res[p] = k_stats(K_df[K_df["method"] == meth], meth, "sealed K perturbation", 160, excluded)
            log(f"K {p}: {K_res[p]['status']} labelled {K_res[p]['scored_labelled_units']}"
                + (f" D={K_res[p]['D']:+.4f}" if "D" in K_res[p] else ""))
    else:
        K_res["substitute"] = k_stats(K_df, K_SEGMENTER, "dry-run substitute (cas_net val)", 80, excluded)
        log(f"K substitute: {K_res['substitute']['status']} labelled "
            f"{K_res['substitute']['scored_labelled_units']} D={K_res['substitute'].get('D', float('nan')):+.4f}")

    # ---- Part L: burden_indep functions (B_alt label + stats), not reimplemented
    orig = BI.index_original_masks(BI.ORIG_DIR)
    scans = BI.read_split_ids(root, split)
    jobs = [(m, s, str(root), str(orig[s])) for m in L_METHODS for s in scans if s in orig]
    with Pool(a.workers) as pool:
        recs = list(pool.imap_unordered(BI.score_unit, jobs))
    units = pd.DataFrame(recs).sort_values(["method", "scan_id"]).reset_index(drop=True)
    L_res = l_block(units, L_scores, len(jobs))
    log(f"L units: {L_res.get('n_ok')} ok of {len(jobs)}; "
        + ("; ".join(f"{k}: {v.get('decision_branch')}" for k, v in (L_res.get('stats') or {}).items())))
    if L_res.get("stats") is not None:
        BI.write_report(out, type("A", (), {"split": split})(), units, L_res["stats"], len(jobs))
    else:
        units.to_csv(out / "units.csv", index=False)

    assumptions = [
        "Part L label and statistics are burden_indep.score_unit and burden_indep.stats_block, unchanged; B_alt is not reimplemented.",
        "L scores: test cas_net/swin_unetr/ffr_unet read from test_eval features/flags (flags.csv score); a method without sealed rows is computed once by the frozen pipeline (hash logged).",
        "K scores: frozen pipeline (sealed_read1.compute_units) on preds/cas_net@K_<p> with qa/v3 labels; the rule is qa graph n_components_ge_10mm.",
        "K D_p and L use patient-block (scan_id) bootstrap, 2,000 resamples, seed 0; no pooling across K perturbations.",
        "Perturbations with fewer than 150 labelled units are INCOMPLETE and carry their count.",
        "Missing or failed units are excluded and counted per method; they are never imputed.",
    ]
    if split == "val":
        assumptions.append("VAL DRY RUN: K uses plain cas_net val units labelled dry-run substitute; no sealed input is read.")
    res = {
        "split": split,
        "code_version": CODE_VERSION,
        "units_note": units_note,
        "n_scans": len(ids),
        "manifest_sha256": man_sha,
        "frozen_sha256": shas,
        "K": K_res,
        "L": L_res,
        "excluded": excluded,
        "hashes": hashes,
        "reference_unperturbed": unperturbed,
        "assumptions": assumptions,
        "runtime_s_before_results": round(time.time() - t_start, 1),
    }
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    write_read2_md(out, split, res)

    prov = {
        "split": split,
        "code_version": CODE_VERSION,
        "code_file_sha256": R1.sha256_file(Path(__file__)),
        "manifest_sha256": man_sha,
        "manifest_path": str(mp) if mp else None,
        "frozen_sha256": shas,
        "results_json_sha256": R1.sha256_file(out / "results.json"),
        "READ2_md_sha256": R1.sha256_file(out / "READ2.md"),
        "hashes": hashes,
        "argv": sys.argv,
        "python": platform.python_version(),
        "runtime_s": round(time.time() - t_start, 1),
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=1))
    log(f"DONE in {time.time() - t_start:.0f}s -> {out}")


def hashlib_sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


if __name__ == "__main__":
    main()
