"""Label robustness to a second annotator (ANALYSIS_PLAN_v1 Addendum I). Secondary, exploratory.

For each unit (segmenter x scan) the v3 failure labels are recomputed against an INDEPENDENT
reference: the original ImageCAS lumen mask (a different annotation team from ImageCAS-X).

1. Transfer: every voxel of the original lumen takes the segment label (1..14) of the nearest
   ImageCAS-X labelled voxel, if that voxel is within ``max_mm`` (spacing-aware EDT); otherwise
   it gets the "Other" label 14.
2. The frozen taxonomy_v3 code (``label_failures``) runs with the relabelled original mask as GT
   and centerlines derived from its skeleton (the code's fallback path; no VTK).
   Outputs: critical_alt, severe_alt, burden_alt.
3. Agreement of v3 critical (qa/v3) with critical_alt: percent agreement, Cohen's kappa,
   Gwet's AC1, PABAK, patient-cluster bootstrap CI (2000, seed 0), per segmenter and pooled.
4. Optional --scores CSV (method, scan_id, score): NR@20%, AUROC and paired frozen-minus-rule
   difference (rule = graph g_n_components_ge_10mm from qa/v3) against critical_alt.

Frozen code, features and thresholds are unchanged. This is a robustness analysis, not a new test.

  python -m coronarysafe.analysis.label_robust --split val --methods cas_net swin_unetr ffr_unet --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to
from scipy import ndimage
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from ..eval.bootstrap import cluster_bootstrap, paired_bootstrap_difference  # noqa: E402
from ..eval.metrics import normalised_recall  # noqa: E402
from ..geometry import bbox_slices  # noqa: E402
from ..labels.taxonomy import label_failures, load_taxonomy_config  # noqa: E402
from ..io import load_nifti  # noqa: E402

ROOT = Path("/root/data")
ICX_GT_DIR = "imagecas_x/ImageCAS-X_dataset_v3"
ORIG_DIR = ROOT / "imagecas" / "orig_labels"
TAXONOMY_CFG = REPO / "configs" / "taxonomy_v3.yaml"
OTHER = 14
MAX_MM = 3.0
BUDGET = 0.2
N_BOOT = 2000
SEED = 0
PAD_MM = 10.0  # crop margin around the union of original lumen and prediction (no effect on labels)


def transfer_labels(orig, gt, spacing, max_mm: float = MAX_MM, other: int = OTHER):
    """Relabel the original binary lumen with ImageCAS-X segment names.

    orig: binary original lumen (any dtype), gt: ImageCAS-X 14-class array, same grid.
    Returns (relabelled uint8 array on the full grid, stats dict).
    """
    orig = np.asarray(orig) > 0
    gt = np.asarray(gt)
    sp = np.asarray(spacing, dtype=float)
    out = np.zeros(orig.shape, dtype=np.uint8)
    if not orig.any():
        return out, {"mapped_fraction": float("nan"), "n_orig": 0}
    labelled = gt > 0
    if not labelled.any():
        raise ValueError("ImageCAS-X GT has no labelled voxels")
    pad = np.ceil(max_mm / sp).astype(int) + 2
    sl = bbox_slices(orig, sp, pad_vox=int(pad.max()))
    o = orig[sl]
    g = gt[sl]
    lab = g > 0
    if lab.any():
        # distance from each voxel to its nearest labelled voxel (indices of the nearest zero of ~lab)
        dist, idx = ndimage.distance_transform_edt(~lab, sampling=tuple(sp), return_indices=True)
        near = dist <= max_mm + 1e-9
        names = g[idx[0], idx[1], idx[2]].astype(np.uint8)
        seg = np.where(near, names, np.uint8(other)).astype(np.uint8)
    else:
        near = np.zeros(o.shape, dtype=bool)
        seg = np.full(o.shape, other, dtype=np.uint8)
    sub = out[sl]
    sub[o] = seg[o]
    out[sl] = sub
    return out, {"mapped_fraction": float(near[o].mean()), "n_orig": int(o.sum())}


def dice(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    den = a.sum() + b.sum()
    return float(2.0 * np.logical_and(a, b).sum() / den) if den else float("nan")


def index_original_masks(orig_dir: Path = ORIG_DIR) -> dict:
    """Map scan id -> original label path. Raises on duplicate ids across groups."""
    found = {}
    for p in sorted(Path(orig_dir).glob("*/*.label.nii.gz")):
        sid = str(int(p.name.split(".")[0]))
        if sid in found:
            raise ValueError(f"duplicate original mask for scan {sid}: {found[sid]} and {p}")
        found[sid] = p
    return found


def read_split_ids(root: Path, split: str) -> list:
    base = Path(root) / ICX_GT_DIR / "filelist"
    ids, seen = [], set()
    for line in (base / f"{split}.txt").read_text().splitlines():
        line = line.strip()
        if line and str(int(line)) not in seen:
            seen.add(str(int(line)))
            ids.append(str(int(line)))
    return ids


def _load_original(path: Path, ref_shape, ref_affine):
    img = nib.load(str(path))
    if tuple(img.shape) == tuple(ref_shape) and np.allclose(img.affine, ref_affine, atol=0.01):
        return np.asanyarray(img.dataobj) > 0
    # different grid: nearest-neighbour resampling onto the ImageCAS-X grid
    res = resample_from_to(img, (tuple(ref_shape), ref_affine), order=0)
    return np.asanyarray(res.dataobj) > 0


RESTRICTED_TERMS = ("F1", "F2")  # restricted critical = these flags only, on base major segments


def restricted_critical(events, cfg):
    """Restricted critical tier (Addendum I, restricted variant). Returns (critical, reasons).

    Keeps only the terms that do not depend on lumen thickness:
      (a') F1 (missed) or F2 (disconnection) on a base major segment (LM, LAD, LCx, RCA, from
           cfg major_base_labels) whose start is within cfg tier distance_mm of the ostium.
           The GT-radius alternative of the full tier (radius >= tier radius_mm) is dropped.
    Dropped: F3c (leak, thick bridge), F3b (detached false positive, thick), and every
    thickness-based clause. F1b, F3a, F3h, F3d are never critical in the full tier either.
    """
    base = set(int(x) for x in cfg["major_base_labels"])
    dist_max = float((cfg.get("tier") or {}).get("distance_mm", 50.0))
    reasons = []
    for e in events:
        if e.get("flag") not in RESTRICTED_TERMS:
            continue
        if not (set(int(x) for x in e.get("label_ids", [])) & base):
            continue
        d = e.get("dist_from_ostium_mm")
        if d is not None and np.isfinite(d) and d <= dist_max:
            reasons.append(f"{e['flag']}:{e.get('segment', '+'.join(e.get('segments', [])))}")
    return bool(reasons), reasons


def score_unit(task):
    """One (method, scan) unit. Returns a flat record; errors are recorded, never raised."""
    method, sid, root, orig_path, cfg = task
    root = Path(root)
    rec = {"method": method, "scan_id": sid, "status": "ok", "error": ""}
    try:
        qa_path = root / "qa" / "v3" / method / f"{sid}.json"
        qa = json.loads(qa_path.read_text())
        r_v3, _ = restricted_critical(qa.get("events", []), cfg)
        rec.update({
            "critical_v3": int(bool(qa["critical"])),
            "severe_v3": int(bool(qa["severe"])),
            "burden_v3": float(qa["burden"]),
            "critical_v3_r": int(r_v3),
            "centerline_source_v3": qa.get("centerline_source"),
            "rule_n_components_ge_10mm": float((qa.get("graph") or {}).get("n_components_ge_10mm", np.nan)),
        })
        gt, sp, aff = load_nifti(root / ICX_GT_DIR / "segmentations" / f"{sid}.coronary.nii.gz")
        gt = np.asarray(gt)
        pred_arr, _, pred_aff = load_nifti(root / "preds" / method / sid / "mask.nii.gz")
        pred = np.asarray(pred_arr) > 0
        if pred.shape != gt.shape:
            raise ValueError(f"prediction shape {pred.shape} != GT {gt.shape}")
        if not np.allclose(np.asarray(pred_aff, float), np.asarray(aff, float), atol=0.01):
            raise ValueError("prediction affine differs from GT affine (allclose, atol 0.01 mm)")
        orig = _load_original(orig_path, gt.shape, aff)
        rec["icx_orig_dice"] = dice(orig, gt > 0)
        relab, st = transfer_labels(orig, gt, sp)
        rec["frac_mapped"] = st["mapped_fraction"]
        rec["n_orig_voxels"] = st["n_orig"]
        # crop to the union of original lumen and prediction (label_failures crops further; no effect)
        sl = bbox_slices(orig | pred, sp, pad_mm=PAD_MM)
        lab = label_failures(relab[sl], None, pred[sl], sp, cfg)
        r_alt, _ = restricted_critical(lab["events"], cfg)
        rec.update({
            "critical_alt": int(bool(lab["critical"])),
            "critical_alt_r": int(r_alt),
            "severe_alt": int(bool(lab["severe"])),
            "burden_alt": float(lab["burden"]),
            "centerline_source_alt": lab["centerline_source"],
            "n_centerline_points_alt": int(lab["n_centerline_points"]),
        })
    except Exception as exc:  # recorded per unit, the run continues
        rec["status"] = "error"
        rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return rec


def agreement_stats(v3, alt) -> dict:
    """Binary agreement: percent agreement, Cohen's kappa, Gwet's AC1, PABAK."""
    a = np.asarray(v3).astype(bool)
    b = np.asarray(alt).astype(bool)
    n = len(a)
    if n == 0:
        return {k: float("nan") for k in ["n", "percent_agreement", "kappa", "ac1", "pabak", "prev_v3", "prev_alt"]}
    pa = float(np.mean(a == b))
    p1, q1 = float(a.mean()), float(b.mean())
    pe_k = p1 * q1 + (1 - p1) * (1 - q1)
    kappa = (pa - pe_k) / (1 - pe_k) if pe_k < 1 else float("nan")
    pi = (p1 + q1) / 2.0
    pe_g = 2.0 * pi * (1 - pi)
    ac1 = (pa - pe_g) / (1 - pe_g) if pe_g < 1 else float("nan")
    return {"n": int(n), "percent_agreement": pa, "kappa": float(kappa), "ac1": float(ac1),
            "pabak": 2.0 * pa - 1.0, "prev_v3": p1, "prev_alt": q1}


def _ci_block(v3, alt, clusters):
    full = agreement_stats(v3, alt)
    out = {"n": full["n"], "prev_v3": full["prev_v3"], "prev_alt": full["prev_alt"], "patients": int(len(set(clusters)))}
    for key in ["percent_agreement", "kappa", "ac1", "pabak"]:
        res = cluster_bootstrap(
            clusters, lambda rows, k=key: agreement_stats(v3[rows], alt[rows])[k], n_boot=N_BOOT, seed=SEED)
        out[key] = {"estimate": full[key], "lo": res["lo"], "hi": res["hi"]}
    return out


def agreement_block(units: pd.DataFrame, v3_col: str = "critical_v3", alt_col: str = "critical_alt") -> dict:
    ok = units[units["status"] == "ok"].reset_index(drop=True)
    res = {"per_segmenter": {}, "pooled": None, "v3_col": v3_col, "alt_col": alt_col}
    for m, d in ok.groupby("method"):
        d = d.reset_index(drop=True)
        res["per_segmenter"][m] = _ci_block(d[v3_col].to_numpy(int), d[alt_col].to_numpy(int),
                                            d["scan_id"].to_numpy(str))
    if len(ok):
        res["pooled"] = _ci_block(ok[v3_col].to_numpy(int), ok[alt_col].to_numpy(int),
                                  ok["scan_id"].to_numpy(str))
    return res


def rescore_block(units: pd.DataFrame, scores: pd.DataFrame) -> dict:
    """Frozen score, rule and NR@20%, AUROC, paired frozen-minus-rule, against critical_alt."""
    ok = units[units["status"] == "ok"]
    out = {}
    for m, d in ok.groupby("method"):
        s = scores[scores["method"] == m][["scan_id", "score"]].drop_duplicates("scan_id")
        d = d.merge(s, on="scan_id", how="inner").reset_index(drop=True)
        if len(d) == 0:
            out[m] = {"available": False, "reason": "no matching scan_id in scores"}
            continue
        y = d["critical_alt"].to_numpy(bool)
        frozen = d["score"].to_numpy(float)
        rule = d["rule_n_components_ge_10mm"].fillna(0).to_numpy(float)
        clusters = d["scan_id"].to_numpy(str)
        entry = {"available": True, "n": int(len(d)), "n_critical_alt": int(y.sum()),
                 "nr20_frozen": float(normalised_recall(frozen, y, BUDGET)),
                 "nr20_rule": float(normalised_recall(rule, y, BUDGET)),
                 "auroc_frozen": float(roc_auc_score(y, frozen)) if 0 < y.sum() < len(y) else float("nan"),
                 "auroc_rule": float(roc_auc_score(y, rule)) if 0 < y.sum() < len(y) else float("nan")}
        res = paired_bootstrap_difference(
            clusters,
            lambda rows: normalised_recall(frozen[rows], y[rows], BUDGET),
            lambda rows: normalised_recall(rule[rows], y[rows], BUDGET),
            n_boot=N_BOOT, seed=SEED)
        entry["paired_frozen_minus_rule"] = {"estimate": float(res["estimate"]), "lo": res["lo"], "hi": res["hi"]}
        out[m] = entry
    return out


def _fmt(x, nd=3):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def write_report(out: Path, args, units: pd.DataFrame, agree: dict, rescore: dict | None):
    ok = units[units["status"] == "ok"]
    title = "restricted critical tier (F1/F2 on LM/LAD/LCx/RCA within 50 mm; no thickness terms)" \
        if getattr(args, "restricted", False) else "full v3 critical tier"
    lines = [f"# Label robustness to a second annotator (Addendum I), split={args.split}", "",
             f"Label: {title}. Prevalence v3 column = ImageCAS-X reference; alt column = original-mask reference.", "",
             "Reference: original ImageCAS lumen masks (independent team), segment names transferred from "
             "ImageCAS-X within 3 mm, frozen taxonomy_v3 with skeleton-derived centerlines. Secondary, exploratory.", "",
             f"Units: {len(units)} requested, {len(ok)} ok, {int((units['status'] != 'ok').sum())} errors.", ""]
    if len(ok):
        lines += [f"Inter-annotator mask Dice (ImageCAS-X vs original, per scan): median "
                  f"{ok.drop_duplicates('scan_id')['icx_orig_dice'].median():.3f}, "
                  f"IQR {ok.drop_duplicates('scan_id')['icx_orig_dice'].quantile(0.25):.3f}-"
                  f"{ok.drop_duplicates('scan_id')['icx_orig_dice'].quantile(0.75):.3f}. "
                  f"Fraction of original voxels mapped within 3 mm: median {ok['frac_mapped'].median():.4f}.", ""]
        lines += [f"v3 centerline source: {ok['centerline_source_v3'].value_counts().to_dict()}. "
                  f"Alt: {ok['centerline_source_alt'].value_counts().to_dict()}. "
                  "The two differ (VTK vs skeleton), a confound that this analysis does not remove.", ""]
    lines += ["## Agreement, v3 critical vs critical_alt (patient-cluster bootstrap 95% CI, 2000, seed 0)", "",
              "| segmenter | n | prev v3 | prev alt | % agree | kappa | Gwet AC1 | PABAK |",
              "|---|---|---|---|---|---|---|---|"]
    rows = list(agree["per_segmenter"].items()) + ([("pooled", agree["pooled"])] if agree["pooled"] else [])
    for name, a in rows:
        cell = lambda k: f"{_fmt(a[k]['estimate'])} [{_fmt(a[k]['lo'])}, {_fmt(a[k]['hi'])}]"
        lines.append(f"| {name} | {a['n']} | {_fmt(a['prev_v3'])} | {_fmt(a['prev_alt'])} | "
                     f"{cell('percent_agreement')} | {cell('kappa')} | {cell('ac1')} | {cell('pabak')} |")
    if rescore is not None:
        lines += ["", "## Frozen model and rule re-scored against critical_alt", "",
                  "| segmenter | n | NR@20 frozen | NR@20 rule | AUROC frozen | AUROC rule | frozen - rule [CI] |",
                  "|---|---|---|---|---|---|---|"]
        for m, e in rescore.items():
            if not e.get("available"):
                lines.append(f"| {m} | - | {e.get('reason')} | | | | |")
                continue
            p = e["paired_frozen_minus_rule"]
            lines.append(f"| {m} | {e['n']} | {_fmt(e['nr20_frozen'])} | {_fmt(e['nr20_rule'])} | "
                         f"{_fmt(e['auroc_frozen'])} | {_fmt(e['auroc_rule'])} | "
                         f"{_fmt(p['estimate'])} [{_fmt(p['lo'])}, {_fmt(p['hi'])}] |")
    lines += ["", "Secondary and exploratory. The frozen model, features and thresholds are unchanged; "
              "this checks label robustness, it is not a new test of S1-S3.", ""]
    (out / "LABEL_ROBUSTNESS.md").write_text("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--scores", default=None, help="CSV with method, scan_id, score (frozen model)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--restricted", action="store_true",
                    help="agreement on the restricted critical tier (F1/F2 proximal, no thickness terms)")
    a = ap.parse_args(argv)

    root = Path(a.root)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_taxonomy_config(TAXONOMY_CFG)
    orig_index = index_original_masks(root / "imagecas" / "orig_labels")
    scans = read_split_ids(root, a.split)

    tasks, missing = [], []
    for m in a.methods:
        for sid in scans:
            if sid not in orig_index:
                missing.append((m, sid, "no original mask"))
                continue
            if not (root / "qa" / "v3" / m / f"{sid}.json").exists() or not (root / "preds" / m / sid / "mask.nii.gz").exists():
                missing.append((m, sid, "no v3 qa or prediction"))
                continue
            tasks.append((m, sid, str(root), str(orig_index[sid]), cfg))

    with Pool(processes=a.workers) as pool:
        recs = pool.map(score_unit, tasks, chunksize=1)
    recs += [{"method": m, "scan_id": s, "status": "missing", "error": why} for m, s, why in missing]
    units = pd.DataFrame(recs).sort_values(["method", "scan_id"]).reset_index(drop=True)
    units.to_csv(out / "units.csv", index=False)

    if a.restricted:
        agree = agreement_block(units, "critical_v3_r", "critical_alt_r")
    else:
        agree = agreement_block(units)
    rescore = None
    if a.scores and not a.restricted:
        scores = pd.read_csv(a.scores, dtype={"scan_id": str})
        rescore = rescore_block(units, scores)
    summary = {"split": a.split, "methods": a.methods, "n_units_ok": int((units["status"] == "ok").sum()),
               "n_units_not_ok": int((units["status"] != "ok").sum()), "agreement": agree,
               "restricted": bool(a.restricted), "rescore_against_critical_alt": rescore,
               "taxonomy_config": str(TAXONOMY_CFG),
               "max_mm": MAX_MM, "other_label": OTHER, "n_boot": N_BOOT, "seed": SEED}
    (out / "agreement.json").write_text(json.dumps(summary, indent=2, default=float))
    write_report(out, a, units, agree, rescore)
    print(json.dumps({"ok": summary["n_units_ok"], "not_ok": summary["n_units_not_ok"],
                      "pooled": agree["pooled"]}, default=float)[:2000], flush=True)
    print("LABEL_ROBUSTNESS_DONE", flush=True)


if __name__ == "__main__":
    main()
