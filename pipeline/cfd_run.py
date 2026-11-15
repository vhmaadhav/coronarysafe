"""Addendum C runner: GT vs reconstruction FFR_sim (1D hyperaemic model) per (method, scan).

Usage (on the droplet, inside cs-infer):
  python pipeline/cfd_run.py --root /root/data --split val --methods cas_net,swin_unetr
  test split runs only with --test-go (the model freeze must come first).

Per unit: /root/data/cfd/units/<split>/<method>/<scan>.json (GT and prediction per vessel, dFFR,
decision flip, unmeasurable flag, rooting method). Summary rows: /root/data/cfd/<split>_results.parquet.
Joins to v3 labels (keyed by method, scan) happen in a separate summary step.
Idempotent: existing unit files are skipped unless --force.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coronarysafe.analysis.hemo1d import analyse_mask, compare_units, crop_inputs  # noqa: E402
from coronarysafe.analysis.harm import nearest_gt_labels  # noqa: E402
from coronarysafe.io import load_nifti  # noqa: E402

def load_scan(root, scan):
    """GT labels, spacing, and the aorta mask (None when m4/aorta/<id> is absent)."""
    gt, sp, _ = load_nifti(Path(root) / "imagecas_x/ImageCAS-X_dataset_v3/segmentations" /
                           f"{scan}.coronary.nii.gz")
    ao_path = Path(root) / "m4/aorta" / f"{scan}.nii.gz"
    aorta = np.asarray(load_nifti(ao_path)[0]) > 0 if ao_path.exists() else None
    return np.asarray(gt), tuple(sp), aorta


def run_unit(gt, sp, aorta, pred_mask):
    """GT and prediction results for one unit, cropped jointly so no prediction voxel is lost."""
    pred = np.asarray(pred_mask, dtype=bool)
    g, p, ao, _ = crop_inputs(gt, pred, aorta, sp)
    lab = nearest_gt_labels(g, sp)
    gt_res = analyse_mask(g > 0, sp, lab, ao)
    pr_res = analyse_mask(p, sp, lab, ao)
    per = compare_units(gt_res, pr_res)
    gt_lumen = gt > 0
    denom = gt_lumen.sum() + pred.sum()
    dice = float(2.0 * np.logical_and(gt_lumen, pred).sum() / denom) if denom else float("nan")
    return {
        "dice": dice,
        "rooting_gt": gt_res["method"], "rooting_pred": pr_res["method"],
        "aorta_available": ao is not None,
        "gt_volume_ml": gt_res["volume_ml"], "pred_volume_ml": pr_res["volume_ml"],
        "gt_flow_ml_s": gt_res["flow_ml_s"], "pred_flow_ml_s": pr_res["flow_ml_s"],
        "vessels": per,
        "gt_vessels": {k: {kk: vv for kk, vv in v.items() if kk != "distal_mm"}
                       for k, v in gt_res["vessels"].items()},
        "pred_vessels": {k: {kk: vv for kk, vv in v.items() if kk != "distal_mm"}
                         for k, v in pr_res["vessels"].items()},
    }


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    raise TypeError(type(o))


def to_rows(scan, method, unit):
    rows = []
    for vessel, r in unit["vessels"].items():
        rows.append({
            "method": method, "scan": scan, "vessel": vessel, "dice": unit["dice"],
            "eligible": r["eligible"], "unmeasurable": r["unmeasurable"], "missing": r["missing"],
            "gt_ffr_distal": r["gt_ffr_distal"], "pred_ffr_distal": r["pred_ffr_distal"],
            "gt_ffr_min": r["gt_ffr_min"], "pred_ffr_min": r["pred_ffr_min"],
            "d_ffr_distal": r.get("d_ffr_distal", np.nan), "d_ffr_min": r.get("d_ffr_min", np.nan),
            "flip": r.get("flip"), "rooting_gt": unit["rooting_gt"],
            "rooting_pred": unit["rooting_pred"], "aorta_available": unit["aorta_available"],
        })
    return rows


def _run_scan(job):
    """One scan: GT loaded once, every method run. Returns (scan, rows)."""
    root, out, split, scan, methods, force = job
    root, out = Path(root), Path(out)
    rows = []
    seg = root / "imagecas_x/ImageCAS-X_dataset_v3/segmentations" / f"{scan}.coronary.nii.gz"
    if not seg.exists():
        return scan, rows
    gt, sp, aorta = load_scan(root, scan)
    for m in methods:
        mpath = root / "preds" / m / scan / "mask.nii.gz"
        if not mpath.exists():
            continue
        unit_path = out / "units" / split / m / f"{scan}.json"
        if unit_path.exists() and not force:
            unit = json.loads(unit_path.read_text())
        else:
            pred = np.asarray(load_nifti(mpath)[0]) > 0
            if pred.shape != gt.shape:
                raise ValueError(f"{scan}/{m}: prediction shape {pred.shape} != GT {gt.shape}")
            unit = run_unit(gt, sp, aorta, pred)
            unit_path.parent.mkdir(parents=True, exist_ok=True)
            unit_path.write_text(json.dumps(unit, default=_jsonable, indent=1))
        rows += to_rows(scan, m, unit)
    return scan, rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/data")
    ap.add_argument("--split", required=True, choices=["val", "test", "train"])
    ap.add_argument("--methods", required=True)
    ap.add_argument("--out", default="/root/data/cfd")
    ap.add_argument("--test-go", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    if args.split == "test" and not args.test_go:
        raise SystemExit("test split waits for the model freeze; pass --test-go after the go")

    root = Path(args.root)
    out = Path(args.out)
    fl = root / "imagecas_x/ImageCAS-X_dataset_v3/filelist" / f"{args.split}.txt"
    scans = [s.strip() for s in fl.read_text().split() if s.strip()]
    if args.limit:
        scans = scans[: args.limit]
    methods = [m for m in args.methods.split(",") if m]
    jobs = [(str(root), str(out), args.split, scan, methods, args.force) for scan in scans]
    rows = []
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for scan, scan_rows in ex.map(_run_scan, jobs):
            rows += scan_rows
            print(f"done {scan}", flush=True)

    import pandas as pd

    df = pd.DataFrame(rows)
    dest = out / f"{args.split}_results.parquet"
    df.to_parquet(dest, index=False)
    print(f"wrote {dest} rows={len(df)}", flush=True)


if __name__ == "__main__":
    main()
