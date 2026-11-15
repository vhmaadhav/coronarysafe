"""Dev-only scoring of M4 (and M3 ostium features) on ImageCAS-X VAL x {cas_net, swin_unetr}.

Never reads FFR-UNet predictions and never reads test-split scans. Usage (droplet):

  python -m coronarysafe.m4.score --root /root/data --model /root/data/m4/model/model.pt \
      --aorta-dir /root/data/m4/aorta --out /root/data/m4/dev_units.json

Writes one record per (method, scan) unit with: critical label (frozen v3 taxonomy), M4 probability,
and M3 ostium features. Metrics are computed by `summarise` (normalised recall@20%, AUROC).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "pipeline"))

DEV_METHODS = ["cas_net", "swin_unetr"]
BUDGET = 0.20


def normalised_recall(score, critical, budget=BUDGET):
    """Actual recall of critical units at the budget, divided by oracle recall (both at the same budget)."""
    score = np.asarray(score, float)
    y = np.asarray(critical, int)
    n = len(y)
    k = int(np.ceil(budget * n))
    n_pos = int(y.sum())
    if n_pos == 0:
        return {"recall": float("nan"), "oracle": float("nan"), "normalised": float("nan"), "k": k, "n": n}
    top = np.argsort(-score, kind="stable")[:k]
    recall = float(y[top].sum()) / n_pos
    oracle = min(k, n_pos) / n_pos
    return {"recall": recall, "oracle": oracle, "normalised": recall / oracle, "k": k, "n": n, "n_pos": n_pos}


def auroc(score, critical):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(critical, int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, np.asarray(score, float)))


def load_model(path, dev):
    import torch
    from .net import PlausibilityNet
    net = PlausibilityNet().to(dev)
    net.load_state_dict(torch.load(path, map_location=dev))
    net.eval()
    return net


def m4_score(net, mask, sp, dev):
    import torch
    from .build import to_grid
    x = torch.from_numpy(to_grid(mask, sp)).float()[None, None].to(dev)
    with torch.no_grad():
        logit, _ = net(x)
    return float(torch.sigmoid(logit).item())


def _unit_job(args):
    """CPU part of one dev unit: labels (provided centerlines, as in the QA stage), M3 features, 96^3 grid.
    Idempotent: returns immediately when the per-unit JSON already exists."""
    import numpy as np
    from pathlib import Path as _P
    from pipeline.label_core import _gt_seg, load_centerlines
    from coronarysafe.io import load_nifti
    from coronarysafe.labels.taxonomy import label_failures, load_taxonomy_config
    from coronarysafe.features.ostium import ostium_features
    from coronarysafe.m4.lr_convention import lr_from_affine
    from coronarysafe.m4.build import to_grid

    method, scan, root, aorta_dir, left_unit, outdir = args
    out_json = _P(outdir) / f"{method}__{scan}.json"
    out_npz = _P(outdir) / f"{method}__{scan}.npz"
    if out_json.exists() and out_npz.exists():
        return method, scan, "cached"
    root = _P(root)
    mpath = root / "preds" / method / scan / "mask.nii.gz"
    apath = _P(aorta_dir) / f"{scan}.nii.gz"
    if not mpath.exists() or not apath.exists():
        out_json.write_text(json.dumps({"method": method, "scan": scan, "status": "missing_input"}))
        return method, scan, "missing_input"
    gt, sp, aff = load_nifti(_gt_seg(root, scan))
    gt = np.asarray(gt)
    marr, _, paff = load_nifti(mpath)
    mask = np.asarray(marr) > 0
    aorta = np.asarray(load_nifti(apath)[0]) > 0
    lr_axis, lr_sign = lr_from_affine(paff, left_unit)  # orientation from the prediction affine, no GT
    polys, _ = load_centerlines(root, scan, gt > 0, aff)
    cfg = load_taxonomy_config(str(REPO / "configs" / "taxonomy_v3.yaml"))
    lab = label_failures(gt, polys, mask, sp, cfg)
    ost = ostium_features(mask, aorta, sp, lr_axis=lr_axis, lr_sign=lr_sign)
    grid = to_grid(mask, sp)
    np.savez_compressed(out_npz, X=grid)
    rec = {
        "method": method, "scan": scan, "status": "ok",
        "critical": int(bool(lab["critical"])), "burden": float(lab["burden"]),
        "m3": {k: v for k, v in ost.items() if k != "attached_dist_mm"},
        "lr_axis": int(lr_axis), "lr_sign": int(lr_sign),
    }
    tmp = out_json.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, default=float))
    tmp.rename(out_json)
    return method, scan, "ok"


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/data")
    ap.add_argument("--model", required=True)
    ap.add_argument("--aorta-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--unit-dir", required=True)
    ap.add_argument("--lr-json", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    from pipeline.label_core import read_split

    left_unit = json.loads(Path(a.lr_json).read_text())["summary"]["left_world_unit"]
    scans = read_split(a.root, "val")  # VAL only
    Path(a.unit_dir).mkdir(parents=True, exist_ok=True)
    jobs = [(m, s, a.root, a.aorta_dir, left_unit, a.unit_dir) for m in DEV_METHODS for s in scans]
    n = len(jobs)
    done = 0
    with mp.get_context("fork").Pool(a.workers) as pool:
        for method, scan, status in pool.imap_unordered(_unit_job, jobs):
            done += 1
            print(f"[{done}/{n}] {method} {scan} {status}", flush=True)

    # GPU pass: M4 probability on each unit's grid (val only)
    import torch
    dev = torch.device("cuda")
    net = load_model(a.model, dev)
    units = []
    for method, scan, *_ in jobs:
        jp = Path(a.unit_dir) / f"{method}__{scan}.json"
        rec = json.loads(jp.read_text())
        if rec.get("status") != "ok":
            units.append(rec)
            continue
        X = np.load(Path(a.unit_dir) / f"{method}__{scan}.npz")["X"]
        x = torch.from_numpy(X).float()[None, None].to(dev)
        with torch.no_grad():
            logit, _ = net(x)
        rec["m4_prob"] = float(torch.sigmoid(logit).item())
        units.append(rec)
    Path(a.out).write_text(json.dumps(units, indent=1, default=float))
    print("units", len(units), "ok", sum(u["status"] == "ok" for u in units), flush=True)


def summarise(units):
    """Per-method and pooled normalised recall@20% and AUROC for the M4 score."""
    res = {}
    for label, sel in [("pooled", lambda u: True)] + [(m, (lambda m: (lambda u: u["method"] == m))(m)) for m in DEV_METHODS]:
        ok = [u for u in units if u.get("status") == "ok" and sel(u)]
        if not ok:
            continue
        y = [u["critical"] for u in ok]
        res[label] = {
            "n_units": len(ok),
            "m4": {**normalised_recall([u["m4_prob"] for u in ok], y), "auroc": auroc([u["m4_prob"] for u in ok], y)},
        }
    return res


if __name__ == "__main__":
    main()
