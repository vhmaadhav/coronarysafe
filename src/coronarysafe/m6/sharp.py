"""Addendum G sharpening: fragmentation rule recomputed on the proximal subtree (<= 60 mm from the aorta/ostium).
Val only (dev). Label-free. Writes /root/data/m6/sharp/<method>__<id>.json.
  python -m coronarysafe.m6.sharp --root /root/data --methods cas_net swin_unetr --out /root/data/m6/sharp --workers 6
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
PROX_MM = 60.0


def _job(args):
    from scipy import ndimage
    from coronarysafe.io import load_nifti
    from coronarysafe.features.graph_features import compute_graph_features

    method, sid, root, aorta_dir, outdir = args
    out = Path(outdir) / f"{method}__{sid}.json"
    if out.exists():
        return method, sid, "cached"
    root = Path(root)
    mask, sp, _ = load_nifti(root / "preds" / method / sid / "mask.nii.gz")
    mask = np.asarray(mask) > 0
    ao = np.asarray(load_nifti(Path(aorta_dir) / f"{sid}.nii.gz")[0]) > 0
    if ao.any():
        d_ao = ndimage.distance_transform_edt(~ao, sampling=tuple(float(s) for s in sp))
        keep = mask & (d_ao <= PROX_MM)
    else:
        keep = np.zeros_like(mask)
    g = compute_graph_features(keep, sp) if keep.any() else {"n_components_ge_10mm": 0, "n_components": 0}
    rec = {"method": method, "scan": sid, "prox_n_components_ge_10mm": float(g.get("n_components_ge_10mm", 0)),
           "prox_n_components": float(g.get("n_components", 0)), "prox_voxels": int(keep.sum())}
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec))
    tmp.rename(out)
    return method, sid, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/data")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--aorta-dir", default="/root/data/m4/aorta")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    from pipeline.label_core import read_split
    Path(a.out).mkdir(parents=True, exist_ok=True)
    scans = read_split(a.root, "val")
    jobs = [(m, s, a.root, a.aorta_dir, a.out) for m in a.methods for s in scans]
    n = len(jobs)
    done = 0
    with mp.get_context("fork").Pool(a.workers) as pool:
        for method, sid, status in pool.imap_unordered(_job, jobs):
            done += 1
            print(f"[{done}/{n}] {method} {sid} {status}", flush=True)
    print("SHARP_DONE", flush=True)


if __name__ == "__main__":
    main()
