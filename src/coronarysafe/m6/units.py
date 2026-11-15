"""M6 feature extraction per unit (label-free). Splits: val (dev) and test (features only; never evaluated here).

  python -m coronarysafe.m6.units --root /root/data --split val --methods cas_net swin_unetr --out /root/data/m6/units --workers 6
Reads only: preds/<method>/<id>/mask.nii.gz, CT volumes, the aorta mask, and the frozen lr convention.
Never reads FFR-UNet predictions or any label file.
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


def _job(args):
    import glob
    import nibabel as nib
    from coronarysafe.io import load_nifti
    from coronarysafe.m4.lr_convention import lr_from_affine
    from coronarysafe.features.proximal import proximal_features

    split, method, sid, root, aorta_dir, left_unit, outdir = args
    out = Path(outdir) / f"{method}__{split}__{sid}.json"
    if out.exists():
        return method, split, sid, "cached"
    root = Path(root)
    mask, sp, paff = load_nifti(root / "preds" / method / sid / "mask.nii.gz")
    mask = np.asarray(mask) > 0
    cts = glob.glob(str(root / "imagecas" / "volumes" / "*" / f"{sid}.img.nii.gz"))
    if not cts:
        return method, split, sid, "missing_ct"
    hu = np.asanyarray(nib.load(cts[0]).dataobj).astype(np.float32)
    if hu.shape != mask.shape:
        return method, split, sid, f"shape_mismatch {hu.shape} {mask.shape}"
    aorta = np.asarray(load_nifti(Path(aorta_dir) / f"{sid}.nii.gz")[0]) > 0
    lr_axis, lr_sign = lr_from_affine(paff, left_unit)
    feats = proximal_features(mask, hu, sp, aorta, lr_axis, lr_sign)
    rec = {"method": method, "scan": sid, "split": split, "status": "ok", **{k: float(v) for k, v in feats.items()}}
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec))
    tmp.rename(out)
    return method, split, sid, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/data")
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--aorta-dir", default="/root/data/m4/aorta")
    ap.add_argument("--lr-json", default="/root/data/m4/lr_convention.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    from pipeline.label_core import read_split

    Path(a.out).mkdir(parents=True, exist_ok=True)
    left = json.loads(Path(a.lr_json).read_text())["summary"]["left_world_unit"]
    scans = read_split(a.root, a.split)
    jobs = [(a.split, m, s, a.root, a.aorta_dir, left, a.out) for m in a.methods for s in scans
            if (Path(a.root) / "preds" / m / s / "mask.nii.gz").exists()]
    n = len(jobs)
    done = 0
    with mp.get_context("fork").Pool(a.workers) as pool:
        for method, split, sid, status in pool.imap_unordered(_job, jobs):
            done += 1
            print(f"[{done}/{n}] {split} {method} {sid} {status}", flush=True)
    print("DONE", n, flush=True)


if __name__ == "__main__":
    main()
