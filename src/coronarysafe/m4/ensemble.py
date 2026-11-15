"""S3 ensemble comparators on TEST (Addendum G). Label-free per scan; GT is never read here.

For each test scan: majority-vote mask (>=3 of nnunet_f0..f4) written to preds/nnunet_mv/<id>/mask.nii.gz
(+ meta.json), and mean pairwise agreement over the 10 fold pairs:
  dice       pairwise Dice
  cldice     pairwise clDice (skeleton-based topology overlap, symmetric harmonic mean)
  b0err      pairwise |b0_i - b0_j|, b0 = number of 26-connected components
Output: <out>/ensemble_comparators.csv.

  python -m coronarysafe.m4.ensemble --root /root/data --out /root/data/test_eval --workers 6
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
FOLDS = [f"nnunet_f{i}" for i in range(5)]


def _cl_pair(sa, ma, sb, mb):
    """clDice between skeleton sa of mask ma and skeleton sb of mask mb, symmetric harmonic mean."""
    from scipy import ndimage

    tprec = (sa & mb_dil(mb)).sum() / max(sa.sum(), 1)
    tsens = (sb & mb_dil(ma)).sum() / max(sb.sum(), 1)
    if tprec + tsens == 0:
        return 0.0
    return float(2 * tprec * tsens / (tprec + tsens))


def mb_dil(m):
    from scipy import ndimage
    return ndimage.binary_dilation(m)


def _job(args):
    import nibabel as nib
    from scipy import ndimage
    from coronarysafe.graph import skeletonize

    sid, root, out = args
    root = Path(root)
    masks, img0 = [], None
    for f in FOLDS:
        img = nib.load(str(root / "preds" / f / sid / "mask.nii.gz"))
        if img0 is None:
            img0 = img
        masks.append(np.asanyarray(img.dataobj) > 0)
    vote = np.sum(masks, axis=0) >= 3
    mvdir = root / "preds" / "nnunet_mv" / sid
    mvdir.mkdir(parents=True, exist_ok=True)
    hdr = img0.header.copy()
    hdr.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(vote.astype(np.uint8), img0.affine, hdr), str(mvdir / "mask.nii.gz"))
    (mvdir / "meta.json").write_text(json.dumps({
        "method": "nnunet_mv", "rule": "majority >=3 of nnunet_f0..f4", "prob_cropped": False}))

    union = np.any(masks, axis=0)
    sl = ndimage.find_objects(union.astype(np.uint8))[0]
    crop = [m[sl] for m in masks]
    sk = [skeletonize(c) for c in crop]
    dil = [ndimage.binary_dilation(c) for c in crop]
    b0 = [ndimage.label(c, structure=np.ones((3, 3, 3)))[1] for c in crop]
    dice, cl, b0e = [], [], []
    for i, j in itertools.combinations(range(5), 2):
        a, b = crop[i], crop[j]
        inter = (a & b).sum()
        denom = a.sum() + b.sum()
        dice.append(float(2 * inter / denom) if denom else 1.0)
        tprec = (sk[i] & dil[j]).sum() / max(sk[i].sum(), 1)
        tsens = (sk[j] & dil[i]).sum() / max(sk[j].sum(), 1)
        cl.append(float(2 * tprec * tsens / (tprec + tsens)) if (tprec + tsens) > 0 else 0.0)
        b0e.append(float(abs(b0[i] - b0[j])))
    return {"scan_id": sid, "mean_pair_dice": float(np.mean(dice)), "mean_pair_cldice": float(np.mean(cl)),
            "mean_pair_b0err": float(np.mean(b0e)), "mv_voxels": int(vote.sum())}


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    from pipeline.label_core import read_split
    scans = read_split(a.root, "test")
    jobs = [(s, a.root, a.out) for s in scans]
    rows = []
    with mp.get_context("fork").Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_job, jobs), 1):
            rows.append(r)
            print(f"[{k}/{len(jobs)}] {r['scan_id']} dice={r['mean_pair_dice']:.3f}", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("scan_id").to_csv(Path(a.out) / "ensemble_comparators.csv", index=False)
    print("ENSEMBLE_DONE", flush=True)


if __name__ == "__main__":
    main()
