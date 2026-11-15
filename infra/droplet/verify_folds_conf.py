"""E2: confidence-based fold verification on CPU.

For each train scan s (first N in the filelist order, the same ids run_infer --limit N uses) and each
nnU-Net fold model k, compute from the saved probabilities:
  lumen_p  : mean probability inside the GT lumen (lower = worse fit)
  band_bce : BCE in a 3 mm band around the GT surface (higher = worse fit)
A candidate fold mapping gives each scan s a held-out fold f(s); the held-out model nnunet_f{f(s)} should
be the worst of the five models for s. Statistic = fraction of scans where the held-out model is the worst.
Null: permute the held-out labels across scans (1000 permutations). p = P(null >= observed).
Usage (container, CPU): python verify_folds_conf.py N OUT_JSON
"""
import glob
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt
from sklearn.model_selection import KFold

sys.path.insert(0, "/root/coronarysafe")
from pipeline.infer_core import CorePaths, N_FOLDS, compute_folds, split_ids  # noqa: E402

DATA = "/root/data"
P = CorePaths(data=DATA, pred=f"{DATA}/preds")
GT_DIR = Path(P.dataset_root) / "segmentations"
PRED = Path(DATA) / "preds"
BAND_MM = 3.0


def find_ct(sid):
    return glob.glob(f"{DATA}/imagecas/volumes/*/{sid}.img.nii.gz")[0]


def metrics_for(sid):
    ref = sitk.ReadImage(find_ct(sid))
    gt_img = sitk.ReadImage(str(GT_DIR / f"{sid}.coronary.nii.gz"))
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(ref)
    r.SetInterpolator(sitk.sitkNearestNeighbor)
    r.SetDefaultPixelValue(0)
    gt = sitk.GetArrayFromImage(r.Execute(gt_img)).transpose(2, 1, 0) > 0
    spacing = np.array(ref.GetSpacing())  # (sx, sy, sz) in mm
    # distance to the GT surface, in mm: inside the lumen and outside it
    d_in = distance_transform_edt(gt, sampling=spacing[::-1])
    d_out = distance_transform_edt(~gt, sampling=spacing[::-1])
    band = np.where(gt, d_in, d_out) <= BAND_MM
    y = gt[band].astype(np.float32)
    out = {}
    for k in range(N_FOLDS):
        prob = nib.load(str(PRED / f"nnunet_f{k}" / sid / "prob.nii.gz")).get_fdata(dtype=np.float32)
        _m = json.loads((PRED / f"nnunet_f{k}" / sid / "meta.json").read_text())
        _bb = _m.get("bbox_voxel_half_open")
        if _bb is not None and tuple(prob.shape) != tuple(gt.shape):
            full = np.zeros(gt.shape, np.float32)
            (x0, x1), (y0, y1), (z0, z1) = _bb
            full[x0:x1, y0:y1, z0:z1] = prob
            prob = full
        lumen_p = float(prob[gt].mean()) if gt.any() else float("nan")
        p = np.clip(prob[band], 1e-6, 1 - 1e-6)
        bce = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()) if band.any() else float("nan")
        out[k] = {"lumen_p": lumen_p, "band_bce": bce}
    return sid, out


def candidate_maps(trainval):
    cands = {"nnunetv2_default_str_sorted_shuffled": compute_folds(trainval)}
    keys_str = sorted(trainval)
    keys_num = sorted(trainval, key=lambda x: int(x))
    for name, keys in (("str", keys_str), ("num", keys_num)):
        for shuffle in (True, False):
            kf = KFold(n_splits=N_FOLDS, shuffle=shuffle, random_state=12345 if shuffle else None)
            m = {}
            for k, (_, v) in enumerate(kf.split(keys)):
                for i in v:
                    m[keys[i]] = k
            cands[f"{name}_sorted_{'shuffled' if shuffle else 'chunks'}"] = m
    return cands


def main():
    n = int(sys.argv[1])
    out_path = sys.argv[2]
    train_ids = split_ids(P, "train")[:n]
    with Pool(4) as pool:
        rows = dict(pool.map(metrics_for, train_ids))
    lumen = np.array([[rows[s][k]["lumen_p"] for k in range(N_FOLDS)] for s in train_ids])
    bce = np.array([[rows[s][k]["band_bce"] for k in range(N_FOLDS)] for s in train_ids])
    worst_lumen = lumen.argmin(axis=1)  # lowest mean prob = worst fit
    worst_bce = bce.argmax(axis=1)      # highest BCE = worst fit
    rng = np.random.default_rng(0)
    result = {"n_scans": len(train_ids), "chance_worst_of_5": 0.2, "candidates": {}}
    for name, fmap in candidate_maps(split_ids(P, "trainval")).items():
        held = np.array([fmap.get(s, -1) for s in train_ids])
        ok = held >= 0
        h = held[ok]
        hit_l = float((worst_lumen[ok] == h).mean())
        hit_b = float((worst_bce[ok] == h).mean())
        null_l, null_b = [], []
        for _ in range(1000):
            perm = rng.permutation(h)
            null_l.append(float((worst_lumen[ok] == perm).mean()))
            null_b.append(float((worst_bce[ok] == perm).mean()))
        p_l = float((np.array(null_l) >= hit_l).mean())
        p_b = float((np.array(null_b) >= hit_b).mean())
        result["candidates"][name] = {
            "n_mapped": int(ok.sum()),
            "heldout_worst_lumen": hit_l, "p_perm_lumen": p_l,
            "heldout_worst_bce": hit_b, "p_perm_bce": p_b,
            "verified": bool(max(hit_l, hit_b) >= 0.9 and min(p_l, p_b) < 0.001),
        }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
