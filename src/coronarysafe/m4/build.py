"""Build M4 training variants from TRAIN ground truth (CPU). Usage (droplet, inside cs-infer):

  python -m coronarysafe.m4.build --root /root/data --scans-file /root/data/m4/train_scans.txt \
      --out /root/data/m4/synth --variants 8 --workers 6

Each variant: corrupted lumen (never shown to labelling), label = frozen taxonomy v3 vs the original GT,
then resampled to a 96^3, 2 mm grid centred on the lumen bounding box. Output: one .npz per scan with
X (V,96,96,96) uint8, critical (V,), burden (V,), clean (V,), ops (V,) as a JSON string list.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "pipeline"))

GRID = 96
VOX_MM = 2.0
CFG_PATH = REPO / "configs" / "taxonomy_v3.yaml"


def to_grid(mask, sp):
    """Resample a binary lumen (native voxels) onto a GRID^3, VOX_MM grid centred on its bbox."""
    sp = np.asarray(sp, dtype=float)
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return np.zeros((GRID,) * 3, np.uint8)
    centre_vox = (coords.min(0) + coords.max(0)) / 2.0
    centre_mm = centre_vox * sp
    ax = (np.arange(GRID) - (GRID - 1) / 2.0) * VOX_MM
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    pts_mm = np.stack([gx, gy, gz], 0) + centre_mm[:, None, None, None]
    vox = pts_mm / sp[:, None, None, None]
    out = ndimage.map_coordinates(mask.astype(np.float32), vox, order=1, mode="constant", cval=0.0)
    return (out > 0.5).astype(np.uint8)


def _one_scan(args):
    scan, root, out_dir, n_var, seed = args
    import nibabel as nib
    from pipeline.label_core import load_centerlines
    from coronarysafe.labels.taxonomy import label_failures, load_taxonomy_config
    from coronarysafe.m4.corrupt import corrupt, sample_plan
    from coronarysafe.io import load_nifti

    out = Path(out_dir) / f"{scan}.npz"
    if out.exists():
        return scan, "skip"
    gt, sp, aff = load_nifti(Path(root) / "imagecas_x/ImageCAS-X_dataset_v3/segmentations" / f"{scan}.coronary.nii.gz")
    gt = np.asarray(gt).astype(np.int32)
    sp = np.asarray(sp, float)
    lumen = gt > 0
    polys, _ = load_centerlines(root, scan, lumen, aff)
    cfg = load_taxonomy_config(CFG_PATH)
    left = cfg["left_labels"]
    right = cfg["right_labels"]
    rng = np.random.default_rng(seed)

    X, crit, burden, clean, ops = [], [], [], [], []
    for v in range(n_var):
        is_clean = v < max(1, n_var // 4)
        plan = sample_plan(rng, clean=is_clean)
        mask, applied = corrupt(gt, sp, rng, plan, left_labels=left, right_labels=right)
        lab = label_failures(gt, polys, mask, sp, cfg)
        X.append(to_grid(mask, sp))
        crit.append(int(bool(lab["critical"])))
        burden.append(float(lab["burden"]))
        clean.append(int(is_clean))
        ops.append(json.dumps(applied))
    np.savez_compressed(out, X=np.stack(X), critical=np.array(crit, np.int8),
                        burden=np.array(burden, np.float32), clean=np.array(clean, np.int8),
                        ops=np.array(ops))
    return scan, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/data")
    ap.add_argument("--scans-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    scans = [s.strip() for s in Path(a.scans_file).read_text().split() if s.strip()]
    jobs = [(s, a.root, a.out, a.variants, a.seed + i) for i, s in enumerate(scans)]
    with Pool(a.workers) as pool:
        for scan, status in pool.imap_unordered(_one_scan, jobs):
            print(scan, status, flush=True)


if __name__ == "__main__":
    main()
