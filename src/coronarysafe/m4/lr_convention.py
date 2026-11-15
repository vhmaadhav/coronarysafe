"""Patient-left direction for the ostium features, fitted on TRAIN ground truth only.

Left tree = LM/LAD/LCx (labels 1,2,3); right tree = RCA (label 9). For each TRAIN scan the world-space
vector from the RCA centroid to the left-tree centroid (affine applied, mm) gives a per-scan direction.
The mean unit vector over train scans is frozen as the patient-left direction (world space).

For any scan (val/test included, no GT used), the left direction is mapped into that scan's voxel axes
through the inverse affine: lr_axis = argmax |component|, lr_sign = sign of that component. Both are
written into the ostium features by the scorer, so orientation is per-scan and derived from its affine.

  fit:    python -m coronarysafe.m4.lr_convention fit --gt-dir ... --scans /root/data/m4/train_scans.txt --out ...
  apply:  lr_from_affine(affine, json_path) -> (axis, sign)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LEFT_LABELS = (1, 2, 3)
RIGHT_LABELS = (9,)


def _world_vec(affine, mask_l, mask_r):
    lv = np.argwhere(mask_l).mean(0)
    rv = np.argwhere(mask_r).mean(0)
    A = affine
    return (A[:3, :3] @ lv + A[:3, 3]) - (A[:3, :3] @ rv + A[:3, 3])


def fit(gt_dir, scans, out_json):
    import nibabel as nib

    unit_vecs, per_scan = [], {}
    for sid in scans:
        img = nib.load(str(Path(gt_dir) / "segmentations" / f"{sid}.coronary.nii.gz"))
        g = np.asanyarray(img.dataobj)
        mL = np.isin(g, LEFT_LABELS)
        mR = np.isin(g, RIGHT_LABELS)
        if not mL.any() or not mR.any():
            per_scan[sid] = None
            continue
        w = _world_vec(img.affine, mL, mR)
        n = np.linalg.norm(w)
        if n < 1e-6:
            per_scan[sid] = None
            continue
        u = w / n
        unit_vecs.append(u)
        per_scan[sid] = u.tolist()
    U = np.array(unit_vecs)
    mean = U.mean(0)
    consensus = float(np.linalg.norm(mean))  # 1.0 when all train scans agree
    left_u = mean / np.linalg.norm(mean)
    summary = {
        "n_scans": len(unit_vecs),
        "left_world_unit": left_u.tolist(),
        "mean_resultant_length": consensus,
        "min_cosine_to_consensus": float((U @ left_u).min()),
        "frac_cosine_gt_0_5": float(((U @ left_u) > 0.5).mean()),
    }
    Path(out_json).write_text(json.dumps({"summary": summary, "per_scan_world_unit": per_scan}, indent=1))
    return summary


def lr_from_affine(affine, left_world_unit):
    """Map the frozen patient-left world direction into this scan's voxel axes (no GT)."""
    inv = np.linalg.inv(np.asarray(affine, float)[:3, :3])
    dv = inv @ np.asarray(left_world_unit, float)
    axis = int(np.argmax(np.abs(dv)))
    sign = 1 if dv[axis] > 0 else -1
    return axis, sign


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--gt-dir", required=True)
    f.add_argument("--scans", required=True)
    f.add_argument("--out", required=True)
    a = ap.parse_args()
    scans = [s.strip() for s in Path(a.scans).read_text().split() if s.strip()]
    print(json.dumps(fit(a.gt_dir, scans, a.out), indent=1))


if __name__ == "__main__":
    main()
