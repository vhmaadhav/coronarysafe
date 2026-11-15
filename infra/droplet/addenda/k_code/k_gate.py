"""Addendum K spec-check gate (CPU). Usage: python -I k_gate.py <pre_regen|post_regen> <noise_dir> <contrast_dir> <blur_dir>
Writes/merges /root/data/addenda/K/spec_gate.json under the given key."""
import json
import os
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, "/root/data/addenda/K/code")
from k_common import DATA, aorta_mask, central_roi, slice_sd, src_path  # noqa: E402

OUT = f"{DATA}/addenda/K/spec_gate.json"
KEY, DIRS = sys.argv[1], {"noise": sys.argv[2], "contrast": sys.argv[3], "blur": sys.argv[4]}
N_GATE = 8


def load(path):
    img = nib.load(path)
    return img, np.asarray(img.dataobj, dtype=np.float32)


def iqr(v):
    return float(np.percentile(v, 75) - np.percentile(v, 25))


def gate_scan(sid, kind, pdir):
    simg, src = load(src_path(sid))
    pimg, prt = load(f"{pdir}/{sid}.img.nii.gz")
    rec = {"scan": sid, "shape_ok": bool(src.shape == prt.shape),
           "affine_ok": bool(np.allclose(simg.affine, pimg.affine, atol=1e-4)),
           "dtype_src": str(simg.get_data_dtype()), "dtype_pert": str(pimg.get_data_dtype())}
    mask = aorta_mask(sid, src.shape)
    roi, zs = central_roi(mask)
    sd_s, sd_p = slice_sd(src, roi, zs), slice_sd(prt, roi, zs)
    rec["roi_slices"] = len(zs)
    rec["roi_voxels"] = int(roi.sum())
    rec["sigma0_src_sd"] = float(np.median(sd_s))
    rec["noise_sd_ratio"] = float(np.median(sd_p) / np.median(sd_s))
    rec["roi_median_src"] = float(np.median(src[roi]))
    rec["roi_median_pert"] = float(np.median(prt[roi]))
    rec["median_diff_HU"] = rec["roi_median_pert"] - rec["roi_median_src"]
    rec["iqr_ratio"] = iqr(prt[roi]) / iqr(src[roi])
    # isotropic 0.6 mm reference: blur the source with sigma_vox = 0.6 / spacing per axis
    zooms = np.asarray(simg.header.get_zooms()[:3], dtype=float)
    ref = ndi.gaussian_filter(src, sigma=0.6 / zooms, mode="nearest")
    rec["blur_sigma_vox"] = [round(float(s), 3) for s in 0.6 / zooms]
    rec["mad_pert_vs_iso06_HU"] = float(np.mean(np.abs(prt[roi] - ref[roi])))
    rec["mad_src_vs_pert_HU"] = float(np.mean(np.abs(prt[roi] - src[roi])))
    del src, prt, ref
    return rec


def decide(kind, rows):
    ok_geo = all(r["shape_ok"] and r["affine_ok"] for r in rows)
    if kind == "noise":
        good = all(1.40 <= r["noise_sd_ratio"] <= 1.60 for r in rows)
        crit = "noise_sd_ratio in [1.40,1.60] per scan"
    elif kind == "blur":
        good = all(r["mad_pert_vs_iso06_HU"] <= 1.0 for r in rows)
        crit = "mean |pert - isotropic 0.6 mm reference| <= 1.0 HU per scan"
    else:
        good = all(abs(r["median_diff_HU"]) <= 5.0 and abs(r["iqr_ratio"] - 0.75) <= 0.05 for r in rows)
        crit = "|median diff| <= 5 HU and IQR ratio in [0.70,0.80] per scan"
    return {"pass": bool(ok_geo and good), "geometry_ok": ok_geo, "criterion": crit}


def main():
    results = {}
    for kind, pdir in DIRS.items():
        rows = [gate_scan(str(s), kind, pdir) for s in GATE_IDS]
        results[kind] = {"dir": pdir, "decision": decide(kind, rows), "scans": rows}
        print(kind, results[kind]["decision"], flush=True)
    doc = json.load(open(OUT)) if os.path.exists(OUT) else {}
    doc[KEY] = results
    doc["gate_scans"] = [str(s) for s in GATE_IDS]
    doc["roi_definition"] = ("aorta = m4/aorta TotalSegmentator mask; per axial slice the central 50% "
                             "(middle half of x and y of the cross-section bbox), slices >= 20 voxels")
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, OUT)
    print("wrote", OUT, KEY, flush=True)


if __name__ == "__main__":
    test = sorted(int(x) for x in open(f"{DATA}/imagecas_x/ImageCAS-X_dataset_v3/filelist/test.txt").read().split())
    GATE_IDS = test[:N_GATE]
    main()
