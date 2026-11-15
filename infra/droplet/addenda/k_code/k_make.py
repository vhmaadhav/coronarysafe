"""Addendum K: regenerate perturbed test CTs per spec (CPU, 8 workers). Usage: python -I k_make.py noise contrast blur
Output: /root/data/imagecas/volumes_shift_K/<kind>/<id>.img.nii.gz (existing files are skipped).
Per-scan parameters are written to /root/data/addenda/K/gen_params_<kind>.json."""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, "/root/data/addenda/K/code")
from k_common import DATA, aorta_mask, central_roi, slice_sd, src_path  # noqa: E402

OUT = f"{DATA}/imagecas/volumes_shift_K"


def job(args):
    kind, sid = args
    dst = f"{OUT}/{kind}/{sid}.img.nii.gz"
    if os.path.exists(dst):
        return {"scan": sid, "status": "skipped"}
    img = nib.load(src_path(sid))
    arr = np.asarray(img.dataobj, dtype=np.float32)
    mask = aorta_mask(sid, arr.shape)
    roi, zs = central_roi(mask)
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=float)
    rec = {"scan": sid, "status": "ok"}
    if kind == "noise":
        sigma0 = float(np.median(slice_sd(arr, roi, zs)))
        sig_add = 1.118 * sigma0
        rng = np.random.default_rng(int(sid))
        out = arr + rng.normal(0.0, sig_add, size=arr.shape)
        rec.update(sigma0=sigma0, sigma_add=sig_add)
    elif kind == "contrast":
        m = float(np.median(arr[mask]))
        out = m + 0.75 * (arr - m)
        rec.update(m_lumen=m)
    elif kind == "blur":
        sig = 0.6 / zooms
        out = ndi.gaussian_filter(arr, sigma=sig, mode="nearest")
        rec.update(sigma_vox=[float(s) for s in sig])
    else:
        raise ValueError(kind)
    dtype = img.get_data_dtype()
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part.nii.gz"
    nib.save(nib.Nifti1Image(out.astype(dtype), img.affine, img.header), tmp)
    os.replace(tmp, dst)
    return rec


def main():
    kinds = sys.argv[1:]
    test = sorted(int(x) for x in open(f"{DATA}/imagecas_x/ImageCAS-X_dataset_v3/filelist/test.txt").read().split())
    for kind in kinds:
        with ProcessPoolExecutor(max_workers=8) as ex:
            recs = list(ex.map(job, [(kind, str(s)) for s in test]))
        n_ok = sum(r["status"] == "ok" for r in recs)
        n_skip = sum(r["status"] == "skipped" for r in recs)
        with open(f"{DATA}/addenda/K/gen_params_{kind}.json", "w") as fh:
            json.dump(recs, fh, indent=1)
        print(f"k_make: {kind} written={n_ok} skipped={n_skip}", flush=True)
    print("k_make: complete", flush=True)


if __name__ == "__main__":
    main()
