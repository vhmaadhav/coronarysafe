"""Degraded-scan stress set on the test CTs (CPU). Writes <data>/imagecas/volumes_shift/<kind>/<id>.img.nii.gz.

kinds (one fixed severity each):
  noise      Gaussian noise, sigma = 40 HU (low dose)
  zblur      Gaussian blur along z, sigma = 1.5 mm (motion); voxels scaled by the header z spacing
  contrast   lumen contrast reduced by 35% toward myocardium: voxels above 150 HU move
             toward MYO_HU = 100 HU, new = HU - 0.35 * (HU - 100)
Headers, affine and dtype are preserved. Idempotent: an existing output is skipped.
Usage (droplet, CPU only): nice -n 19 python infra/droplet/make_shift.py [DATA_ROOT]
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.infer_core import CorePaths, split_ids  # noqa: E402

MYO_HU = 100.0
LUMEN_HU = 150.0
import os
KINDS = os.environ.get("KINDS", "noise,zblur,contrast").split(",")


def transform(kind: str, data: np.ndarray, zoom_z: float, rng) -> np.ndarray:
    if kind == "noise":
        return data + rng.normal(0.0, 40.0, size=data.shape)
    if kind == "zblur":
        return gaussian_filter1d(data, sigma=1.5 / zoom_z, axis=2, mode="nearest")
    if kind == "contrast":
        lumen = data > LUMEN_HU
        out = data.copy()
        out[lumen] = data[lumen] - 0.35 * (data[lumen] - MYO_HU)
        return out
    raise ValueError(kind)


def main() -> None:
    data_root = sys.argv[1] if len(sys.argv) > 1 else "/root/data"
    P = CorePaths(data=data_root, pred=f"{data_root}/preds")
    test_ids = split_ids(P, "test")
    done = skipped = 0
    for kind in KINDS:
        out_dir = Path(data_root) / "imagecas" / "volumes_shift" / kind
        out_dir.mkdir(parents=True, exist_ok=True)
        for sid in test_ids:
            src = Path(P.volumes_root) / f"{sid}.img.nii.gz"
            dst = out_dir / f"{sid}.img.nii.gz"
            if dst.exists():
                skipped += 1
                continue
            if not src.exists():
                src = next(Path(P.volumes_root).glob(f"*/{sid}.img.nii.gz"))
            img = nib.load(str(src))
            rng = np.random.default_rng(int(sid) if str(sid).isdigit() else abs(hash(sid)) % (2**32))
            arr = np.asarray(img.dataobj, dtype=np.float32)
            out = transform(kind, arr, float(img.header.get_zooms()[2]), rng)
            dtype = img.get_data_dtype()
            if np.issubdtype(dtype, np.integer):
                out = np.clip(np.rint(out), np.iinfo(dtype).min, np.iinfo(dtype).max)
            nib.save(nib.Nifti1Image(out.astype(dtype), img.affine, img.header), str(dst))
            done += 1
        print(f"make_shift: {kind} done={done} skipped={skipped}", flush=True)
    print("make_shift: complete", flush=True)


if __name__ == "__main__":
    main()
