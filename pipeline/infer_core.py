"""Method-agnostic inference core for the CoronarySafe QA study.

Used by infra/modal/infer.py (Modal volume mounted at /data) and infra/droplet/run_infer.py (MI300X
droplet, data under /root/data). Nothing here assumes a host: every path and the device
come from a CorePaths object. Heavy imports (torch, nnunetv2, SimpleITK, the ImageCAS-X
repo) are lazy, so importing this module needs only the standard library.

Reuses ImageCAS-X (kitbransby/ImageCAS-X, MIT, pinned commit below) for network classes,
resampling, HU window, 50%-overlap Gaussian tiling, mirror TTA and postprocessing, so
predictions match the benchmark. nnU-Net runs through nnunetv2's own predictor.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_URL = "https://github.com/kitbransby/ImageCAS-X"
REPO_COMMIT = "dbc7343187adf45ca9306dc3460eebc70f92b204"  # ls-remote HEAD, 2026-11-13

PROB_CROP_THRESHOLD_BYTES = 40 * 1024 * 1024
CROP_PAD_VOXELS = 16

NET_WEIGHTS = {"cas_net": "cas_net.pt", "swin_unetr": "swin_unetr.pt", "ffr_unet": "ffr_unet.pt"}
N_FOLDS = 5
FOLD_SEED = 12345
NNUNET_METHODS = [f"nnunet_f{k}" for k in range(N_FOLDS)]
ALL_METHODS = list(NET_WEIGHTS) + NNUNET_METHODS
OOF_METHOD = "nnunet_oof"  # pseudo-method for run(): expands to each scan's OOF fold
EVAL_METRICS = ["dice", "cl_dice", "betti_error_1", "betti_error_2"]
SPLITS = ("train", "val", "test", "valtest", "trainval", "all_nonexcluded")


@dataclass(frozen=True)
class CorePaths:
    """Where things live on one host. data: the data root (/data on Modal, /root/data on
    the droplet). pred: prediction root. device: 'cuda' (also ROCm) or 'cpu'."""

    data: str
    pred: str
    device: str = "cuda"
    repo_dir: str = "/opt/ImageCAS-X"

    @property
    def dataset_root(self) -> str:
        return f"{self.data}/imagecas_x/ImageCAS-X_dataset_v3"

    @property
    def filelist_dir(self) -> str:
        return f"{self.dataset_root}/filelist"

    @property
    def volumes_root(self) -> str:
        return f"{self.data}/imagecas/volumes"

    @property
    def weights(self) -> str:
        return f"{self.data}/weights/pretrained_weights"

    @property
    def nnunet_model_dir(self) -> str:
        return f"{self.weights}/nnunet/nnUNetTrainer__nnUNetPlans__3d_fullres"

    @property
    def meta_root(self) -> str:
        return f"{self.data}/meta"

    @property
    def fold_table(self) -> str:
        return f"{self.meta_root}/nnunet_folds.csv"

    @property
    def eval_root(self) -> str:
        return f"{self.data}/eval"

    @property
    def runs_dir(self) -> str:
        return f"{self.pred}/_runs"


# ---------------------------------------------------------------------------
# Methods, splits, scan lookup
# ---------------------------------------------------------------------------


def is_variant(m: str) -> bool:
    """`nnunet_f<k>+tta` (mirror TTA, written to preds/nnunet_f<k>_tta) or `<method>@<kind>`
    (CT volumes_shift/<kind>, written to preds/<method>@<kind>)."""
    if m.endswith("+tta"):
        return m[:-4] in NNUNET_METHODS
    if "@" in m:
        base, kind = m.split("@", 1)
        return base in ALL_METHODS and bool(kind)
    return False


def parse_methods(spec: str, allow_oof: bool = False) -> list:
    out = [m.strip() for m in spec.split(",") if m.strip()]
    allowed = ALL_METHODS + ([OOF_METHOD] if allow_oof else [])
    bad = [m for m in out if m not in allowed and not is_variant(m)]
    if bad:
        raise ValueError(f"unknown methods {bad}; choose from {allowed}")
    return out


def read_ids(path: str) -> list:
    with open(path, encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def split_ids(P: CorePaths, split: str) -> list:
    """Scan ids for a split. exclude.txt is removed from every split, as in ImageCAS-X config."""
    fl = Path(P.filelist_dir)
    excl = set(read_ids(str(fl / "exclude.txt"))) if (fl / "exclude.txt").exists() else set()
    train = [i for i in read_ids(str(fl / "train.txt")) if i not in excl]
    val = [i for i in read_ids(str(fl / "val.txt")) if i not in excl]
    test = [i for i in read_ids(str(fl / "test.txt")) if i not in excl]
    table = {
        "train": train,
        "val": val,
        "test": test,
        "valtest": val + test,
        "trainval": train + val,
        "all_nonexcluded": train + val + test,
    }
    if split not in table:
        raise ValueError(f"unknown split {split!r}; choose from {list(table)}")
    return table[split]


_CT_INDEX: dict = {}


def find_ct(P: CorePaths, scan_id: str) -> str:
    """Locate <scan_id>.img.nii.gz anywhere under the staged volume folder."""
    root = os.environ.get("CS_VOLUMES_ROOT") or P.volumes_root
    if root not in _CT_INDEX:
        index = {}
        pattern = f"{glob.escape(root)}/**/*.img.nii.gz"
        for path in glob.glob(pattern, recursive=True):
            name = os.path.basename(path)[: -len(".img.nii.gz")]
            index.setdefault(name, []).append(path)
        _CT_INDEX[root] = index
    hits = _CT_INDEX[root].get(str(scan_id), [])
    if not hits:
        raise FileNotFoundError(
            f"no CT volume for scan {scan_id} under {root} (not staged yet?)"
        )
    if len(hits) > 1:
        raise RuntimeError(f"scan {scan_id} matches several volumes: {hits}")
    return hits[0]


def img_spacing(ct_path: str):
    import nibabel as nib

    return [float(s) for s in nib.load(ct_path).header.get_zooms()[:3]]


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cfg(P: CorePaths, method: str):
    """ImageCAS-X config for a method. The data root is the dataset folder, so filelists
    and GT resolve relative to it. Volumes are located by glob instead."""
    os.environ["ImageCAS_X_data_path"] = P.dataset_root  # DATA_PATH_ENV in utils/config.py
    from utils.config import BenchmarkConfig

    cfg = BenchmarkConfig.from_json(f"{P.repo_dir}/configs/{method}.json")
    cfg.validate()
    cfg.data.volume_dir = P.volumes_root
    return cfg


# ---------------------------------------------------------------------------
# Model bundles (loaded once per process and device)
# ---------------------------------------------------------------------------


def build_net(P: CorePaths, method: str, device: str) -> dict:
    import torch
    from models.registry import build_model
    from postprocessing.pipeline import build_postprocessing
    from preprocessing.pipeline import build_preprocessing

    cfg = load_cfg(P, method)
    ckpt = f"{P.weights}/{NET_WEIGHTS[method]}"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)
    cfg.model.checkpoint = ckpt
    model = build_model(cfg).to(torch.device(device)).eval()
    return {
        "cfg": cfg,
        "model": model,
        "pre": build_preprocessing(cfg),
        "post": build_postprocessing(cfg),
        "weights": ckpt,
        "weights_sha256": sha256(ckpt),
    }


_NET_CACHE: dict = {}


def net_bundle(P: CorePaths, method: str) -> dict:
    key = (P.data, P.device, method)
    if key not in _NET_CACHE:
        _NET_CACHE[key] = build_net(P, method, P.device)
    return _NET_CACHE[key]


def build_nnunet_predictor(P: CorePaths, fold: int, device: str):
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=(device == "cuda"),
        device=torch.device(device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        P.nnunet_model_dir, use_folds=(fold,), checkpoint_name="checkpoint_best.pth"
    )
    return predictor


_NNUNET_CACHE: dict = {}


def nnunet_bundle(P: CorePaths, fold: int) -> dict:
    key = (P.data, P.device, fold)
    if key not in _NNUNET_CACHE:
        ckpt = f"{P.nnunet_model_dir}/fold_{fold}/checkpoint_best.pth"
        _NNUNET_CACHE[key] = {
            "predictor": build_nnunet_predictor(P, fold, P.device),
            "weights": ckpt,
            "weights_sha256": sha256(ckpt),
        }
    return _NNUNET_CACHE[key]


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def flip_name(axes: tuple) -> str:
    return {(): "none", (2,): "flipX", (3,): "flipY", (2, 3): "flipXY"}[tuple(axes)]


def mask_bbox(mask, pad: int = CROP_PAD_VOXELS):
    import numpy as np

    idx = np.argwhere(mask)
    if idx.size == 0:
        return None
    lo = np.maximum(idx.min(axis=0) - pad, 0)
    hi = np.minimum(idx.max(axis=0) + 1 + pad, mask.shape)
    return [[int(a), int(b)] for a, b in zip(lo, hi)]


def save_prob_nifti(arr, affine, path: str) -> None:
    """NIfTI-1 has no float16 type, so probabilities are rounded to float16 precision and
    stored as float32."""
    import nibabel as nib
    import numpy as np

    rounded = np.asarray(arr).astype(np.float16).astype(np.float32)
    nib.save(nib.Nifti1Image(rounded, affine), path)


def write_case(out_dir: Path, affine, mask, fg, flips: dict, meta: dict) -> dict:
    """Write mask, prob (optionally cropped), flips and meta. Arrays are (X, Y, Z)."""
    import nibabel as nib
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), str(out_dir / "mask.nii.gz"))

    prob_path = out_dir / "prob.nii.gz"
    bbox = None
    if fg is not None:
        save_prob_nifti(fg, affine, str(prob_path))
        if prob_path.stat().st_size > PROB_CROP_THRESHOLD_BYTES:
            bbox = mask_bbox(mask)
            if bbox is not None:
                (x0, x1), (y0, y1), (z0, z1) = bbox
                crop_affine = affine.copy()
                crop_affine[:3, 3] = affine[:3, :3] @ np.array([x0, y0, z0]) + affine[:3, 3]
                save_prob_nifti(fg[x0:x1, y0:y1, z0:z1], crop_affine, str(prob_path))
        meta["prob_bytes"] = prob_path.stat().st_size
    elif prob_path.exists():
        prob_path.unlink()

    flips_dir = out_dir / "flips"
    if flips_dir.exists():
        shutil.rmtree(flips_dir)
    if flips:
        flips_dir.mkdir()
        for name, m in flips.items():
            nib.save(nib.Nifti1Image(m.astype(np.uint8), affine), str(flips_dir / f"{name}.nii.gz"))

    meta["bbox_voxel_half_open"] = bbox
    meta["prob_cropped"] = bbox is not None
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def gpu_name(device: str) -> str:
    if fake_mode():
        return "fake"
    import torch

    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return device


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def fake_mode() -> bool:
    """CORONARYSAFE_FAKE_MODEL=1: no torch, no weights. Writes a deterministic dummy mask so the
    queue, idempotency and failure isolation can be tested on a laptop. Never set on the droplet."""
    return os.environ.get("CORONARYSAFE_FAKE_MODEL") == "1"


def fake_predict(P: CorePaths, method: str, scan_id: str, ct_path: str, tta: bool) -> dict:
    import nibabel as nib
    import numpy as np

    img = nib.load(ct_path)
    vol = np.asanyarray(img.dataobj, dtype=np.float32)  # (X, Y, Z), same order as the masks
    mask = vol > float(np.percentile(vol, 75))
    fg = (mask.astype(np.float32) * 0.8 + 0.1).astype(np.float32)
    flips = {n: mask.copy() for n in ("none", "flipX", "flipY", "flipXY")} if tta else {}
    return {
        "mask": mask,
        "fg": fg,
        "prob_def": "fake",
        "flips": flips,
        "runtime_s": 0.0,
        "weights": "fake",
        "weights_sha256": "fake",
        "spacing_processed": [float(s) for s in img.header.get_zooms()[:3]],
        "prob_channel": 0,
        "nnunet_fold": nnunet_fold_of(P, scan_id) if method in NNUNET_METHODS else None,
    }


def predict_network(P: CorePaths, method: str, scan_id: str, ref_img, tta: bool) -> dict:
    import numpy as np
    import torch
    from inference import (
        _forward_full_volume,
        _mirror_axis_combos,
        _resample_probs_to_original_space,
    )

    b = net_bundle(P, method)
    cfg, model = b["cfg"], b["model"]
    device = torch.device(P.device)

    sample = b["pre"]({"sitk_img": ref_img, "sitk_mask": None, "scan_id": scan_id})
    vol = torch.from_numpy(np.asarray(sample["volume"], dtype=np.float32))[None, None]
    spacing = tuple(float(s) for s in sample["spacing"])

    # Same mirror combinations as ImageCAS-X inference (identity, X, Y, XY), averaged on logits.
    combos = _mirror_axis_combos() if tta else [()]
    acc, flips = None, {}
    _sync(P)
    t0 = time.perf_counter()
    for axes in combos:
        x = torch.flip(vol, dims=axes) if axes else vol
        lg = _forward_full_volume(model, x, cfg, device)
        if axes:
            lg = torch.flip(lg, dims=axes)
        acc = lg if acc is None else acc + lg
        if tta:
            pf = _resample_probs_to_original_space(torch.sigmoid(lg[0]).numpy(), spacing, ref_img)
            flips[flip_name(axes)] = b["post"]({"pred": pf, "scan_id": scan_id})["mask"]
            del pf
    logits = acc / len(combos)
    # Mask path exactly as the benchmark: sigmoid per channel, then argmax (2-ch) or threshold.
    prob = torch.sigmoid(logits[0]).numpy()
    prob_orig = _resample_probs_to_original_space(prob, spacing, ref_img)
    mask = b["post"]({"pred": prob_orig, "scan_id": scan_id})["mask"]
    # Calibrated foreground probability. For a 2-channel head the argmax decision is
    # l1 > l0, so the matching probability is sigmoid(l1 - l0), not sigmoid(l1).
    if logits.shape[1] == 2:
        fg_raw = torch.sigmoid(logits[0, 1] - logits[0, 0]).numpy()[None]
        prob_def = "sigmoid(l1-l0)"
    else:
        fg_raw = torch.sigmoid(logits[0, :1]).numpy()
        prob_def = "sigmoid(l0)"
    fg = _resample_probs_to_original_space(fg_raw, spacing, ref_img)[0]
    _sync(P)
    runtime = time.perf_counter() - t0

    return {
        "mask": mask,
        "fg": fg,
        "prob_def": prob_def,
        "flips": flips,
        "runtime_s": runtime,
        "weights": b["weights"],
        "weights_sha256": b["weights_sha256"],
        "spacing_processed": spacing,
        "prob_channel": 1 if prob_orig.shape[0] == 2 else 0,
        "nnunet_fold": None,
    }


def predict_nnunet(P: CorePaths, method: str, scan_id: str, ct_path: str, ref_img, tta: bool):
    import numpy as np
    import SimpleITK as sitk

    fold = int(method.rsplit("_f", 1)[1])
    nb = nnunet_bundle(P, fold)
    predictor = nb["predictor"]
    predictor.use_mirroring = bool(tta)  # nnU-Net's built-in mirroring (all allowed axes)

    tmp = Path("/tmp/nnunet_run") / f"{method}_{scan_id}"
    shutil.rmtree(tmp, ignore_errors=True)
    inp, out = tmp / "in", tmp / "out"
    inp.mkdir(parents=True)
    out.mkdir(parents=True)
    in_file = inp / f"{scan_id}_0000.nii.gz"
    os.symlink(ct_path, in_file)

    t0 = time.perf_counter()
    predictor.predict_from_files(
        [[str(in_file)]],
        [str(out / scan_id)],
        save_probabilities=True,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
    )
    runtime = time.perf_counter() - t0

    seg = sitk.ReadImage(str(out / f"{scan_id}.nii.gz"))
    mask = sitk.GetArrayFromImage(seg).transpose(2, 1, 0).astype(np.uint8)
    with np.load(out / f"{scan_id}.npz") as f:
        probs = f["probabilities"]  # (2, Z, Y, X) softmax, original geometry, label order
    fg = probs[1].transpose(2, 1, 0).astype(np.float32)
    shutil.rmtree(tmp, ignore_errors=True)

    return {
        "mask": mask,
        "fg": fg,
        "prob_def": "softmax_channel1",
        "flips": {},  # nnU-Net mirrors internally; per-flip masks are not exposed
        "runtime_s": runtime,
        "weights": nb["weights"],
        "weights_sha256": nb["weights_sha256"],
        "spacing_processed": [float(s) for s in predictor.configuration_manager.spacing],
        "prob_channel": 1,
        "nnunet_fold": nnunet_fold_of(P, scan_id),
    }


def _sync(P: CorePaths) -> None:
    import torch

    if P.device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def consistency(mask, fg) -> dict:
    """Mask and probability must agree: mean fg inside the mask > 0.5 > mean fg in a
    3-voxel shell outside it. Failure is recorded in meta, not raised."""
    import numpy as np
    from scipy.ndimage import binary_dilation

    m = np.asarray(mask).astype(bool)
    if not m.any():
        return {"ok": False, "warning": "empty mask: consistency not evaluable"}
    shell = binary_dilation(m, iterations=3) & ~m
    mean_in = float(fg[m].mean())
    mean_shell = float(fg[shell].mean()) if shell.any() else float("nan")
    ok = bool(mean_in > 0.5 > mean_shell)
    out = {"mean_fg_in_mask": mean_in, "mean_fg_shell3": mean_shell, "ok": ok}
    if not ok:
        out["warning"] = (
            f"consistency failed: mean fg in mask {mean_in:.3f}, in 3-voxel shell {mean_shell:.3f}"
        )
    return out


def predict_impl(
    P: CorePaths,
    method: str,
    scan_id: str,
    save_probs: bool = True,
    tta: bool = False,
    gpu_requested: str = "",
) -> dict:
    """One method on one scan. Idempotent: skipped when meta.json is current and matches tta."""
    import nibabel as nib

    base, kind = method, None
    if method.endswith("+tta"):
        base, tta = method[:-4], True
        method = base + "_tta"
    elif "@" in method:
        base, kind = method.split("@", 1)

    out_dir = Path(P.pred) / method / str(scan_id)
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        # Predictions written before the prob_def/consistency fields are stale: recompute.
        fresh = old.get("consistency") is not None and old.get("prob_def") is not None
        if (
            fresh
            and bool(old.get("tta")) == bool(tta)
            and (old.get("save_probs") or not save_probs)
        ):
            return {"method": method, "scan_id": scan_id, "status": "skipped"}

    os.environ.setdefault("nnUNet_raw", "/tmp/nnunet/raw")
    os.environ.setdefault("nnUNet_preprocessed", "/tmp/nnunet/preprocessed")
    os.environ.setdefault("nnUNet_results", "/tmp/nnunet/results")

    if kind:
        ct_path = f"{P.data}/imagecas/volumes_shift/{kind}/{scan_id}.img.nii.gz"
        if not Path(ct_path).exists():
            raise FileNotFoundError(f"shifted CT missing: {ct_path}")
    else:
        ct_path = find_ct(P, scan_id)
    affine = nib.load(ct_path).affine
    if fake_mode():
        if os.environ.get("CORONARYSAFE_FAKE_FAIL") == str(scan_id):
            raise RuntimeError(f"injected failure for scan {scan_id} (CORONARYSAFE_FAKE_FAIL)")
        res = fake_predict(P, method, str(scan_id), ct_path, tta)
        expected = tuple(int(s) for s in nib.load(ct_path).shape)
    else:
        import SimpleITK as sitk

        ref_img = sitk.ReadImage(ct_path)
        if base in NET_WEIGHTS:
            res = predict_network(P, base, scan_id, ref_img, tta)
        else:
            res = predict_nnunet(P, base, scan_id, ct_path, ref_img, tta)
        expected = tuple(int(s) for s in ref_img.GetSize())
    if res["mask"].shape != expected:
        raise RuntimeError(
            f"{method}/{scan_id}: mask shape {res['mask'].shape} != CT shape {expected}"
        )

    meta = {
        "method": method,
        "scan_id": scan_id,
        "ct_path": ct_path,
        "weights": res["weights"],
        "weights_sha256": res["weights_sha256"],
        "repo": REPO_URL,
        "repo_commit": REPO_COMMIT,
        "runtime_s": round(res["runtime_s"], 3),
        "gpu_requested": gpu_requested,
        "gpu_name": gpu_name(P.device),
        "spacing_original": [float(s) for s in img_spacing(ct_path)],
        "spacing_processed": res["spacing_processed"],
        "shape": list(expected),
        "tta": bool(tta),
        "flips": sorted(res["flips"]),
        "save_probs": bool(save_probs),
        "prob_channel": res["prob_channel"],
        "prob_def": res["prob_def"],
        "nnunet_fold": res["nnunet_fold"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    cons = consistency(res["mask"], res["fg"])
    meta["consistency"] = cons
    if not cons["ok"]:
        meta["warnings"] = [cons.get("warning", "consistency check failed")]
        print(f"[warn] {method}/{scan_id}: {meta['warnings'][0]}")
    write_case(out_dir, affine, res["mask"], res["fg"] if save_probs else None, res["flips"], meta)
    return {"method": method, "scan_id": scan_id, "status": "done", "runtime_s": meta["runtime_s"]}


# ---------------------------------------------------------------------------
# nnU-Net fold reconstruction
# ---------------------------------------------------------------------------


def compute_folds(identifiers) -> dict:
    """Mirror nnunetv2: np.sort the train+val identifiers, then
    KFold(5, shuffle=True, random_state=12345) (nnunetv2.utilities.crossval_split).
    The validation indices of fold k are the cases for which fold k is held out."""
    import numpy as np
    from sklearn.model_selection import KFold

    keys = [str(k) for k in np.sort(np.array(list(identifiers)))]
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    fold_of = {}
    for k, (_, val_idx) in enumerate(kf.split(keys)):
        for i in val_idx:
            fold_of[keys[i]] = k
    return fold_of


def fold_table(P: CorePaths) -> dict:
    """{scan_id: fold} from <meta>/nnunet_folds.csv, else recomputed from the filelists."""
    if os.path.exists(P.fold_table):
        table = {}
        with open(P.fold_table, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["nnunet_fold"] != "":
                    table[row["scan_id"]] = int(row["nnunet_fold"])
        return table
    return compute_folds(split_ids(P, "trainval"))


_FOLD_CACHE: dict = {}


def nnunet_fold_of(P: CorePaths, scan_id: str, table: dict | None = None):
    """nnU-Net fold (0-4) whose validation split holds scan_id, or None for test and unknown ids.

    RECONSTRUCTION, not a verified fact: the nnU-Net splits_final.json is not in the
    weights zip. Verify empirically: out-of-fold Dice (scan in fold k, model k never
    trained on it) should be lower than Dice on scans model k was trained on.
    """
    if table is None:
        if P.fold_table not in _FOLD_CACHE:
            _FOLD_CACHE[P.fold_table] = fold_table(P)
        table = _FOLD_CACHE[P.fold_table]
    return table.get(str(scan_id))


def verify_scans(P: CorePaths) -> list:
    """16 train scans, seed 0, at least 3 per reconstructed fold (3 per fold plus one more)."""
    import random

    table = fold_table(P)
    train = split_ids(P, "train")
    by_fold = {k: sorted(s for s in train if table.get(s) == k) for k in range(N_FOLDS)}
    rng = random.Random(0)
    chosen = []
    for k in range(N_FOLDS):
        chosen += rng.sample(by_fold[k], 3)
    rest = [s for k in range(N_FOLDS) for s in by_fold[k] if s not in chosen]
    chosen.append(rng.choice(rest))
    return sorted(chosen)


def eval_scan(P: CorePaths, method: str, scan_id: str) -> dict:
    """Metrics for one (method, scan) against ImageCAS-X GT, using the repo's metric code."""
    mask_path = Path(P.pred) / method / str(scan_id) / "mask.nii.gz"
    if not mask_path.exists():
        return {"status": "missing_prediction"}
    if fake_mode():  # no ImageCAS-X GT on the laptop; the value is a placeholder, never a result
        return {"status": "ok", "dice": 0.5, "cl_dice": 0.5, "tta": None, "fake": True}
    import numpy as np
    import SimpleITK as sitk
    from evaluate import _load_gt
    from utils.metrics import compute_centerline_metrics
    from utils.metrics import evaluate as compute_metrics

    cfg = load_cfg(P, "cas_net")  # only the data block is used
    pred_img = sitk.ReadImage(str(mask_path))
    pred = sitk.GetArrayFromImage(pred_img).transpose(2, 1, 0).astype(np.uint8)
    gt, gt_pts = _load_gt(cfg, str(scan_id), pred_img)
    spacing = pred_img.GetSpacing()

    row = {"status": "ok"}
    row.update(compute_metrics(pred, gt, EVAL_METRICS, spacing, warn=False))
    try:
        cl = compute_centerline_metrics(pred, pred_img, gt_pts)
        row["centerline_md"] = cl["centerline_md"]
        row["centerline_hd95"] = cl["centerline_hd95"]
    except ValueError as exc:  # empty prediction or GT centerline
        row["status"] = f"centerline_nan: {exc}"
        row["centerline_md"] = float("nan")
        row["centerline_hd95"] = float("nan")
    meta = json.loads((mask_path.parent / "meta.json").read_text())
    row["tta"] = meta.get("tta")
    return row


def verify_summary(scans, oof_of: dict, dice: dict) -> dict:
    """Per scan: does the reconstructed OOF fold give the lowest Dice, and OOF minus the mean
    of the other four folds. dice[scan] is a list of N_FOLDS floats (NaN when missing)."""
    import numpy as np

    n_lowest, diffs, rows = 0, [], []
    for s in scans:
        oof, d = oof_of[s], np.asarray(dice[s], dtype=float)
        finite = np.isfinite(d)
        lowest_is_oof = bool(finite.any()) and int(np.nanargmin(d)) == oof
        others = [d[k] for k in range(N_FOLDS) if k != oof]
        diff = float(d[oof] - np.nanmean(others)) if np.isfinite(d[oof]) else float("nan")
        n_lowest += int(lowest_is_oof)
        diffs.append(diff)
        rows.append({"scan": s, "oof": oof, "dice": [float(x) for x in d],
                     "lowest_is_oof": lowest_is_oof, "oof_minus_other_mean": diff})
    return {
        "n_scans": len(scans),
        "n_lowest_is_oof": n_lowest,
        "mean_oof_minus_mean_other_folds": float(np.nanmean(diffs)),
        "rows": rows,
    }


def print_verify_table(summary: dict) -> None:
    print(
        f"{'scan':>8} {'oof':>4} "
        + " ".join(f"{'f' + str(k):>8}" for k in range(N_FOLDS))
        + f" {'lowest=oof':>11} {'oof-mean(other)':>16}"
    )
    for r in summary["rows"]:
        cells = " ".join(
            f"{r['dice'][k]:>7.4f}{'*' if k == r['oof'] else ' '}" for k in range(N_FOLDS)
        )
        print(
            f"{r['scan']:>8} {r['oof']:>4} {cells} {str(r['lowest_is_oof']):>11} "
            f"{r['oof_minus_other_mean']:>16.4f}"
        )
    print("(* = reconstructed OOF fold for that scan)")
    print(
        f"[verify] lowest Dice is the reconstructed OOF fold for {summary['n_lowest_is_oof']}/"
        f"{summary['n_scans']} scans; mean OOF minus mean other-fold Dice = "
        f"{summary['mean_oof_minus_mean_other_folds']:.4f}"
    )


def mean_sd(values) -> str:
    import numpy as np

    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return "n/a"
    sd = arr.std(ddof=1) if arr.size > 1 else 0.0
    return f"{arr.mean():.4f} +- {sd:.4f} (n={arr.size})"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
