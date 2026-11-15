"""Diagnostic for the mismatch between the original ImageCAS lumen masks and ImageCAS-X (val only).

Addendum I support work (ANALYSIS_PLAN_v1 Addendum I). Measures voxel-level agreement and the size and
location of the mismatch, and writes JSON summaries. It changes no analysis, label or threshold. It was
run on the droplet against '/root/data' (see docs/RUNBOOK.md); paths are droplet paths by default.
"""
import glob, itertools, json, sys
from pathlib import Path
import numpy as np, nibabel as nib
from nibabel.processing import resample_from_to
from scipy import ndimage
from coronarysafe.geometry import bbox_slices, dist_to
from coronarysafe.io import load_nifti
from coronarysafe.labels.taxonomy import label_failures, load_taxonomy_config
from coronarysafe.analysis.label_robust import _load_original, dice, agreement_stats, _ci_block, read_split_ids
from multiprocessing import Pool

ROOT = Path("/root/data")
ICX = ROOT / "imagecas_x/ImageCAS-X_dataset_v3"
OUT = ROOT / "label_robust/diag"
OUT.mkdir(parents=True, exist_ok=True)
CFG = load_taxonomy_config(Path(__file__).resolve().parents[3] / "configs" / "taxonomy_v3.yaml")

def find(patterns):
    for p in patterns:
        hit = sorted(glob.glob(p))
        if hit:
            return Path(hit[0])
    return None

def ginfo(img):
    z = np.array(img.header.get_zooms()[:3], float)
    return {"shape": [int(s) for s in img.shape[:3]], "zooms": [round(float(v), 4) for v in z],
            "origin": [round(float(v), 3) for v in img.affine[:3, 3]],
            "direction": np.round(img.affine[:3, :3] / z[None, :], 3).tolist()}

def same(a, b):
    return tuple(a.shape[:3]) == tuple(b.shape[:3]) and np.allclose(a.affine, b.affine, atol=0.01)

def lab_info(arr, sp):
    v, c = np.unique(arr, return_counts=True)
    ml = float(np.prod(sp)) / 1000.0
    return {int(a): {"vox": int(b), "ml": round(float(b) * ml, 2)} for a, b in zip(v, c)}

def centroid_mm(m, sp):
    return np.argwhere(m).mean(0) * sp

def case(sid):
    rec = {"scan": sid}
    ct_p = find([f"{ROOT}/imagecas/volumes/*/{sid}.nii*", f"{ROOT}/imagecas/volumes/{sid}.nii*"])
    or_p = find([f"{ROOT}/imagecas/orig_labels/*/{sid}.label.nii.gz"])
    x_p = ICX / "segmentations" / f"{sid}.coronary.nii.gz"
    a_p = ROOT / "m4/aorta" / f"{sid}.nii.gz"
    ct = nib.load(str(ct_p)) if ct_p else None
    X = nib.load(str(x_p))
    O = nib.load(str(or_p))
    rec["paths"] = {"ct": str(ct_p), "orig": str(or_p), "x": str(x_p), "aorta_exists": a_p.exists()}
    rec["grid"] = {"CT": ginfo(ct) if ct else None, "X": ginfo(X), "orig_raw": ginfo(O)}
    rec["match"] = {"CT_eq_X": bool(ct is not None and same(ct, X)), "orig_eq_X": same(O, X),
                    "orig_eq_CT": bool(ct is not None and same(O, ct))}
    Xa = np.asanyarray(X.dataobj)
    sp = np.array(X.header.get_zooms()[:3], float)
    Oraw = np.asanyarray(O.dataobj)
    rec["labels"] = {"X": lab_info(Xa, sp), "orig_raw": lab_info(Oraw, np.array(O.header.get_zooms()[:3], float))}
    Oa = _load_original(or_p, Xa.shape, X.affine)
    Xb = Xa > 0
    rec["dice_X_orig"] = dice(Oa, Xb)
    rec["vox_X"] = int(Xb.sum()); rec["vox_orig_on_X"] = int(Oa.sum())
    sl = bbox_slices(Xb | Oa, sp, pad_mm=10.0)
    Oc, Xc = Oa[sl], Xb[sl]
    dO = dist_to(Oc, sp); dX = dist_to(Xc, sp)
    rec["frac_X_within_orig"] = {str(d): float((dO[Xc] <= d + 1e-9).mean()) for d in (0, 1, 2)}
    rec["frac_orig_within_X"] = {str(d): float((dX[Oc] <= d + 1e-9).mean()) for d in (0, 1, 2)}
    rec["centroid_mm"] = {"X": centroid_mm(Xb, sp).round(2).tolist(), "orig": centroid_mm(Oa, sp).round(2).tolist()}
    off = centroid_mm(Oa, sp) - centroid_mm(Xb, sp)
    rec["centroid_offset_mm"] = {"vec": off.round(2).tolist(), "norm": round(float(np.linalg.norm(off)), 2)}
    flips = {}
    for combo in itertools.product([0, 1], repeat=3):
        F = Oa
        for ax, f in enumerate(combo):
            if f:
                F = np.flip(F, axis=ax)
        flips["flip" + "".join("xyz"[i] for i, f in enumerate(combo) if f) if any(combo) else "none"] = dice(F, Xb)
    flips["transpose_xy"] = dice(np.transpose(Oa, (1, 0, 2)), Xb) if Oa.shape[0] == Oa.shape[1] else float("nan")
    rec["dice_variants"] = {k: round(float(v), 4) for k, v in flips.items()}
    rec["best_variant"] = max(rec["dice_variants"], key=lambda k: (rec["dice_variants"][k] if np.isfinite(rec["dice_variants"][k]) else -1))
    if a_p.exists():
        A = nib.load(str(a_p))
        rec["grid"]["aorta"] = ginfo(A)
        Aa = (np.asanyarray(A.dataobj) > 0) if same(A, X) else (resample_from_to(A, (Xa.shape, X.affine), order=0).dataobj > 0)
        Ac = np.asarray(Aa)[sl]
        dA = dist_to(Ac, sp) if Ac.any() else np.full(Ac.shape, np.inf)
        touch = {}
        for name, M, dm in (("X", Xc, None), ("orig", Oc, None)):
            d = dA[M]
            lab, n = ndimage.label(M)
            sizes = np.bincount(lab.ravel())[1:] if n else np.array([0])
            touch[name] = {"min_dist_to_aorta_mm": round(float(d.min()), 2) if d.size else None,
                           "frac_within_1mm_aorta": round(float((d <= 1).mean()), 4) if d.size else None,
                           "touches_aorta_1mm": bool((d <= 1).any()) if d.size else False,
                           "n_components": int(n), "largest_cc_vox": int(sizes.max()),
                           "largest_cc_ml": round(float(sizes.max()) * float(np.prod(sp)) / 1000.0, 2),
                           "largest_cc_frac": round(float(sizes.max() / max(M.sum(), 1)), 4)}
        rec["aorta"] = touch
    else:
        rec["aorta"] = None
    return rec

def conf_unit(task):
    method, sid = task
    rec = {"method": method, "scan_id": sid}
    try:
        qa = json.loads((ROOT / "qa/v3" / method / f"{sid}.json").read_text())
        rec["critical_v3"] = int(bool(qa["critical"])); rec["v3_source"] = qa.get("centerline_source")
        gt, sp, _ = load_nifti(ICX / "segmentations" / f"{sid}.coronary.nii.gz")
        pred = np.asarray(load_nifti(ROOT / "preds" / method / sid / "mask.nii.gz")[0]) > 0
        lab = label_failures(np.asarray(gt), None, pred, sp, CFG)
        rec["critical_skel"] = int(bool(lab["critical"])); rec["skel_source"] = lab["centerline_source"]
        rec["status"] = "ok"
    except Exception as exc:
        rec["status"] = "error"; rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return rec

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "diag":
        cases = [134, 742, 565, 936, 879, 183]
        res = [case(str(c)) for c in cases]
        (OUT / "diag_cases.json").write_text(json.dumps(res, indent=1, default=float))
        for r in res:
            print(json.dumps(r, default=float), flush=True)
    elif mode == "confound":
        ids = read_split_ids(ROOT, "val")
        tasks = [(m, s) for m in ["cas_net", "swin_unetr"] for s in ids]
        with Pool(processes=8) as pool:
            recs = pool.map(conf_unit, tasks, chunksize=1)
        import pandas as pd
        df = pd.DataFrame(recs)
        df.to_csv(OUT / "confound_units.csv", index=False)
        ok = df[df.status == "ok"]
        summ = {"n_ok": int(len(ok)), "n_err": int((df.status != "ok").sum()), "pooled": None, "per_method": {}}
        v, s, c = ok.critical_v3.to_numpy(int), ok.critical_skel.to_numpy(int), ok.scan_id.to_numpy(str)
        summ["pooled"] = {k: v_ for k, v_ in _ci_block(v, s, c).items()}
        for m, d in ok.groupby("method"):
            summ["per_method"][m] = _ci_block(d.critical_v3.to_numpy(int), d.critical_skel.to_numpy(int), d.scan_id.to_numpy(str))
        (OUT / "confound_summary.json").write_text(json.dumps(summ, indent=1, default=float))
        print(json.dumps(summ, default=float)[:3000], flush=True)
        print("CONFOUND_DONE", flush=True)
