"""Build compact browser meshes for the CoronarySafe showcase (runs on the droplet, CPU).

For each case: prediction surface, reference-only voxels (missed by the prediction) and
prediction-only voxels (spurious), plus critical-event markers. Output: one JSON per case
with base64 Float32 positions (mm, centred) and Uint32 indices.
"""
import base64, json, sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure

OUT = Path("/root/site_out"); OUT.mkdir(exist_ok=True)
GT_DIR = Path("/root/data/imagecas_x/ImageCAS-X_dataset_v3/segmentations")
CASES = [
    ("hero", "cas_net", "645"),
    ("caught", "ffr_unet", "993"),
    ("clean", "cas_net", "448"),
    ("missed", "ffr_unet", "473"),
]
SEG = {1: "LM", 2: "LAD", 3: "LCx", 4: "D1", 5: "D2", 6: "OM1", 7: "OM2", 8: "IM", 9: "RCA",
       10: "R-PDA", 11: "R-PLA", 12: "L-PDA", 13: "L-PLA", 14: "Other"}

fl = pd.read_csv("/root/data/test_eval/flags.csv", dtype={"scan_id": str})
ft = pd.read_csv("/root/data/test_eval/features.csv", dtype={"scan_id": str})
d = fl.merge(ft[["method", "scan_id", "critical", "dice"]], on=["method", "scan_id"])
d["rank"] = d.groupby("method")["score"].rank(ascending=False, method="first").astype(int)


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def surface(vol, sp, origin, step=2, sigma=0.8, min_vox=30):
    if vol.sum() < min_vox:
        return None
    f = ndimage.gaussian_filter(vol.astype(np.float32), sigma)
    if f.max() < 0.5:
        return None
    v, faces, _, _ = measure.marching_cubes(f, 0.5, spacing=tuple(sp), step_size=step)
    v = (v - origin).astype(np.float32)
    return {"pos": b64(v.ravel()), "idx": b64(faces.astype(np.uint32).ravel()),
            "nv": int(len(v)), "nf": int(len(faces))}


summary = []
for key, method, scan in CASES:
    gt_img = nib.load(str(GT_DIR / f"{scan}.coronary.nii.gz"))
    gt = np.asarray(gt_img.dataobj).astype(np.int16)
    pr = (np.asarray(nib.load(f"/root/data/preds/{method}/{scan}/mask.nii.gz").dataobj) > 0)
    sp = np.array(gt_img.header.get_zooms()[:3], float)
    g = gt > 0
    both = g | pr
    idx = np.argwhere(both)
    lo = np.maximum(idx.min(0) - 4, 0); hi = np.minimum(idx.max(0) + 5, both.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    g, pr, gtl = g[sl], pr[sl], gt[sl]
    centre = (np.array(g.shape) * sp) / 2.0
    # tolerance ~1 voxel in-plane so thin boundary disagreements are not painted
    pr_d = ndimage.binary_dilation(pr, iterations=2)
    g_d = ndimage.binary_dilation(g, iterations=2)
    missed = g & ~pr_d
    spur = pr & ~g_d
    # keep only meaningful blobs (>= 40 voxels) to avoid speckle
    def clean(m, n=40):
        lab, k = ndimage.label(m)
        if k == 0:
            return m
        sizes = ndimage.sum(m, lab, range(1, k + 1))
        keep = np.isin(lab, 1 + np.flatnonzero(sizes >= n))
        return keep
    missed, spur = clean(missed), clean(spur)
    q = json.load(open(f"/root/data/qa/v3/{method}/{scan}.json"))
    # predicted connected components (26-conn), largest first
    lab, k = ndimage.label(pr, structure=np.ones((3, 3, 3)))
    sizes = ndimage.sum(pr, lab, range(1, k + 1)) if k else np.array([])
    order = 1 + np.argsort(-sizes)
    comps = []
    for i, cid in enumerate(order[:6]):
        if sizes[cid - 1] < 60:
            break
        comps.append(surface(lab == cid, sp, centre))
    rest = pr & ~np.isin(lab, order[:len(comps)])
    frag = surface(rest, sp, centre, step=1, sigma=0.6, min_vox=20)
    # nearest predicted component for every reference voxel (to locate disconnections)
    _, ind = ndimage.distance_transform_edt(~pr, sampling=sp, return_indices=True)
    near = lab[tuple(ind)]
    markers = []
    for e in q.get("events", []):
        if e.get("flag") != "F2" or e.get("severity") != "severe":
            continue
        sel = np.isin(gtl, e.get("label_ids", [])) & g
        if sel.sum() == 0:
            continue
        nl = np.where(sel, near, 0)
        bnd = np.zeros_like(sel)
        for ax in range(3):
            a = np.roll(nl, 1, axis=ax)
            bnd |= sel & np.roll(sel, 1, axis=ax) & (a != nl)
        if bnd.sum() == 0:
            continue
        bl, nb = ndimage.label(bnd, structure=np.ones((3, 3, 3)))
        for j in range(1, min(nb, 3) + 1):
            c = np.argwhere(bl == j).mean(0)
            markers.append({"pos": (c * sp - centre).round(2).tolist(), "flag": "F2", "subtype": "disconnection",
                            "segment": "+".join(e.get("segments", [])), "length_mm": round(float(e.get("length_mm", 0)), 1),
                            "dist_from_ostium_mm": None if e.get("dist_from_ostium_mm") is None else round(float(e["dist_from_ostium_mm"]), 1)})
    for e in q.get("events", []):
        if e.get("severity") != "severe" or "centroid_idx" not in e:
            continue
        c = np.array(e["centroid_idx"], float) - lo
        markers.append({"pos": (c * sp - centre).round(2).tolist(), "flag": e["flag"],
                        "subtype": e.get("subtype", ""), "segment": e.get("segment") or "+".join(e.get("segments", [])),
                        "length_mm": round(float(e.get("length_mm", 0)), 1),
                        "dist_from_ostium_mm": None if e.get("dist_from_ostium_mm") is None else round(float(e["dist_from_ostium_mm"]), 1)})
    row = d[(d.method == method) & (d.scan_id == scan)].iloc[0]
    rec = {"key": key, "method": method, "scan": scan, "rank": int(row["rank"]), "n": 160,
           "flagged": bool(row.flagged_top20), "critical": bool(row.critical), "dice": round(float(row.dice), 3),
           "score": round(float(row.score), 4), "critical_reasons": q.get("critical_reasons") or [],
           "markers": markers,
           "meshes": {"components": comps, "fragments": frag, "missed": surface(missed, sp, centre, step=1, sigma=0.6),
                      "spurious": surface(spur, sp, centre, step=1, sigma=0.6),
                      "ref": surface(g, sp, centre, step=2)},
           "extent_mm": (np.array(g.shape) * sp).round(1).tolist()}
    (OUT / f"{key}.json").write_text(json.dumps(rec))
    sz = (OUT / f"{key}.json").stat().st_size / 1e6
    summary.append((key, method, scan, rec["rank"], rec["flagged"], rec["critical"], rec["dice"], len(markers),
                    (len(comps), {k: (v["nf"] if isinstance(v, dict) and v else 0) for k, v in rec["meshes"].items() if k != "components"}), round(sz, 2)))
for s in summary:
    print(s)
