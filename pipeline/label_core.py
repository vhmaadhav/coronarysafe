"""Shared labelling core for the CoronarySafe QA study (CPU only).

Used by modal/label_features.py (Modal) and droplet/run_labels.py (droplet CPU pool).
Nothing here
assumes a host: the data root is passed in. Runs the coronarysafe package (src/) on predictions
and writes
QA JSON to <root>/qa/<QA_VERSION>/<method>/<scan>.json. Taxonomy: configs/taxonomy_v3.yaml.

Conventions (unchanged from the Modal runs)
  * VTK centerlines are LPS world coordinates, converted to RAS before the NIfTI affine maps
    them to voxels. A case whose centerline points are not >= 95% inside the dilated GT lumen
    falls back to the GT skeleton.
  * GT is used only for labels and evaluation. Priors are fitted on the train split only.
"""

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]
_SRC = REPO / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:  # droplet: the repo is not pip-installed
    sys.path.insert(0, str(_SRC))
# Modal container location; on the droplet the repo copy is used unless this file exists.
CFG_REMOTE_DEFAULT = "/root/configs/taxonomy_v3.yaml"


GT_DIR = "imagecas_x/ImageCAS-X_dataset_v3"

QA_VERSION = "v3"

CFG_REMOTE = "/root/configs/taxonomy_v3.yaml"

CL_MIN_INSIDE = 0.95

LPS_TO_RAS = (-1.0, -1.0, 1.0)


def read_split(root, split):
    """Scan ids of a filelist split ('train', 'val', 'test', 'valtest' or 'all'), in file order."""
    base = Path(root) / GT_DIR / "filelist"
    names = {"all": ["train", "val", "test"], "valtest": ["val", "test"]}.get(split, [split])
    ids, seen = [], set()
    for name in names:
        for line in (base / f"{name}.txt").read_text().splitlines():
            line = line.strip()
            if line and str(int(line)) not in seen:
                seen.add(str(int(line)))
                ids.append(str(int(line)))
    return ids


def _gt_seg(root, scan):
    return Path(root) / GT_DIR / "segmentations" / f"{scan}.coronary.nii.gz"


def _centerline_path(root, scan, side):
    return Path(root) / GT_DIR / "centerlines" / f"{scan}.coronary_{side}_centerline.vtk"


def _to_full(arr, bbox, shape, dtype):
    """Place an array cropped to a half-open bbox into a full-grid array (zeros elsewhere)."""
    arr = np.asarray(arr).astype(dtype)
    if bbox is None or tuple(arr.shape) == tuple(shape):
        return arr
    (x0, x1), (y0, y1), (z0, z1) = bbox
    if arr.shape != (x1 - x0, y1 - y0, z1 - z0):
        raise ValueError(f"cropped array {arr.shape} does not match bbox {bbox}")
    full = np.zeros(shape, dtype=dtype)
    full[x0:x1, y0:y1, z0:z1] = arr
    return full


def load_centerlines(root, scan, lumen, affine):
    """Index-space polylines for the left and right GT centerlines, or (None, status)."""
    from coronarysafe.io import load_centerline_vtk, world_to_index

    polys = []
    for side in ("left", "right"):
        path = _centerline_path(root, scan, side)
        if not path.exists():
            continue
        pts, lines, _ = load_centerline_vtk(path, prefer_vtk=False)
        idx = world_to_index(np.asarray(pts, dtype=float) * np.asarray(LPS_TO_RAS), affine)
        polys += [idx[np.asarray(line)] for line in lines if len(line) > 0]
    if not polys:
        return None, "missing"
    allp = np.vstack(polys)
    ii = np.rint(allp).astype(int)
    inb = np.all((ii >= 0) & (ii < np.array(lumen.shape)), axis=1)
    dil = ndimage.binary_dilation(lumen)
    frac = float(dil[tuple(ii[inb].T)].mean()) if inb.any() else 0.0
    if frac < CL_MIN_INSIDE:
        return None, f"fallback_outside_lumen({frac:.3f})"
    return polys, "vtk"


def _metrics(mask, lumen, sp):
    from coronarysafe.geometry import bbox_slices
    from coronarysafe.graph import skeletonize

    sl = bbox_slices(mask | lumen, sp, pad_mm=2.0)
    m = mask[sl]
    g = lumen[sl]
    denom = m.sum() + g.sum()
    dice = float(2.0 * np.logical_and(m, g).sum() / denom) if denom else float("nan")
    sk_m = skeletonize(m)
    sk_g = skeletonize(g)
    dil_m = ndimage.binary_dilation(m)
    dil_g = ndimage.binary_dilation(g)
    tprec = float((sk_m & dil_g).sum() / sk_m.sum()) if sk_m.sum() else float("nan")
    tsens = float((sk_g & dil_m).sum() / sk_g.sum()) if sk_g.sum() else float("nan")
    if np.isfinite(tprec) and np.isfinite(tsens) and tprec + tsens > 0:
        cldice = 2.0 * tprec * tsens / (tprec + tsens)
    else:
        cldice = float("nan")
    b0_pred = int(ndimage.label(m, structure=np.ones((3, 3, 3)))[1])
    b0_gt = int(ndimage.label(g, structure=np.ones((3, 3, 3)))[1])
    return {
        "dice": dice,
        "cldice": cldice,
        "b0_pred": b0_pred,
        "b0_gt": b0_gt,
        "b0_error": b0_pred - b0_gt,
        "skeleton_empty_warning": bool(m.any() and sk_m.sum() == 0),
        "skeleton_sparse_warning": bool(m.any() and sk_m.sum() < m.sum() ** (1.0 / 3.0)),
    }


def compute_case(root, method, scan, pred_method):
    """Compute the QA record for one (method, scan). Raises on missing or inconsistent data."""
    from coronarysafe.features.graph_features import compute_graph_features
    from coronarysafe.features.priors import zscore
    from coronarysafe.features.uncertainty import tta_disagreement, uncertainty_features
    from coronarysafe.io import load_nifti
    from coronarysafe.labels.taxonomy import label_failures, load_taxonomy_config

    seg = _gt_seg(root, scan)
    if not seg.exists():
        raise FileNotFoundError(str(seg))
    gt, sp, aff = load_nifti(seg)
    gt = np.asarray(gt)
    lumen = gt > 0

    pdir = Path(root) / "preds" / pred_method / scan
    if not (pdir / "meta.json").exists():
        raise FileNotFoundError(f"no prediction at {pdir}")
    pmeta = json.loads((pdir / "meta.json").read_text())
    bbox = pmeta.get("bbox_voxel_half_open") if pmeta.get("prob_cropped") else None

    mask_arr, _, _ = load_nifti(pdir / "mask.nii.gz")
    if tuple(mask_arr.shape) != tuple(gt.shape):
        raise ValueError(f"mask shape {mask_arr.shape} != GT shape {gt.shape}")
    mask = np.asarray(mask_arr) > 0

    prob = None
    if (pdir / "prob.nii.gz").exists():
        prob_arr, _, _ = load_nifti(pdir / "prob.nii.gz")
        prob = _to_full(prob_arr, bbox, gt.shape, np.float32)

    flip_masks = []
    for f in sorted((pdir / "flips").glob("*.nii.gz")) if (pdir / "flips").exists() else []:
        fa, _, _ = load_nifti(f)
        flip_masks.append(_to_full(fa, bbox, gt.shape, bool) > 0)

    polys, cl_status = load_centerlines(root, scan, lumen, aff)
    cfg_path = (
        CFG_REMOTE if Path(CFG_REMOTE).exists() else str(REPO / "configs" / "taxonomy_v3.yaml")
    )
    cfg = load_taxonomy_config(cfg_path)
    lab = label_failures(gt, polys, mask, sp, cfg)

    met = _metrics(mask, lumen, sp)
    graph = compute_graph_features(mask, sp)
    unc = uncertainty_features(mask, prob, sp) if prob is not None else {}
    tta = tta_disagreement([mask] + flip_masks) if flip_masks else float("nan")

    priors_path = Path(root) / "meta" / "priors_v1.json"
    z = {}
    if priors_path.exists():
        z = zscore(graph, json.loads(priors_path.read_text()))

    return {
        "centerline_source": lab["centerline_source"],
        "centerline_status": cl_status,
        "centerline_fallback": cl_status != "vtk",
        "flags": lab["flags"],
        "severe": lab["severe"],
        "minor": lab["minor"],
        "critical": lab["critical"],
        "critical_reasons": lab["critical_reasons"],
        "burden": lab["burden"],
        "missed_major_mm": lab["missed_major_mm"],
        "events": lab["events"],
        "segments": lab["segments"],
        "sensitivity": lab["sensitivity"],
        "dominance": lab["dominance"],
        "n_centerline_points": lab["n_centerline_points"],
        "dice": met["dice"],
        "cldice": met["cldice"],
        "b0_pred": met["b0_pred"],
        "b0_gt": met["b0_gt"],
        "b0_error": met["b0_error"],
        "skeleton_empty_warning": met["skeleton_empty_warning"],
        "skeleton_sparse_warning": met["skeleton_sparse_warning"],
        "graph": graph,
        "uncertainty": unc,
        "tta_disagreement": tta,
        "n_flips": len(flip_masks),
        "z": z,
        "priors_used": bool(z),
        "prob_present": prob is not None,
        "pred_warnings": pmeta.get("warnings", []),
    }


def process_case(root, method, scan_id, pred_method=None, force=False):
    """Idempotent per-case job. Writes qa/<QA_VERSION>/<method>/<scan>.json. Errors are recorded."""
    scan = str(int(scan_id))
    pred_method = pred_method or method
    out = Path(root) / "qa" / QA_VERSION / method / f"{scan}.json"
    if out.exists() and not force:
        if json.loads(out.read_text()).get("status") == "ok":
            return {"status": "skipped", "method": method, "scan_id": scan}
    t0 = time.time()
    try:
        result = compute_case(root, method, scan, pred_method)
        result["status"] = "ok"
    except Exception as exc:  # per-case isolation: record and continue
        result = {"status": "error", "error": repr(exc), "traceback": traceback.format_exc()}
    result.update(
        method=method,
        scan_id=scan,
        pred_method=pred_method,
        qa_version=QA_VERSION,
        runtime_s=round(time.time() - t0, 2),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=_jsonable))
    return {"status": result["status"], "method": method, "scan_id": scan}


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not serialisable: {type(o)}")


def _flatten(d):
    """One summary row per unit: scalars, every flag and subtype, per-flag severity, tier,
    burden, sensitivity and feature groups."""
    flags = set(d.get("flags", []))
    events = d.get("events", [])
    row = {
        "method": d["method"],
        "scan_id": d["scan_id"],
        "status": d["status"],
        "pred_method": d.get("pred_method"),
        "severe": bool(d.get("severe", False)),
        "minor": bool(d.get("minor", False)),
        "critical": bool(d.get("critical", False)),
        "critical_reasons": ";".join(d.get("critical_reasons", []) or []),
        "burden": d.get("burden"),
        "missed_major_mm": d.get("missed_major_mm"),
        "flags": ",".join(sorted(flags)),
        "dice": d.get("dice"),
        "cldice": d.get("cldice"),
        "b0_error": d.get("b0_error"),
        "b0_pred": d.get("b0_pred"),
        "b0_gt": d.get("b0_gt"),
        "skeleton_empty_warning": d.get("skeleton_empty_warning"),
        "centerline_fallback": d.get("centerline_fallback"),
        "centerline_status": d.get("centerline_status"),
        "n_flips": d.get("n_flips"),
        "tta_disagreement": d.get("tta_disagreement"),
        "runtime_s": d.get("runtime_s"),
        "error": d.get("error"),
    }
    for f in ("F1", "F1b", "F2", "F3a", "F3b", "F3c", "F3d", "F3h", "F3"):
        row[f"has_{f}"] = f in flags
    for f in ("F1", "F2", "F3a", "F3b", "F3c", "F3d", "F3h", "F3"):
        row[f"{f}_severe"] = any(e["flag"] == f and e["severity"] == "severe" for e in events)
    for name, v in (d.get("sensitivity") or {}).items():
        row[f"sens_{name}_severe"] = bool(v.get("severe"))
        row[f"sens_{name}_critical"] = bool(v.get("critical"))
    for k, v in (d.get("graph") or {}).items():
        row[f"g_{k}"] = v
    for k, v in (d.get("uncertainty") or {}).items():
        row[f"u_{k}"] = v
    for k, v in (d.get("z") or {}).items():
        row[f"z_{k}"] = v
    return row


def _report_block(sub):
    """Counts and rates for one group of rows (status ok)."""
    n = len(sub)
    if n == 0:
        return "units=0"
    parts = [
        f"units={n}",
        f"severe={int(sub.severe.sum())}",
        f"critical={int(sub.critical.sum())} ({sub.critical.mean():.3f})",
        f"minor={int(sub.minor.sum())}",
        f"F1b_degraded={int(sub.has_F1b.sum())}",
    ]
    subtypes = " ".join(
        f"{f}={int(sub['has_' + f].sum())}" for f in ("F1", "F2", "F3a", "F3b", "F3c", "F3d", "F3h")
    )
    parts.append("flagged: " + subtypes)
    parts.append(f"patients_with_critical={sub[sub.critical].scan_id.nunique()}")
    return " | ".join(parts)


def _build_report(df, methods, split):
    import pandas as pd  # noqa: F401

    ok = df[df.status == "ok"].copy()
    lines = [f"G0/v3 report, split={split}, methods={','.join(methods)}"]
    lines.append(f"rows={len(df)} ok={len(ok)} errors={int((df.status == 'error').sum())}")
    lines.append("")
    lines.append("per method:")
    for m in methods:
        lines.append(f"  {m}: " + _report_block(ok[ok.method == m]))
    lines.append("overall: " + _report_block(ok))

    lines.append("")
    lines.append("critical rate by Image Quality / Disease (overall):")
    for col in ("image_quality", "disease"):
        if col in ok and ok[col].notna().any():
            for key, g in ok.groupby(col, dropna=False):
                lines.append(f"  {col}={key}: units={len(g)} critical={g.critical.mean():.3f}")

    lines.append("")
    lines.append("burden distribution (overall, all units):")
    b = ok.burden.dropna()
    if len(b):
        q = b.quantile([0.25, 0.5, 0.75])
        lines.append(
            f"  q25={q.iloc[0]:.1f} median={q.iloc[1]:.1f} q75={q.iloc[2]:.1f} "
            f"max={b.max():.1f} zero={100 * (b == 0).mean():.1f}%"
        )
    for m in methods:
        bm = ok[ok.method == m].burden.dropna()
        if len(bm):
            lines.append(f"  {m}: median={bm.median():.1f} zero={100 * (bm == 0).mean():.1f}%")

    lines.append("")
    lines.append("critical agreement between methods (scans present in each):")
    if len(methods) >= 2:
        piv = ok.pivot_table(index="scan_id", columns="method", values="critical", aggfunc="first")
        a, b_ = methods[0], methods[1]
        if a in piv and b_ in piv:
            both_rows = piv[[a, b_]].dropna()
            both = int((both_rows[a].astype(bool) & both_rows[b_].astype(bool)).sum())
            either = int((both_rows[a].astype(bool) | both_rows[b_].astype(bool)).sum())
            n = len(both_rows)
            lines.append(
                f"  {a} vs {b_}: scans={n} both={both} either={either} "
                f"both/either={both / either:.3f}"
                if either
                else f"  {a} vs {b_}: scans={n} none critical"
            )

    lines.append("")
    lines.append("sensitivity (critical units, overall):")
    lines.append(f"  main: {int(ok.critical.sum())}")
    for c in sorted(c for c in ok.columns if c.startswith("sens_") and c.endswith("_critical")):
        lines.append(f"  {c[5:-9]}: {int(ok[c].fillna(False).astype(bool).sum())}")

    lines.append("")
    lines.append(
        f"skeleton-empty warnings: "
        f"{int(ok.skeleton_empty_warning.fillna(False).astype(bool).sum())}; "
        f"centerline fallbacks: {int(ok.centerline_fallback.fillna(False).astype(bool).sum())}"
    )
    return "\n".join(lines)


def summarise_local(root, methods, split):
    """Droplet version of modal summarise. Writes qa/<QA_VERSION>/table_<split>.parquet, returns the
    report text. Needs pandas, pyarrow, openpyxl. Descriptors come from the ImageCAS-X folder."""
    import pandas as pd

    ids = set(read_split(root, split))
    rows = []
    for m in methods:
        d = Path(root) / "qa" / QA_VERSION / m
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            rec = json.loads(f.read_text())
            if rec.get("scan_id") in ids and rec.get("status") == "ok":
                rows.append(_flatten(rec))
    df = pd.DataFrame(rows)
    if df.empty:
        return f"no QA records for split={split} methods={methods}"
    desc_path = Path(root) / GT_DIR / "Descriptors.xlsx"
    desc_map = {}
    if desc_path.exists():  # absent only in fake test roots
        desc = pd.read_excel(desc_path, engine="openpyxl")
        desc_map = {
            str(int(r["Scan ID"])): {
                "Image Quality": r.get("Image Quality"),
                "Dominance": r.get("Dominance"),
                "Disease": r.get("Disease"),
            }
            for _, r in desc.iterrows()
        }
    df["image_quality"] = df.scan_id.map(lambda s: desc_map.get(s, {}).get("Image Quality"))
    df["dominance"] = df.scan_id.map(lambda s: desc_map.get(s, {}).get("Dominance"))
    df["disease"] = df.scan_id.map(lambda s: desc_map.get(s, {}).get("Disease"))
    out = Path(root) / "qa" / QA_VERSION / f"table_{split}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    report = _build_report(df, methods, split)
    return report + f"\ntable: {out} ({len(df)} rows)"
