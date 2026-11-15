"""CPU-only Modal app: run the coronarysafe package on predictions and write QA JSON.

Steps (run from the repository root, see the entrypoint below):
    modal run infra/modal/label_features.py --step priors
    modal run --detach infra/modal/label_features.py --step run --methods cas_net,nnunet_oof --split test
    modal run infra/modal/label_features.py --step summarise --methods cas_net,nnunet_oof --split test

Data on the Volume `coronarysafe-data` at /data:
    imagecas_x/ImageCAS-X_dataset_v3/segmentations/<id>.coronary.nii.gz  15-class GT
    imagecas_x/ImageCAS-X_dataset_v3/centerlines/<id>.coronary_{left,right}_centerline.vtk
    imagecas_x/ImageCAS-X_dataset_v3/filelist/{train,val,test}.txt, Descriptors.xlsx
    preds/<method>/<id>/{mask,prob}.nii.gz, meta.json, flips/*.nii.gz (tta only)
    meta/nnunet_folds.csv (scan_id, filelist_split, nnunet_fold)

Outputs: meta/priors_v1.json, qa/v2/<method>/<id>.json, qa/v2/table_<split>.parquet.

Conventions
  * VTK centerlines are LPS world coordinates. They are converted to RAS (x, y negated)
    before the NIfTI affine maps them to voxel indices. A case whose centerline points do
    not lie (>= 95%) inside the dilated GT lumen falls back to the GT skeleton and is flagged.
  * GT is used only for labels and evaluation. Priors are fitted on the train split only.
"""

import csv
import json
from pathlib import Path

import modal
import nibabel as nib  # noqa: F401  (image dependency)
import numpy as np

# Shared labelling core (pipeline/label_core.py), also used by infra/droplet/run_labels.py.
from pipeline.label_core import (  # noqa: E402
    CFG_REMOTE,
    GT_DIR,
    QA_VERSION,
    _build_report,
    _flatten,
    _gt_seg,
    _jsonable,
    process_case,
    read_split,
)
from scipy import ndimage  # noqa: F401

APP_NAME = "coronarysafe-label-features"
VOLUME_NAME = "coronarysafe-data"
DATA = "/data"
DEFAULT_METHODS = "cas_net,swin_unetr,ffr_unet,nnunet_oof"

REPO = Path(__file__).resolve().parents[2]

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy",
    "scipy",
    "scikit-image",
    "networkx",
    "nibabel",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "pyyaml",
    "openpyxl",
)
image = image.env({"PYTHONPATH": "/root/pkgs:/root"})  # /root holds pipeline/ (shared label core)
if modal.is_local():  # local files last, so later build steps do not invalidate the cache
    image = image.add_local_dir(
        REPO / "src" / "coronarysafe", "/root/pkgs/coronarysafe", ignore=["__pycache__"]
    ).add_local_file(REPO / "configs" / "taxonomy_v3.yaml", CFG_REMOTE).add_local_dir(
        REPO / "pipeline", "/root/pipeline", ignore=["__pycache__"]
    )

# ---------------------------------------------------------------- helpers (local or remote)


























# ----------------------------------------------------------------------- Modal functions


@app.function(image=image, volumes={DATA: volume}, cpu=2, memory=8192, timeout=3600)
def priors_scan(scan_id: str) -> dict:
    """Graph features of one train-split GT lumen (binarised). Fitting uses train only."""
    from coronarysafe.features.graph_features import compute_graph_features
    from coronarysafe.io import load_nifti

    scan = str(int(scan_id))
    try:
        gt, sp, _ = load_nifti(_gt_seg(DATA, scan))
        return {"scan_id": scan, "features": compute_graph_features(np.asarray(gt) > 0, sp)}
    except Exception as exc:
        return {"scan_id": scan, "error": repr(exc)}


@app.function(image=image, volumes={DATA: volume}, cpu=2, memory=8192, timeout=7200)
def fit_priors():
    """Fit per-feature mean/std over the train-split GT masks; write meta/priors_v1.json."""
    from coronarysafe.features.priors import fit_priors as fit_priors_stats

    volume.reload()
    ids = read_split(DATA, "train")
    results = list(priors_scan.map(ids))
    good = [r["features"] for r in results if "features" in r]
    bad = [r for r in results if "error" in r]
    priors = fit_priors_stats(good)
    out = Path(DATA) / "meta" / "priors_v1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(priors, indent=1, default=_jsonable))
    volume.commit()
    print(f"priors: fitted on {len(good)} train GT masks, {len(bad)} errors -> {out}")
    for r in bad[:10]:
        print(f"  error {r['scan_id']}: {r['error']}")
    return {"n_fitted": len(good), "n_errors": len(bad), "features": sorted(priors)}


@app.function(
    image=image,
    volumes={DATA: volume},
    cpu=2,
    memory=16384,
    timeout=7200,
    max_containers=50,
)
def process(method: str, scan_id: str, pred_method: str = None) -> dict:
    """QA for one (method, scan). pred_method defaults to method (nnunet_oof uses nnunet_f<k>)."""
    volume.reload()
    res = process_case(DATA, method, scan_id, pred_method=pred_method)
    volume.commit()
    return res


@app.function(image=image, volumes={DATA: volume}, timeout=6 * 3600)
def run(methods: str, split: str) -> dict:
    """Fan out process() over methods x split. nnunet_oof uses each scan's fold from the table."""
    volume.reload()
    ms = [m.strip() for m in methods.split(",") if m.strip()]
    ids = read_split(DATA, split)
    folds = {}
    fold_csv = Path(DATA) / "meta" / "nnunet_folds.csv"
    if fold_csv.exists():
        with open(fold_csv) as fh:
            for row in csv.DictReader(fh):
                if row.get("nnunet_fold", "").strip() != "":
                    folds[str(int(row["scan_id"]))] = row["nnunet_fold"].strip()
    jobs, no_pred, no_fold = [], 0, 0
    for m in ms:
        for s in ids:
            if m == "nnunet_oof":
                if s not in folds:
                    no_fold += 1
                    continue
                pred = f"nnunet_f{folds[s]}"
            else:
                pred = m
            if not (Path(DATA) / "preds" / pred / s / "meta.json").exists():
                no_pred += 1
                continue
            jobs.append((m, s, pred if m == "nnunet_oof" else None))
    print(f"run: split={split} jobs={len(jobs)} no_prediction_yet={no_pred} no_fold={no_fold}")
    results = list(process.starmap(jobs, order_outputs=False)) if jobs else []
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"run: results {counts}")
    return {"jobs": len(jobs), "no_prediction_yet": no_pred, "results": counts}


@app.function(image=image, volumes={DATA: volume}, cpu=2, memory=8192, timeout=3600)
def summarise(methods: str, split: str) -> str:
    """Write qa/v2/table_<split>.parquet and print the G0 report."""
    import pandas as pd

    volume.reload()
    ms = [m.strip() for m in methods.split(",") if m.strip()]
    ids = set(read_split(DATA, split))
    rows = []
    for m in ms:
        d = Path(DATA) / "qa" / QA_VERSION / m
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            rec = json.loads(f.read_text())
            if rec.get("scan_id") not in ids:
                continue
            rows.append(_flatten(rec))
    df = pd.DataFrame(rows)
    if df.empty:
        report = f"no QA records for split={split} methods={ms}"
        print(report)
        return report
    desc_map = {}
    import openpyxl  # noqa: F401

    desc = pd.read_excel(Path(DATA) / GT_DIR / "Descriptors.xlsx", engine="openpyxl")
    for _, r in desc.iterrows():
        desc_map[str(int(r["Scan ID"]))] = {
            "Image Quality": r.get("Image Quality"),
            "Dominance": r.get("Dominance"),
            "Disease": r.get("Disease"),
        }
    df["image_quality"] = df.scan_id.map(lambda s: desc_map.get(s, {}).get("Image Quality"))
    df["dominance"] = df.scan_id.map(lambda s: desc_map.get(s, {}).get("Dominance"))
    df["disease"] = df.scan_id.map(lambda s: desc_map.get(s, {}).get("Disease"))
    out = Path(DATA) / "qa" / QA_VERSION / f"table_{split}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    volume.commit()
    report = _build_report(df, ms, split)
    print(report)
    print(f"table: {out} ({len(df)} rows)")
    return report


@app.local_entrypoint()
def main(step: str = "priors", methods: str = DEFAULT_METHODS, split: str = "test"):
    if step == "priors":
        print(fit_priors.remote())
    elif step == "run":
        print(run.remote(methods, split))
    elif step == "summarise":
        print(summarise.remote(methods, split))
    else:
        raise SystemExit(f"unknown step {step!r}: use priors, run or summarise")
