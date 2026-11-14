"""GPU inference for the CoronarySafe QA study, on Modal.

Thin Modal layer. The method-agnostic core (model bundles, prediction, outputs, fold
reconstruction, evaluation) lives in pipeline/infer_core.py and is shared with the
MI300X droplet runner (droplet/run_infer.py).

Steps (local_entrypoint `main`):
    check        CPU. Imports the pinned stack and builds all four model heads on CPU.
    folds        CPU. Writes /data/meta/nnunet_folds.csv (nnU-Net 5-fold reconstruction).
    verify_folds 16 train scans x 5 nnU-Net folds, then the OOF check. Writes
                 /data/meta/fold_verification.json.
    smoke        One val scan with cas_net and nnunet_f0, then evaluated. Skips with exit 0
                 if no CT volume is staged yet.
    run          Batch predict for --methods over --split. Accepts nnunet_oof.
    evaluate     Per-scan metrics and mean +- sd per method, to /data/eval/<method>_<split>.csv.

Method names: cas_net, swin_unetr, ffr_unet, nnunet_f0 .. nnunet_f4 (and nnunet_oof in run).
"""

import csv
import json
import sys
from pathlib import Path

import modal

# Make pipeline/ importable when this file is run locally from modal/ (in the container the
# pipeline directory is added at /root/pipeline, next to this file).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.infer_core import (  # noqa: E402
    ALL_METHODS,
    NET_WEIGHTS,
    NNUNET_METHODS,
    N_FOLDS,
    OOF_METHOD,
    REPO_COMMIT,
    REPO_URL,
    CorePaths,
    compute_folds,
    eval_scan,
    fold_table,
    find_ct,
    mean_sd,
    nnunet_fold_of,
    parse_methods,
    predict_impl,
    print_verify_table,
    split_ids,
    utc_stamp,
    verify_scans,
    verify_summary,
)

APP_NAME = "coronarysafe-infer"
VOLUME_NAME = "coronarysafe-data"
DATA = "/data"

GPU = "L40S"  # module constant; switch to "A100" or "H100" here
MAX_CONTAINERS = 4
HOUR = 3600

REPO_DIR = "/opt/ImageCAS-X"
PRED_ROOT = f"{DATA}/preds"
PAPER_CASNET_DSC = "CAS-Net lumen DSC 91.2 +- 2.8 on test (ImageCAS-X paper)"
DEFAULT_METHODS = (
    "cas_net,swin_unetr,ffr_unet,nnunet_f0,nnunet_f1,nnunet_f2,nnunet_f3,nnunet_f4"
)

P = CorePaths(data=DATA, pred=PRED_ROOT, device="cuda", repo_dir=REPO_DIR)

_LOCAL_PIPELINE = _REPO_ROOT / "pipeline"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git", "libgl1", "libglib2.0-0", "libxt6", "libxrender1", "libsm6", "libice6", "libx11-6"
    )
    .pip_install(
        "torch==2.5.1",
        "numpy==1.26.4",
        "monai==1.4.0",
        "nnunetv2==2.6.2",
        "nibabel==5.3.2",
        "scikit-image==0.24.0",
        "SimpleITK==2.4.0",
        "scipy==1.14.1",
        "scikit-learn==1.5.2",
        "vtk==9.3.1",
        "tqdm",
        "matplotlib",
        "openpyxl",
        "pandas",
    )
    .run_commands(
        f"git clone {REPO_URL} {REPO_DIR} && cd {REPO_DIR} && git checkout {REPO_COMMIT}",
        f"pip install --no-deps {REPO_DIR}",
    )
    .env(
        {
            "nnUNet_raw": "/tmp/nnunet/raw",
            "nnUNet_preprocessed": "/tmp/nnunet/preprocessed",
            "nnUNet_results": "/tmp/nnunet/results",
        }
    )
)
if _LOCAL_PIPELINE.is_dir():  # present when building locally; the container has it at /root/pipeline
    image = image.add_local_dir(str(_LOCAL_PIPELINE), remote_path="/root/pipeline")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# ---------------------------------------------------------------------------
# Modal functions
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu=GPU,
    volumes={DATA: volume},
    timeout=2 * HOUR,
    cpu=4,
    memory=32768,
    max_containers=MAX_CONTAINERS,
)
def predict(method: str, scan_id: str, save_probs: bool = True, tta: bool = False) -> dict:
    """One method on one scan. Idempotent: skipped when meta.json is current and matches tta."""
    if method not in ALL_METHODS:
        raise ValueError(f"unknown method {method!r}")
    volume.reload()
    result = predict_impl(P, method, str(scan_id), save_probs, tta, gpu_requested=GPU)
    volume.commit()
    return result


@app.function(image=image, volumes={DATA: volume}, timeout=12 * HOUR, cpu=2, memory=4096)
def run(methods: str, split: str, tta: bool = False) -> dict:
    """Fan predict over methods x split with bounded concurrency. One failure does not stop the batch."""
    volume.reload()
    method_list = parse_methods(methods, allow_oof=True)
    ids = split_ids(P, split)
    jobs, no_fold = [], 0
    table = fold_table(P) if OOF_METHOD in method_list else {}
    for m in method_list:
        if m == OOF_METHOD:
            for s in ids:
                k = nnunet_fold_of(P, s, table)
                if k is None:
                    no_fold += 1
                    continue
                jobs.append((f"nnunet_f{k}", s, True, tta))
        else:
            jobs.extend((m, s, True, tta) for s in ids)
    if no_fold:
        print(f"[run] {OOF_METHOD}: skipped {no_fold} scans with no nnU-Net fold (test/unknown)")
    print(f"[run] {len(jobs)} jobs: methods={method_list} split={split} ({len(ids)} scans) tta={tta}")

    results = list(predict.starmap(jobs, return_exceptions=True))

    done, skipped, failures = 0, 0, []
    for (m, s, _, _), r in zip(jobs, results):
        if isinstance(r, BaseException):
            failures.append({"method": m, "scan_id": s, "error": f"{type(r).__name__}: {r}"})
        elif r.get("status") == "skipped":
            skipped += 1
        else:
            done += 1

    stamp = utc_stamp()
    record = {
        "started_utc": stamp,
        "methods": method_list,
        "split": split,
        "tta": tta,
        "n_jobs": len(jobs),
        "done": done,
        "skipped": skipped,
        "failed": len(failures),
        "failures": failures,
    }
    Path(P.runs_dir).mkdir(parents=True, exist_ok=True)
    path = f"{P.runs_dir}/{stamp}.json"
    Path(path).write_text(json.dumps(record, indent=2))
    volume.commit()
    print(f"[run] done={done} skipped={skipped} failed={len(failures)} -> {path}")
    return record


@app.function(
    image=image,
    volumes={DATA: volume},
    timeout=2 * HOUR,
    cpu=4,
    memory=16384,
    max_containers=MAX_CONTAINERS,
)
def eval_one(method: str, scan_id: str) -> dict:
    volume.reload()
    return eval_scan(P, method, scan_id)


@app.function(image=image, volumes={DATA: volume}, timeout=6 * HOUR, cpu=2, memory=4096)
def evaluate(methods: str, split: str = "valtest", scan_ids: list | None = None) -> None:
    """Per-scan metrics for each method, written to /data/eval/<method>_<split>.csv."""
    volume.reload()
    method_list = parse_methods(methods)
    ids = list(scan_ids) if scan_ids else split_ids(P, split)
    folds = fold_table(P)
    Path(P.eval_root).mkdir(parents=True, exist_ok=True)
    metric_cols = ["dice", "cl_dice", "betti_error_1", "betti_error_2",
                   "centerline_md", "centerline_hd95"]

    for m in method_list:
        results = list(eval_one.starmap([(m, s) for s in ids], return_exceptions=True))
        rows = []
        for s, r in zip(ids, results):
            if isinstance(r, BaseException):
                row = {"status": f"error: {type(r).__name__}: {r}"}
            else:
                row = dict(r)
            row.update(
                {"method": m, "split": split, "scan_id": s, "nnunet_fold": folds.get(str(s), "")}
            )
            rows.append(row)

        fields = ["method", "split", "scan_id", "tta", "status", "nnunet_fold"] + metric_cols
        out_csv = Path(P.eval_root) / f"{m}_{split}.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        print(f"[evaluate] {m} split={split}: {len(rows)} scans -> {out_csv}")
        for col in metric_cols:
            print(f"    {col:16s} {mean_sd([r.get(col) for r in rows])}")
        if m == "cas_net" and split == "test":
            print(f"    (published: {PAPER_CASNET_DSC})")
        if m in NNUNET_METHODS:
            k = int(m.rsplit("_f", 1)[1])
            held = [r.get("dice") for r in rows if r.get("nnunet_fold") == k]
            seen = [r.get("dice") for r in rows if r.get("nnunet_fold") not in ("", None, k)]
            print(
                f"    fold {k}: Dice held out (fold=={k}) {mean_sd(held)}; "
                f"Dice on its training cases (fold!={k}) {mean_sd(seen)}. "
                f"Held-out should be lower if the reconstruction is right."
            )
    volume.commit()


@app.function(image=image, volumes={DATA: volume}, timeout=30 * 60, cpu=2, memory=4096)
def make_fold_table() -> None:
    """Write /data/meta/nnunet_folds.csv: nnU-Net 5-fold reconstruction for train+val, blank for test."""
    volume.reload()
    train, val, test = split_ids(P, "train"), split_ids(P, "val"), split_ids(P, "test")
    train_val = sorted(set(train) | set(val))
    folds = compute_folds(train_val)

    meta = json.loads((Path(P.nnunet_model_dir) / "dataset.json").read_text())
    if meta.get("numTraining") != len(train_val):
        print(
            f"[folds] WARNING numTraining={meta.get('numTraining')} but train+val={len(train_val)}"
        )

    Path(P.meta_root).mkdir(parents=True, exist_ok=True)
    train_set = set(train)
    counts = {k: 0 for k in range(N_FOLDS)}
    with open(P.fold_table, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["scan_id", "filelist_split", "nnunet_fold"])
        for s in train_val:
            counts[folds[s]] += 1
            writer.writerow([s, "train" if s in train_set else "val", folds[s]])
        for s in test:
            writer.writerow([s, "test", ""])
    volume.commit()
    print(f"[folds] wrote {P.fold_table}: {len(train_val)} train+val, {len(test)} test")
    print(f"[folds] per-fold validation sizes: {counts}")


@app.function(image=image, volumes={DATA: volume}, timeout=30 * 60, cpu=4, memory=16384)
def check_env() -> None:
    """CPU-only check: imports, and all four heads build on CPU. No GPU, no predictions."""
    import glob
    import importlib

    import torch

    from pipeline.infer_core import build_nnunet_predictor, build_net

    for mod in [
        "inference",
        "models.registry",
        "preprocessing.pipeline",
        "postprocessing.pipeline",
        "utils.metrics",
        "evaluate",
        "nnunetv2.inference.predict_from_raw_data",
        "monai",
        "nibabel",
        "skimage",
        "SimpleITK",
        "vtk",
        "vtkmodules.util.numpy_support",
    ]:
        importlib.import_module(mod)
        print(f"[check] import ok: {mod}")
    print(f"[check] torch {torch.__version__} cuda build {torch.version.cuda}")
    for m in NET_WEIGHTS:
        b = build_net(P, m, "cpu")
        n = sum(p.numel() for p in b["model"].parameters())
        print(f"[check] {m}: {n / 1e6:.2f} M params, weights sha256 {b['weights_sha256'][:12]}")
    for k in range(N_FOLDS):
        build_nnunet_predictor(P, k, "cpu")
        print(f"[check] nnunet fold {k} predictor initialised on CPU")
    cts = glob.glob(f"{glob.escape(P.volumes_root)}/**/*.img.nii.gz", recursive=True)
    print(f"[check] CT volumes staged: {len(cts)} under {P.volumes_root}")


@app.function(image=image, volumes={DATA: volume}, timeout=10 * 60, cpu=1, memory=2048)
def pick_smoke_scan():
    """First val scan whose CT volume is staged, or None."""
    volume.reload()
    if not Path(P.volumes_root).exists():
        return None
    for s in split_ids(P, "val"):
        try:
            find_ct(P, s)
            return s
        except FileNotFoundError:
            continue
    return None


@app.function(image=image, volumes={DATA: volume}, timeout=6 * HOUR, cpu=2, memory=4096)
def verify_folds() -> dict:
    """Fold reconstruction check: 16 train scans x 5 folds. If the reconstruction is right,
    the OOF model (the one whose held-out fold holds the scan) gives the lowest Dice."""
    volume.reload()
    scans = verify_scans(P)
    table = fold_table(P)
    jobs = [(f"nnunet_f{k}", s, True, False) for s in scans for k in range(N_FOLDS)]
    print(f"[verify] {len(jobs)} predictions on {len(scans)} train scans")
    pred_results = list(predict.starmap(jobs, return_exceptions=True))
    failures = [
        f"{m}/{s}: {type(r).__name__}: {r}"
        for (m, s, _, _), r in zip(jobs, pred_results)
        if isinstance(r, BaseException)
    ]

    ev_jobs = [(f"nnunet_f{k}", s) for s in scans for k in range(N_FOLDS)]
    ev_results = list(eval_one.starmap(ev_jobs, return_exceptions=True))
    dice = {s: [float("nan")] * N_FOLDS for s in scans}
    for (m, s), r in zip(ev_jobs, ev_results):
        k = int(m.rsplit("_f", 1)[1])
        if isinstance(r, BaseException):
            failures.append(f"eval {m}/{s}: {type(r).__name__}: {r}")
        elif r.get("dice") is not None:
            dice[s][k] = float(r["dice"])

    summary = verify_summary(scans, {s: table[s] for s in scans}, dice)
    print_verify_table(summary)
    summary.update({"seed": 0, "failures": failures})
    out = Path(P.meta_root) / "fold_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    volume.commit()
    print(f"[verify] wrote {out}; failures: {len(failures)}")
    return summary


@app.local_entrypoint()
def main(
    step: str = "smoke",
    methods: str = DEFAULT_METHODS,
    split: str = "valtest",
    tta: bool = False,
    scan_ids: str = "",
):
    if step == "check":
        check_env.remote()
    elif step == "folds":
        make_fold_table.remote()
    elif step == "verify_folds":
        verify_folds.remote()
    elif step == "smoke":
        scan = pick_smoke_scan.remote()
        if scan is None:
            print(
                "SMOKE SKIPPED: no CT volumes staged under /data/imagecas/volumes yet. "
                "Nothing ran on GPU; re-run after the Kaggle volume staging finishes."
            )
            return
        print(f"[smoke] scan {scan}, tta={tta}")
        for m in ("cas_net", "nnunet_f0"):
            print(predict.remote(m, scan, True, tta))
        evaluate.remote("cas_net,nnunet_f0", "smoke", [scan])
    elif step == "run":
        print(run.remote(methods, split, tta))
    elif step == "evaluate":
        ids = [s.strip() for s in scan_ids.split(",") if s.strip()] or None
        evaluate.remote(methods, split, ids)
    else:
        raise SystemExit(
            f"unknown step {step!r}; choose check, folds, verify_folds, smoke, run or evaluate"
        )
