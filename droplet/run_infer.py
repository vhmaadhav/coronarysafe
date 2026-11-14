"""Job queue for inference on the MI300X droplet (spot or on-demand, preemption-safe).

A job is (method, scan, tta). Jobs come from --methods x --split (nnunet_oof expands to each
scan's out-of-fold model), or from --verify16 (16 train scans x 5 nnU-Net folds, then the OOF
check). Worker processes share the single GPU and pin their torch threads. predict_impl is
idempotent (meta.json), so re-running a queue item only computes what is missing. A failed job
is recorded and does not stop the others. Progress rows go to docs/RUNLOG.md every --progress-s
seconds, and every item ends with a summary row.

Budget: the running cost is (now - /root/.droplet_start) x GPU_RATE. Above --budget-usd the
workers are terminated and the process exits 3 (hard stop). The clock runs while the droplet
exists.

Usage (on the droplet, from /root/coronarysafe; droplet/run_all.sh calls this):
    python droplet/run_infer.py --verify16 --name verify_folds
    python droplet/run_infer.py --methods nnunet_f0,nnunet_f1 --split test --name nnunet_test
    python droplet/run_infer.py --dry-run --methods cas_net --split test
    python droplet/run_infer.py --methods cas_net --split test --limit 1 --evaluate --name smoke
    python droplet/run_infer.py --summary

Flags: --tta --workers 0(auto) --budget-usd 90 --progress-s 300 --data /root/data --limit N
       --evaluate --runlog PATH
Exit codes: 0 ok, 1 workers died, 2 bad arguments, 3 budget hard stop, 4 dry-run missing inputs.
"""

import argparse
import csv
import glob
import json
import multiprocessing as mp
import os
import queue
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.infer_core import (  # noqa: E402
    N_FOLDS,
    NET_WEIGHTS,
    NNUNET_METHODS,
    OOF_METHOD,
    CorePaths,
    eval_scan,
    fake_mode,
    find_ct,
    fold_table,
    parse_methods,
    predict_impl,
    print_verify_table,
    split_ids,
    utc_stamp,
    verify_scans,
    verify_summary,
)

START_FILE = Path("/root/.droplet_start")
GPU_LABEL = "MI300X"
HARD_STOP_RC = 3
DRY_RUN_RC = 4
# RAM and VRAM estimates per worker (predictor plus preprocessing). Measure on the first run.
RAM_GB_NNUNET = 12
RAM_GB_NET = 8
VRAM_GB_PER_WORKER = 24


def start_epoch() -> float:
    if os.environ.get("DROPLET_START_EPOCH"):
        return float(os.environ["DROPLET_START_EPOCH"])
    if START_FILE.exists():
        return float(START_FILE.read_text().strip())
    uptime = float(Path("/proc/uptime").read_text().split()[0])
    print(f"[run_infer] WARNING {START_FILE} missing; counting from boot", flush=True)
    return time.time() - uptime


def total_hours() -> float:
    return (time.time() - start_epoch()) / 3600.0


def running_cost() -> float:
    return total_hours() * float(os.environ.get("GPU_RATE", "1.99"))


def runlog_row(path: Path, *cols) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(str(c) for c in cols) + " |\n")


def vram_gb() -> float:
    """Largest GPU VRAM from amdgpu sysfs. GPU_VRAM_GB overrides (the MI300X has 192 GB)."""
    if os.environ.get("GPU_VRAM_GB"):
        return float(os.environ["GPU_VRAM_GB"])
    vals = []
    for p in glob.glob("/sys/class/drm/card*/device/mem_info_vram_total"):
        try:
            vals.append(int(Path(p).read_text().strip()) / 1e9)
        except (OSError, ValueError):
            pass
    return max(vals) if vals else 0.0


def mem_available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    return 0.0


def plan_workers(requested: int, methods: list) -> tuple:
    """(workers, threads_per_worker, note). Auto (requested 0): min(6, VRAM cap, RAM cap).

    nnU-Net jobs use more RAM per worker, so the RAM cap uses that estimate for such items.
    Threads leave 4 vCPUs for the parent process and the data loaders.
    """
    cpus = os.cpu_count() or 4
    gpu = vram_gb()
    vram_cap = max(1, int(gpu // VRAM_GB_PER_WORKER)) if gpu else 6
    base = requested if requested > 0 else min(6, vram_cap)
    per = RAM_GB_NNUNET if any(m.startswith("nnunet") for m in methods) else RAM_GB_NET
    ram_cap = max(1, int(mem_available_gb() * 0.8 // per))
    workers = max(1, min(base, vram_cap, ram_cap))
    threads = max(1, (cpus - 4) // workers)
    note = f"vram_cap={vram_cap} ram_cap={ram_cap} (per-worker {per} GB) cpus={cpus}"
    return workers, threads, note


def worker(P: CorePaths, jobq, resq, threads: int) -> None:
    """Loads model bundles lazily and keeps them for the life of the process."""
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    try:
        import torch

        torch.set_num_threads(threads)
    except ImportError:
        pass  # fake mode and dry runs have no torch
    while True:
        job = jobq.get()
        if job is None:
            break
        method, scan, tta = job
        try:
            r = predict_impl(P, method, scan, True, tta, gpu_requested=GPU_LABEL)
            resq.put((method, scan, r["status"], None))
        except BaseException as exc:  # one bad job must not stop the queue
            resq.put((method, scan, "failed", f"{type(exc).__name__}: {exc}"))


def build_jobs(P: CorePaths, methods: list, split: str, tta: bool, scans=None, limit: int = 0):
    ids = list(scans) if scans is not None else split_ids(P, split)
    if limit:
        ids = ids[:limit]
    table = fold_table(P) if OOF_METHOD in methods else {}
    jobs, no_fold = [], 0
    for m in methods:
        if m == OOF_METHOD:
            for s in ids:
                k = table.get(str(s))
                if k is None:
                    no_fold += 1
                    continue
                jobs.append((f"nnunet_f{k}", s, tta))
        else:
            jobs.extend((m, s, tta) for s in ids)
    return jobs, no_fold


def check_inputs(P: CorePaths, jobs: list) -> list:
    """Every path the jobs will read. Returns the missing ones (empty when all exist)."""
    missing = []
    for fl in ("train.txt", "val.txt", "test.txt"):
        if not (Path(P.filelist_dir) / fl).exists():
            missing.append(f"filelist {P.filelist_dir}/{fl}")
    used = {m for m, _, _ in jobs}
    for m in sorted(used):
        if m in NET_WEIGHTS and not (Path(P.weights) / NET_WEIGHTS[m]).exists():
            missing.append(f"weights {P.weights}/{NET_WEIGHTS[m]}")
        if m in NNUNET_METHODS:
            k = int(m.rsplit("_f", 1)[1])
            ck = Path(P.nnunet_model_dir) / f"fold_{k}" / "checkpoint_best.pth"
            if not ck.exists():
                missing.append(f"nnU-Net fold checkpoint {ck}")
    if any(m in NNUNET_METHODS for m in used):
        for f in ("dataset.json", "plans.json"):
            if not (Path(P.nnunet_model_dir) / f).exists():
                missing.append(f"nnU-Net {P.nnunet_model_dir}/{f}")
    if any(m in NET_WEIGHTS for m in used) and not (Path(P.repo_dir) / "configs").exists():
        missing.append(f"ImageCAS-X configs under {P.repo_dir}")
    for s in sorted({str(s) for _, s, _ in jobs}):
        try:
            find_ct(P, s)
        except (FileNotFoundError, RuntimeError) as exc:
            missing.append(str(exc))
    return missing


def run_queue(P, jobs, label, workers, budget, progress_s, runlog) -> dict:
    """Returns a record. record['rc'] is HARD_STOP_RC when the budget was exceeded."""
    total = len(jobs)
    methods = sorted({m for m, _, _ in jobs})
    workers, threads, note = plan_workers(workers, methods)
    started = time.time()
    stamp_start = utc_stamp()
    h0 = total_hours()
    runlog_row(
        runlog,
        stamp_start,
        f"start {label}",
        f"{total} jobs, {workers} workers x {threads} threads ({note})",
        "-",
        f"{h0:.3f}",
        f"{running_cost():.2f}",
        "running",
    )
    print(
        f"[run_infer] {label}: {total} jobs, {workers} workers x {threads} threads, "
        f"budget ${budget}; {note}",
        flush=True,
    )

    ctx = mp.get_context("spawn")
    jobq, resq = ctx.Queue(), ctx.Queue()
    for j in jobs:
        jobq.put(j)
    for _ in range(workers):
        jobq.put(None)
    procs = [
        ctx.Process(target=worker, args=(P, jobq, resq, threads), daemon=True)
        for _ in range(workers)
    ]
    for p in procs:
        p.start()

    counts = {"done": 0, "skipped": 0, "failed": 0}
    failures = []
    rc = 0
    last_progress = time.time()
    while counts["done"] < total:
        try:
            method, scan, status, err = resq.get(timeout=5)
        except queue.Empty:
            pass
        else:
            counts["done"] += 1
            if status == "failed":
                counts["failed"] += 1
                failures.append({"method": method, "scan_id": scan, "error": err})
                print(f"[run_infer] FAILED {method}/{scan}: {err}", flush=True)
            elif status == "skipped":
                counts["skipped"] += 1
        now = time.time()
        cost = running_cost()
        if cost > budget:
            for p in procs:
                p.terminate()
            rc = HARD_STOP_RC
            runlog_row(
                runlog,
                utc_stamp(),
                f"hard stop {label}",
                f"cost ${cost:.2f} > budget ${budget:.2f}; workers terminated; "
                f"done {counts['done']}/{total}",
                f"{(now - started) / 3600:.3f}",
                f"{total_hours():.3f}",
                f"{cost:.2f}",
                rc,
            )
            print(f"[run_infer] HARD STOP: cost ${cost:.2f} exceeds ${budget:.2f}", flush=True)
            break
        if now - last_progress >= progress_s:
            runlog_row(
                runlog,
                utc_stamp(),
                f"progress {label}",
                f"done {counts['done']}/{total}, skipped {counts['skipped']}, "
                f"failed {counts['failed']}",
                f"{(now - started) / 3600:.3f}",
                f"{total_hours():.3f}",
                f"{cost:.2f}",
                "running",
            )
            print(f"[run_infer] progress {counts['done']}/{total} cost ${cost:.2f}", flush=True)
            last_progress = now
        if not any(p.is_alive() for p in procs) and counts["done"] < total:
            missing = total - counts["done"]
            counts["failed"] += missing
            failures.append({"error": f"{missing} jobs never ran: all workers exited"})
            rc = 1
            break

    for p in procs:
        p.join(timeout=30)
    wall = (time.time() - started) / 3600.0
    word = {0: "ok", 1: "workers died", HARD_STOP_RC: "hard stop"}.get(rc, str(rc))
    runlog_row(
        runlog,
        utc_stamp(),
        f"summary {label}",
        f"{word}: done {counts['done']}/{total}, skipped {counts['skipped']}, "
        f"failed {counts['failed']}",
        f"{wall:.3f}",
        f"{total_hours():.3f}",
        f"{running_cost():.2f}",
        rc,
    )
    record = {
        "label": label,
        "started_utc": stamp_start,
        "jobs": total,
        "workers": workers,
        "threads": threads,
        "counts": counts,
        "failures": failures,
        "rc": rc,
        "gpu": GPU_LABEL,
        "cost_usd_at_end": running_cost(),
    }
    print(f"[run_infer] {label} finished rc={rc} counts={counts}", flush=True)
    return record


def verify16(P, workers, budget, progress_s, runlog) -> int:
    scans = verify_scans(P)
    jobs, _ = build_jobs(P, NNUNET_METHODS, "train", False, scans=scans)
    rec = run_queue(P, jobs, "verify_folds", workers, budget, progress_s, runlog)
    if rec["rc"] != 0:
        return rec["rc"]
    table = fold_table(P)
    dice = {s: [float("nan")] * N_FOLDS for s in scans}
    for s in scans:
        for k, m in enumerate(NNUNET_METHODS):
            r = eval_scan(P, m, s)
            if r.get("dice") is not None:
                dice[s][k] = float(r["dice"])
    summary = verify_summary(scans, {s: table[s] for s in scans}, dice)
    print_verify_table(summary)
    summary.update({"seed": 0, "failures": rec["failures"]})
    out = Path(P.meta_root) / "fold_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"[run_infer] wrote {out}")
    return 0


def evaluate_jobs(P, jobs, label) -> None:
    """Per-scan metrics against GT, written to <eval>/<label>.csv. Prints the mean per method."""
    rows = []
    for m, s, tta in jobs:
        r = eval_scan(P, m, s)
        rows.append(
            {
                "method": m,
                "scan_id": s,
                "tta": tta,
                "status": r.get("status"),
                "dice": r.get("dice"),
                "cl_dice": r.get("cl_dice"),
            }
        )
    out = Path(P.eval_root) / f"{label}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "scan_id", "tta", "status", "dice", "cl_dice"])
        w.writeheader()
        w.writerows(rows)
    for m in sorted({r["method"] for r in rows}):
        vals = [r["dice"] for r in rows if r["method"] == m and isinstance(r["dice"], float)]
        mean = sum(vals) / len(vals) if vals else float("nan")
        print(f"[eval] {m}: dice mean {mean:.4f} over {len(vals)} scans -> {out}", flush=True)


def summary_row(P, runlog) -> None:
    counts = {}
    for meta in glob.glob(f"{P.pred}/*/*/meta.json"):
        m = Path(meta).parent.parent.name
        counts[m] = counts.get(m, 0) + 1
    text = ", ".join(f"{m}={n}" for m, n in sorted(counts.items())) or "none"
    runlog_row(
        runlog,
        utc_stamp(),
        "summary predictions on disk",
        text,
        "-",
        f"{total_hours():.3f}",
        f"{running_cost():.2f}",
        "-",
    )
    print(f"[run_infer] predictions on disk: {text}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--verify16", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="first N scans of the split (smoke runs)")
    ap.add_argument("--name", default="")
    ap.add_argument("--workers", type=int, default=0, help="0 = auto (VRAM and RAM caps, max 6)")
    ap.add_argument("--budget-usd", type=float, default=float(os.environ.get("BUDGET_USD", "90")))
    ap.add_argument("--progress-s", type=float, default=300.0)
    ap.add_argument("--data", default=os.environ.get("DATA_ROOT", "/root/data"))
    ap.add_argument("--runlog", default=str(ROOT / "docs" / "RUNLOG.md"))
    args = ap.parse_args()

    data = args.data
    P = CorePaths(
        data=data,
        pred=f"{data}/preds",
        device="cpu" if fake_mode() else "cuda",
        repo_dir=os.environ.get("IMAGECAS_DIR", "/root/ImageCAS-X"),
    )
    runlog = Path(args.runlog)

    if args.summary:
        summary_row(P, runlog)
        return 0

    if args.verify16:
        if args.dry_run:
            scans = verify_scans(P)
            jobs, _ = build_jobs(P, NNUNET_METHODS, "train", False, scans=scans)
            missing = check_inputs(P, jobs)
            print(
                f"[dry-run] verify_folds: {len(jobs)} jobs over {len(scans)} train scans; "
                f"missing {len(missing)}"
            )
            for x in missing:
                print("  MISSING", x)
            return DRY_RUN_RC if missing else 0
        return verify16(P, args.workers, args.budget_usd, args.progress_s, runlog)

    methods = parse_methods(args.methods, allow_oof=True)
    jobs, no_fold = build_jobs(P, methods, args.split, args.tta, limit=args.limit)
    if no_fold:
        print(f"[run_infer] {OOF_METHOD}: skipped {no_fold} scans with no nnU-Net fold", flush=True)
    label = args.name or f"{'+'.join(methods)}@{args.split}{'-tta' if args.tta else ''}"

    if args.dry_run:
        by_method = {}
        for m, _, _ in jobs:
            by_method[m] = by_method.get(m, 0) + 1
        missing = check_inputs(P, jobs)
        _, _, note = plan_workers(args.workers, sorted(by_method))
        print(
            f"[dry-run] {label}: {len(jobs)} jobs {by_method} tta={args.tta} "
            f"budget=${args.budget_usd} (workers auto: {note})"
        )
        for x in missing:
            print("  MISSING", x)
        print(f"[dry-run] inputs {'OK' if not missing else 'MISSING ' + str(len(missing))}")
        return DRY_RUN_RC if missing else 0

    rec = run_queue(P, jobs, label, args.workers, args.budget_usd, args.progress_s, runlog)
    runs = Path(P.runs_dir)
    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"{utc_stamp()}_{label.replace('/', '_')}.json").write_text(json.dumps(rec, indent=2))
    if args.evaluate and rec["rc"] == 0:
        evaluate_jobs(P, jobs, label)
    return rec["rc"]


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
