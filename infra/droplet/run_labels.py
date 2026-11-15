"""CPU labelling on the droplet: the same QA code as infra/modal/label_features.py, in a process pool.

Writes <data>/qa/<QA_VERSION>/<method>/<scan>.json per case (idempotent: an existing
status=ok record is skipped), then the split report and parquet table. Errors are recorded per
case and do not stop the pool. nnunet_oof is labelled from each scan's out-of-fold prediction
(nnunet_f<k>), as on Modal.

Usage (droplet, from /root/coronarysafe; infra/droplet/run_labels.sh wraps this):
    python infra/droplet/run_labels.py --methods cas_net,swin_unetr,ffr_unet,nnunet_oof --split test
    python infra/droplet/run_labels.py --methods cas_net --split val --limit 1 --workers 1   (smoke)
Flags: --workers 0 (auto: min(16, cpus-4)), --force, --limit N, --data /root/data
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

# One BLAS thread per process: the pool provides the parallelism.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.infer_core import OOF_METHOD, CorePaths, fold_table, split_ids  # noqa: E402
from pipeline.label_core import (  # noqa: E402
    QA_VERSION,
    process_case,
    summarise_local,
)

POOL_CAP = 16  # 20 vCPU droplet: 16 labelling processes leave room for the GPU feeders


def _job(args):
    root, method, scan, pred_method = args
    try:
        res = process_case(root, method, scan, pred_method=pred_method)
    except Exception as exc:  # process_case records errors itself; this catches anything else
        return {"status": "error", "method": method, "scan_id": scan, "error": repr(exc)}
    return res


def build_items(root: str, methods: list, split: str, limit: int):
    scans = list(split_ids(CorePaths(data=root, pred=f"{root}/preds"), split))
    if limit:
        scans = scans[:limit]
    table = fold_table(CorePaths(data=root, pred=f"{root}/preds")) if OOF_METHOD in methods else {}
    items, skipped = [], 0
    for m in methods:
        for s in scans:
            if m == OOF_METHOD:
                k = table.get(str(s))
                if k is None:
                    skipped += 1
                    continue
                items.append((root, OOF_METHOD, str(s), f"nnunet_f{k}"))
            else:
                items.append((root, m, str(s), None))
    return items, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="cas_net,swin_unetr,ffr_unet,nnunet_oof")
    ap.add_argument("--split", default="test")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="recompute even when status=ok exists")
    ap.add_argument("--data", default=os.environ.get("DATA_ROOT", "/root/data"))
    args = ap.parse_args()

    root = args.data
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    items, no_fold = build_items(root, methods, args.split, args.limit)
    if no_fold:
        print(f"[labels] {OOF_METHOD}: {no_fold} scans have no nnU-Net fold, skipped", flush=True)
    if args.force:
        for root_, m, s, _ in items:
            out = Path(root_) / "qa" / QA_VERSION / m / f"{int(s)}.json"
            out.unlink(missing_ok=True)

    cpus = os.cpu_count() or 4
    workers = args.workers or max(1, min(POOL_CAP, cpus - 4))
    print(
        f"[labels] {len(items)} cases, methods={methods} split={args.split}, {workers} workers",
        flush=True,
    )
    t0 = time.time()
    counts = {"ok": 0, "skipped": 0, "error": 0}
    errors = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_job, items, chunksize=1), 1):
            st = res.get("status", "error")
            counts[st if st in counts else "error"] = counts.get(st, 0) + 1
            if st == "error":
                errors.append(res)
                print(
                    f"[labels] ERROR {res.get('method')}/{res.get('scan_id')}: {res.get('error')}",
                    flush=True,
                )
            if i % 25 == 0 or i == len(items):
                print(f"[labels] {i}/{len(items)} {counts} {time.time() - t0:.0f}s", flush=True)

    report = summarise_local(root, methods, args.split)
    out = Path(root) / "qa" / QA_VERSION / f"report_{args.split}.txt"
    out.write_text(report + "\n")
    print(report)
    runs = Path(root) / "qa" / QA_VERSION / "_runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"labels_{args.split}_{int(time.time())}.json").write_text(
        json.dumps(
            {
                "methods": methods,
                "split": args.split,
                "counts": counts,
                "errors": errors[:50],
                "workers": workers,
                "wall_s": round(time.time() - t0, 1),
            },
            indent=2,
            default=str,
        )
    )
    print(f"[labels] counts {counts}; wall {time.time() - t0:.0f}s; report {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
