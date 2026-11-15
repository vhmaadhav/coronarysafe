"""Addendum K labelling: the frozen per-unit pipeline (pipeline.label_core.process_case, the same call
droplet/run_labels.py makes) for one method over one split, in a spawn pool. Writes
qa/v3/<method>/<scan>.json. Deliberately does NOT call summarise_local (no report/table on test).
Usage: python -I k_label.py <method> <split> [workers]"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, "/root/coronarysafe")
from pipeline.infer_core import CorePaths, split_ids  # noqa: E402
from pipeline.label_core import process_case  # noqa: E402

DATA = "/root/data"


def _job(args):
    method, sid = args
    try:
        return process_case(DATA, method, sid, pred_method=None)
    except Exception as exc:
        return {"status": "error", "method": method, "scan_id": sid, "error": repr(exc)}


def main():
    from multiprocessing import get_context
    method, split = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    ids = [str(s) for s in split_ids(CorePaths(data=DATA, pred=f"{DATA}/preds"), split)]
    t0 = time.time()
    counts = {}
    with get_context("spawn").Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_job, [(method, s) for s in ids], chunksize=1), 1):
            st = res.get("status", "error")
            counts[st] = counts.get(st, 0) + 1
            if st == "error":
                print(f"k_label ERROR {method}/{res.get('scan_id')}: {res.get('error')}", flush=True)
            if i % 20 == 0 or i == len(ids):
                print(f"k_label {method} {i}/{len(ids)} {counts} {time.time() - t0:.0f}s", flush=True)
    print(f"k_label: {method} {split} done counts={counts} wall={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
