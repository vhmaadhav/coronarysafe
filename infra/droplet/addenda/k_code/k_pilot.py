"""Timing pilot: frozen CAS-Net inference (cas_net, tta, as in preds/cas_net) on 3 VAL scans via the
repo's predict_impl. Writes only to /root/data/addenda/K/pilot_preds (not the main preds tree)."""
import sys
import time

sys.path.insert(0, "/root/coronarysafe")
from pipeline.infer_core import CorePaths, predict_impl, split_ids  # noqa: E402

DATA = "/root/data"
P = CorePaths(data=DATA, pred=f"{DATA}/addenda/K/pilot_preds", repo_dir="/root/ImageCAS-X")
ids = sorted(split_ids(CorePaths(data=DATA, pred=f"{DATA}/preds"), "val"), key=int)[:3]
for sid in ids:
    t0 = time.time()
    r = predict_impl(P, "cas_net", str(sid), True, True, gpu_requested="MI300X")
    print(f"pilot scan={sid} status={r['status']} seconds={time.time() - t0:.1f}", flush=True)
