"""svOneDSolver cross-check on GT trees: fallback (hemo1d) vs svOneDSolver, same tree and outlets.

Usage (droplet, cs-infer): python pipeline/svone_crosscheck.py --root /root/data --n 20 --workdir /root/data/cfd/svone
For each train GT scan: build the coupled hemo1d network, export it to a legacy svOneDSolver deck,
run OneDSolver, parse segment pressures, and compare distal FFR of LAD, LCx and RCA.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cfd_run import load_scan  # noqa: E402
from coronarysafe.analysis import hemo1d as h  # noqa: E402
from coronarysafe.analysis import svone_adapter as sv  # noqa: E402
from coronarysafe.analysis.harm import nearest_gt_labels  # noqa: E402
from coronarysafe.graph import branch_graph  # noqa: E402

SOLVER = "/root/coronarysafe-cfd/third_party/svOneDSolver_build/bin/OneDSolver"


def hemo_element_ffr(gt_crop, sp, ao):
    lab = nearest_gt_labels(gt_crop, sp)
    mask = gt_crop > 0
    B = branch_graph(mask, sp, prune_mm=h.SPUR_MM)
    roots, method = h.pick_roots(B, None if ao is None else None)
    net, _ = h.build_tree(B, roots, sp, lab, h.H_MM, mask.astype(float))
    if net is None:
        return None, None, method
    V = h.volume_ml(net)
    h.assign_outlets(net, h.K_FLOW * V ** 0.75 * 1e-6)
    Q, x = h.solve(net)
    return net, np.clip(x / h.P_AORTA, 0, 1), method


def vessel_distal(net, ffr):
    out = {}
    for name, lab in h.FLOW_VESSELS.items():
        idx = np.where(net["label"] == lab)[0]
        if idx.size == 0:
            out[name] = None
            continue
        out[name] = {"hemo": float(ffr[idx[np.argmax(net["depth"][idx])]])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/data")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--workdir", default="/root/data/cfd/svone")
    ap.add_argument("--scan", default="")
    args = ap.parse_args()
    root = Path(args.root)
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    if args.scan:
        scans = [args.scan]
    else:
        ids = [s.strip() for s in (root / "m4/train_scans.txt").read_text().split() if s.strip()]
        scans = ids[: args.n]
    results = {}
    for scan in scans:
        gt, sp, aorta = load_scan(root, scan)
        from coronarysafe.analysis.hemo1d import crop_inputs
        g, _, ao, _ = crop_inputs(gt, np.zeros(gt.shape, dtype=bool), aorta, sp)
        net, ffr, method = hemo_element_ffr(g, sp, ao)
        if net is None:
            results[scan] = {"status": "no_tree"}
            continue
        try:
            deck, meta = sv.build_deck(net, name=f"s{scan}")
        except ValueError as e:
            results[scan] = {"status": f"adapter: {e}"}
            continue
        sdir = work / scan
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "model.in").write_text(deck)
        proc = subprocess.run([SOLVER, "model.in"], cwd=str(sdir), capture_output=True, text=True, timeout=3600)
        files = sorted(os.listdir(sdir))
        rec = {"status": f"solver_rc_{proc.returncode}", "files": files, "stderr_tail": proc.stderr[-400:]}
        hd = vessel_distal(net, ffr)
        try:
            pr = sv.read_pressures(str(sdir), [0] + [meta["seg_of"][c] for c in range(len(meta["chains"]))] if meta["trunk"] else [meta["seg_of"][c] for c in range(len(meta["chains"]))])
            sv_ffr = sv.ffr_by_element(net, meta, pr)
            for name, d in hd.items():
                if d is None:
                    continue
                idx = np.where(net["label"] == h.FLOW_VESSELS[name])[0]
                dist = idx[np.argmax(net["depth"][idx])]
                d["svone"] = float(sv_ffr[dist])
                d["abs_diff"] = abs(d["svone"] - d["hemo"])
            rec["status"] = "ok"
        except (FileNotFoundError, ValueError, IndexError) as e:
            rec["status"] = f"parse: {e}"
        rec["vessels"] = hd
        rec["rooting"] = method
        results[scan] = rec
        print(scan, rec["status"], flush=True)
    (work / "crosscheck.json").write_text(json.dumps(results, indent=1))
    print("wrote", work / "crosscheck.json", flush=True)


if __name__ == "__main__":
    main()
