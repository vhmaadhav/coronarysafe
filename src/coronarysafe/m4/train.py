"""Train the M4 plausibility CNN on synthetic corruptions of TRAIN GT only (bf16 on MI300X).

  python -m coronarysafe.m4.train --data /root/data/m4/synth --out /root/data/m4/model

80/20 split by patient (scan id) inside the TRAIN scans. Early stopping on the 20% held-out
patients (loss). No val or test scan is ever read here. Writes model.pt, train_log.json and
sha256 of the weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .net import PlausibilityNet


def load_split(data_dir, seed=0, holdout_frac=0.2):
    files = sorted(Path(data_dir).glob("*.npz"))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files))
    n_hold = int(round(holdout_frac * len(files)))
    hold_idx = set(order[:n_hold].tolist())
    parts = {"train": ([], [], []), "hold": ([], [], [])}
    for i, f in enumerate(files):
        d = np.load(f)
        key = "hold" if i in hold_idx else "train"
        parts[key][0].append(d["X"])
        parts[key][1].append(d["critical"])
        parts[key][2].append(d["burden"])
    out = {}
    for k, (xs, cs, bs) in parts.items():
        out[k] = (np.concatenate(xs), np.concatenate(cs), np.concatenate(bs))
    return out, [f.stem for f in files]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = torch.device("cuda")

    split, scan_ids = load_split(a.data, a.seed)
    def ds(name):
        X, c, b = split[name]
        return TensorDataset(torch.from_numpy(X), torch.from_numpy(c.astype(np.float32)),
                             torch.from_numpy(np.log1p(b).astype(np.float32)))
    tr = DataLoader(ds("train"), batch_size=a.batch, shuffle=True, num_workers=2, drop_last=True)
    ho = DataLoader(ds("hold"), batch_size=a.batch, shuffle=False, num_workers=2)

    net = PlausibilityNet().to(dev).to(memory_format=torch.channels_last_3d)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    bce = nn.BCEWithLogitsLoss()
    huber = nn.SmoothL1Loss()

    best, best_ep, bad, log = float("inf"), -1, 0, []
    for ep in range(a.epochs):
        net.train()
        t0 = time.time()
        tot = 0.0
        for x, c, b in tr:
            x = x.to(dev, non_blocking=True).float().unsqueeze(1).contiguous(memory_format=torch.channels_last_3d)
            c, b = c.to(dev), b.to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logit, reg = net(x)
            loss = bce(logit.float(), c) + 0.5 * huber(reg.float(), b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        net.eval()
        hl, n = 0.0, 0
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for x, c, b in ho:
                x = x.to(dev).float().unsqueeze(1).contiguous(memory_format=torch.channels_last_3d)
                logit, reg = net(x)
                l = bce(logit.float(), c.to(dev)) + 0.5 * huber(reg.float(), b.to(dev))
                hl += l.item() * len(c)
                n += len(c)
        hl /= max(n, 1)
        log.append({"epoch": ep, "train_loss": tot / max(len(tr), 1), "holdout_loss": hl,
                    "sec": round(time.time() - t0, 1)})
        print(log[-1], flush=True)
        if hl < best - 1e-4:
            best, best_ep, bad = hl, ep, 0
            torch.save(net.state_dict(), out / "model.pt")
        else:
            bad += 1
            if bad >= a.patience:
                break

    weights = out / "model.pt"
    meta = {
        "best_epoch": best_ep, "best_holdout_loss": best, "n_train_scans": int(len(scan_ids) * 0.8),
        "n_variants_train": int(len(split["train"][1])), "n_variants_holdout": int(len(split["hold"][1])),
        "sha256_weights": _sha256(weights), "grid": 96, "voxel_mm": 2.0, "seed": a.seed,
        "input": "binary corrupted lumen, 96^3, 2 mm, centred on lumen bbox (coronarysafe.m4.build.to_grid)",
        "outputs": ["critical logit", "log1p burden"],
    }
    (out / "train_log.json").write_text(json.dumps({"meta": meta, "log": log}, indent=2))
    print("best", best_ep, best, flush=True)


if __name__ == "__main__":
    main()
