"""Assemble the v3 QA table (one row per unit) from the per-unit JSON records written by the label stage.

Used for the DEV split only (val). Columns follow eval.triage: scan_id, method, status, critical (bool),
burden, dice, severe, g_* (graph features) and u_* (uncertainty features).

  python -m coronarysafe.m4.qa_table --qa-root /root/data/qa/v3 --methods cas_net swin_unetr \
      --scans /root/data/imagecas_x/ImageCAS-X_dataset_v3/filelist/val.txt --out /root/data/m4/table_val.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_scans(path):
    return [str(int(x)) for x in Path(path).read_text().split() if x.strip()]


def build(qa_root, methods, scans):
    rows = []
    for m in methods:
        for s in scans:
            p = Path(qa_root) / m / f"{s}.json"
            if not p.exists():
                raise FileNotFoundError(str(p))
            d = json.loads(p.read_text())
            r = {
                "scan_id": s,
                "method": m,
                "status": d.get("status"),
                "critical": bool(d["critical"]),
                "burden": float(d["burden"]),
                "severe": bool(d["severe"]),
                "dice": float(d["dice"]),
            }
            for k, v in (d.get("graph") or {}).items():
                r[f"g_{k}"] = float(v) if v is not None else float("nan")
            for k, v in (d.get("uncertainty") or {}).items():
                r[f"u_{k}"] = float(v) if v is not None else float("nan")
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-root", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--scans", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = build(a.qa_root, a.methods, _read_scans(a.scans))
    if str(a.out).endswith(".parquet"):
        raise SystemExit("parquet needs pyarrow compatible with this container's numpy; use a .csv path")
    df.to_csv(a.out, index=False)
    print(df.shape, sorted(c for c in df.columns if c.startswith("u_")))


if __name__ == "__main__":
    main()
