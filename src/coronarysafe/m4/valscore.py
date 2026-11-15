"""Frozen-model score extension on the VAL split (Addendum L phase 2, dev scope only).

VAL ONLY: --split accepts "val" alone; scan ids are asserted disjoint from filelist/test.txt.
For methods outside the original dev scope {cas_net, swin_unetr}, computes the FROZEN model's score:

  0. the frozen directory must contain frozen_model.json, m4_model.pt and lr_convention.json; all three
     are sha256-verified against Addendum H (table copied from m4/test_eval.py EXPECTED_SHA) before use.
     lr_convention is read ONLY from that directory.
  1. reuses coronarysafe.m4.score._unit_job for the per-unit record (m3 ostium features, v3 label,
     96^3 grid), wrapped per unit so an exception is recorded in that unit's JSON, not raised.
     Every unit JSON carries the frozen-bundle provenance; cached units without a matching
     provenance are refused.
  2. runs M4 on the chosen device (default cpu) to get m4_prob -> m4_logit with the frozen clamp.
  3. assembles the frozen 22-feature vector (g_* from qa/v3 graph dict, m3_* from the unit record,
     m4_logit). A feature whose SOURCE KEY IS ABSENT flags the unit (status missing_feature) and the
     unit is excluded from scores; the count is reported per feature. A present NaN value takes the
     frozen median imputation (as in the frozen apply_frozen), and nothing else is imputed.

Outputs: --out/scores.csv (method, scan_id, score) and --out/units.json.

  python -m coronarysafe.m4.valscore --split val --root /root/data --methods ffr_unet nnunet_f0 \
      nnunet_f1 nnunet_f2 nnunet_f3 nnunet_f4 --unit-dir /root/data/m4/val_units_L \
      --aorta-dir /root/data/m4/aorta --frozen-dir /root/data/m4/frozen --out /root/data/m4/val_L \
      --device cpu --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(REPO))

from coronarysafe.m4.score import _unit_job, load_model  # noqa: E402

SPLIT = "val"  # dev scope only
CODE_VERSION = "valscore-L-phase2-v2"
# Addendum H sha256 table: verbatim copy of m4/test_eval.py EXPECTED_SHA (not imported, to keep the
# sealed-test module out of the val path).
EXPECTED_SHA = {
    "frozen_model.json": "56a666ef39ed652b85e65fafb18101841dc8963cecff4b27df5ef18de0f7aa5d",
    "m4_model.pt": "baf841cfba4d0f08b21cbead0239e930d1f8db19a744720733b94afbfb30a266",
    "lr_convention.json": "53ea24bfdfc8978c6ec02b979789bda1a068dad527457cc1fad7b0d182a3f6db",
}
M4_CLAMP = 1e-6


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen_dir(frozen_dir: Path) -> dict:
    """Hard-fail unless all three frozen files match Addendum H. Returns {name: sha256}."""
    shas = {}
    for name, want in EXPECTED_SHA.items():
        got = sha256_of(Path(frozen_dir) / name)
        if got != want:
            raise SystemExit(f"ABORT: {name} sha256 {got} != Addendum H {want}")
        shas[name] = got
    frozen = json.loads((Path(frozen_dir) / "frozen_model.json").read_text())
    if frozen.get("m4_weights_sha256") != EXPECTED_SHA["m4_model.pt"]:
        raise SystemExit("ABORT: frozen_model.json m4_weights_sha256 disagrees with Addendum H")
    return shas


def provenance_for(shas: dict) -> dict:
    return {"split": SPLIT, "frozen_sha256": shas, "code": CODE_VERSION}


def m4_logit(prob: float) -> float:
    """Frozen convention (m4/test_eval.py unit_features): log(max(p, 1e-6) / max(1 - p, 1e-6))."""
    p = float(prob)
    return math.log(max(p, M4_CLAMP) / max(1.0 - p, M4_CLAMP))


def assemble_features(qa: dict, unit: dict, frozen_features: list):
    """Map the frozen feature names onto (qa unit JSON, score unit record).

    Returns (values, missing): values holds floats (NaN kept for the frozen imputation);
    missing lists features whose source key is absent (the unit is then flagged, not imputed).
    """
    graph = qa.get("graph") or {}
    m3 = unit.get("m3") or {}
    out, missing = {}, []
    for name in frozen_features:
        if name.startswith("g_"):
            src, key = graph, name[2:]
        elif name.startswith("m3_"):
            src, key = m3, name[3:]
        elif name == "m4_logit":
            if unit.get("m4_prob") is None:
                missing.append(name)
            else:
                out[name] = m4_logit(float(unit["m4_prob"]))
            continue
        else:
            raise ValueError(f"unknown frozen feature {name}")
        if key not in src or src[key] is None:
            missing.append(name)
        else:
            out[name] = float(src[key])
    return out, missing


def apply_frozen_lr(values: dict, frozen: dict) -> float:
    """Frozen impute-median (NaN only) -> standardise -> sigmoid(coef·x + intercept). No fitting ever."""
    feats = frozen["features"]
    impute = frozen["preprocessing"]["impute_median"]
    mean, std = frozen["preprocessing"]["scale_mean"], frozen["preprocessing"]["scale_std"]
    coef = frozen["logistic"]["coef"]
    intercept = float(frozen["logistic"]["intercept"])
    x = np.empty(len(feats), dtype=float)
    for i, name in enumerate(feats):
        if name not in values or values[name] is None:
            raise ValueError(f"feature {name} absent: flag the unit instead of scoring it")
        raw = float(values[name])
        if math.isnan(raw):  # frozen imputation applies to NaN values only
            raw = float(impute[name])
        if not math.isfinite(raw):
            raise ValueError(f"feature {name} is infinite")
        x[i] = (raw - mean[name]) / std[name]
    return float(
        1.0 / (1.0 + math.exp(-(intercept + float(x @ np.array([coef[n] for n in feats])))))
    )


def _safe_unit_job(args):
    """_unit_job per unit: exceptions are recorded in the unit JSON; the provenance is stored."""
    method, scan, root, aorta_dir, left_unit, outdir, provenance = args
    out_json = Path(outdir) / f"{method}__{scan}.json"
    try:
        _, _, ret = _unit_job((method, scan, root, aorta_dir, left_unit, outdir))
        if out_json.exists():
            rec = json.loads(out_json.read_text())
        else:
            rec = {"method": method, "scan": scan, "status": ret}
    except Exception as exc:  # recorded per unit, the run continues
        rec = {
            "method": method,
            "scan": scan,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}"[:300],
        }
    rec["provenance"] = provenance
    tmp = out_json.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, default=float))
    tmp.rename(out_json)
    return method, scan, rec.get("status")


def refuse_unmatched_cache(unit_dir: Path, jobs, provenance: dict) -> None:
    """Refuse cached unit JSONs whose provenance differs from the current frozen bundle."""
    bad = []
    for method, scan, *_ in jobs:
        jp = Path(unit_dir) / f"{method}__{scan}.json"
        if jp.exists():
            rec = json.loads(jp.read_text())
            if rec.get("provenance") != provenance:
                bad.append(jp.name)
    if bad:
        raise SystemExit(
            f"refusing {len(bad)} cached unit(s) without matching frozen provenance "
            f"(e.g. {bad[:3]}); use a new --unit-dir or delete them explicitly"
        )


def _m4_pass(device: str, net, unit_dir: Path, keys) -> dict:
    """M4 probability for each unit NPZ of this run (idempotent against existing units)."""
    import torch

    out = {}
    for method, scan in keys:
        jp = Path(unit_dir) / f"{method}__{scan}.json"
        if not jp.exists():
            continue
        rec = json.loads(jp.read_text())
        if rec.get("status") not in ("ok", "cached"):
            continue
        if "m4_prob" in rec:
            out[(method, scan)] = float(rec["m4_prob"])
            continue
        X = np.load(jp.with_suffix(".npz"))["X"]
        t = torch.from_numpy(X).float()[None, None].to(device)
        with torch.no_grad():
            logit, _ = net(t)
        p = float(torch.sigmoid(logit).item())
        rec["m4_prob"] = p
        tmp = jp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, default=float))
        tmp.rename(jp)
        out[(method, scan)] = p
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=[SPLIT], required=True, help="val only; test is refused")
    ap.add_argument("--root", default="/root/data")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--unit-dir", required=True)
    ap.add_argument("--aorta-dir", required=True)
    ap.add_argument("--frozen-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    if a.split == "test":
        raise SystemExit("valscore never runs on the test split")

    from pipeline.label_core import read_split

    root = Path(a.root)
    frozen_dir = Path(a.frozen_dir)
    shas = verify_frozen_dir(frozen_dir)
    provenance = provenance_for(shas)
    frozen = json.loads((frozen_dir / "frozen_model.json").read_text())
    left_unit = json.loads((frozen_dir / "lr_convention.json").read_text())["summary"]["left_world_unit"]

    scans = read_split(root, SPLIT)
    test_ids = set(read_split(root, "test"))
    overlap = set(scans) & test_ids
    if overlap:
        raise SystemExit(f"ABORT: {len(overlap)} val scan ids overlap filelist/test.txt")

    Path(a.unit_dir).mkdir(parents=True, exist_ok=True)
    jobs = [(m, s) for m in a.methods for s in scans]
    refuse_unmatched_cache(Path(a.unit_dir), jobs, provenance)

    args = [(m, s, a.root, a.aorta_dir, left_unit, a.unit_dir, provenance) for m, s in jobs]
    with Pool(a.workers) as pool:
        done = 0
        for method, scan, status in pool.imap_unordered(_safe_unit_job, args):
            done += 1
            if done % 20 == 0 or status not in ("ok", "cached"):
                print(f"[{done}/{len(jobs)}] {method} {scan} {status}", flush=True)

    import torch

    m4_path = frozen_dir / "m4_model.pt"
    net = load_model(m4_path, torch.device(a.device))
    probs = _m4_pass(a.device, net, Path(a.unit_dir), jobs)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, units = [], []
    missing_count: dict = {}
    for m, s in jobs:
        jp = Path(a.unit_dir) / f"{m}__{s}.json"
        rec = json.loads(jp.read_text()) if jp.exists() else {"method": m, "scan": s, "status": "missing_input"}
        rec["score"] = None
        if rec.get("status") not in ("ok", "cached") or (m, s) not in probs:
            units.append(rec)
            continue
        qa_path = root / "qa" / "v3" / m / f"{s}.json"
        if not qa_path.exists():
            rec["status"] = "missing_qa"
            units.append(rec)
            continue
        qa = json.loads(qa_path.read_text())
        values, missing = assemble_features(qa, rec, frozen["features"])
        if missing:
            rec["status"] = "missing_feature"
            rec["missing_features"] = missing
            for name in missing:
                missing_count[f"{m}:{name}"] = missing_count.get(f"{m}:{name}", 0) + 1
            units.append(rec)
            continue
        rec["score"] = apply_frozen_lr(values, frozen)
        units.append(rec)
        rows.append({"method": m, "scan_id": str(s), "score": rec["score"]})
    import pandas as pd

    pd.DataFrame(rows, columns=["method", "scan_id", "score"]).to_csv(out / "scores.csv", index=False)
    summary = {
        "provenance": provenance,
        "n_requested": len(jobs),
        "n_scored": len(rows),
        "status_counts": {k: sum(u.get("status") == k for u in units) for k in
                          sorted({str(u.get("status")) for u in units})},
        "missing_feature_counts": missing_count,
        "units": units,
    }
    (out / "units.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"units {len(jobs)}, scored {len(rows)}, excluded {len(jobs) - len(rows)}")
    print(json.dumps({"missing_feature_counts": missing_count}, indent=1))
    if not rows:
        raise SystemExit("no units scored — check qa/v3 coverage and unit errors in units.json")


if __name__ == "__main__":
    main()
