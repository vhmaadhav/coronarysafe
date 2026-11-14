"""Extract reference-derived labels and prediction-only graph features on CPU."""

from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from pipeline.label_core import _flatten, compute_case


def label_one(args):
    root, method, scan = args
    return _flatten({
        **compute_case(root, method, scan, method, mask_only=True),
        "method": method,
        "pred_method": method,
        "scan_id": scan,
        "status": "ok",
    })


def cohort(root, split, methods, output):
    jobs = []
    for method in methods:
        manifest = root / "cohort" / f"{split}_{method}.txt"
        if manifest.is_file():
            for scan in manifest.read_text().split():
                if not scan.isdecimal():
                    raise ValueError("invalid case ID in cohort manifest")
                jobs.append((root, method, scan))
    if not jobs:
        raise ValueError(f"no saved {split} predictions")
    print(f"Labelling {len(jobs)} saved {split} masks on CPU", flush=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(label_one, jobs))
    pd.DataFrame(rows).to_parquet(output, index=False)
    print(f"Labelled {len(rows)} of {len(jobs)} saved {split} masks", flush=True)
