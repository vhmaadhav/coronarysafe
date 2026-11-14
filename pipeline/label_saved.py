"""Extract reference-derived labels and prediction-only graph features on CPU."""

from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from pipeline.label_core import _flatten, compute_case


def label_one(args):
    root, method, scan = args
    identity = {"method": method, "pred_method": method, "scan_id": scan}
    try:
        record = compute_case(root, method, scan, method, mask_only=True)
    except (OSError, EOFError, ValueError) as error:
        return _flatten({**identity, "status": "error", "error": type(error).__name__})
    return _flatten({**record, **identity, "status": "ok"})


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
    table = pd.DataFrame(rows)
    usable = table.loc[table.status.eq("ok")]
    usable.to_parquet(output, index=False)
    counts = {"submitted": len(table), "labelled": len(usable), "excluded": len(table)-len(usable)}
    print(f"{split} mask counts: {counts}", flush=True)
    return counts
