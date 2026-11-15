"""Prepare a private Kaggle CPU input dataset and notebook from saved QA records."""

import argparse
import importlib.metadata
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from coronarysafe.eval.triage import GRAPH, UNCERTAINTY, load_table

ROOT = Path(__file__).resolve().parents[2]


def transfer_row(record):
    return {
        "scan_id": str(record["scan_id"]),
        "method": record["method"],
        "status": record["status"],
        "critical": record["critical"],
        "severe": record["severe"],
        "dice": record["dice"],
        **{"g_" + key: value for key, value in record["graph"].items()},
        **{"u_" + key: value for key, value in record["uncertainty"].items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--third-model", type=Path)
    parser.add_argument("--smoke-record", type=Path, action="append", default=[])
    parser.add_argument("--owner", required=True)
    parser.add_argument("--dataset", default="coronarysafe-cpu-qa-inputs")
    parser.add_argument("--kernel", default="coronarysafe-cpu-validation")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--saved-mask-source", help="optional private Kaggle dataset slug")
    args = parser.parse_args()
    payload = args.bundle / "input"
    notebook = args.bundle / "notebook"
    payload.mkdir(parents=True, exist_ok=True)
    notebook.mkdir(parents=True, exist_ok=True)

    tables = {"pilot": load_table(args.pilot)}
    if args.smoke_record:
        records = [transfer_row(json.loads(path.read_text())) for path in args.smoke_record]
        tables["smoke"] = pd.DataFrame(records)
    for name, path in (
        ("development", args.development),
        ("transfer", args.third_model),
    ):
        if path is not None:
            tables[name] = load_table(path, require_uncertainty=False)

    identities = {}
    for name, table in tables.items():
        if name == "development":
            test_ids = set(tables["pilot"].scan_id)
            if "transfer" in tables:
                test_ids |= set(tables["transfer"].scan_id)
            if test_ids.intersection(table.scan_id):
                raise ValueError("development contains ImageCAS-X test patients")
        ids = []
        for original in table.scan_id:
            identity = str(original)
            ids.append(identities.setdefault(identity, f"p{len(identities) + 1:05d}"))
        table = table.drop(columns="scan_id").assign(patient_group=ids)
        columns = [
            "patient_group", "method", "status", "critical", "severe", "dice",
            "centerline_fallback", *GRAPH, *UNCERTAINTY,
        ]
        table[[column for column in columns if column in table]].to_parquet(
            payload / f"{name}.parquet", index=False,
        )

    with zipfile.ZipFile(payload / "code.bundle", "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted((ROOT / "src" / "coronarysafe").rglob("*.py")):
            archive.write(file, file.relative_to(ROOT / "src"))
        for name in (
            "pipeline/label_core.py", "configs/taxonomy_v3.yaml", "pipeline/label_saved.py",
        ):
            archive.write(ROOT / name, name)
    manifest = {name: len(table) for name, table in tables.items()}
    manifest["builder_versions"] = {
        package: importlib.metadata.version(package)
        for package in ("numpy", "pandas", "pyarrow", "scikit-learn")
    }
    (payload / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    dataset_id = f"{args.owner}/{args.dataset}"
    (payload / "dataset-metadata.json").write_text(json.dumps({
        "title": "CoronarySafe CPU QA inputs",
        "id": dataset_id,
        "licenses": [{"name": "other"}],
    }, indent=2) + "\n")
    shutil.copyfile(ROOT / "infra" / "kaggle" / "run_cpu.py", notebook / "run_cpu.py")
    sources = [dataset_id] + ([args.saved_mask_source] if args.saved_mask_source else [])
    (notebook / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{args.owner}/{args.kernel}",
        "title": "CoronarySafe CPU Validation",
        "code_file": "run_cpu.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": sources,
        "competition_sources": [],
        "kernel_sources": [],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
