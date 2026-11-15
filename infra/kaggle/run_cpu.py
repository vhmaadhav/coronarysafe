"""Reproduce CoronarySafe rankings on Kaggle CPU from private feature tables."""

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def main():
    search_root = Path(os.environ.get("CORONARYSAFE_INPUT", "/kaggle/input"))
    data = search_root
    if not (data / "pilot.parquet").is_file():
        matches = list(data.rglob("pilot.parquet"))
        if len(matches) != 1:
            raise RuntimeError(f"expected one pilot table, found {len(matches)}")
        data = matches[0].parent

    output = Path(os.environ.get("CORONARYSAFE_OUTPUT", "/kaggle/working"))
    output.mkdir(parents=True, exist_ok=True)
    code = output / "coronarysafe_code"
    if (data / "code.bundle").is_file():
        with zipfile.ZipFile(data / "code.bundle") as archive:
            archive.extractall(code)
    else:
        matches = list(data.rglob("coronarysafe/__init__.py"))
        if len(matches) != 1:
            raise RuntimeError(f"expected one extracted code bundle, found {len(matches)}")
        shutil.copytree(matches[0].parent.parent, code, dirs_exist_ok=True)
    if any(Path("/dev").glob("nvidia[0-9]*")):
        raise RuntimeError("expected a Kaggle CPU session")
    environment = dict(os.environ, PYTHONPATH=str(code), OPENBLAS_NUM_THREADS="1")
    command = [sys.executable, "-m", "coronarysafe.eval.triage"]
    subprocess.run(
        command + [str(data / "pilot.parquet"), "--output", str(output / "pilot-triage.json")],
        env=environment, check=True,
    )

    dev, transfer = (data / name for name in ("development.parquet", "transfer.parquet"))
    mask_inputs = list(search_root.rglob("saved-inputs.bundle"))
    if not mask_inputs:
        mask_inputs = [p.parent.parent for p in search_root.rglob("cohort/test_ffr_unet.txt")]
    if len(mask_inputs) > 1:
        raise RuntimeError("expected only one saved mask bundle")
    mask_counts = {}
    if mask_inputs:
        sys.path.insert(0, str(code))
        from pipeline.label_saved import cohort

        with tempfile.TemporaryDirectory() as scratch:
            scratch = Path(scratch)
            mask_root = mask_inputs[0]
            if mask_root.is_file():
                with zipfile.ZipFile(mask_root) as archive:
                    archive.extractall(scratch / "masks")
                mask_root = scratch / "masks"
            dev, transfer = scratch / "dev.parquet", scratch / "transfer.parquet"
            mask_counts = {
                "development": cohort(mask_root, "val", ("cas_net", "swin_unetr"), dev),
                "transfer": cohort(mask_root, "test", ("ffr_unet",), transfer),
            }
            run_transfer(command, dev, transfer, output, environment)
    elif dev.is_file() and transfer.is_file():
        run_transfer(command, dev, transfer, output, environment)
    else:
        print("Paired development and transfer feature tables were unavailable.")
    results = {
        "backend": "kaggle_cpu",
        "mask_counts": mask_counts,
        "versions": {p: importlib.metadata.version(p) for p in
                     ("numpy", "scipy", "pandas", "pyarrow", "scikit-learn")},
    }
    if (data / "smoke.parquet").is_file():
        import pandas as pd

        smoke = pd.read_parquet(data / "smoke.parquet")
        results["ffr_unet_smoke"] = {
            "rows": len(smoke),
            "case_status": smoke["status"].value_counts().to_dict(),
            "critical": int(smoke["critical"].sum()),
            "mean_dice": float(smoke["dice"].mean()),
            "min_dice": float(smoke["dice"].min()),
            "max_dice": float(smoke["dice"].max()),
        }
    (output / "run-summary.json").write_text(json.dumps(results, indent=2) + "\n")


def run_transfer(command, dev, transfer, output, environment):
    subprocess.run(
        command + [
            str(dev), "--independent-test", str(transfer),
            "--output", str(output / "transfer-triage.json"),
        ],
        env=environment, check=True,
    )


if __name__ == "__main__":
    main()
