"""Stage the coronary CTA data into the Modal Volume `coronarysafe-data`, mounted at /data.

Steps (see modal/README.md for the commands):
    labels   Zenodo 23894668: ImageCAS-X labels/filelists (+ weights). CPU, internet.
    volumes  Kaggle xiaoweixumedicalai/imagecas split archives, one container per 200-case group.
    inspect  Print the manifest and the weights tree.

Layout under /data:
    raw/zenodo/                 downloaded Zenodo zips (+ .verified markers)
    imagecas_x/                 unzipped ImageCAS-X dataset (labels, filelists, Descriptors)
    weights/                    unzipped pretrained weights, one folder per method
    imagecas/volumes/           NIfTI CT volumes moved out of the Kaggle archives
    imagecas/orig_labels/       original ImageCAS label NIfTIs, if the archives contain any
    imagecas/meta/              any non-NIfTI files shipped in the archives
    manifest/                   labels_tree.txt, volumes.csv, groups/<group>.json (per-group markers)
"""

import csv
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import modal

APP_NAME = "coronarysafe-stage-data"
VOLUME_NAME = "coronarysafe-data"
DATA = "/data"

ZENODO_RECORD = 23894668
ZENODO_FILES = ["ImageCAS-X_dataset_v3.zip", "pretrained_weights.zip"]

KAGGLE_DATASET = "xiaoweixumedicalai/imagecas"
GROUPS = ["1-200", "201-400", "401-600", "601-800", "801-1000"]

HOUR = 3600
KAGGLE_SECRET = "kaggle"  # provides KAGGLE_USERNAME / KAGGLE_KEY

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("p7zip-full", "unzip", "zip")
    .pip_install("kaggle==2.2.4", "six", "nibabel", "pandas", "openpyxl", "requests")
)


# ---------------------------------------------------------------------------
# Step 1: labels (Zenodo)
# ---------------------------------------------------------------------------

def _download_verified(url: str, dest: Path, size: int, md5: str) -> None:
    """Stream url to dest via a .part file; check byte count and md5 before renaming."""
    import requests

    part = dest.with_name(dest.name + ".part")
    digest = hashlib.md5()
    received = 0
    with requests.get(url, stream=True, timeout=(30, 300)) as resp:
        resp.raise_for_status()
        with open(part, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                digest.update(chunk)
                received += len(chunk)
    if received != size:
        raise RuntimeError(f"{dest.name}: got {received} bytes, Zenodo says {size}")
    if digest.hexdigest() != md5:
        raise RuntimeError(f"{dest.name}: md5 {digest.hexdigest()} != Zenodo {md5}")
    part.rename(dest)


def _unzip(archive: Path, target: Path, marker: Path) -> None:
    if marker.exists():
        print(f"skip unzip, marker present: {marker}")
        return
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(["unzip", "-q", "-o", str(archive), "-d", str(target)], check=True)
    marker.write_text(archive.name + "\n")


def _find_listing(root: str, depth: int = 4) -> str:
    result = subprocess.run(
        ["find", root, "-maxdepth", str(depth), "-printf", "%10s  %p\\n"],
        capture_output=True,
        text=True,
    )
    return result.stdout


def _print_label_summary(base: Path) -> None:
    print("=== directory tree (find, depth 4) ===")
    print(_find_listing(DATA, 4))

    print("=== filelist/*.txt line counts ===")
    for txt in sorted(base.rglob("*.txt")):
        if txt.parent.name == "filelist":
            with open(txt, "r", errors="replace") as fh:
                n = sum(1 for _ in fh)
            print(f"{n:8d}  {txt}")

    print("=== Descriptors (first rows, first 8 columns) ===")
    desc = sorted(list(base.rglob("Descriptors.xlsx")) + list(base.rglob("Descriptors.csv")))
    if not desc:
        print("no Descriptors.xlsx / .csv found under", base)
        return
    import pandas as pd

    path = desc[0]
    try:
        df = pd.read_excel(path) if path.suffix == ".xlsx" else pd.read_csv(path)
        print(f"{path}  shape={df.shape}")
        print(df.iloc[:5, :8].to_string())
    except Exception as exc:  # summary only; never fail the stage on a read problem
        print(f"could not read {path}: {exc}")


@app.function(image=image, volumes={DATA: volume}, timeout=2 * HOUR, cpu=2, memory=4096)
def stage_labels() -> None:
    import requests

    base = Path(DATA)
    raw = base / "raw" / "zenodo"
    manifest = base / "manifest"
    raw.mkdir(parents=True, exist_ok=True)
    manifest.mkdir(parents=True, exist_ok=True)

    meta = requests.get(f"https://zenodo.org/api/records/{ZENODO_RECORD}", timeout=60)
    meta.raise_for_status()
    files = {f["key"]: f for f in meta.json()["files"]}
    missing = [name for name in ZENODO_FILES if name not in files]
    if missing:
        raise RuntimeError(f"Zenodo record {ZENODO_RECORD} has no files {missing}; has {sorted(files)}")

    for name in ZENODO_FILES:
        info = files[name]
        dest = raw / name
        marker = raw / f".{name}.verified"
        if marker.exists() and dest.exists():
            print(f"skip download, verified: {dest}")
            continue
        md5 = info["checksum"].split(":")[-1]  # API gives "md5:<hex>"
        print(f"downloading {name} ({info['size']} bytes) ...")
        _download_verified(info["links"]["self"], dest, int(info["size"]), md5)
        marker.write_text(md5 + "\n")
        volume.commit()

    _unzip(raw / "ImageCAS-X_dataset_v3.zip", base / "imagecas_x", base / "imagecas_x" / ".unzipped")
    _unzip(raw / "pretrained_weights.zip", base / "weights", base / "weights" / ".unzipped")

    listing = _find_listing(DATA, 4)
    (manifest / "labels_tree.txt").write_text(listing)
    _print_label_summary(base)
    volume.commit()


# ---------------------------------------------------------------------------
# Step 2: volumes (Kaggle split archives)
# ---------------------------------------------------------------------------

def _write_kaggle_json() -> None:
    """Some kaggle CLI versions ignore env vars; also write ~/.kaggle/kaggle.json from the secret."""
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if not cfg.exists():
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(json.dumps({"username": os.environ["KAGGLE_USERNAME"], "key": os.environ["KAGGLE_KEY"]}))
        cfg.chmod(0o600)


def _kaggle_fetch(archives: Path, name: str) -> None:
    """Download one file of the Kaggle dataset. The CLI may wrap it as <name>.zip; unwrap that."""
    _write_kaggle_json()
    r = subprocess.run(
        ["kaggle", "datasets", "download", KAGGLE_DATASET, "-f", name, "-p", str(archives)],
        capture_output=True, text=True,
    )
    print(r.stdout[-2000:])
    if r.returncode != 0:
        raise RuntimeError(f"kaggle download {name} failed ({r.returncode}): {r.stderr[-3000:]}")
    plain = archives / name
    if plain.exists():
        return
    wrapper = archives / (name + ".zip")
    if wrapper.exists():
        subprocess.run(["unzip", "-q", "-o", str(wrapper), "-d", str(archives)], check=True)
        wrapper.unlink()
        if plain.exists():
            return
    raise RuntimeError(f"Kaggle download of {name} produced neither {plain} nor {wrapper}")


def _nifti_row(group: str, kind: str, path: Path) -> dict:
    import nibabel as nib

    row = {"group": group, "kind": kind, "file": str(path), "bytes": path.stat().st_size,
           "shape": "", "spacing": "", "error": ""}
    try:
        img = nib.load(str(path))  # header only; voxel data is not read
        row["shape"] = "x".join(str(s) for s in img.shape)
        row["spacing"] = "x".join(f"{z:.4g}" for z in img.header.get_zooms())
    except Exception as exc:
        row["error"] = str(exc)[:200]
    return row


@app.function(
    image=image,
    volumes={DATA: volume},
    secrets=[modal.Secret.from_name(KAGGLE_SECRET)],
    timeout=6 * HOUR,
    ephemeral_disk=512 * 1024,  # MiB; Modal minimum is 512 GiB. One group is ~18 GB of archives + extract
    cpu=4,
    memory=16384,
)
def _stage_group(group: str) -> list:
    base = Path(DATA)
    group_marker = base / "manifest" / "groups" / f"{group}.json"
    if group_marker.exists():
        print(f"skip group {group}, marker present")
        return json.loads(group_marker.read_text())

    scratch = Path("/scratch") / group
    shutil.rmtree(scratch, ignore_errors=True)
    archives = scratch / "archives"
    extract = scratch / "extract"
    archives.mkdir(parents=True)
    extract.mkdir()

    # The final part is .change2zip on Kaggle; the other parts are .z01..z04.
    parts = [f"{group}.z0{i}" for i in range(1, 5)] + [f"{group}.change2zip"]
    for name in parts:
        print(f"[{group}] downloading {name}")
        _kaggle_fetch(archives, name)
    final = archives / f"{group}.change2zip"
    final.rename(archives / f"{group}.zip")
    print(f"[{group}] archives: " + ", ".join(f"{p.name}={p.stat().st_size}" for p in sorted(archives.iterdir())))

    # 7z reads PKZIP split sets when the .zip and .z01.. parts sit together.
    r = subprocess.run(["7z", "x", str(archives / f"{group}.zip"), f"-o{extract}", "-y"])
    if r.returncode != 0:
        # Fallback: join the split set into one archive with Info-ZIP, then unzip.
        print(f"[{group}] 7z failed ({r.returncode}); joining with zip -s 0")
        shutil.rmtree(extract, ignore_errors=True)
        extract.mkdir()
        joined = scratch / f"{group}_joined.zip"
        subprocess.run(["zip", "-s", "0", str(archives / f"{group}.zip"), "--out", str(joined)], check=True)
        subprocess.run(["unzip", "-q", "-o", str(joined), "-d", str(extract)], check=True)
        joined.unlink()
    shutil.rmtree(archives)  # bound disk use before moving the extracted data
    print(f"[{group}] extracted; moving files into the volume")

    volumes_dir = base / "imagecas" / "volumes"
    labels_dir = base / "imagecas" / "orig_labels"
    meta_dir = base / "imagecas" / "meta"
    rows = []
    for src in sorted(p for p in extract.rglob("*") if p.is_file()):
        rel = src.relative_to(extract)
        low = str(rel).lower()
        if low.endswith((".nii", ".nii.gz")):
            if "label" in low or "mask" in low:
                kind, target_root = "orig_label", labels_dir
            else:
                kind, target_root = "volume", volumes_dir
        else:
            kind, target_root = "meta", meta_dir
        dest = target_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            print(f"[{group}] exists, not overwriting: {dest}")
            continue
        shutil.move(str(src), str(dest))
        if kind != "meta":
            rows.append(_nifti_row(group, kind, dest))

    shutil.rmtree(scratch, ignore_errors=True)
    group_marker.parent.mkdir(parents=True, exist_ok=True)
    group_marker.write_text(json.dumps(rows))
    volume.commit()
    print(f"[{group}] done: {len(rows)} NIfTI files")
    return rows


@app.function(image=image, volumes={DATA: volume}, timeout=6 * HOUR)
def stage_volumes() -> None:
    rows: list = []
    for group_rows in _stage_group.map(GROUPS):
        rows.extend(group_rows)

    manifest = Path(DATA) / "manifest"
    manifest.mkdir(parents=True, exist_ok=True)
    out = manifest / "volumes.csv"
    fields = ["group", "kind", "file", "bytes", "shape", "spacing", "error"]
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    volume.commit()
    print(f"wrote {out} with {len(rows)} rows")
    for kind in ("volume", "orig_label"):
        print(f"  {kind}: {sum(1 for r in rows if r['kind'] == kind)}")


# ---------------------------------------------------------------------------
# Step 3: inspect
# ---------------------------------------------------------------------------

@app.function(image=image, volumes={DATA: volume}, timeout=30 * 60)
def inspect_data() -> None:
    manifest = Path(DATA) / "manifest"
    print("=== /data/manifest ===")
    if not manifest.exists():
        print("no manifest yet")
    else:
        for path in sorted(manifest.rglob("*")):
            if path.is_file():
                print(f"--- {path} ({path.stat().st_size} bytes)")
                if path.suffix in {".csv", ".txt"}:
                    with open(path, "r", errors="replace") as fh:
                        for i, line in enumerate(fh):
                            if i >= 40:
                                print("...")
                                break
                            print(line.rstrip())

    print("=== /data/weights (find, depth 4) ===")
    print(_find_listing(f"{DATA}/weights", 4))


@app.local_entrypoint()
def main(step: str = "labels") -> None:
    steps = {
        "labels": stage_labels.remote,
        "volumes": stage_volumes.remote,
        "inspect": inspect_data.remote,
    }
    if step not in steps:
        raise SystemExit(f"unknown step {step!r}; choose one of: {', '.join(steps)}")
    steps[step]()
