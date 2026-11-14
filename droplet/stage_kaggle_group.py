"""Stage one Kaggle split group (for example 1-200) of xiaoweixumedicalai/imagecas under DATA_ROOT.

Same logic and layout as the Modal _stage_group in modal/stage_data.py:
    download <group>.z01..z04 and <group>.change2zip with the kaggle CLI, rename the last to .zip,
    extract with 7z, fall back to joining the split set with `zip -s 0`, then move the
    NIfTI files to <data>/imagecas/volumes/ (or orig_labels/ for label-like names) and other files
    to imagecas/meta/.
The archives are deleted after extraction. Idempotent: a group marker in
manifest/groups/ skips the group.

Needs ~/.kaggle/kaggle.json (checked by stage.sh).
Usage: python droplet/stage_kaggle_group.py GROUP [DATA_ROOT]
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

KAGGLE_DATASET = "xiaoweixumedicalai/imagecas"


def kaggle_fetch(archives: Path, name: str) -> None:
    r = subprocess.run(
        ["kaggle", "datasets", "download", KAGGLE_DATASET, "-f", name, "-p", str(archives)],
        capture_output=True,
        text=True,
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


def nifti_row(group: str, kind: str, path: Path) -> dict:
    import nibabel as nib

    row = {
        "group": group,
        "kind": kind,
        "file": str(path),
        "bytes": path.stat().st_size,
        "shape": "",
        "spacing": "",
        "error": "",
    }
    try:
        img = nib.load(str(path))  # header only
        row["shape"] = "x".join(str(s) for s in img.shape)
        row["spacing"] = "x".join(f"{z:.4g}" for z in img.header.get_zooms())
    except Exception as exc:
        row["error"] = str(exc)[:200]
    return row


def main() -> None:
    group = sys.argv[1]
    data = Path(sys.argv[2] if len(sys.argv) > 2 else "/root/data")
    marker = data / "manifest" / "groups" / f"{group}.json"
    if marker.exists():
        print(f"[{group}] skip, marker present")
        return
    scratch = data / "_scratch" / group
    shutil.rmtree(scratch, ignore_errors=True)
    archives, extract = scratch / "archives", scratch / "extract"
    archives.mkdir(parents=True)
    extract.mkdir()

    parts = [f"{group}.z0{i}" for i in range(1, 5)] + [f"{group}.change2zip"]
    for name in parts:
        print(f"[{group}] downloading {name}", flush=True)
        kaggle_fetch(archives, name)
    (archives / f"{group}.change2zip").rename(archives / f"{group}.zip")

    # 7z reads PKZIP split sets when the .zip and .z01 parts sit together.
    r = subprocess.run(["7z", "x", str(archives / f"{group}.zip"), f"-o{extract}", "-y"])
    if r.returncode != 0:
        print(f"[{group}] 7z failed ({r.returncode}); joining with zip -s 0", flush=True)
        shutil.rmtree(extract, ignore_errors=True)
        extract.mkdir()
        joined = scratch / f"{group}_joined.zip"
        subprocess.run(
            ["zip", "-s", "0", str(archives / f"{group}.zip"), "--out", str(joined)], check=True
        )
        subprocess.run(["unzip", "-q", "-o", str(joined), "-d", str(extract)], check=True)
        joined.unlink()
    shutil.rmtree(archives)  # archives are no longer needed once extracted
    print(f"[{group}] extracted; moving files", flush=True)

    volumes_dir = data / "imagecas" / "volumes"
    labels_dir = data / "imagecas" / "orig_labels"
    meta_dir = data / "imagecas" / "meta"
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
            rows.append(nifti_row(group, kind, dest))

    shutil.rmtree(scratch, ignore_errors=True)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(rows))
    print(f"[{group}] done: {len(rows)} NIfTI files", flush=True)


if __name__ == "__main__":
    main()
