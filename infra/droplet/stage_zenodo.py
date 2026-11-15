"""Stage the ImageCAS-X labels and pretrained weights from Zenodo 23894668 under DATA_ROOT.

Same layout as infra/modal/stage_data.py (the Modal volume):
    <data>/raw/zenodo/                      downloaded zips, plus .verified markers
    <data>/imagecas_x/             unzipped ImageCAS-X dataset (labels, filelists, Descriptors)
    <data>/weights/                         unzipped pretrained weights
Downloads are checked against the byte count and md5 that Zenodo reports. Idempotent.

Usage: python infra/droplet/stage_zenodo.py [DATA_ROOT]     (default /root/data)
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import requests

RECORD = 23894668
FILES = ["ImageCAS-X_dataset_v3.zip", "pretrained_weights.zip"]


def download_verified(url: str, dest: Path, size: int, md5: str) -> None:
    """Resumable: retries and continues from the .part file (Zenodo drops long connections)."""
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, 201):
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink()
            have = 0
        if have == size:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, stream=True, timeout=(30, 300), headers=headers) as resp:
                if have and resp.status_code != 206:
                    part.unlink()
                    have = 0
                resp.raise_for_status()
                with open(part, "ab" if have else "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
        except (requests.RequestException, OSError) as exc:
            print(f"{dest.name}: attempt {attempt} broke at {have} bytes ({exc.__class__.__name__}); resuming", flush=True)
    received = part.stat().st_size
    if received != size:
        raise RuntimeError(f"{dest.name}: got {received} bytes, Zenodo says {size}")
    digest = hashlib.md5()
    with open(part, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != md5:
        raise RuntimeError(f"{dest.name}: md5 {digest.hexdigest()} != Zenodo {md5}")
    part.rename(dest)


def unzip(archive: Path, target: Path, marker: Path) -> None:
    if marker.exists():
        print(f"skip unzip, marker present: {marker}")
        return
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(["unzip", "-q", "-o", str(archive), "-d", str(target)], check=True)
    marker.write_text(archive.name + "\n")


def main() -> None:
    data = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/data")
    raw = data / "raw" / "zenodo"
    raw.mkdir(parents=True, exist_ok=True)

    meta = requests.get(f"https://zenodo.org/api/records/{RECORD}", timeout=60)
    meta.raise_for_status()
    files = {f["key"]: f for f in meta.json()["files"]}
    missing = [name for name in FILES if name not in files]
    if missing:
        raise RuntimeError(f"Zenodo record {RECORD} has no files {missing}; has {sorted(files)}")

    for name in FILES:
        info = files[name]
        dest = raw / name
        marker = raw / f".{name}.verified"
        if marker.exists() and dest.exists():
            print(f"skip download, verified: {dest}")
            continue
        md5 = info["checksum"].split(":")[-1]
        print(f"downloading {name} ({info['size']} bytes) ...", flush=True)
        download_verified(info["links"]["self"], dest, int(info["size"]), md5)
        marker.write_text(md5 + "\n")

    unzip(raw / "ImageCAS-X_dataset_v3.zip", data / "imagecas_x", data / "imagecas_x" / ".unzipped")
    unzip(raw / "pretrained_weights.zip", data / "weights", data / "weights" / ".unzipped")
    print("zenodo: done")


if __name__ == "__main__":
    main()
