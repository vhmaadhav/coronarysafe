#!/usr/bin/env bash
# M3 aorta prior: TotalSegmentator `total` task, roi_subset aorta only (Apache-2.0). Runs INSIDE cs-infer.
# Usage: bash m4_aorta.sh <scan-list-file> [workers]
# Output: /root/data/m4/aorta/<id>.nii.gz (skips existing). Logs: /root/data/m4/logs/aorta_<id>.log
set -euo pipefail
LIST="$1"
WORKERS="${2:-4}"
OUT=/root/data/m4/aorta
LOGS=/root/data/m4/logs
mkdir -p "$OUT" "$LOGS" /root/data/m4/ts_tmp

one() {
  id="$1"
  [ -s "/root/data/m4/aorta/${id}.nii.gz" ] && { echo "skip $id"; return 0; }
  ct=$(find /root/data/imagecas/volumes -name "${id}.img.nii.gz" | head -1)
  [ -n "$ct" ] || { echo "missing CT $id"; return 1; }
  tmp="/root/data/m4/ts_tmp/${id}"
  /root/venv-infer/bin/TotalSegmentator -i "$ct" -o "$tmp" --task total --roi_subset aorta --fast --device gpu \
    > "/root/data/m4/logs/aorta_${id}.log" 2>&1
  cp "$tmp/aorta.nii.gz" "/root/data/m4/aorta/${id}.nii.gz.tmp" && mv "/root/data/m4/aorta/${id}.nii.gz.tmp" "/root/data/m4/aorta/${id}.nii.gz"
  rm -rf "$tmp"
  echo "done $id"
}
export -f one
export OUT LOGS
xargs -P "$WORKERS" -I{} bash -c 'one {}' < "$LIST"
echo "aorta batch finished: $(ls /root/data/m4/aorta/*.nii.gz 2>/dev/null | wc -l) masks"
