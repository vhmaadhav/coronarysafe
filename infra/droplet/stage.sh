#!/usr/bin/env bash
# Usage (on the DROPLET, after setup.sh):   bash /root/coronarysafe/infra/droplet/stage.sh 2>&1 | tee -a /root/stage.log
#
# Stages the Modal-volume layout under /root/data (a symlink to the scratch disk, see env.sh):
#   raw/zenodo/ + imagecas_x/ + weights/   Zenodo labels and weights (byte-count and md5 checked)
#   imagecas/volumes/, orig_labels/, meta/ Kaggle split groups, five in parallel
# Needs /root/.kaggle/kaggle.json. Idempotent and preemption-safe: finished groups and verified Zenodo files are
# skipped; an interrupted group restarts from scratch (its archives are in _scratch/<group>, removed on restart).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=env.sh
. "$ROOT/infra/droplet/env.sh"

GROUPS=("1-200" "201-400" "401-600" "601-800" "801-1000")
NEED_GB="${NEED_GB:-150}"
die() { echo "stage: ERROR: $*" >&2; exit 2; }

KAGGLE_JSON="$HOME/.kaggle/kaggle.json"
if [ ! -s "$KAGGLE_JSON" ]; then
  die "$KAGGLE_JSON is missing. From the laptop: scp -i ~/.ssh/amd_devcloud ~/.kaggle/kaggle.json root@<ip>:/root/.kaggle/kaggle.json"
fi
chmod 600 "$KAGGLE_JSON"
setup_data_root || die "data root setup failed"
[ -x "$VENV/bin/python" ] || die "$VENV/bin/python not found. Run infra/droplet/setup.sh first."

# Free-space check on the scratch disk (the data root is a symlink into it).
free_gb=$(df -BG --output=avail "$DATA_ROOT/" | tail -1 | tr -dc '0-9')
if [ "$free_gb" -lt "$NEED_GB" ]; then
  die "only ${free_gb} GB free on $(readlink -f "$DATA_ROOT"); need about ${NEED_GB} GB"
fi
echo "stage: $(df -h --output=avail,target "$DATA_ROOT/" | tail -1) free"

export PATH="$VENV/bin:$PATH"  # the kaggle CLI lives in the venv
echo "stage: Zenodo labels and weights"
cs_exec env DATA_ROOT="$DATA_ROOT" PATH="$VENV/bin:$PATH" "$PY_CS" "$ROOT/infra/droplet/stage_zenodo.py" "$DATA_ROOT"

echo "stage: Kaggle groups in parallel: ${GROUPS[*]}"
mkdir -p "$DATA_ROOT/_logs"
pids=()
for g in "${GROUPS[@]}"; do
  cs_exec env PATH="$VENV/bin:$PATH" "$PY_CS" "$ROOT/infra/droplet/stage_kaggle_group.py" "$g" "$DATA_ROOT" \
    > "$DATA_ROOT/_logs/kaggle_${g}.log" 2>&1 &
  pids+=("$!")
done
failed=()
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then failed+=("${GROUPS[$i]}"); fi
done
if [ "${#failed[@]}" -gt 0 ]; then
  die "groups failed: ${failed[*]}. Logs: $DATA_ROOT/_logs/kaggle_<group>.log. Re-run to resume."
fi

n_ct=$(find "$DATA_ROOT/imagecas/volumes" -name '*.img.nii.gz' 2>/dev/null | wc -l)
echo "stage: done. CT volumes: $n_ct (expect 1000)"
[ "$n_ct" -ge 1000 ] || die "only $n_ct CT volumes, expected 1000. Check $DATA_ROOT/_logs and re-run."
echo "stage: next: bash $ROOT/infra/droplet/run_all.sh"
