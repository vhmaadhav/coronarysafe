#!/usr/bin/env bash
# Sourced by setup.sh, stage.sh, run_all.sh, run_labels.sh, resume.sh and fetch.sh. Not run directly.
#
# Host layout decisions (see README):
#   SCRATCH      the largest writable NVMe mount (the AMD MI300X plan has a 5 TB scratch disk).
#                Override with SCRATCH=/path. All data goes under $SCRATCH/coronarysafe-data.
#   /root/data   symlink to that data root, so every script can keep using /root/data.
#   CS_CONTAINER name of the Docker container when running inside the AMD PyTorch Quick Start
#                image (preferred path). Empty when running on the bare host.

set -euo pipefail

DROPLET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$DROPLET_DIR/../.." && pwd)"
VENV="${VENV:-/root/venv-infer}"
CS_CONTAINER="${CS_CONTAINER:-}"
CS_IMAGE_TAG="${CS_IMAGE_TAG:-cs-infer}"
# On the host, pick up the running Quick Start container automatically (set by setup.sh path A).
if [ -z "$CS_CONTAINER" ] && [ "${CS_INSIDE:-0}" != 1 ] && command -v docker >/dev/null 2>&1 \
   && docker ps --format '{{.Names}}' 2>/dev/null | grep -qx cs-infer; then
  CS_CONTAINER=cs-infer
fi

find_scratch() {
  # Largest mount with at least 1 TB available, outside the boot and system filesystems.
  if [ -n "${SCRATCH:-}" ]; then echo "$SCRATCH"; return; fi
  df -B1 --output=target,avail,fstype 2>/dev/null | tail -n +2 \
    | awk '$3 !~ /^(tmpfs|overlay|squashfs|devtmpfs|efivarfs|vfat)$/ && $1 !~ /^\/(boot|dev|proc|sys|run)/ {print $2, $1}' \
    | sort -rn | head -1 | awk '$1 >= 1000000000000 {print $2}'
}

# Exec a command inside the Quick Start container if one is in use, else on the host.
cs_exec() {
  if [ -n "$CS_CONTAINER" ]; then
    docker exec -i -e DATA_ROOT="${DATA_ROOT:-/root/data}" -w "$REPO_DIR" "$CS_CONTAINER" "$@"
  else
    "$@"
  fi
}

PY_HOST="$VENV/bin/python"
if [ -n "$CS_CONTAINER" ]; then PY_CS="$VENV/bin/python"; else PY_CS="$PY_HOST"; fi

# Data root: $SCRATCH/coronarysafe-data, linked from /root/data. Created once, idempotent.
setup_data_root() {
  local scratch
  scratch="$(find_scratch)"
  if [ -z "$scratch" ]; then
    echo "env: no scratch mount >= 1 TB found; set SCRATCH=/path and re-run" >&2
    return 1
  fi
  mkdir -p "$scratch/coronarysafe-data"
  if [ -L /root/data ]; then
    [ "$(readlink -f /root/data)" = "$(readlink -f "$scratch/coronarysafe-data")" ] || {
      echo "env: /root/data points elsewhere: $(readlink -f /root/data)" >&2; return 1; }
  elif [ -e /root/data ]; then
    echo "env: /root/data exists and is not a symlink; move it aside and re-run" >&2; return 1
  else
    ln -s "$scratch/coronarysafe-data" /root/data
  fi
  export DATA_ROOT=/root/data
  export SCRATCH_DIR="$scratch"
}

# nnU-Net and ROCm environment. nnUNet_compile=f because torch.compile is not supported on ROCm here.
export_runtime_env() {
  export nnUNet_raw="/root/nnunet/raw"
  export nnUNet_preprocessed="/root/nnunet/preprocessed"
  export nnUNet_results="/root/nnunet/results"
  export nnUNet_compile=f
  export MIOPEN_USER_DB_PATH="/root/.cache/miopen"
  export MIOPEN_CUSTOM_CACHE_DIR="/root/.cache/miopen"
  export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
  mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results" "$MIOPEN_USER_DB_PATH"
}
