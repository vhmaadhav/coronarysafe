#!/usr/bin/env bash
# Usage (on the DROPLET, after stage.sh and run_all.sh have written the predictions):
#   bash /root/coronarysafe/infra/droplet/run_labels.sh cas_net,swin_unetr,ffr_unet,nnunet_oof test
#   bash /root/coronarysafe/infra/droplet/run_labels.sh cas_net val 1      # smoke: first scan only
# CPU only. Modal is not needed for labelling any more. Idempotent and preemption-safe (existing
# status=ok records are skipped). Writes <data>/qa/v3/<method>/<scan>.json and report_<split>.txt.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=env.sh
. "$ROOT/infra/droplet/env.sh"

METHODS="${1:-cas_net,swin_unetr,ffr_unet,nnunet_oof}"
SPLIT="${2:-test}"
LIMIT="${3:-0}"
setup_data_root >/dev/null 2>&1 || true
export DATA_ROOT="${DATA_ROOT:-/root/data}"
cs_exec "$PY_CS" "$ROOT/infra/droplet/run_labels.py" --methods "$METHODS" --split "$SPLIT" --limit "$LIMIT" --workers "${LABEL_WORKERS:-4}"
echo "run_labels: done. Next: bash $ROOT/infra/droplet/fetch.sh <droplet-ip>  (qa/ comes back to the laptop)"
