#!/usr/bin/env bash
# Usage (on the LAPTOP, before destroying the droplet):
#   bash infra/droplet/fetch.sh <droplet-ip> [--with-probs]
#
# Copies back to <repo>/data/droplet/: predictions (mask.nii.gz, meta.json, flips/, _runs/), the QA
# records and reports (qa/), meta/ (fold verification), and docs/RUNLOG.md. Probability maps
# (prob.nii.gz, large) only with --with-probs. Verifies the file counts match before it reminds you to
# destroy the droplet. Safe to re-run (rsync resumes).
set -euo pipefail

IP="${1:-}"
[ -n "$IP" ] || { echo "usage: bash infra/droplet/fetch.sh <droplet-ip> [--with-probs]" >&2; exit 2; }
WITH_PROBS=0
[ "${2:-}" = "--with-probs" ] && WITH_PROBS=1

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="$REPO/data/droplet"
KEY="${DROPLET_KEY:-$HOME/.ssh/amd_devcloud}"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new)
RSYNC=(rsync -az --partial --info=stats0 -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new")
mkdir -p "$DEST/preds" "$DEST/qa" "$DEST/meta"

echo "fetch: predictions (masks, meta, flips, run records)"
PRED_FILTERS=(--include='*/' --include='meta.json' --include='mask.nii.gz' --include='flips/***' --include='_runs/***')
if [ "$WITH_PROBS" = 1 ]; then
  PRED_FILTERS+=(--include='prob.nii.gz')
  echo "fetch: including prob maps (large)"
fi
PRED_FILTERS+=(--exclude='*')
"${RSYNC[@]}" "${PRED_FILTERS[@]}" "root@$IP:/root/data/preds/" "$DEST/preds/"

echo "fetch: QA records and reports"
"${RSYNC[@]}" "root@$IP:/root/data/qa/" "$DEST/qa/"

echo "fetch: meta (fold verification, priors) and RUNLOG"
"${RSYNC[@]}" "root@$IP:/root/data/meta/" "$DEST/meta/" || echo "fetch: no meta directory yet"
"${RSYNC[@]}" "root@$IP:/root/coronarysafe/docs/RUNLOG.md" "$DEST/RUNLOG.md" || echo "fetch: no RUNLOG yet"

# Count check: remote and local meta.json counts must agree (catches a partial rsync).
remote_n=$("${SSH[@]}" "root@$IP" 'find /root/data/preds -name meta.json | wc -l')
local_n=$(find "$DEST/preds" -name meta.json | wc -l)
echo "fetch: meta.json remote=$remote_n local=$local_n"
if [ "$remote_n" != "$local_n" ]; then
  echo "fetch: MISMATCH; re-run fetch.sh before destroying the droplet" >&2
  exit 1
fi
echo "fetch: done. Files in $DEST"
echo "fetch: =============================================================="
echo "fetch: DESTROY THE DROPLET in the AMD console now (stopping does NOT stop billing)."
echo "fetch: =============================================================="
