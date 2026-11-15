#!/usr/bin/env bash
# Usage (on the DROPLET, after setup.sh and stage.sh; or let resume.sh call it):
#   cd /root/coronarysafe && BUDGET_USD=90 nohup bash infra/droplet/run_all.sh > /root/run_all.out 2>&1 &
#   DRY_RUN=1 bash infra/droplet/run_all.sh        # list every item and check every input path, no GPU
#   RUN_OOF=1 bash infra/droplet/run_all.sh        # also run the optional out-of-fold item (last)
#
# Queue, in order. Each item is idempotent (meta.json), so a re-run or a spot restart computes only what is missing.
#   (a) verify_folds     16 train scans x nnunet_f0..f4, probs saved (fold reconstruction check)
#   (b) nnunet_test      nnunet_f0..f4 on test, no TTA
#   (c) tta_valtest      cas_net, swin_unetr, ffr_unet on val+test with mirror TTA (recomputed here: the
#                        Modal test predictions had no TTA and are not on this droplet)
#   (d) nnunet_val       nnunet_f0..f4 on val, no TTA
#   (e) nnunet_oof       OPTIONAL (RUN_OOF=1), last: nnunet_oof on train+val. Unverified fold mapping.
# Budget: BUDGET_USD (default 90). Exit 3 is a hard stop; the queue then halts and nothing is killed silently.
# Powering off does NOT stop billing: destroy the droplet in the AMD console when the fetch is verified.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# shellcheck source=env.sh
. "$ROOT/infra/droplet/env.sh"
export_runtime_env
export BUDGET_USD="${BUDGET_USD:-90}"
DRY_RUN="${DRY_RUN:-0}"
RUN_OOF="${RUN_OOF:-0}"
mkdir -p docs

item() {  # item NAME ARGS...: run one queue item; stop the queue only on a hard stop
  local name="$1"; shift
  echo "run_all: $(date -u +%H:%M:%S) start $name"
  local extra=()
  [ "$DRY_RUN" = 1 ] && extra+=(--dry-run)
  cs_exec "$PY_CS" infra/droplet/run_infer.py --name "$name" "$@" "${extra[@]}"
  local rc=$?
  echo "run_all: $name exit $rc"
  if [ "$rc" -eq 3 ]; then
    echo "run_all: HARD STOP at budget \$$BUDGET_USD; queue halted. Raise BUDGET_USD only with approval, then re-run."
    exit 3
  fi
  return 0
}

ALL_NNUNET=nnunet_f0,nnunet_f1,nnunet_f2,nnunet_f3,nnunet_f4
item verify_folds --verify16
item nnunet_test --methods "$ALL_NNUNET" --split test
item tta_valtest --methods cas_net,swin_unetr,ffr_unet --split valtest --tta
item nnunet_val --methods "$ALL_NNUNET" --split val
if [ "$RUN_OOF" = 1 ]; then
  item nnunet_oof_trainval --methods nnunet_oof --split trainval
else
  echo "run_all: nnunet_oof skipped (set RUN_OOF=1 to run it after the other items)"
fi

if [ "$DRY_RUN" = 0 ]; then
  cs_exec "$PY_CS" infra/droplet/run_infer.py --summary
  echo "run_all: queue complete. Fold check: $DATA_ROOT/meta/fold_verification.json"
  echo "run_all: fetch from the LAPTOP now: bash infra/droplet/fetch.sh <droplet-ip>"
  echo "run_all: REMINDER: DESTROY the droplet in the AMD console once the fetch is verified."
fi
