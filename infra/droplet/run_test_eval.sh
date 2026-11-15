#!/usr/bin/env bash
# Sealed test pass (Addendum H). Runs INSIDE cs-infer from the repo copy:
#   bash /root/coronarysafe-m4/droplet/run_test_eval.sh
# Order: S3 ensemble comparators + nnU-Net majority vote -> v3 labels for nnunet_mv (test) -> test_eval (test split).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG=/root/data/test_eval/logs
mkdir -p "$LOG" /root/data/test_eval
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$ROOT/src:$ROOT/pipeline:$ROOT"
export DATA_ROOT=/root/data
PY=/root/venv-infer/bin/python

nice -n 10 "$PY" -m coronarysafe.m4.ensemble --root /root/data --out /root/data/test_eval --workers 6 >> "$LOG/ensemble.log" 2>&1
bash "$ROOT/infra/droplet/run_labels.sh" nnunet_mv test >> "$LOG/labels_nnunet_mv.log" 2>&1
nice -n 10 "$PY" -m coronarysafe.m4.test_eval --split test --out /root/data/test_eval >> "$LOG/test_eval.log" 2>&1
echo TEST_EVAL_DONE | tee -a "$LOG/test_eval.log"
