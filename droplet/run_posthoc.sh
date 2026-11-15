#!/bin/bash
# Post-freeze descriptive analyses (descriptive only; no refit). Run INSIDE the cs-infer container:
#   docker exec cs-infer bash /root/coronarysafe-m5/droplet/run_posthoc.sh <IN_DIR> <OUT_DIR> [CLINICAL_PARQUET]
# Example (test, after the freeze):
#   setsid nohup docker exec cs-infer bash /root/coronarysafe-m5/droplet/run_posthoc.sh \
#     /root/data/test_eval /root/data/posthoc /root/data/m5/clinical_test_vessel_rows.parquet
set -euo pipefail
IN=${1:?usage: run_posthoc.sh IN_DIR OUT_DIR [CLINICAL_PARQUET]}
OUT=${2:?usage: run_posthoc.sh IN_DIR OUT_DIR [CLINICAL_PARQUET]}
CLIN=${3:-/root/data/m5/clinical_test_vessel_rows.parquet}
cd /root/coronarysafe-m5
export PYTHONPATH=/root/coronarysafe-m5/src
export OMP_NUM_THREADS=1
nice -n 10 /root/venv-infer/bin/python -m coronarysafe.analysis.posthoc \
  --in "$IN" --out "$OUT" --clinical "$CLIN" --workers 6
echo POSTHOC_DONE "$(date -u +%T)"
