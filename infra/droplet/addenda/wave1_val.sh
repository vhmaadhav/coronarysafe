#!/bin/bash
# CoronarySafe Wave 1 (VAL ONLY): Addendum L phase 1 + phase 2 (valscore) and Addendum N (N1, N2).
# Never touches the test split. Runs inside container cs-infer, cwd = src, <= 8 workers.
LOG=/root/wave1_val.log
echo $$ > /root/wave1_val.pid
ROOT=/root/data
SRC=/root/coronarysafe-li/src
DX=(docker exec -e PYTHONPATH=$SRC -w $SRC cs-infer /root/venv-infer/bin/python -u)
DEV=$ROOT/m4/dev_units.json
FR=$ROOT/m4/frozen
L1=$ROOT/addenda/L/val_phase1
VS=$ROOT/addenda/L/valscores
L2=$ROOT/addenda/L/val_phase2
NN=$ROOT/addenda/N
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "$(ts) $*" >> "$LOG"; }
run() { local name=$1; shift; log "START $name"; "$@" >> "$LOG" 2>&1; local rc=$?; log "END $name rc=$rc"; return $rc; }

: > "$LOG"
log "WAVE1 start pid=$$ host=$(hostname)"
mkdir -p "$L1" "$VS" "$L2" "$NN"

# (a0) frozen dev scores, frozen coefficients only (no fitting)
run dev_scores "${DX[@]}" -m coronarysafe.analysis.rule_stability dev-scores \
  --root "$ROOT" --dev-units "$DEV" --frozen-dir "$FR" --out "$L1/scores_dev160.csv"

# (a) L phase 1: 160 dev-scope units (cas_net, swin_unetr)
run L_phase1 "${DX[@]}" -m coronarysafe.analysis.burden_indep --split val \
  --methods cas_net swin_unetr --scores "$L1/scores_dev160.csv" --out "$L1" --workers 8 --root "$ROOT"

# (b) frozen-model scores on val x {ffr_unet, nnunet_f0..f4}, CPU
run valscore "${DX[@]}" -m coronarysafe.m4.valscore --split val --root "$ROOT" \
  --methods ffr_unet nnunet_f0 nnunet_f1 nnunet_f2 nnunet_f3 nnunet_f4 \
  --unit-dir "$VS/unit_jsons" --aorta-dir "$ROOT/m4/aorta" --frozen-dir "$FR" \
  --out "$VS" --device cpu --workers 8

# (c) L phase 2: 640 val units = 160 dev + 480 valscore
if [ -s "$VS/scores.csv" ]; then
  { head -1 "$L1/scores_dev160.csv"; tail -n +2 "$L1/scores_dev160.csv"; tail -n +2 "$VS/scores.csv"; } > "$L2/scores_640.csv"
  log "combined scores_640 rows=$(( $(wc -l < "$L2/scores_640.csv") - 1 ))"
  run L_phase2 "${DX[@]}" -m coronarysafe.analysis.burden_indep --split val \
    --methods cas_net swin_unetr ffr_unet nnunet_f0 nnunet_f1 nnunet_f2 nnunet_f3 nnunet_f4 \
    --scores "$L2/scores_640.csv" --out "$L2" --workers 8 --root "$ROOT"
else
  log "SKIP L_phase2: no valscore scores.csv"
fi

# (d) Addendum N: N1 rule stability (val), N2 in-sample ablation (160 dev units)
run N "${DX[@]}" -m coronarysafe.analysis.rule_stability n --root "$ROOT" --dev-units "$DEV" \
  --unit-dir "$VS/unit_jsons" --frozen-dir "$FR" --out "$NN"

log WAVE1_DONE
