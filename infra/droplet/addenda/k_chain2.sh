#!/bin/bash
# Addendum K chain v2: noise inference continues (orphaned from k_chain.sh, PID of its docker exec passed as $1);
# contrast and blur inference run IN PARALLEL now (GPU freed by amdbench). Then labels. Same code paths as k_chain.sh.
LOG=/root/k_chain.log
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[k_chain2] $(ts) $*" >> "$LOG"; }
DC="docker exec -e DATA_ROOT=/root/data -w /root/coronarysafe cs-infer"
NOISE_PID=$1
log "START pid=$$ parallel contrast+blur; waiting noise pid=$NOISE_PID"
infer_label() {
  P=$1
  log "INFER_START $P"
  $DC /root/venv-infer/bin/python -u infra/droplet/run_infer.py --methods "cas_net@K_$P" --split test --tta --workers 3 --name "k_$P" --budget-usd 90 > "/root/k_chain_infer_$P.log" 2>&1
  log "INFER_END $P rc=$? preds=$(ls /root/data/preds/cas_net@K_$P/*/meta.json 2>/dev/null | wc -l)"
  log "LABEL_START $P"
  $DC /root/venv-infer/bin/python -u /root/data/addenda/K/code/k_label.py "cas_net@K_$P" test 6 > "/root/k_chain_label_$P.log" 2>&1
  log "LABEL_END $P rc=$? qa_files=$(ls /root/data/qa/v3/cas_net@K_$P/*.json 2>/dev/null | wc -l)"
}
infer_label contrast & C=$!
infer_label blur & B=$!
while kill -0 "$NOISE_PID" 2>/dev/null; do sleep 30; done
log "INFER_END noise preds=$(ls /root/data/preds/cas_net@K_noise/*/meta.json 2>/dev/null | wc -l)"
log "LABEL_START noise"
$DC /root/venv-infer/bin/python -u /root/data/addenda/K/code/k_label.py "cas_net@K_noise" test 6 > "/root/k_chain_label_noise.log" 2>&1
log "LABEL_END noise rc=$? qa_files=$(ls /root/data/qa/v3/cas_net@K_noise/*.json 2>/dev/null | wc -l)"
wait $C $B
log "K_CHAIN_DONE"
