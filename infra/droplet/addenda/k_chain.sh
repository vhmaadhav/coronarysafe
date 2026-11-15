#!/bin/bash
# Addendum K chain (droplet). CAS-Net (--tta, as preds/cas_net) on the 3 K-perturbed test sets, then
# the per-unit labeller (process_case, qa/v3). Milestones: /root/k_chain.log (UTC). Inference and
# label stdout: /root/k_chain_<stage>_<pert>.log. Order: noise -> contrast -> blur.
LOG=/root/k_chain.log
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[k_chain] $(ts) $*" >> "$LOG"; }
DC="docker exec -e DATA_ROOT=/root/data -w /root/coronarysafe cs-infer"
DEADLINE=$(TZ=UTC date -d "today 20:30" +%s)
exec 9>/root/gpu.lock
flock 9
log "START pid=$$ deadline_start=20:30Z"
for P in noise contrast blur; do
  if [ "$(date -u +%s)" -ge "$DEADLINE" ]; then log "SKIPPED $P (deadline)"; continue; fi
  log "INFER_START $P"
  $DC /root/venv-infer/bin/python -u infra/droplet/run_infer.py --methods "cas_net@K_$P" --split test --tta --workers 3 --name "k_$P" --budget-usd 90 > "/root/k_chain_infer_$P.log" 2>&1
  log "INFER_END $P rc=$?"
  log "PREDS $P meta_files=$(ls /root/data/preds/cas_net@K_$P/*/meta.json 2>/dev/null | wc -l)"
  log "LABEL_START $P"
  $DC /root/venv-infer/bin/python -u /root/data/addenda/K/code/k_label.py "cas_net@K_$P" test 8 > "/root/k_chain_label_$P.log" 2>&1
  log "LABEL_END $P rc=$? qa_files=$(ls /root/data/qa/v3/cas_net@K_$P/*.json 2>/dev/null | wc -l)"
done
log "K_CHAIN_DONE"
