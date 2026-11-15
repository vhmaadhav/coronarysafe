#!/bin/bash
# Autonomous overnight GPU chain for CoronarySafe (no agent or laptop needed). Log: /root/auto_chain.log
cd /root/coronarysafe
log(){ echo "[chain] $(date -u +%H:%M) $*"; }
RUN(){ docker exec -e DATA_ROOT=/root/data -w /root/coronarysafe cs-infer /root/venv-infer/bin/python droplet/run_infer.py "$@"; log "finished run_infer $*"; }
LBL(){ docker exec -e DATA_ROOT=/root/data -e LABEL_WORKERS=8 -w /root/coronarysafe cs-infer bash droplet/run_labels.sh "$@"; log "finished labels $*"; }
# 0) stop the old priority_queue driver so it does not start pq4+ itself; its running pq3 child keeps going
kill 204508 2>/dev/null && log "stopped priority_queue.sh driver"
# 1) wait for pq3 (cas/swin test tta) to finish
while pgrep -f "droplet/run_infer.py --name pq3_test_tta" >/dev/null; do sleep 60; done; log "pq3 done"
# 2) stop the 2-worker early nnU-Net run and finish nnU-Net test with full workers (idempotent)
P=$(pgrep -f "droplet/run_infer.py --name pq4_early_nnu_test"); [ -n "$P" ] && kill $P && sleep 20 && log "stopped pq4_early"
RUN --name pq4_nnu_test --methods nnunet_f0,nnunet_f1,nnunet_f2,nnunet_f3,nnunet_f4 --split test --workers 5
LBL nnunet_f0,nnunet_f1,nnunet_f2,nnunet_f3,nnunet_f4 test &
# 3) E2 fold verification: first 100 train scans x 5 folds, then confidence-based verification
RUN --name e2_train100 --methods nnunet_f0,nnunet_f1,nnunet_f2,nnunet_f3,nnunet_f4 --split train --limit 100 --workers 5
docker exec -w /root/coronarysafe -e PYTHONPATH=/root/coronarysafe:/root/coronarysafe/src cs-infer nice -n 5 /root/venv-infer/bin/python droplet/verify_folds_conf.py 100 /root/data/results/fold_verification_full.json > /root/e2_verify.log 2>&1
V=$(python3 -c "import json;print(json.load(open('/root/data/results/fold_verification_full.json')).get('verified'))" 2>/dev/null); log "E2 verified=$V"
# 4) E3 out-of-fold nnU-Net on train+val (only if verified), then label
if [ "$V" = "True" ]; then RUN --name e3_oof --methods nnunet_oof --split trainval --workers 5; LBL nnunet_oof trainval & fi
# 5) pq5 ffr val tta, pq6 nnU-Net val
RUN --name pq5_ffr_val --methods ffr_unet --split val --tta --workers 4
LBL ffr_unet val &
RUN --name pq6_nnu_val --methods nnunet_f0,nnunet_f1,nnunet_f2,nnunet_f3,nnunet_f4 --split val --workers 5
# 6) E1 degraded-scan robustness (cas_net + ffr_unet, 3 kinds, tta)
for k in noise zblur contrast; do RUN --name e1_$k --methods cas_net@$k,ffr_unet@$k --split test --tta --volumes-root /root/data/imagecas/volumes_shift/$k --workers 4; done
log "ALL DONE"
