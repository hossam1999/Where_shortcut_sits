#!/bin/bash
# A42 reported clean check (NOT part of the A39 verdict): each EPVT seed and each arm (trained at bias b) scored once on
# test_bias_0_1. Bias 1 now; bias 0.9 after the A39 chain's bias-0.9 arms exist; then the combined table.
export WTSS_DATA=${WTSS_DATA:-/workspace/data}; D=$WTSS_DATA; Q=/root/Where_shortcut_sits/scripts/replication; PYW=/workspace/venv-wtss/bin/python; PYE=/workspace/envs/epvt/bin/python
export EPVT_EVAL_ROOT=/workspace/third_party/bissoto_ag/trap_sets/ EPVT_IMFOLDER=$D/epvt/isic2019_plain/ EPVT_ROOT=$D/epvt/ WANDB_MODE=disabled
cd /root/Where_shortcut_sits
gpu(){ local need=$1; shift; until ! ps -eo args | grep -q "[g]pu_wait.sh 15000 .*--net isnet"; do sleep 30; done; MAX_OTHER_JOBS=1 $Q/gpu_wait.sh $need "$@"; }
step(){ local n=$1; shift; echo "$(date -u +%F_%T) start $n"; "$@" || { echo "$(date -u +%F_%T) FAILED P6CLEAN $n"; exit 1; }; echo "$(date -u +%F_%T) done $n"; }
for B in 1 0.9; do
  if [ $B = 0.9 ]; then until grep -q "done armd_bias09" /workspace/logs/m2_p6_chain.log; do sleep 120; done; fi
  for s in 0 1 2 3 4; do R=$D/epvt/runs/epvt_a36_bias${B}_1_seed$s
    [ -s $R/epvt__test_bias_0_1.json ] || step "epvt_bias${B}_seed${s}_test0" bash -c "cd /workspace/third_party/epvt && $(declare -f gpu); Q=$Q; gpu 5000 $PYE $Q/p6_epvt_score.py --ckpt $R/epvt.pkl --test test_bias_0_1.csv >> $R/score.log 2>&1"
  done
  step clean_a_bias$B gpu 6000 $PYE $Q/p6_arms.py --stage clean_a --bias $B
  step clean_bd_bias$B gpu 6000 $PYW $Q/p6_arms.py --stage clean_bd --bias $B
done
step clean_both $PYW $Q/p6_arms.py --stage clean --bias 1 0.9
echo "$(date -u +%F_%T) P6 clean check done"
