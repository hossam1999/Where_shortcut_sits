#!/bin/bash
# A43: P1 location test, seeds 3232 and 3221, tube inside and outside (their Standard config, their training code; Masked
# scored from each, A5); arms a-c (p1_location_arms.py) and d (armd_train p1_loc_*). Per seed: split listing -> PSPNet ROI +
# placements -> train (both positions at once) -> score -> arms. Every result is staged for the pusher as soon as it exists.
#   [SEEDS="3232 3221"] WTSS_DATA=/workspace/data nohup scripts/replication/m2_p1_loc.sh > /workspace/logs/m2_p1_loc.log 2>&1 &
export WTSS_DATA=${WTSS_DATA:-/workspace/data}; D=$WTSS_DATA; Q=/root/Where_shortcut_sits/scripts/replication
PV=/workspace/envs/viviano/bin/python; PW=/workspace/venv-wtss/bin/python
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONPATH=/workspace/third_party/viviano:/workspace/third_party/viviano/activmask
cd /root/Where_shortcut_sits; mkdir -p $D/replication/viviano
step(){ local n=$1; shift; echo "$(date -u +%F_%T) start $n"; "$@" || { echo "$(date -u +%F_%T) FAILED P1LOC $n"; return 1; }; echo "$(date -u +%F_%T) done $n"; }
stage(){ local s=$1 src=$2 dst=$3; mkdir -p $(dirname $D/results_push/results/replication/viviano/$dst); cp $src $D/results_push/results/replication/viviano/$dst; }
seed_job(){ local s=$1; local L=$D/replication/viviano/location_s$s
  export P1_SEED=$s
  [ -s $L/nospc_$s.csv ] || step "list_s$s" $PV $Q/p1_location.py list --position inside || return 1
  [ -s $L/placements_$s.csv ] || step "prep_s$s" $PW $Q/p1_location_prep.py || return 1
  for pos in inside outside; do
    [ -s $L/$pos/results/xray-location_$pos/best_model_$s.pth.tar ] && [ -s $L/$pos/results/xray-location_$pos/last_model_$s.pth.tar ] && continue
    ( mkdir -p $L/$pos; step "train_s${s}_$pos" $PV $Q/p1_location.py train --position $pos >> $L/$pos/train.log 2>&1 ) &
  done; wait
  for pos in inside outside; do
    [ -s $L/$pos/scores.csv.gz ] || step "score_s${s}_$pos" $PV $Q/p1_location.py score --position $pos || return 1
    stage $s $L/$pos/scores.csv.gz scores/location_${pos}_$s.csv.gz
    [ -s $L/$pos/arms/summary.json ] || step "arms_abc_s${s}_$pos" $PW $Q/p1_location_arms.py --position $pos || return 1
    for a in a_mask_cmc_their_net b_dino518_mask_cmc c_umte; do stage $s $L/$pos/arms/preds_$a.csv.gz scores/location_${pos}_${a}_$s.csv.gz; done
    for st in train features heads; do step "armd_${st}_s${s}_$pos" $PW $Q/armd_train.py --setting p1_loc_$pos --stage $st --workers 8 || return 1; done
    stage $s $L/$pos/arms/preds_d_effb0_mask_cmc.csv.gz scores/location_${pos}_d_effb0_mask_cmc_$s.csv.gz
  done
  echo "$(date -u +%F_%T) seed $s done"; }
for s in ${SEEDS:-3232 3221}; do ( seed_job $s || echo "FAILED P1LOC seed $s" ) > $D/replication/viviano/p1loc_seed$s.log 2>&1 & done; wait
grep -h "FAILED" $D/replication/viviano/p1loc_seed*.log && echo "FAILED P1LOC (see seed logs)" || echo "$(date -u +%F_%T) P1 location seeds done"
