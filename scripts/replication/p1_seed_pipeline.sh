#!/bin/bash
# Score one P1 seed's trained models and run our arms on them (preliminary per-seed head-to-head).
#   p1_seed_pipeline.sh <xray|rsna> <seed>
# Their methods (their code, P1 env): Standard; ActDiff; Masked = for cardiomegaly the Standard model scored with test
# masking (A5), for RSNA their Masked model as configured (label-dependent test input, A11). Our arms (our env): ROIs,
# arm a features (cardiomegaly: their circle on every split; RSNA: PSPNet lungs through their Masked net, A11), arms a/b/c.
# Every GPU step waits for free memory (gpu_wait.sh). Outputs under $WTSS_DATA/replication/viviano/{preds,arms}.
set -euo pipefail
t=$1; s=$2
Q=/root/Where_shortcut_sits/scripts/replication; V=/workspace/third_party/viviano
D=${WTSS_DATA:-/workspace/data}; R=$D/replication/viviano/runs; O=$D/replication/viviano/preds; A=$D/replication/viviano/arms/${t}_$s
mkdir -p $O $A
model(){ echo $R/${t}_$1_$s/results/${t}-seeds_$1/best_model_$s.pth.tar; }
done_run(){ [ -f $R/${t}_$1_$s/results/${t}-seeds_$1/stats_$s.pkl ]; }
score(){  # <method config> <model method> <out name> [extra args]
  local cfg=$1 mm=$2 name=$3; shift 3
  [ -f $O/$name.csv.gz ] && return 0
  until done_run $mm; do sleep 300; done
  $Q/gpu_wait.sh 3000 bash -c ". /workspace/envs/viviano/bin/activate; export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONPATH=$V:$V/activmask; \
    cd $V && python -u $Q/viviano_score.py --config $V/activmask/config/${t}-seeds/${t}-seeds_$cfg.yml --seed $s \
    --model $(model $mm) --out $O/$name $*"
}
score resnet resnet ${t}_standard_$s
if [ $t = xray ]; then
  score resnet resnet ${t}_masked_$s --mask_all on            # A5: Masked = Standard model, test (and every split) masked
  FA=$O/${t}_masked_${s}_features.npz                         # arm a: circle-masked features, every split
else
  score resnet_maskall resnet_maskall ${t}_masked_$s          # their RSNA Masked, as configured (A11: label-dependent)
fi
P1ENV="WTSS_DATA=$D PYTHONPATH=/root/Where_shortcut_sits/src /workspace/venv-wtss/bin/python -u $Q/p1_arms.py --task $t --seed $s --split_csv $O/${t}_standard_$s.csv.gz"
[ -f $A/roi224_test.npz ] || $Q/gpu_wait.sh 3000 bash -c "$P1ENV --stage rois"
if [ $t = rsna ]; then                                        # A12: arm a = lungs (A11) through the ActDiff network
  score resnet_actdiff resnet_actdiff ${t}_arma_lungs_$s --mask_all off --external_roi $A
  FA=$O/${t}_arma_lungs_${s}_features.npz
fi
[ -f $A/summary.json ] || $Q/gpu_wait.sh 5500 bash -c "$P1ENV --stage arms --arm_a_features $FA"
score resnet_actdiff resnet_actdiff ${t}_actdiff_$s
echo "pipeline done: $t $s"
