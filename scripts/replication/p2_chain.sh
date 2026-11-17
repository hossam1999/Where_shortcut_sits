#!/bin/bash
# P2 long chain after Vanilla (A12): score Vanilla (+ features for arm a), their activation preprocessing for the CAV
# (class 0 only: RR-ClArC uses cav_scope [0], so the other classes' files are never read; same CAV, less GPU), their
# RR-ClArC grid (lambda in {10, 50} x layers identity_1 / identity_2 / last_conv; 10 epochs each, their configs with
# our paths and our Vanilla as ckpt_path), and per-image scores of each corrected model from last.ckpt (as their
# evaluation). Selection among the grid: mean of clean + biased validation accuracy (section 3), done on CPU afterwards.
# Each step is skipped when its output exists.   nohup scripts/replication/p2_chain.sh > /workspace/logs/p2_chain.log 2>&1 &
set -uo pipefail
cd /root/Where_shortcut_sits
Q=scripts/replication; D=/workspace/data; P=/workspace/third_party/pahde; PY=/workspace/envs/pahde/bin/python
R=$D/replication/pahde; V=$R/runs/vanilla_resnet50d_seed0; CK=$V/checkpoints/config/checkpoint_resnet50d_last.pth
export WANDB_MODE=disabled
fail(){ echo "$(date +%F_%T) FAILED: $1 -- chain stopped, P3 chain NOT started"; exit 1; }
until [ -f $CK ] && ! ps -eo args | grep -q "[v]anilla_resnet50d_seed0/config.yaml"; do sleep 300; done
echo "$(date +%F_%T) Vanilla finished: $CK"
mkdir -p $R/scores $R/mitigation
[ -f $R/scores/vanilla_seed0.csv.gz ] || WTSS_DATA=$D $Q/gpu_wait.sh 4000 $PY -u $Q/p2_score.py --config $V/config.yaml \
  --ckpt $CK --out $R/scores/vanilla_seed0.csv.gz --features || fail "Vanilla scoring"
# their mitigation configs, regenerated with our local_config, then pointed at our Vanilla
cd $P && PYTHONPATH=$P $PY config_files/bias_mitigation_controlled/config_generator_mitigation_isic_attacked.py && cd - > /dev/null
for lamb in 10 50; do for layer in identity_1 identity_2 last_conv; do
  name=resnet50d_RRClarc_lamb${lamb}_adam_lr0.001_${layer}
  $PY - $P/config_files/bias_mitigation_controlled/isic_attacked/local/$name.yaml $R/mitigation/$name.yaml $CK <<'PY'
import sys, yaml
src, dst, ck = sys.argv[1:]
c = yaml.safe_load(open(src)); c["ckpt_path"] = ck
yaml.safe_dump(c, open(dst, "w"))
PY
done; done
# activations of class 0 for the CAV (their run_collect_relevances_and_activations, split "all")
first=$R/mitigation/resnet50d_RRClarc_lamb10_adam_lr0.001_identity_1.yaml
[ -f $R/precomputed_data/global_relevances_and_activations/isic_attacked_microscope-0.2/resnet50d/class_0_all.hdf5 ] || \
  (cd $P && WTSS_DATA=$D $OLDPWD/$Q/gpu_wait.sh 6000 $PY -u -c "
import sys; sys.argv=['x']
from utils.helper import load_config
from experiments.preprocessing.global_collect_relevances_and_activations import run_collect_relevances_and_activations
c = load_config('$first'); run_collect_relevances_and_activations({**c, 'class_idx': 0, 'split': 'all'}, False)") || fail "class-0 activations"
for f in $R/mitigation/*.yaml; do
  name=$(basename $f .yaml)
  [ -f $R/checkpoints_corrected/$name/last.ckpt ] || (cd $P && WTSS_DATA=$D $OLDPWD/$Q/gpu_wait.sh 6000 \
    $PY -u -m experiments.mitigation_experiments.start_model_correction --config_file $f > $R/mitigation/$name.log 2>&1) || fail "fine-tune $name"
  [ -f $R/scores/$name.csv.gz ] || WTSS_DATA=$D $Q/gpu_wait.sh 4000 $PY -u $Q/p2_score.py --config $f \
    --ckpt $R/checkpoints_corrected/$name/last.ckpt --out $R/scores/$name.csv.gz || fail "scoring $name"
done
# hand over to ISNet only when all 6 corrected models are scored (2026-11-17: a failed step used to fall through to here)
n=$(ls $R/scores/resnet50d_RRClarc_*.csv.gz 2>/dev/null | wc -l); [ "$n" -eq 6 ] || fail "only $n of 6 RR-ClArC score files"
echo "$(date +%F_%T) P2 chain done; starting the P3 ISNet chain (alone)"
nohup $Q/p3_chain.sh > /workspace/logs/p3_chain.log 2>&1 &
