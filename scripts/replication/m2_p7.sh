#!/bin/bash
# P7 (RR-ClArC, A28): per architecture their preprocessing (activations for the CAV) -> their RR-ClArC correction (10 epochs)
# -> our one-time scoring (p7_score.py). Each step skips finished work; GPU via gpu_wait (alone). Fills the GPU while the
# phase-5 pre-registration waits for approval (author). Log: /workspace/logs/m2_p7.log (watched by the watchdog).
#   WTSS_DATA=/workspace/data nohup scripts/replication/m2_p7.sh > /workspace/logs/m2_p7.log 2>&1 &
D=${WTSS_DATA:-/workspace/data}; Q=/root/Where_shortcut_sits/scripts/replication; RC=/workspace/third_party/rrclarc; PY=/workspace/envs/r2r/bin/python
export WANDB_MODE=offline WTSS_DATA=$D
$PY $Q/p7_configs.py || { echo "FAILED P7 configs"; exit 1; }
cd $RC
for arch in ${ARCHS:-resnet18 efficientnet_b0 vgg16}; do
  C=config_files/correcting_isic_attacked/local/${arch}_RRClarc_p7.yaml
  p=$(grep "^p_artifact:" $C | awk '{print $2}'); L=$(grep "^layer_name:" $C | awk '{print $2}')
  A=results/global_relevances_and_activations/isic_attacked_lsb-$p/$arch/${L}_class_0_all.pth
  if [ ! -s $A ]; then echo "$(date -u +%F_%T) preprocessing $arch"
    # inference-only relevance/activation pass at batch ${PREP_BS:-16} (VGG-16's LRP pass at 64 exceeds 16 GB); the output path
    # depends on dataset / model / layer only, and per-image activations do not depend on the batch size
    CP=${C%.yaml}_prep.yaml; sed "s/^batch_size: .*/batch_size: ${PREP_BS:-16}/" $C > $CP
    MAX_OTHER_JOBS=0 $Q/gpu_wait.sh 6000 $PY -m experiments.preprocessing.run_preprocessing --config_file $CP || { echo "FAILED P7 preprocessing $arch"; exit 1; }; fi
  if [ ! -s checkpoints/${arch}_RRClarc_p7/last.ckpt ]; then echo "$(date -u +%F_%T) correction $arch"
    MAX_OTHER_JOBS=0 $Q/gpu_wait.sh 6000 $PY -m model_training.start_model_correction --config_file $C || { echo "FAILED P7 correction $arch"; exit 1; }; fi
  if [ ! -s $D/results_push/results/replication/rrclarc/p7/${arch}_summary.csv ]; then echo "$(date -u +%F_%T) scoring $arch"
    MAX_OTHER_JOBS=0 $Q/gpu_wait.sh 6000 $PY $Q/p7_score.py $arch || { echo "FAILED P7 scoring $arch"; exit 1; }; fi
  echo "$(date -u +%F_%T) done $arch"
done
echo "$(date -u +%F_%T) P7 runs done"
