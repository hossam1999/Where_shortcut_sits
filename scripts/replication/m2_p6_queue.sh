#!/bin/bash
# Machine 2, P6 step 1 (A29): their ERM and EPVT, once each (seed 0), on Bissoto trap sets bias {0,0.3,0.5,0.7,0.9,1}, split 1,
# README hparams (batch 26 per domain, lr 5e-6, lr_classifier 5e-5, wd_classifier 1e-5, EPVT prompt_dim 10) with the PAPER's
# protocol (author, before any P6 result): 100 epochs in their code's epoch unit (int(min domain in-split / 26) steps; +1 final
# step as in their 1501), early stopping patience 22 epochs on val_bias ROC-AUC. BIASES (default "1"): bias 1 first, timed.
# Waits until P4 Seg-ERM splits 1-3 are scored (author: P4 first; pair only if measured not to slow P4), then runs alone.
#   nohup scripts/replication/m2_p6_queue.sh > /workspace/logs/m2_p6_queue.log 2>&1 &
D=${WTSS_DATA:-/workspace/data}; Q=/root/Where_shortcut_sits/scripts/replication; E=/workspace/third_party/epvt; PY=/workspace/envs/epvt/bin/python
export EPVT_GRAD_CKPT=1 EPVT_PATIENCE=22 WANDB_MODE=disabled EPVT_EVAL_ROOT=/workspace/third_party/bissoto_ag/trap_sets/ EPVT_IMFOLDER=$D/epvt/isic2019_plain/ EPVT_ROOT=$D/epvt/
# Start rule (author 2026-11-18): P6_START=pair (default) -> start once A31 reruns 1-2 are scored and split 3 runs, sharing the GPU
# (kept only if m2_pair_measure.sh measures total throughput > 1.0); P6_START=after -> wait for all three reruns, run alone.
R=$D/replication/saab/runs
if [ "${P6_START:-pair}" = pair ]; then
  until [ -s $R/seg_pil_split1/preds_test.csv ] && [ -s $R/seg_pil_split2/preds_test.csv ] && grep -q "'epoch': 1," $D/replication/saab/seg_pil_split3.log 2>/dev/null; do sleep 60; done
  MOJ=1; NEED=6500
else
  until [ -s $R/seg_pil_split1/preds_test.csv ] && [ -s $R/seg_pil_split2/preds_test.csv ] && [ -s $R/seg_pil_split3/preds_test.csv ]; do sleep 120; done
  MOJ=0; NEED=8000
fi
H='{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier":1e-5}'
HP='{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier": 1e-5, "prompt_dim":10}'
for b in ${BIASES:-1}; do
  [ -d $D/epvt/bias${b}_1 ] || WTSS_DATA=$D /workspace/venv-wtss/bin/python $Q/p6_epvt_group.py --bias $b --split 1 > /dev/null
  ST=$(/workspace/venv-wtss/bin/python -c "import json;c=json.load(open('$D/epvt/bias${b}_1/counts.json'))['per_domain'];print(100*int(min((v-int(v*0.2))/26 for v in c.values()))+1)")
  for m in erm epvt; do
    O=$D/epvt/runs/${m}_bias${b}_1; [ -s $O/${m}__test_bias_${b}_1.json ] && continue
    mkdir -p $O; echo "$(date -u +%F_%T) start $m bias $b"
    if [ $m = erm ]; then CMD="$PY -m domainbed.scripts.train_erm --data_dir=$D/epvt/bias${b}_1/ --steps $ST --dataset SKIN --test_env 0 --algorithm ERM --output_dir $O --hparams '$H' --exp erm"
    else CMD="$PY -m domainbed.scripts.train_epvt --data_dir=$D/epvt/bias${b}_1/ --steps $ST --dataset SKIN --test_env 0 --algorithm DoPrompt_group_decompose --output_dir $O --hparams '$HP' --exp epvt"; fi
    (cd $E && EPVT_SPLIT=${b}_1 MAX_OTHER_JOBS=$MOJ $Q/gpu_wait.sh $NEED bash -c "$CMD > $O/train.log 2>&1") || { echo "$(date -u +%F_%T) FAILED train $m bias $b"; exit 1; }
    (cd $E && EPVT_SPLIT=${b}_1 $PY $Q/p6_epvt_score.py --ckpt $O/$m.pkl --test test_bias_${b}_1.csv >> $O/score.log 2>&1) || { echo "FAILED score $m $b"; exit 1; }
    echo "$(date -u +%F_%T) done $m bias $b: $(tail -1 $O/score.log)"
  done
done
echo "$(date -u +%F_%T) P6 step-1 runs done"
