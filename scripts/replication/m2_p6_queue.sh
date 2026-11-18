#!/bin/bash
# Machine 2, P6 step 1 (A29): their ERM and EPVT, once each (seed 0), on Bissoto trap sets bias {0,0.3,0.5,0.7,0.9,1}, split 1,
# exact README hparams (1501 steps, batch 26 per domain, lr 5e-6, lr_classifier 5e-5, wd_classifier 1e-5, EPVT prompt_dim 10).
# Waits until P4 Seg-ERM splits 1-3 are scored (author: P4 first; pair only if measured not to slow P4), then runs alone.
#   nohup scripts/replication/m2_p6_queue.sh > /workspace/logs/m2_p6_queue.log 2>&1 &
D=${WTSS_DATA:-/workspace/data}; Q=/root/Where_shortcut_sits/scripts/replication; E=/workspace/third_party/epvt; PY=/workspace/envs/epvt/bin/python
export WANDB_MODE=disabled EPVT_EVAL_ROOT=/workspace/third_party/bissoto_ag/trap_sets/ EPVT_IMFOLDER=$D/epvt/isic2019_plain/ EPVT_ROOT=$D/epvt/
until [ -s $D/replication/saab/runs/seg_split1/preds_test.csv ] && [ -s $D/replication/saab/runs/seg_split2/preds_test.csv ] && [ -s $D/replication/saab/runs/seg_split3/preds_test.csv ]; do sleep 120; done
H='{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier":1e-5}'
HP='{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier": 1e-5, "prompt_dim":10}'
for b in 1 0 0.9 0.7 0.5 0.3; do
  [ -d $D/epvt/bias${b}_1 ] || WTSS_DATA=$D /workspace/venv-wtss/bin/python $Q/p6_epvt_group.py --bias $b --split 1 > /dev/null
  for m in erm epvt; do
    O=$D/epvt/runs/${m}_bias${b}_1; [ -s $O/${m}__test_bias_${b}_1.json ] && continue
    mkdir -p $O; echo "$(date -u +%F_%T) start $m bias $b"
    if [ $m = erm ]; then CMD="$PY -m domainbed.scripts.train_erm --data_dir=$D/epvt/bias${b}_1/ --steps 1501 --dataset SKIN --test_env 0 --algorithm ERM --output_dir $O --hparams '$H' --exp erm"
    else CMD="$PY -m domainbed.scripts.train_epvt --data_dir=$D/epvt/bias${b}_1/ --steps 1501 --dataset SKIN --test_env 0 --algorithm DoPrompt_group_decompose --output_dir $O --hparams '$HP' --exp epvt"; fi
    (cd $E && EPVT_SPLIT=${b}_1 $Q/gpu_wait.sh 12000 bash -c "$CMD > $O/train.log 2>&1") || { echo "$(date -u +%F_%T) FAILED train $m bias $b"; exit 1; }
    (cd $E && EPVT_SPLIT=${b}_1 $PY $Q/p6_epvt_score.py --ckpt $O/$m.pkl --test test_bias_${b}_1.csv >> $O/score.log 2>&1) || { echo "FAILED score $m $b"; exit 1; }
    echo "$(date -u +%F_%T) done $m bias $b: $(tail -1 $O/score.log)"
  done
done
echo "$(date -u +%F_%T) P6 step-1 runs done"
