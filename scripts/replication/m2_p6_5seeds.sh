#!/bin/bash
# A34: ERM and EPVT at bias 1, split 1, seeds 1-4 (seed 0 = the A33 run), A33 step count; then 5-seed means vs the gate.
# Start manually AFTER P8's gate and our P8 arms, before P7.   nohup scripts/replication/m2_p6_5seeds.sh > /workspace/logs/m2_p6_5seeds.log 2>&1 &
D=${WTSS_DATA:-/workspace/data}; Q=/root/Where_shortcut_sits/scripts/replication; E=/workspace/third_party/epvt; PY=/workspace/envs/epvt/bin/python
export EPVT_GRAD_CKPT=1 EPVT_PATIENCE=22 WANDB_MODE=disabled EPVT_EVAL_ROOT=/workspace/third_party/bissoto_ag/trap_sets/ EPVT_IMFOLDER=$D/epvt/isic2019_plain/ EPVT_ROOT=$D/epvt/ EPVT_SPLIT=1_1
ST=$(/workspace/venv-wtss/bin/python -c "import json;c=json.load(open('$D/epvt/bias1_1/counts.json'))['per_domain'];print(100*int(min(v/26 for v in c.values()))+1)")
H='{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier":1e-5}'
HP='{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier": 1e-5, "prompt_dim":10}'
for s in ${SEEDS:-1 2 3 4}; do for m in erm epvt; do
  O=$D/epvt/runs/${m}_bias1_1_seed$s; [ -s $O/${m}__test_bias_1_1.json ] && continue; mkdir -p $O; echo "$(date -u +%F_%T) $m seed $s"
  if [ $m = erm ]; then CMD="$PY -m domainbed.scripts.train_erm --data_dir=$D/epvt/bias1_1/ --steps $ST --dataset SKIN --test_env 0 --algorithm ERM --output_dir $O --hparams '$H' --exp erm --seed $s"
  else CMD="$PY -m domainbed.scripts.train_epvt --data_dir=$D/epvt/bias1_1/ --steps $ST --dataset SKIN --test_env 0 --algorithm DoPrompt_group_decompose --output_dir $O --hparams '$HP' --exp epvt --seed $s"; fi
  (cd $E && MAX_OTHER_JOBS=1 $Q/gpu_wait.sh 8000 bash -c "$CMD > $O/train.log 2>&1") || { echo "FAILED $m $s"; exit 1; }
  (cd $E && $PY $Q/p6_epvt_score.py --ckpt $O/$m.pkl --test test_bias_1_1.csv >> $O/score.log 2>&1) || { echo "FAILED score $m $s"; exit 1; }
  echo "$(date -u +%F_%T) done $m seed $s: $(tail -1 $O/score.log)"
done; done
echo "$(date -u +%F_%T) A34 seeds done"
