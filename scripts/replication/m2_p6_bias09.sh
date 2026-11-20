#!/bin/bash
# A39: ERM and EPVT (A36 mixup) at bias 0.9, split 1, seeds 0-4 (all new), A33 step rule (100 epochs in their unit, patience
# 22), scored once on test_bias_0.9_1; then the bias-0.9 gate (5-seed means within 0.05 of the digitised ERM 0.762 / EPVT 0.792).
D=${WTSS_DATA:-/workspace/data}; Q=/root/Where_shortcut_sits/scripts/replication; E=/workspace/third_party/epvt; PY=/workspace/envs/epvt/bin/python
export EPVT_GRAD_CKPT=1 EPVT_PATIENCE=22 WANDB_MODE=disabled EPVT_EVAL_ROOT=/workspace/third_party/bissoto_ag/trap_sets/ EPVT_IMFOLDER=$D/epvt/isic2019_plain/ EPVT_ROOT=$D/epvt/ EPVT_SPLIT=0.9_1
grep -q "A36 (paper Eq. 3)" $E/domainbed/algorithms.py || { echo "FAILED P6B09 A36 patch missing"; exit 1; }
[ -s $D/epvt/bias0.9_1/counts.json ] || /workspace/venv-wtss/bin/python $Q/p6_epvt_group.py --bias 0.9 --split 1 || { echo "FAILED P6B09 grouping"; exit 1; }
ST=$(/workspace/venv-wtss/bin/python -c "import json;c=json.load(open('$D/epvt/bias0.9_1/counts.json'))['per_domain'];print(100*int(min(v/26 for v in c.values()))+1)")
H='{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier":1e-5}'
HP='{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier": 1e-5, "prompt_dim":10}'
for s in 0 1 2 3 4; do for m in erm epvt; do
  if [ $m = epvt ]; then O=$D/epvt/runs/epvt_a36_bias0.9_1_seed$s; else O=$D/epvt/runs/erm_bias0.9_1_seed$s; fi
  [ -s $O/${m}__test_bias_0.9_1.json ] && continue; mkdir -p $O
  until ! ps -eo args | grep -q "[g]pu_wait.sh 15000 .*--net isnet"; do sleep 30; done   # phase 5's ISNet has priority
  echo "$(date -u +%F_%T) $m bias 0.9 seed $s (steps $ST)"
  if [ $m = erm ]; then CMD="$PY -m domainbed.scripts.train_erm --data_dir=$D/epvt/bias0.9_1/ --steps $ST --dataset SKIN --test_env 0 --algorithm ERM --output_dir $O --hparams '$H' --exp erm --seed $s"
  else CMD="$PY -m domainbed.scripts.train_epvt --data_dir=$D/epvt/bias0.9_1/ --steps $ST --dataset SKIN --test_env 0 --algorithm DoPrompt_group_decompose --output_dir $O --hparams '$HP' --exp epvt --seed $s"; fi
  (cd $E && MAX_OTHER_JOBS=1 $Q/gpu_wait.sh 7000 bash -c "$CMD > $O/train.log 2>&1") || { echo "FAILED P6B09 $m $s"; exit 1; }
  (cd $E && $PY $Q/p6_epvt_score.py --ckpt $O/$m.pkl --test test_bias_0.9_1.csv >> $O/score.log 2>&1) || { echo "FAILED P6B09 score $m $s"; exit 1; }
  echo "$(date -u +%F_%T) done $m bias 0.9 seed $s: $(tail -1 $O/score.log)"
done; done
/workspace/venv-wtss/bin/python - $D <<'PY'
import json, sys, numpy as np
R = sys.argv[1] + "/epvt/runs/"
erm = [json.load(open(R + f"erm_bias0.9_1_seed{s}/erm__test_bias_0.9_1.json"))["roc_auc"] for s in range(5)]
epvt = [json.load(open(R + f"epvt_a36_bias0.9_1_seed{s}/epvt__test_bias_0.9_1.json"))["roc_auc"] for s in range(5)]
r = {"erm": erm, "epvt_a36": epvt, "erm_mean": float(np.mean(erm)), "epvt_mean": float(np.mean(epvt)),
     "gate_erm": abs(np.mean(erm) - 0.762) <= 0.05, "gate_epvt": abs(np.mean(epvt) - 0.792) <= 0.05}
r["gate_erm"], r["gate_epvt"] = bool(r["gate_erm"]), bool(r["gate_epvt"])
json.dump(r, open(R + "A39_bias09_gate.json", "w"), indent=1); print(json.dumps(r))
PY
echo "$(date -u +%F_%T) P6 bias 0.9 seeds done"
