#!/bin/bash
cd /root/wtss
source /venv/main/bin/activate
export PYTHONPATH=/root/wtss/src TQDM_DISABLE=1 WTSS_DATA=/root/data
python scripts/run_thyroid_traps.py --cohort ovary --backbone dino518 > logs/ovary_dino518.log 2>&1
A="erm mask balanced mask_balanced dfr mask_dfr mte mte_balanced mte_aug mte_protect mte_protect_balanced jtt mask_jtt umte_jtt"
python scripts/run_thyroid_traps.py --cohort ovary --backbone dino518 --generic --tag universal --save_val --arms $A > logs/ovary_dino518_universal.log 2>&1
echo DONE >> logs/ovary_dino518_universal.log
python scripts/run_synthetic.py --cohort ovary --backbone dino518 --artifact caliper --tag main --workers 4 --arms erm mask inpaint balanced dfr leace --proposed > logs/synth_ovary_dino518.log 2>&1
echo "rc=$?" >> logs/synth_ovary_dino518.log
