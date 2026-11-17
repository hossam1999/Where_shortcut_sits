#!/bin/bash
# Heartbeat (author's rule, 2026-11-17): every 2 h, even if nothing finished, refresh the auto block of
# docs/REPLICATION_RESUME.md (GPU jobs, run progress, downloads), export new per-image test scores, refresh the artifact
# manifest, and commit + push to both branches. Runs forever:  nohup scripts/replication/heartbeat.sh &
# One pass only:  scripts/replication/heartbeat.sh once
cd /root/Where_shortcut_sits
D=/workspace/data
block(){
  echo "<!-- HEARTBEAT START -->"
  echo "### Heartbeat $(date -u +'%F %H:%M UTC') (auto-generated)"
  echo; echo "GPU jobs (PID, MiB, command):"; echo '```'
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | while IFS=, read p m; do
    [ -d /proc/$p ] && echo "$p $m $(ps -o etime=,args= -p $p | cut -c1-150)"; done
  echo '```'
  echo "- P2 Vanilla: $(grep -o 'Epoch [0-9]*:' $D/replication/pahde/runs/vanilla_resnet50d_seed0/train.log 2>/dev/null | tail -1) of 300; checkpoints: $(ls $D/replication/pahde/runs/vanilla_resnet50d_seed0/checkpoints/*/ 2>/dev/null | tr '\n' ' ')"
  for r in $D/replication/isnet/runs/isnet_*; do [ -f $r/train_log.csv ] && echo "- P3 $(basename $r): last epoch $(tail -1 $r/train_log.csv | cut -d, -f1) of 47"; done
  for r in $D/replication/saab/runs/*/; do [ -f $r/log.csv ] && echo "- P4 $(basename $r): last epoch $(tail -1 $r/log.csv | cut -d, -f1) of 149"; done
  echo "- BIMCV: $(($(wc -l < $D/bimcv/fetch_log.tsv)-1)) of 531 archives ($(grep -c FAIL $D/bimcv/fetch_log.tsv) SHA1 failures)"
  echo "- Disk: $(df -h /workspace | tail -1 | awk '{print $3" used, "$4" free"}')"
  echo "<!-- HEARTBEAT END -->"
}
beat(){
  exec 8>/tmp/wtss_git.lock; flock 8
  WTSS_DATA=$D /workspace/venv-wtss/bin/python scripts/replication/export_scores.py > /dev/null 2>&1
  WTSS_DATA=$D nice /workspace/venv-wtss/bin/python scripts/replication/artifact_manifest.py > /dev/null 2>&1
  python3 - "$(block)" <<'PY'
import sys, re
p = "docs/REPLICATION_RESUME.md"; s = open(p).read()
s = re.sub(r"<!-- HEARTBEAT START -->.*<!-- HEARTBEAT END -->", lambda m: sys.argv[1], s, flags=re.S)
open(p, "w").write(s)
PY
  git add docs/REPLICATION_RESUME.md results/replication 2>/dev/null
  git commit -q -m "Heartbeat $(date -u +'%F %H:%M UTC')

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" 2>/dev/null
  for i in 1 2 3; do
    git pull -q --rebase origin claude/festive-bohr-95evo5 && git push -q origin HEAD:claude/festive-bohr-95evo5 \
      && git push -q origin HEAD:stage-reports && break
    sleep 30
  done
  flock -u 8
}
if [ "${1:-}" = once ]; then beat; exit; fi
while true; do beat; sleep 7200; done
