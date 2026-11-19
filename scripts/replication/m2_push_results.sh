#!/bin/bash
# Machine 2 has no persistent disk (author, 2026-11-19): push every result the moment it exists. Every 2 min, finished results
# are copied into a separate detached git worktree (never the main checkout, so in-progress edits there are never committed),
# committed and pushed to claude/festive-bohr-95evo5 and stage-reports. Only small result files (csv, csv.gz, json, jsonl,
# txt, log, md; < 10 MB) — never checkpoints (.pth/.pkl/.pt), features or data.
#   Sources (a run is copied only once it is finished):
#   - P8 DISC/ERM runs: $WTSS_DATA/disc/output/ISIC/{ERM,DISC}/*-trapset_id=k with 100 epochs in test.csv
#       -> results/replication/disc/runs/{erm,disc}_trapset<k>/ {train,val,test}.csv, log.txt
#   - P6 runs: $WTSS_DATA/epvt/runs/<run>/ once its *__test_*.json exists, + A34_gate.json
#       -> results/replication/epvt/runs/<run>/
#   - pairing measurements /workspace/logs/pair_*.json -> results/replication/machine2/pairing/
#   - anything a script drops into $WTSS_DATA/results_push/<repo path> (our P8 arms, phase 5, P7) -> <repo path>
#   WTSS_DATA=/workspace/data nohup scripts/replication/m2_push_results.sh > /workspace/logs/m2_push.log 2>&1 &
D=${WTSS_DATA:-/workspace/data}; REPO=/root/Where_shortcut_sits; W=/root/wtss_push; B=claude/festive-bohr-95evo5
mkdir -p $D/results_push
[ -d $W ] || git -C $REPO worktree add -q --detach $W origin/$B
cd $W || exit 1
cp_small(){ # src dst: copy one file if small and allowed
  local s=$1 t=$2
  [[ "$s" =~ \.(csv|csv\.gz|json|jsonl|txt|log|md)$ ]] || return
  [ $(stat -c %s "$s") -lt 10000000 ] || { echo "$(date -u +%T) skip large $s"; return; }
  mkdir -p "$(dirname $t)"; cmp -s "$s" "$t" || cp "$s" "$t"
}
while :; do
  git fetch -q origin && git reset -q --hard origin/$B || { echo "$(date -u +%T) PUSH FAILED: fetch/reset"; sleep 120; continue; }
  # P8 runs
  for m in ERM DISC; do for k in 1 2 3 4 5; do
    d=$(ls -d $D/disc/output/ISIC/$m/*-trapset_id=$k 2>/dev/null | head -1); [ -n "$d" ] || continue
    [ -s $d/test.csv ] && [ $(wc -l < $d/test.csv) -ge 101 ] || continue
    for f in train.csv val.csv test.csv log.txt; do [ -f $d/$f ] && cp_small $d/$f results/replication/disc/runs/$(echo $m | tr A-Z a-z)_trapset$k/$f; done
  done; done
  # P6 runs
  for d in $D/epvt/runs/*/; do r=$(basename $d)
    ls $d/*__test_*.json >/dev/null 2>&1 || continue
    for f in $d/*.json $d/*.csv.gz $d/train.log $d/out.txt $d/err.txt $d/score.log $d/results.jsonl; do [ -f "$f" ] && cp_small $f results/replication/epvt/runs/$r/$(basename $f); done
  done
  [ -f $D/epvt/runs/A34_gate.json ] && cp_small $D/epvt/runs/A34_gate.json results/replication/epvt/A34_gate.json
  # pairing measurements
  for f in /workspace/logs/pair_*.json; do [ -f "$f" ] && cp_small $f results/replication/machine2/pairing/$(basename $f); done
  # staged results from our scripts
  (cd $D/results_push && find . -type f) | while read -r f; do cp_small "$D/results_push/$f" "${f#./}"; done
  git add -A results
  if ! git diff --cached --quiet; then
    n=$(git diff --cached --name-only | wc -l); top=$(git diff --cached --name-only | cut -d/ -f1-4 | sort -u | head -4 | tr '\n' ' ')
    git commit -qm "machine 2 (auto): results — $n files ($top)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
    ok=0; for i in 1 2 3; do
      git push -q origin HEAD:$B && { ok=1; break; }
      git fetch -q origin && git rebase -q origin/$B || { git rebase --abort; break; }
    done
    if [ $ok = 1 ]; then git push -q origin HEAD:stage-reports || echo "$(date -u +%T) PUSH FAILED: stage-reports"
      echo "$(date -u +%T) pushed $n files: $top"
    else echo "$(date -u +%T) PUSH FAILED: $B"; fi
  fi
  sleep 120
done
