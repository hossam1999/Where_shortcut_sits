#!/usr/bin/env bash
# Round-9 search driver (docs/ROUND9_SEARCH_LEDGER.md): reference arms, then the 12 candidates in order; after every
# candidate: evaluate, append to the ledger, commit, push. Stops at 08:15 or when two candidates meet every component.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" TQDM_DISABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
L=results/round9_search/logs; mkdir -p $L
S=scripts/round9_search; BR=claude/festive-bohr-95evo5
DEADLINE=$(date -d "2026-09-29 08:15" +%s)
python $S/search.py --candidate baselines > $L/baselines.log 2>&1 && python $S/search.py --candidate baselines --evaluate >> $L/baselines.log 2>&1
git add results/round9_search/*.csv 2>/dev/null; git commit -q -m "Round 9 search: reference arms on development data

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin $BR
i=0; full=0
for c in mask_ba full_ba mask_cmc_ba full_cmc_ba mask_poe mask_la mask_moments mask_condadv mask_cnc mask_cfc mask_vrex mask_irm; do
  i=$((i+1))
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then echo "deadline reached before $c"; break; fi
  echo "=== $(date '+%F %T') $c"
  python $S/search.py --candidate $c > $L/$c.log 2>&1 && python $S/search.py --candidate $c --evaluate >> $L/$c.log 2>&1
  python $S/ledger_append.py $c $i >> $L/$c.log 2>&1 || printf '\n### %s. %s\n- crashed (see results/round9_search/logs/%s.log).\n' $i $c $c >> docs/ROUND9_SEARCH_LEDGER.md
  git add docs/ROUND9_SEARCH_LEDGER.md results/round9_search/*.csv 2>/dev/null
  git commit -q -m "Round 9 search: candidate $i ($c), development proxy

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin $BR
  full=$(python - <<PY
import pandas as pd
s = pd.read_csv("results/round9_search/summary.csv")
s = s[s.candidate == s.arm]
print(int(((s.met == s.components) & (s.candidate != "baselines")).sum()))
PY
)
  echo "candidates meeting every component: $full"
  if [ "$full" -ge 2 ]; then echo "stopping: two candidates meet every proxy component"; break; fi
done
echo "=== $(date '+%F %T') search finished"
