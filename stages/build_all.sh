#!/usr/bin/env bash
# Regenerate every stage report from the result files, compile the six PDFs and run the number audit.
#   bash stages/build_all.sh
# Needs: the Python environment of the repository (PYTHONPATH=src), tectonic, pdfinfo. Exit code 1 if a build or the
# audit fails.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
status=0
for d in stages/stage1_* stages/stage2_* stages/stage3_* stages/stage4_* stages/stage5_* stages/stage6_*; do
  n=$(basename "$d" | cut -d_ -f1)
  echo "== $n: tables and figures"
  python -W ignore "$d/make_$n.py" || { echo "!! make_$n.py failed"; status=1; }
  echo "== $n: LaTeX"
  (cd "$d" && tectonic -X compile "$n.tex" > "/tmp/${n}_tectonic.txt" 2>&1) || { echo "!! $n.tex failed (see /tmp/${n}_tectonic.txt)"; status=1; }
  [ -f "$d/$n.pdf" ] && echo "   $(pdfinfo "$d/$n.pdf" | awk '/Pages/{print $2}') pages"
done
echo "== number audit (stage reports and paper)"
python scripts/verify/audit_numbers_final.py --docs all || status=1
exit $status
