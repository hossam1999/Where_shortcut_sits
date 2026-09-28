#!/usr/bin/env bash
# Regenerate every stage report from the result files, compile the seven PDFs and run the number audit.
#   bash stages/build_all.sh
# Needs: the Python environment of the repository (PYTHONPATH=src), a LaTeX engine (tectonic, or pdflatex + bibtex),
# pdfinfo. Figures that need the raw images (example panels) are kept from the last run when the data are not on the
# machine. Exit code 1 if a build or the audit fails.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
status=0
compile() {  # $1 = stage folder, $2 = stageN
  if command -v tectonic >/dev/null && tectonic -X compile "$2.tex" > "/tmp/${2}_latex.txt" 2>&1; then return 0; fi
  pdflatex -interaction=nonstopmode "$2.tex" > "/tmp/${2}_latex.txt" 2>&1 && bibtex "$2" >> "/tmp/${2}_latex.txt" 2>&1
  pdflatex -interaction=nonstopmode "$2.tex" >> "/tmp/${2}_latex.txt" 2>&1
  pdflatex -interaction=nonstopmode "$2.tex" >> "/tmp/${2}_latex.txt" 2>&1
  local rc=$?; rm -f "$2".{aux,bbl,blg,out,toc,log}; return $rc
}
for d in stages/stage1_* stages/stage2_* stages/stage3_* stages/stage4_* stages/stage5_* stages/stage6_* stages/stage7_*; do
  n=$(basename "$d" | cut -d_ -f1)
  echo "== $n: tables and figures"
  python -W ignore "$d/make_$n.py" || { echo "!! make_$n.py failed"; status=1; }
  echo "== $n: LaTeX"
  (cd "$d" && compile "$d" "$n") || { echo "!! $n.tex failed (see /tmp/${n}_latex.txt)"; status=1; }
  [ -f "$d/$n.pdf" ] && echo "   $(pdfinfo "$d/$n.pdf" | awk '/Pages/{print $2}') pages"
done
echo "== combined full report"
python stages/combined/make_combined.py || status=1
(cd stages/combined && compile stages/combined full_report) || { echo "!! full_report.tex failed"; status=1; }
[ -f stages/combined/full_report.pdf ] && echo "   $(pdfinfo stages/combined/full_report.pdf | awk '/Pages/{print $2}') pages"
echo "== number audit (stage reports and paper)"
python scripts/verify/audit_numbers_final.py --docs all || status=1
exit $status
