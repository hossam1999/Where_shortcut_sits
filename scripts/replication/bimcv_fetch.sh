#!/bin/bash
# Fetch only the BIMCV COVID-19+ archives that hold P3's rebuilt test images (scripts/replication/bimcv_select.py),
# one at a time: download -> verify SHA1 (owners' sha1sum.txt) -> extract only the selected PNG + JSON -> delete archive.
# Resumable (done list). Log: bytes downloaded, SHA1 result, files kept, per archive.
# Licence: BIMCV-COVID19 Research Use Agreement (academic use, cite, no redistribution). Never commit images.
set -uo pipefail
B=${WTSS_DATA:-/workspace/data}/bimcv; M=$B/meta; T=$B/tmp; O=$B/images
U="https://b2drop.bsc.es/public.php/webdav"; AUTH=(--http-user=BIMCV-COVID19-cIter_1_2_3 --http-passwd=)
mkdir -p $T $O; touch $B/done.txt; LOG=$B/fetch_log.tsv
[ -s $LOG ] || printf "archive\tbytes\tsha1\tkept_png\tkept_json\n" > $LOG
python3 - "$B" <<'PY'
import csv,sys,collections,os
B=sys.argv[1]; d=collections.defaultdict(list)
for r in csv.DictReader(open(f"{B}/selection.csv")):
    d[r["archive"]] += [r["path"], r["path"][:-4] + ".json"]
os.makedirs(f"{B}/members", exist_ok=True)
for a, m in d.items(): open(f"{B}/members/{a}.txt", "w").write("\n".join(m) + "\n")
PY
for f in $B/members/*.txt; do
  a=$(basename $f .txt); grep -qx "$a" $B/done.txt && continue
  rm -f $T/$a $T/$a.aria2
  aria2c -q -x8 -s8 --max-tries=10 --retry-wait=30 "${AUTH[@]}" -d $T -o $a "$U/$a" || { echo "download failed $a" >&2; continue; }
  bytes=$(stat -c %s $T/$a); want=$(awk -v n="$a" '$2==n{print $1}' $M/sha1sum.txt); got=$(sha1sum $T/$a | cut -d' ' -f1)
  if [ "$want" != "$got" ]; then printf "%s\t%s\tFAIL\t0\t0\n" $a $bytes >> $LOG; rm -f $T/$a; continue; fi
  tar -xzf $T/$a -C $O -T $f 2>/dev/null
  np=$(grep -c '\.png$' $f); kp=0; kj=0
  while read p; do [ -f "$O/$p" ] && { [[ $p == *.png ]] && kp=$((kp+1)) || kj=$((kj+1)); }; done < $f
  printf "%s\t%s\tOK\t%s\t%s\n" $a $bytes $kp $kj >> $LOG
  rm -f $T/$a; echo $a >> $B/done.txt
done
echo "BIMCV fetch finished: $(grep -c OK $LOG) archives OK, $(grep -c FAIL $LOG) failed"
