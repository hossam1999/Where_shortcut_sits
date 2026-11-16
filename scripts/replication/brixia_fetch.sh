#!/bin/bash
# BrixIA DICOM archive (15 split parts) from the authors' private OSF link. The link and the file listing are read from
# /workspace/secrets (outside the repository, never committed). Each part is checked against the SHA-256 that OSF reports;
# then the parts are concatenated and extracted. Licence: research use only, no re-identification.
set -uo pipefail
S=/workspace/secrets; D=${WTSS_DATA:-/workspace/data}/brixia; P=$D/x/data/dicom
VO=$(grep -o 'view_only=[0-9a-f]*' $S/brixia_url); R=https://files.de-1.osf.io/v1/resources/2j8d5/providers/osfstorage
python3 - $S/brixia_dicom_listing.json > $D/parts.tsv <<'PY'
import json,sys
for x in json.load(open(sys.argv[1]))['data']:
    a=x['attributes']
    if a['name'].startswith('dicom_archive_v2'): print(a['name'], a['path'], a['extra']['hashes']['sha256'], a['size'], sep='\t')
PY
while IFS=$'\t' read name path sha size; do
  f=$P/$name
  if [ -f $f ] && [ "$(sha256sum $f | cut -d' ' -f1)" = "$sha" ]; then echo "$name OK (present)"; continue; fi
  rm -f $f; aria2c -q -x4 --max-tries=10 --retry-wait=30 -d $P -o $name "$R$path?$VO"
  [ "$(sha256sum $f | cut -d' ' -f1)" = "$sha" ] && echo "$name OK (downloaded)" || echo "$name SHA256 MISMATCH"
done < $D/parts.tsv
if ! grep -q MISMATCH $D/fetch.log 2>/dev/null; then
  mkdir -p $D/dicom && cat $(cut -f1 $D/parts.tsv | sort | sed "s#^#$P/#") | tar -xz -C $D/dicom && echo "EXTRACTED $(find $D/dicom -name '*.dcm' | wc -l) dcm"
fi
