"""Check the re-downloaded phase-5 cohorts against docs/DATA_MANIFEST.md: file count, total bytes and the listing SHA-256
(sorted "relative path<TAB>size" lines, as scripts/verify/data_manifest.py computes them).
    WTSS_DATA=/workspace/data python scripts/data/verify_phase5.py
"""
import hashlib
import os
import re
from pathlib import Path

D = Path(os.environ["WTSS_DATA"])
ROOT = Path(__file__).resolve().parents[2]
ROWS = {"TN3K": ["us/tncd/datasets.rar", "us/tncd/ds"], "MMOTU": ["ovary/mmotu"], "SEE-AI": ["capsule/seeai"],
        "Capsule expert contamination masks": ["capsule/datasets"]}
man = (ROOT / "docs/DATA_MANIFEST.md").read_text().splitlines()
ok = True
for name, paths in ROWS.items():
    fs = []
    for rel in paths:
        p = D / rel
        fs += [p] if p.is_file() else sorted(q for q in p.rglob("*") if q.is_file()) if p.is_dir() else []
    fs = sorted(fs)
    lines = sorted(f"{f.relative_to(D)}\t{f.stat().st_size}" for f in fs)
    h = hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
    ref = next(l for l in man if l.startswith("| " + name) or (name in l and paths[0] in l))
    cells = [c.strip() for c in ref.split("|")]
    n_ref, size_ref, h_ref = int(cells[-4]), int(cells[-3]), cells[-2].strip("`")
    match = (len(fs), sum(f.stat().st_size for f in fs), h) == (n_ref, size_ref, h_ref)
    ok &= match
    print(f"{name}: files {len(fs)} (manifest {n_ref}), bytes {sum(f.stat().st_size for f in fs)} (manifest {size_ref}), "
          f"listing hash {'MATCH' if h == h_ref else 'differs'} -> {'OK' if match else 'CHECK'}")
print("ALL OK" if ok else "SOME DIFFER")
