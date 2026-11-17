"""results/replication/ARTIFACT_MANIFEST.csv: every large replication output kept only under $WTSS_DATA (checkpoints,
features, caches, full predictions), with path, size and SHA-256, so we know exactly what would be lost with the disk.
Raw datasets are listed in docs/DATA_MANIFEST.md instead. Hashes are cached by (path, size, mtime), so reruns are cheap.

    python scripts/replication/artifact_manifest.py
"""
import csv
import hashlib
import os
from pathlib import Path

D = Path(os.environ.get("WTSS_DATA", "/workspace/data"))
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "replication" / "ARTIFACT_MANIFEST.csv"
EXT = (".pt", ".pth", ".tar", ".ckpt", ".npz", ".npy", ".csv.gz", ".csv", ".pkl", ".hdf5", ".h5")
SKIP = ("/smoke", "/timing/", "runs_superseded", "runs_cancelled")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    old = {}
    if OUT.exists():
        for r in csv.DictReader(open(OUT)):
            old[(r["path"], r["size"], r["mtime"])] = r["sha256"]
    rows = []
    for p in sorted((D / "replication").rglob("*")):
        s = str(p)
        if not p.is_file() or not s.endswith(EXT) or any(k in s for k in SKIP) or p.stat().st_size < 100_000:
            continue
        st = p.stat()
        key = (str(p.relative_to(D)), str(st.st_size), str(int(st.st_mtime)))
        rows.append({"path": key[0], "size": key[1], "mtime": key[2], "sha256": old.get(key) or sha(p)})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "size", "mtime", "sha256"])
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} artifacts, {sum(int(r['size']) for r in rows) / 1e9:.1f} GB -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
