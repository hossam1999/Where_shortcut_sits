"""A43 rebuild check (before any P1 training): the rebuilt seed-1234 location test split must match the committed seed-1234
per-image files exactly - image ids, labels, tube flags and row order, for every test set (corr, rev, clean) and both
positions. The test sets are rebuilt with their JointDataset (ratio 0.5, mode "test", seed 1234; 32f7b7d split code + port
patch) and our tube presence hash, exactly as p1_location.py's TubeJointDataset (keys and tube flags only; no image is read).
Runs in P1's environment:  /workspace/envs/viviano/bin/python scripts/replication/p1_location_rebuild_check.py
Exit 0 = exact match; 1 = mismatch (P1 stops and is reported). Writes results/replication/viviano/location_rebuild_check.md.
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
V = "/workspace/third_party/viviano"
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [V, f"{V}/activmask", str(ROOT / "src"), str(ROOT / "scripts" / "replication")]
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import activmask.datasets.xray as X
from wtss.utils import stable_int
import p1_location as PL

assert PL.SEED == 1234
lines, ok = ["# P1 location test — rebuild check of seed 1234's test split (A43)", "",
             "| position | test set | committed rows | rebuilt rows | ids + order | labels | tube flags |", "|---|---|---|---|---|---|---|"], True
ds = X.JointDataset(mode="test", seed=1234, ratio=0.5, verbose=False, **PL.PATHS)
keys = [f"{s}/" + (str(ds.dataset1.csv["Image Index"].iloc[i]) if s == 0 else str(ds.dataset2.csv["ImageID"].iloc[i])) for s, i in zip(ds.site, ds.imageids)]
labels = np.asarray(ds.labels).astype(int)
for pos in ("inside", "outside"):
    c = pd.read_csv(ROOT / f"results/replication/viviano/scores/location_{pos}_1234.csv.gz")
    c = c[c.method == "standard"]
    for env in ("corr", "rev", "clean"):
        p0, p1 = PL.P_TUBE[env]
        tube = np.array([stable_int("p1loc", k, int(y), 1234, env) / (2**32 - 1) < (p1 if y == 1 else p0) for k, y in zip(keys, labels)]).astype(int)
        q = c[c.test_set == env]
        same_ids = len(q) == len(keys) and (q.image_id.astype(str).to_numpy() == np.asarray(keys)).all()
        same_y = len(q) == len(keys) and (q.label.to_numpy() == labels).all()
        same_t = len(q) == len(keys) and (q.tube.to_numpy() == tube).all()
        ok &= bool(same_ids and same_y and same_t)
        lines.append(f"| {pos} | {env} | {len(q)} | {len(keys)} | {'match' if same_ids else 'MISMATCH'} | {'match' if same_y else 'MISMATCH'} | {'match' if same_t else 'MISMATCH'} |")
lines += ["", f"**Result: {'EXACT MATCH' if ok else 'MISMATCH — P1 stops'}.** Data: NIH 224 and PadChest 224 verified against "
          "`docs/DATA_MANIFEST.md` (file count, bytes, listing SHA256)."]
out = ROOT / "results/replication/viviano/location_rebuild_check.md"; out.write_text("\n".join(lines) + "\n"); print("\n".join(lines))
sys.exit(0 if ok else 1)
