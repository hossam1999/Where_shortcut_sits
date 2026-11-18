"""Export per-image TEST scores of finished replication runs to small committed files (author's rule, 2026-11-17: every
table must be rebuildable if the machine is lost). Columns: image_id, label, method, test_set, score (+ logits where the
paper's metric needs them). No pixels, no paths. Writes results/replication/<paper>/scores/<run>.csv.gz.

    python scripts/replication/export_scores.py          # exports everything that exists and is not yet exported
"""
import glob
import os
from pathlib import Path

import pandas as pd

D = os.environ.get("WTSS_DATA", "/workspace/data")
ROOT = Path(__file__).resolve().parents[2]


def write(paper, run, df):
    out = ROOT / "results" / "replication" / paper / "scores" / f"{run}.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, compression={"method": "gzip", "mtime": 0})   # mtime 0: byte-stable, no spurious diffs
    print("wrote", out.relative_to(ROOT), len(df))


def p1():
    for f in sorted(glob.glob(f"{D}/replication/viviano/preds/*.csv.gz")):
        name = Path(f).name[:-7]
        task, method, seed = name.split("_")[0], "_".join(name.split("_")[1:-1]), name.split("_")[-1]
        d = pd.read_csv(f)
        d = d[d.split == "test"]
        write("viviano", name, pd.DataFrame({"image_id": d.image_id, "label": d.y, "site": d.site, "method": method,
                                             "test_set": "reversed", "logit0": d.logit0, "logit1": d.logit1,
                                             "score": d.logit1 - d.logit0}))
    for a in sorted(glob.glob(f"{D}/replication/viviano/arms/*/preds_*.csv.gz")):
        task_seed, arm = Path(a).parent.name, Path(a).name[6:-7]
        d = pd.read_csv(a)
        write("viviano", f"{task_seed.split('_')[0]}_{arm}_{task_seed.split('_')[1]}",
              pd.DataFrame({"image_id": d.image_id, "label": d.y, "site": d.site, "method": arm, "test_set": "reversed",
                            "score": d.score}))


def p1_location():
    for f in sorted(glob.glob(f"{D}/replication/viviano/location/*/scores.csv.gz")):
        pos = Path(f).parent.name
        d = pd.read_csv(f)
        write("viviano", f"location_{pos}_1234", d[d.test_set.isin(["corr", "rev", "clean"])])   # test sets only
    for f in sorted(glob.glob(f"{D}/replication/viviano/location/*/arms/preds_*.csv.gz")):   # our arms (A16), test sets only
        pos, arm = Path(f).parents[1].name, Path(f).name[6:-7]
        write("viviano", f"location_{pos}_{arm}_1234", pd.read_csv(f))
    for f in sorted(glob.glob(f"{D}/replication/viviano/location/*/scores_last.csv.gz")):   # A17, POST HOC (epoch 100)
        d = pd.read_csv(f)
        write("viviano", f"location_{Path(f).parent.name}_last_1234", d[d.test_set.isin(["corr", "rev", "clean"])])
    for f in sorted(glob.glob(f"{D}/replication/viviano/location/*/arms_last/preds_*.csv.gz")):
        pos, arm = Path(f).parents[1].name, Path(f).name[6:-7]
        write("viviano", f"location_{pos}_last_{arm}_1234", pd.read_csv(f))


def p2():
    for f in sorted(glob.glob(f"{D}/replication/pahde/scores/*.csv.gz")):
        d = pd.read_csv(f)
        write("pahde", Path(f).name[:-7], d[d.split == "test"])   # test rows only (val/train stay under $WTSS_DATA)
    for f in sorted(glob.glob(f"{D}/replication/pahde/scores_a18/resnet50d_*.csv.gz")):   # A18 RR-ClArC runs
        d = pd.read_csv(f)
        write("pahde", "a18_" + Path(f).name[:-7], d[d.split == "test"])
    for f in sorted(glob.glob(f"{D}/replication/pahde/arms/preds_*.csv.gz")):   # our arms: test rows only by construction
        write("pahde", "arm_" + Path(f).name[6:-7], pd.read_csv(f))


def p3():
    for r in sorted(glob.glob(f"{D}/replication/isnet/runs/*/preds_*.csv.gz")):
        run, test = Path(r).parent.name, Path(r).name[6:-7]
        if run.startswith("smoke"):
            continue
        d = pd.read_csv(r)
        write("isnet", f"{run}_{test}", d.assign(method=run, test_set=test).rename(columns={"y": "label"}))


def p4():
    for r in sorted(glob.glob(f"{D}/replication/saab/runs/*/preds_test.csv")):
        run = Path(r).parent.name
        d = pd.read_csv(r)
        write("saab", run, pd.DataFrame({"image_id": d.id, "label": d.target, "method": run, "test_set": "trap_test",
                                         "score": d.score}))


def armd():
    for f in sorted(glob.glob(f"{D}/replication/armd/*/preds_test.csv.gz")):   # arm d (A20/A21), test rows only
        write("armd", Path(f).parent.name, pd.read_csv(f))


if __name__ == "__main__":
    p1(); p1_location(); p2(); p3(); p4(); armd()
