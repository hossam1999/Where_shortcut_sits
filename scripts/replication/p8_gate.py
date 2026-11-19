"""P8 (DISC, ICML 2023) gate (A32, A35, A37): per trap set, the test ROC-AUC of their run at the epoch with the highest
validation ROC-AUC (their run_expt.py: `curr_val_perf > best_val_perf`, so the first maximum), on DISC's 618-image test
part (A35); the 5-set means vs the paper (ERM 0.364, DISC 0.551; reproduced if both within 0.03).
A37 pre-specified reading: if the corrected ERM mean misses 0.364 by > 0.03, the gate is reported as failed with the
statement that the published numbers may carry the shared-cache bug of their released code; within 0.03, the match stands.
Reads only their run CSVs ($WTSS_DATA/disc/output/ISIC/{ERM,DISC}/*-trapset_id=k/{val,test}.csv); writes
$WTSS_DATA/results_push/results/replication/disc/gate.{json,md} (pushed by m2_push_results.sh).

    python scripts/replication/p8_gate.py
"""
import glob, json, os, sys
import numpy as np, pandas as pd

D = os.environ.get("WTSS_DATA", "/workspace/data")
PAPER = {"ERM": 0.364, "DISC": 0.551}
TOL = 0.03
OUT = f"{D}/results_push/results/replication/disc"


def run(m, k):
    d = glob.glob(f"{D}/disc/output/ISIC/{m}/*-trapset_id={k}")
    assert len(d) == 1, (m, k, d)
    v, t = pd.read_csv(f"{d[0]}/val.csv"), pd.read_csv(f"{d[0]}/test.csv")
    assert len(v) == len(t) == 100, (m, k, len(v), len(t))
    i = int(np.argmax(v.roc_auc.values))
    return {"method": m, "trapset": k, "best_epoch": int(v.epoch.iloc[i]), "val_auroc": float(v.roc_auc.iloc[i]),
            "test_auroc": float(t.roc_auc.iloc[i])}


def main():
    rows = [run(m, k) for m in ("ERM", "DISC") for k in (1, 2, 3, 4, 5)]
    df = pd.DataFrame(rows)
    res = {"per_set": rows}
    for m in ("ERM", "DISC"):
        x = df[df.method == m].test_auroc
        res[m] = {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "paper": PAPER[m],
                  "diff": float(x.mean() - PAPER[m]), "within": bool(abs(x.mean() - PAPER[m]) <= TOL)}
    res["gate_passed"] = res["ERM"]["within"] and res["DISC"]["within"]
    res["a37_reading"] = ("ERM mean misses 0.364 by > 0.03: gate FAILED; the published numbers may carry the shared-cache "
                          "bug of DISC's released code (A37 hypothesis)") if not res["ERM"]["within"] else \
                         "ERM mean within 0.03 of 0.364: the ERM match stands (A37)"
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/gate.json", "w"), indent=1)
    md = ["# P8 gate (DISC, ICML 2023; A32, A35, A37) — machine 2", "",
          "Their code and commands; one data root per trap set (A37, caches verified row by row); per set the test ROC-AUC "
          "(618 images, DISC's split) at the epoch with the highest validation ROC-AUC (154 images).", "",
          "| method | trap set 1 | 2 | 3 | 4 | 5 | mean ± SD | paper | Δ | within 0.03 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for m in ("ERM", "DISC"):
        x = df[df.method == m].sort_values("trapset")
        md.append(f"| {m} | " + " | ".join(f"{a:.3f} (ep {e})" for a, e in zip(x.test_auroc, x.best_epoch)) +
                  f" | {res[m]['mean']:.3f} ± {res[m]['sd']:.3f} | {PAPER[m]:.3f} | {res[m]['diff']:+.3f} | {'yes' if res[m]['within'] else 'no'} |")
    md += ["", f"**Gate: {'PASSED' if res['gate_passed'] else 'FAILED'}.** {res['a37_reading']}."]
    open(f"{OUT}/gate.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(main())
