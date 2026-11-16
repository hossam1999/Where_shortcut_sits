"""P1 gate (author's instruction, 2026-11-16): after the first corrected seed (1234) of Standard and ActDiff on each task,
compare their test AUROC (their own metric, at their selected epoch: best_test_score in stats_<seed>.pkl) with the paper.
Within paper +- 2 SD for all four -> write GATE_OK (the queue then starts the other seeds). Otherwise write GATE_FAIL and
P1 stays paused for the author. Prints a table either way.
"""
import os
import pickle
import sys

R = os.path.expandvars("$WTSS_DATA/replication/viviano/runs")
PAPER = {("xray", "resnet"): (0.70, 0.05), ("rsna", "resnet"): (0.20, 0.03),
         ("xray", "resnet_actdiff"): (0.73, 0.03), ("rsna", "resnet_actdiff"): (0.68, 0.05)}
rows, done = [], True
for (t, m), (mu, sd) in PAPER.items():
    f = f"{R}/{t}_{m}_1234/results/{t}-seeds_{m}/stats_1234.pkl"
    if not os.path.exists(f):
        done = False; rows.append((t, m, None, mu, sd, None)); continue
    st = pickle.load(open(f, "rb"))
    v = float(st[-1]["best_test_score"])
    rows.append((t, m, v, mu, sd, abs(v - mu) <= 2 * sd))
for t, m, v, mu, sd, ok in rows:
    print(f"{t:5s} {m:15s} ours {'pending' if v is None else f'{v:.3f}'}  paper {mu:.2f} +- {sd:.2f}  "
          f"band [{mu-2*sd:.2f}, {mu+2*sd:.2f}]  {'' if ok is None else ('within' if ok else 'OUTSIDE')}")
if not done:
    sys.exit(2)
open(f"{R}/../{'GATE_OK' if all(r[5] for r in rows) else 'GATE_FAIL'}", "w").write("\n".join(map(str, rows)) + "\n")
print("GATE_OK" if all(r[5] for r in rows) else "GATE_FAIL")
