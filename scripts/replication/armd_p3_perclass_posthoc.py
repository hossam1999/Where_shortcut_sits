"""POST HOC, exploratory (author, 2026-11-18): arm d on P3 90/10 inside with mode = per_class (span of the per-class
mark directions) instead of the registered selection (which chose pooled). Same features, same A21 C grid, same validation
rule (macro-F1 on P3's validation split with its marks); test scored once. Purpose: tell whether the registered result's
failure on the deceiving test comes from the pooled mode or from selection on the biased validation. CPU only.
The registered result (`results/replication/armd/p3_9010_inside.md`) is unchanged.

    python scripts/replication/armd_p3_perclass_posthoc.py
Writes results/replication/armd/p3_9010_inside_posthoc_perclass.{md,csv}.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "round8"), str(ROOT / "scripts" / "replication")]
import armd_train as A  # noqa: E402
import candidates as CA  # noqa: E402

S = A.P3()
out = Path(f"{A.D}/replication/armd/{S.KEY}")
F = np.load(out / "features.npz")
ytr, atr = S.labels("train")
P = CA.Projector(CA.cmc_directions(F["train"], ytr, atr, "per_class"))
Z = {k: P(F[k]) for k in S.sets}
sel, best = [], None
for C in A.CS:
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=0).fit(Z["train"], ytr)
    v = float(f1_score(S.labels("valid")[0], clf.predict(Z["valid"]), average="macro"))
    sel.append({"mode": "per_class", "k": P.Q.shape[1], "C": C, "val_macro_f1": v})
    if best is None or v > best[1]:
        best = (clf, v, C)
clf, v, C = best
edge = "lowest" if C == A.CS[0] else "highest" if C == A.CS[-1] else "no"
rows = []
for k in [k for k in S.sets if k.startswith("test_")]:
    d = S.part[k]; y = d.y.to_numpy(); p = clf.predict(Z[k])
    for var, keep in (("primary", np.ones(len(y), bool)), ("size-matched", d.size_matched.to_numpy().astype(bool))):
        rows.append({"test": k[5:], "variant": var, "n": int(keep.sum()), "macro_f1": float(f1_score(y[keep], p[keep], average="macro"))})
res = pd.DataFrame(rows); OUT = ROOT / "results/replication/armd"
pd.DataFrame(sel).to_csv(OUT / "p3_9010_inside_posthoc_perclass_selection.csv", index=False); res.to_csv(OUT / "p3_9010_inside_posthoc_perclass.csv", index=False)
reg = pd.read_csv(OUT / "p3_9010_inside.csv")
md = ["# POST HOC, exploratory — arm d on P3 90/10 inside with mode = per_class (author, 2026-11-18)", "",
      "**Not a registered result.** Same EfficientNet-B0 features as the registered arm d; mode fixed to per_class "
      f"(span of the {P.Q.shape[1]} per-class mark directions); C by the same validation rule (macro-F1 on P3's validation "
      f"split with its marks, A21 grid): C = {C:g} (edge: {edge}; validation macro-F1 {v:.3f}). Test scored once. "
      "The registered row (pooled, C = 0.03) is shown for comparison.", "",
      "| test | variant | registered (pooled, C = 0.03) | POST HOC per_class |", "|---|---|---|---|"]
for _, r in res.iterrows():
    g = reg[(reg.test == r.test) & (reg.variant == r.variant)].macro_f1.iloc[0]
    md.append(f"| {r.test} | {r.variant} | {g:.3f} | {r.macro_f1:.3f} |")
md += ["", "Validation macro-F1 by C (per_class): " + ", ".join(f"{s['C']:g}: {s['val_macro_f1']:.3f}" for s in sel)]
(OUT / "p3_9010_inside_posthoc_perclass.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
