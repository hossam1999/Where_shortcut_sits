"""A38 (author, 2026-11-20): every reproduction gate re-evaluated with UNROUNDED values at the original 0.03 and the widened
0.05 tolerance (P2-P8; P1 keeps its +/- 1 SD rule). Values are read from the committed result files; published targets as
registered. No result changes. Writes results/replication/GATES_TOLERANCE.{md,csv}."""
import json
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[2] / "results/replication"
rows = []
def add(paper, method, metric, ours, pub, src, note=""):
    g = abs(ours - pub)
    rows.append({"paper": paper, "method": method, "metric": metric, "ours": ours, "published": pub, "gap": g,
                 "passes_0.03": g <= 0.03, "passes_0.05": g <= 0.05, "source": src, "note": note})
p2 = pd.read_csv(R / "pahde/p2_reproduction_a18.csv")
x = p2[(p2.method.str.startswith("RR-ClArC")) & (p2.test == "biased")].iloc[0]
add("P2 Pahde 2025", "RR-ClArC (A18)", "biased-test accuracy", float(x.accuracy), 0.76, "pahde/p2_reproduction_a18.csv")
c = pd.read_csv(R / "isnet/covidisnet_check.csv")
for _, x in c.iterrows():
    add("P3 Bassi 2024", f"released COVIDIsNet.pt, {x.variant.split(' (')[0]} test (A9)", "deceiving macro-F1", float(x.macro_f1), 0.773, "isnet/covidisnet_check.csv")
g = pd.read_csv(R / "saab/p4_gate_seg_pil.csv")
for m, pub, note in (("Binary-ERM", 0.359, ""), ("Seg-ERM", 0.727, "A31 rerun; also misses the notebook value 0.772")):
    add("P4 Saab 2022", f"{m} (3 trap sets)", "robust AUROC", float(g[g.method == m].robust_auroc.mean()), pub, "saab/p4_gate_seg_pil.csv", note)
s = pd.read_csv(R / "r2r/step1.csv"); s = s[(s.checkpoint == "checkpoint_resnet18_isic_attacked.pth") & (s.split == "test")]
for var, pa, pf in (("ch", 0.895, 0.821), ("clean", 0.895, 0.821), ("attacked", 0.356, 0.390)):
    x = s[s.variant == var].iloc[0]
    add("P5 R2R (step 1)", f"released Vanilla, {var} test", "accuracy", float(x.accuracy), pa, "r2r/step1.csv")
    add("P5 R2R (step 1)", f"released Vanilla, {var} test", "macro-F1", float(x.f1_macro), pf, "r2r/step1.csv")
e = json.load(open(R / "epvt/A34_gate.json"))
add("P6 Yan 2023 (EPVT)", "ERM, bias 1, 5 seeds", "AUROC", e["erm_mean"], 0.653, "epvt/A34_gate.json", "target digitised (±≈0.005)")
add("P6 Yan 2023 (EPVT)", "EPVT (A36), bias 1, 5 seeds", "AUROC", e["epvt_mean"], 0.746, "epvt/A34_gate.json", "target digitised (±≈0.005)")
for a, pub in (("vgg16", 0.570), ("resnet18", 0.712), ("efficientnet_b0", 0.756)):
    t = pd.read_csv(R / f"rrclarc/p7/{a}_summary.csv"); x = t[(t.split == "test") & (t.variant == "attacked")].iloc[0]
    add("P7 Dreyer 2024 (RR-ClArC)", f"RR-ClArC {a}", "biased-test accuracy", float(x.accuracy), pub, f"rrclarc/p7/{a}_summary.csv")
d = json.load(open(R / "disc/gate.json"))
add("P8 Wu 2023 (DISC)", "ERM, 5 trap sets", "AUROC", d["ERM"]["mean"], 0.364, "disc/gate.json")
add("P8 Wu 2023 (DISC)", "DISC, 5 trap sets", "AUROC", d["DISC"]["mean"], 0.551, "disc/gate.json")
t = pd.DataFrame(rows); t.to_csv(R / "GATES_TOLERANCE.csv", index=False)
gate = {"P2 Pahde 2025": "RR-ClArC", "P3 Bassi 2024": "A9 check (either variant)", "P4 Saab 2022": "both", "P5 R2R (step 1)": "every number",
        "P6 Yan 2023 (EPVT)": "both", "P7 Dreyer 2024 (RR-ClArC)": "per architecture", "P8 Wu 2023 (DISC)": "both"}
md = ["# Reproduction gates at 0.03 and 0.05 (A38; unrounded values; no result changes)", "",
      "The tolerance was widened from 0.03 to 0.05 by the author on 2026-11-20, **after the first gate results**, for every "
      "paper alike (P2–P8; P1 keeps its ±1 SD rule). Values are read unrounded from the committed result files.", "",
      "| paper | method | metric | ours | published | gap | passes at 0.03 | passes at 0.05 | source |", "|---|---|---|---|---|---|---|---|---|"]
for _, x in t.iterrows():
    md.append(f"| {x.paper} | {x.method} | {x.metric} | {x.ours:.6f} | {x.published:.3f} | {x.gap:.6f} | {'yes' if x['passes_0.03'] else 'no'} | "
              f"{'yes' if x['passes_0.05'] else 'no'} | `{x.source}`{(' — ' + x.note) if x.note else ''} |")
md += ["", "**Gate verdicts (registered combination rule per paper):**", "",
       "| paper | rule | at 0.03 | at 0.05 |", "|---|---|---|---|"]
for p, rule in gate.items():
    q = t[t.paper == p]
    f = (lambda c: q[c].any()) if rule.startswith("A9") else (lambda c: q[c].all())
    if rule == "per architecture":
        v3 = ", ".join(f"{m.split()[-1]} {'yes' if a else 'no'}" for m, a in zip(q.method, q["passes_0.03"]))
        v5 = ", ".join(f"{m.split()[-1]} {'yes' if a else 'no'}" for m, a in zip(q.method, q["passes_0.05"]))
    else:
        v3, v5 = ("reproduced" if f("passes_0.03") else "not reproduced"), ("reproduced" if f("passes_0.05") else "not reproduced")
    md.append(f"| {p} | {rule} | {v3} | {v5} |")
md += ["", "Descriptive published-number row (P8 table): **CBDebug (Enouen & Galhotra, ICLR 2026) 0.580 ± 0.116** on the same "
       "Bissoto trap sets — *not reproduced: no ISIC code released*.", "",
       "P6 note: the EPVT gap (0.049712) is inside 0.05 by less than the digitisation uncertainty of the published figure "
       "(±≈0.005); the pass is stated with that caveat."]
(R / "GATES_TOLERANCE.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
