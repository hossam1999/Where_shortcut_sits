"""results/SUMMARY.md — expected (docs/REPLICATION_SPEC.md) vs obtained, with MATCH / MISMATCH.

Rule (spec §5): MATCH = same sign, same CI-excludes-0 verdict, |Δ| ≤ 0.02; "SIGN+CI" = sign and CI verdict
match but |Δ| > 0.02; MISMATCH otherwise. Rows whose inputs do not exist yet are listed as PENDING.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from wtss import paths

R = paths.RESULTS
rows: list[dict] = []


def add(exp_id, quantity, expected, got=None, lo=None, hi=None, e_lo=None, e_hi=None, source=""):
    if got is None or (isinstance(got, float) and np.isnan(got)):
        rows.append(dict(E=exp_id, quantity=quantity, expected=expected, obtained="PENDING", verdict="PENDING", source=source))
        return
    ok_sign = np.sign(expected) == np.sign(got) or abs(expected) < 0.005
    ok_ci = True if e_lo is None or lo is None else ((e_lo > 0 or e_hi < 0) == (lo > 0 or hi < 0))
    near = abs(expected - got) <= 0.02
    v = "MATCH" if ok_sign and ok_ci and near else ("SIGN+CI" if ok_sign and ok_ci else "MISMATCH")
    ob = f"{got:+.3f}" + (f" [{lo:+.3f}, {hi:+.3f}]" if lo is not None else "")
    ex = f"{expected:+.3f}" + (f" [{e_lo:+.3f}, {e_hi:+.3f}]" if e_lo is not None else "")
    rows.append(dict(E=exp_id, quantity=quantity, expected=ex, obtained=ob, verdict=v, source=source))


def boot(path, arm, ov=None, **kw):
    if not Path(path).exists():
        return (None,) * 3
    b = pd.read_csv(path)
    col = "method_a" if "method_a" in b else "arm"
    q = b[b[col] == arm]
    if ov is not None:
        q = q[np.isclose(q.overlap, ov)]
    for k, v in kw.items():
        q = q[q[k] == v]
    if q.empty:
        return (None,) * 3
    return q.seed_delta_mean.iloc[0], q.ci95_lo.iloc[0], q.ci95_hi.iloc[0]


S = R / "synthetic" / "isic2018"
main = S / "dino518_ruler_fixed_corr_main" / "bootstrap_vs_erm.csv"
# E1
for ov, e, lo, hi in [(0.0, .184, .146, .224), (0.25, -.058, -.100, -.018), (0.5, -.097, -.138, -.056),
                      (0.75, -.111, -.153, -.068), (1.0, -.128, -.170, -.086)]:
    add("E1", f"mask − ERM rev @{int(ov * 100)}%", e, *boot(main, "mask", ov), e_lo=lo, e_hi=hi, source=str(main))
li = S / "dino518_ruler_fixed_corr_main" / "location_interaction.csv"
if li.exists():
    q = pd.read_csv(li); q = q[q.method == "mask"].iloc[0]
    add("E1", "interaction 0% − 100% (mask)", .312, q.seed_delta_mean, q.ci95_lo, q.ci95_hi, .272, .354, str(li))
# E2
for bb, vals in [("dino224", [.364, -.040, -.076, -.102, -.092]), ("dermlip224", [.160, -.291, -.300, -.313, -.268])]:
    p = S / f"{bb}_ruler_fixed_corr_main" / "bootstrap_vs_erm.csv"
    for ov, e in zip((0, .25, .5, .75, 1.0), vals):
        add("E2", f"{bb} mask − ERM @{int(ov * 100)}%", e, *boot(p, "mask", ov), source=str(p))
# E3
p = S / "dino518_ruler_variable_corr_stress" / "bootstrap_vs_erm.csv"
for ov, e, lo, hi in [(0.0, .147, .109, .185), (0.5, -.094, -.140, -.048), (1.0, -.155, -.199, -.112)]:
    add("E3", f"variable ruler mask − ERM @{int(ov * 100)}%", e, *boot(p, "mask", ov), e_lo=lo, e_hi=hi, source=str(p))
for ov, e in [(0.0, .103), (0.5, .171), (1.0, .222)]:
    add("E3", f"variable ruler balanced − ERM @{int(ov * 100)}%", e, *boot(p, "balanced", ov), source=str(p))
# E4
p = S / "dino518_ruler_fixed_occlusion_occlusion" / "bootstrap_vs_erm.csv"
for ov, e, lo, hi in [(0.0, .035, .006, .065), (0.5, .033, -.000, .066), (1.0, .039, .010, .067)]:
    add("E4", f"occlusion (50/50) mask − ERM @{int(ov * 100)}%", e, *boot(p, "mask", ov), e_lo=lo, e_hi=hi, source=str(p))
# E5
p = S / "dino518_ruler_fixed_corr_dilation" / "bootstrap_vs_erm.csv"
for ov, arm, e in [(0.5, "dilate0", -.097), (0.5, "dilate10", -.138), (0.5, "dilate25", -.153), (0.5, "dilate50", -.142),
                   (1.0, "dilate0", -.128), (1.0, "dilate10", -.143), (1.0, "dilate25", -.144), (1.0, "dilate50", -.138)]:
    add("E5", f"{arm} − ERM @{int(ov * 100)}%", e, *boot(p, arm, ov), source=str(p))
# E6
cf = S / "dino518_ruler_fixed_corr_main" / "counterfactual_summary.csv"
if cf.exists():
    c = pd.read_csv(cf).set_index(["method", "overlap"]).abs_delta_p
    for (m, ov, e) in [("erm", 1.0, .235), ("mask", 1.0, .490), ("mask", 0.0, .000), ("erm", 0.0, .128), ("erm", 0.5, .195)]:
        add("E6", f"|Δp| {m} @{int(ov * 100)}%", e, float(c.get((m, ov), np.nan)), source=str(cf))
# E8/E9
for arm, ov, e, lo, hi in [("inpaint", 0.5, .211, .187, .234), ("inpaint", 1.0, .244, .218, .269),
                           ("inpaint_consistency", 0.5, .046, None, None), ("inpaint_consistency_lam0", 0.5, .036, None, None),
                           ("leace", 1.0, .294, .265, .325), ("balanced", 1.0, .281, .249, .315), ("dfr", 1.0, .279, .240, .319),
                           ("groupdro", 1.0, .197, .168, .226)]:
    add("E8/E9", f"{arm} − ERM @{int(ov * 100)}%", e, *boot(main, arm, ov), e_lo=lo, e_hi=hi, source=str(main))
# E13 / E15
for bb, E, items in [("dino518", "E13", [("trapA", "mask", -.064, -.079, -.049), ("trapB", "mask", .124, .105, .143),
                                        ("trapA", "inpaint", .101, .094, .107), ("trapB", "inpaint", .108, .098, .117),
                                        ("trapA", "balanced", .213, .190, .235), ("trapB", "balanced", .234, .217, .252),
                                        ("trapA", "dfr", .202, .176, .228), ("trapB", "dfr", .192, .168, .216),
                                        ("trapA", "leace_paired", .058, .053, .063), ("trapB", "leace_paired", .078, .070, .085),
                                        ("trapA", "leace_unpaired", .316, None, None)]),
                     ("dermlip224", "E15", [("trapA", "mask", -.069, -.088, -.051), ("trapA", "balanced", .126, .115, .138),
                                            ("trapA", "dfr", .114, .091, .138), ("trapA", "leace_paired", .031, .028, .034),
                                            ("trapA", "leace_unpaired", .192, None, None)])]:
    p = R / "spec_e13" / f"{bb}_spec" / "bootstrap_vs_erm.csv"
    for trap, arm, e, lo, hi in items:
        add(E, f"{trap} {arm} − ERM (rev)", e, *boot(p, arm, trap=trap, env="test_rev", source="all"), e_lo=lo, e_hi=hi, source=str(p))
    c3 = R / "spec_e13" / f"{bb}_spec" / "C3_crossover.json"
    if E == "E13":
        if c3.exists():
            j = json.loads(c3.read_text())
            add("E13", "C3 crossover B − A (mask)", .188, j["seed_delta_mean"], j["ci95_lo"], j["ci95_hi"], .179, .194, str(c3))
        for src, e, lo, hi in [("HAM", -.030, -.059, -.003), ("BCN", -.078, -.099, -.058)]:
            add("E13", f"trapA mask − ERM, {src} only", e, *boot(p, "mask", trap="trapA", env="test_rev", source=src),
                e_lo=lo, e_hi=hi, source=str(p))

d = pd.DataFrame(rows)
counts = d.verdict.value_counts().to_dict()
txt = ["# Replication summary (generated by scripts/make_summary.py)", "",
       "Rule: MATCH = same sign, same CI-excludes-0 verdict, |Δ| ≤ 0.02; SIGN+CI = sign and CI verdict match, "
       "|Δ| > 0.02; MISMATCH otherwise.", "", f"Verdict counts: {counts}", "", d.to_markdown(index=False)]
(R / "SUMMARY.md").write_text("\n".join(txt))
print(counts)
print(d[["E", "quantity", "expected", "obtained", "verdict"]].to_string())
