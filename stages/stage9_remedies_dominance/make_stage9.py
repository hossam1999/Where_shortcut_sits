"""Stage 9 tables, macros and figure: is there a remedy that dominates ROI masking? Round 8 (registered), the round-9
development search (exploratory) and round 9 (registered); docs/PREREGISTRATION_ROUND8.md, docs/ROUND9_SEARCH_LEDGER.md,
docs/PREREGISTRATION_ROUND9.md. Every number is read from results/round8/, results/round9_search/ and results/round9/.
  PYTHONPATH=src python stages/stage9_remedies_dominance/make_stage9.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import stagelib as L  # noqa: E402

R8, R9, RS = (L.ROOT / "results" / d for d in ("round8", "round9", "round9_search"))
T, F = HERE / "tables", HERE / "figures"
T.mkdir(exist_ok=True); F.mkdir(exist_ok=True)
M = L.Macros("sNine")
BLUE, ORANGE, GREEN, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#898781"
COH = {"thyroid": "thyroid split", "capsule": "capsule", "isic_BCN": "dermoscopy, BCN held out",
       "isic_HAM": "dermoscopy, HAM held out", "isic_MSK": "dermoscopy, MSK held out", "isic2020": "ISIC 2019 $\\to$ 2020",
       "ovary": "ovary (held out)", "isic": "dermoscopy hair"}
TRAPC = {"thyroid": "thyroid", "capsule": "capsule", "ovary": "ovary (held out)", "isic": "dermoscopy hair"}
NAME = {"mask_cmc": "class-conditional mean constraint (\\texttt{mask\\_cmc})", "full_cmc": "mean constraint on the full image (\\texttt{full\\_cmc})",
        "mask_bal": "partial reweighting (\\texttt{mask\\_bal})", "locrand_loc": "location-matched pasting (\\texttt{locrand\\_loc})",
        "mask_condadv": "conditional adversarial debiasing (\\texttt{mask\\_condadv})", "mask_irm": "IRMv1 (\\texttt{mask\\_irm})",
        "mask_vrex": "V-REx (\\texttt{mask\\_vrex})"}


def csv(p: Path):
    if not p.exists():
        L.MISSING.append(str(p.relative_to(L.ROOT)))
        return None
    return pd.read_csv(p)


def cir(r, est="estimate"):
    return L.ci(float(r[est]), float(r["ci95_lo"]), float(r["ci95_hi"]))


def label(r):
    return {"met": "met", "loss": "\\textbf{loss}", "inconclusive": "inconclusive"}.get(str(r.label), str(r.label))


# ------------------------------------------------------------------------------------------------ registrations
def experiments():
    M.add("RegEight", L.reg("docs/PREREGISTRATION_ROUND8.md"))
    M.add("RegNine", L.reg("docs/PREREGISTRATION_ROUND9.md"))
    M.add("Ledger", L.reg("docs/ROUND9_SEARCH_LEDGER.md"))
    M.add("Design", L.reg("docs/ROUND8_DESIGN.md"))
    rows = [["Design and scoreboard of every existing arm", "\\posthoc{design; no test scoring of new models}",
             L.reg("docs/ROUND8_DESIGN.md"), "--"],
            ["Round 8: four primary candidates (fixed sequence), sweeps (D5), fine-tuned (D6), replication (R)",
             "\\registered{round 8}; one amendment before any run", L.reg("docs/PREREGISTRATION_ROUND8.md"),
             "dominance: every component of D1--D4 met (intersection--union)"],
            ["Search over twelve candidates on development data", "\\posthoc{exploratory; development data only}",
             L.reg("docs/ROUND9_SEARCH_LEDGER.md"), "ranking only"],
            ["Round 9: three candidates from the search (fixed sequence)", "\\registered{round 9}; the last remedy round",
             L.reg("docs/PREREGISTRATION_ROUND9.md"), "as round 8"]]
    L.table(T / "experiments.tex", ["Experiment", "Status", "First commit (local time)", "Support criterion"], rows,
            "Experiments of this stage. Each registration was committed before the data it tests were generated (round 8: "
            "confirmation seeds 8101--8505; round 9: 9101--9505). Commit times are set by the committing machine.",
            "tab:exp", align="p{5.0cm}p{3.6cm}p{2.8cm}p{3.6cm}", size="\\scriptsize")
    crit = [["D1", "all pairs of the six unaltered test sets (thyroid split, capsule, BCN, HAM, MSK, ISIC 2019 $\\to$ 2020)",
             "not worse: lower bound $>-0.01$"],
            ["D2", "shortcut-conflicting pairs (thyroid, BCN, MSK, ISIC 2020) and Trap~A $\\min$(reversed, correlated) in four cohorts",
             "better: lower bound $>0$"],
            ["D3", "Trap~B and clean tests, per cell (16) and pooled over cohorts (4)", "not worse: $-0.03$ per cell, $-0.01$ pooled"],
            ["D4", "every trap", "no shortcut flipping: correlated $\\ge$ reversed $-0.02$"]]
    L.table(T / "criterion.tex", ["", "Cells", "Test (candidate $-$ masking, one-sided 0.025)"], crit,
            "The registered criterion for ``dominates masking''. A candidate dominates only if every component is met; "
            "easy pairs, where any method that stops using the shortcut must lose, are not in it.", "tab:crit",
            align="lp{9.0cm}p{5.0cm}", size="\\scriptsize")


# ------------------------------------------------------------------------------------------------ round 8
def round8():
    v = csv(R8 / "confirm" / "verdicts.csv")
    c = csv(R8 / "confirm" / "components.csv")
    if v is None or c is None:
        return
    rows = []
    for r in v[v.scope == "all cohorts"].itertuples():
        comp = c[(c.scope == "all cohorts") & (c.candidate == r.candidate)]
        loss = comp[comp.label == "loss"]
        rows.append([NAME[r.candidate], f"{int(r.components_met)} of {int(r.components)}",
                     "yes" if r.all_D2_met else "no", ", ".join(f"{COH.get(x.cohort, x.cohort)} ({x.component.split()[0]})"
                                                               for x in loss.itertuples()) or "none",
                     {"tested": "tested, not rejected"}.get(r.sequence, r.sequence)])
        M.add(f"Met{L.Macros.clean(r.candidate)}", int(r.components_met))
    M.add("NComp", int(v.components.iloc[0]))
    L.table(T / "r8_verdicts.tex", ["Candidate", "Components met", "All of D2 met", "Losses", "Fixed sequence"], rows,
            "Round 8 on the confirmation data: no candidate dominates masking. The sequence stopped at its first candidate; the "
            "others are reported unadjusted.", "tab:rEight", align="p{4.6cm}p{1.6cm}p{1.3cm}p{5.0cm}p{2.6cm}", size="\\scriptsize")
    g = lambda cand, comp, coh: c[(c.scope == "all cohorts") & (c.candidate == cand) & (c.component == comp) & (c.cohort == coh)].iloc[0]
    rows = []
    for comp, cohs in (("D1 all pairs", ("thyroid", "capsule", "isic_BCN", "isic_HAM", "isic_MSK", "isic2020")),
                       ("D2 hard pairs", ("thyroid", "isic_BCN", "isic_MSK", "isic2020")),
                       ("D2 Trap A min(rev,corr)", ("thyroid", "capsule", "ovary", "isic"))):
        for coh in cohs:
            a, b = g("mask_cmc", comp, coh), g("mask_bal", comp, coh)
            rows.append([{"D1 all pairs": "D1 all pairs", "D2 hard pairs": "D2 conflicting pairs",
                          "D2 Trap A min(rev,corr)": "D2 Trap~A"}[comp], COH[coh] if "Trap" not in comp else TRAPC[coh],
                         cir(a), label(a), cir(b), label(b)])
            k = L.Macros.clean(comp.split()[0] + comp.split()[1] + coh)
            M.add(f"Cmc{k}", cir(a)); M.add(f"Bal{k}", cir(b))
    for comp in ("D3 Trap B reversed (pooled)", "D3 clean (Trap A models) (pooled)", "D3 clean (Trap B models) (pooled)"):
        a, b = (c[(c.scope == "all cohorts") & (c.candidate == x) & (c.component == comp)].iloc[0] for x in ("mask_cmc", "mask_bal"))
        rows.append(["D3 pooled", comp.replace("D3 ", "").replace(" (pooled)", ""), cir(a), label(a), cir(b), label(b)])
    L.table(T / "r8_components.tex", ["Component", "Cell", "\\texttt{mask\\_cmc} $-$ masking", "", "\\texttt{mask\\_bal} $-$ masking", ""],
            rows, "Round 8, the two candidates that won everywhere masking fails: the D1 and D2 components and the pooled D3 "
            "components (AUROC, crossed 95\\% CIs); every per-cell D3 component and D4 were met by both.", "tab:rEightc",
            align="p{2.3cm}p{3.3cm}p{2.9cm}p{1.4cm}p{2.9cm}p{1.4cm}", size="\\scriptsize")
    # ISIC 2020 decomposition for mask_cmc (round 8)
    d = csv(R8 / "confirm" / "descriptive_all_cells.csv")
    if d is not None:
        q = d[(d.source == "natural/isic2020") & (d.arm == "mask_cmc") & (d.env == "clean")].set_index("cell")
        for cell in ("easy", "hard", "all"):
            r = q.loc[cell]
            M.add(f"CmcIsicTwoZero{cell}", L.ci(r.arm_minus_mask, r.ci95_lo, r.ci95_hi))
    # sweeps (D5), fine-tuned (D6), replication (R), checks
    for cand in ("mask_cmc", "mask_bal", "full_cmc"):
        r = v[(v.scope == "D5") & (v.candidate == cand)].iloc[0]
        M.add(f"DFive{L.Macros.clean(cand)}", f"{int(r.components_met)} of {int(r.components)}")
    r = v[(v.scope == "D6")].iloc[0]
    M.add("DSix", f"{int(r.components_met)} of {int(r.components)}")
    rep = csv(R8 / "confirm" / "replication.csv")
    if rep is not None:
        for cand in ("mask_cmc", "mask_bal"):
            q = rep[rep.candidate == cand]
            M.add(f"Rep{L.Macros.clean(cand)}", f"{int(q.met_holm.sum())} of {len(q)}")
            M.add(f"RepMin{L.Macros.clean(cand)}", L.f3(float(q.estimate.min())))
    pr = []
    for coh in ("thyroid", "capsule", "ovary", "isic2019"):
        f = R8 / "checks" / f"locrand_probe_{coh}.json"
        if f.exists():
            pr.append(json.loads(f.read_text())["pasted_vs_real_auroc"])
    M.add("ProbeMin", L.f3(min(pr))); M.add("ProbeMax", L.f3(max(pr)))


# ------------------------------------------------------------------------------------------------ search
def search():
    s = csv(RS / "summary.csv")
    if s is None:
        return
    lab = {"mask_ba": "backdoor adjustment", "full_ba": "backdoor adjustment, full image", "mask_cmc_ba": "constraint + backdoor",
           "full_cmc_ba": "constraint + backdoor, full image", "mask_poe": "product of experts", "mask_la": "group logit adjustment",
           "mask_moments": "moment matching", "mask_condadv": "conditional adversarial debiasing", "mask_cnc": "Correct-n-Contrast",
           "mask_cfc": "counterfactual contrastive", "mask_vrex": "V-REx", "mask_irm": "IRMv1"}
    flip = {"mask_ba", "mask_poe", "mask_la"}
    kept = {"mask_condadv": "1st", "mask_irm": "2nd", "mask_vrex": "3rd"}
    rows = []
    for r in s[s.candidate != "baselines"].itertuples():
        rows.append([lab[r.arm], f"{int(r.met)} of {int(r.components)}", int(r.losses),
                     "no (flips the shortcut)" if r.arm in flip else "yes", kept.get(r.arm, "--")])
    b = s[s.candidate == "baselines"].set_index("arm")
    for ref in ("mask_cmc", "mask_bal"):
        rows.append([f"round 8 \\texttt{{{ref.replace('_', chr(92) + '_')}}} (reference)", f"{int(b.loc[ref].met)} of {int(b.loc[ref].components)}",
                     int(b.loc[ref].losses), "yes", "--"])
    L.table(T / "search.tex", ["Search candidate (frozen masked features)", "Development components met", "Losses", "Eligible",
                               "Registered in round 9"], rows,
            "The exploratory search: twelve candidates on development data only (development seeds, validation folds of the "
            "unaltered training sets; never a test set used for confirmation, never ovary). Round 8's candidates are shown on the "
            "same development criterion for comparison.", "tab:search", align="p{5.4cm}p{2.4cm}p{1.0cm}p{2.8cm}p{2.2cm}",
            size="\\scriptsize")
    M.add("NSearch", int((s.candidate != "baselines").sum()))
    M.add("SearchBal", int(b.loc["mask_bal"].met))


# ------------------------------------------------------------------------------------------------ round 9
def round9():
    v = csv(R9 / "verdicts.csv")
    c = csv(R9 / "components.csv")
    dec = csv(R9 / "isic2020_decomposition.csv")
    st = csv(R9 / "stability.csv")
    if v is None or c is None:
        return
    rows = []
    for r in v[v.scope == "all cohorts"].itertuples():
        comp = c[(c.scope == "all cohorts") & (c.candidate == r.candidate)]
        loss = comp[comp.label == "loss"]
        role = "reference (round 8)" if r.role == "reference" else {"tested": "tested, not rejected"}.get(r.sequence, r.sequence)
        rows.append([NAME[r.candidate], f"{int(r.components_met)} of {int(r.components)}", "yes" if r.all_D2_met else "no",
                     ", ".join(f"{COH.get(x.cohort, x.cohort)} ({x.component.split()[0]})" for x in loss.itertuples()) or "none", role])
        M.add(f"NineMet{L.Macros.clean(r.candidate)}", int(r.components_met))
    L.table(T / "r9_verdicts.tex", ["Candidate", "Components met", "All of D2 met", "Losses", "Role"], rows,
            "Round 9 on new confirmation data (seeds 9101--9505): no candidate dominates masking. Round 8's two best candidates, "
            "refitted with their frozen recipes, are descriptive references on the same data.", "tab:rNine",
            align="p{4.6cm}p{1.6cm}p{1.3cm}p{5.2cm}p{2.4cm}", size="\\scriptsize")
    g = lambda cand, comp, coh: c[(c.scope == "all cohorts") & (c.candidate == cand) & (c.component == comp) & (c.cohort == coh)].iloc[0]
    for cand in ("mask_irm", "mask_vrex", "mask_cmc", "mask_condadv"):
        for comp, coh in (("D1 all pairs", "isic2020"), ("D2 hard pairs", "isic2020"), ("D1 all pairs", "thyroid")):
            M.add(f"Nine{L.Macros.clean(cand + comp.split()[0] + coh)}", cir(g(cand, comp, coh)))
    rows = []
    for coh in ("thyroid", "capsule", "ovary", "isic"):
        rows.append([TRAPC[coh]] + [cir(g(x, "D2 Trap A min(rev,corr)", coh)) for x in ("mask_irm", "mask_vrex", "mask_cmc", "mask_bal")])
    for coh in ("thyroid", "isic_BCN", "isic_MSK", "isic2020"):
        rows.append([f"conflicting pairs, {COH[coh]}"] + [cir(g(x, "D2 hard pairs", coh)) for x in ("mask_irm", "mask_vrex", "mask_cmc", "mask_bal")])
    L.table(T / "r9_gains.tex", ["Where masking fails", "IRMv1", "V-REx", "\\texttt{mask\\_cmc} (ref.)", "\\texttt{mask\\_bal} (ref.)"], rows,
            "Round 9: gains over masking where masking fails (Trap~A $\\min$(reversed, correlated); conflicting pairs), crossed 95\\% CIs.",
            "tab:rNineg", size="\\scriptsize", resize=True)
    if dec is not None:
        rows = []
        for r in dec.itertuples():
            rows.append([r.arm.replace("_", "\\_"), L.f3(r.easy_part), L.f3(r.hard_part), L.f3(r.same_artifact_part), L.f3(r.all_pairs_delta)])
        L.table(T / "r9_decomp.tex", ["Arm", "Easy pairs", "Conflicting pairs", "Same-artifact pairs", "All pairs"], rows,
                "ISIC 2019 $\\to$ 2020, round 9: the all-pairs difference to masking split exactly into the contributions of the "
                "pair types (share of pairs $\\times$ difference). Every arm, including the unmasked model (ERM), loses mostly on "
                "the same-artifact pairs, i.e.\\ disease ranking, not only on the easy pairs.", "tab:decomp", size="\\scriptsize")
        e = dec.set_index("arm")
        M.add("DecErm", L.f3(e.loc["erm", "all_pairs_delta"])); M.add("DecIrm", L.f3(e.loc["mask_irm", "all_pairs_delta"]))
        M.add("DecWorst", L.f3(e.all_pairs_delta.min())); M.add("DecBest", L.f3(e.all_pairs_delta.max()))
        M.add("DecErmSame", L.f3(e.loc["erm", "same_artifact_part"]))
    rep = csv(R9 / "replication.csv")
    if rep is not None:
        for cand in ("mask_irm", "mask_vrex", "mask_condadv"):
            q = rep[rep.candidate == cand]
            M.add(f"NineRep{L.Macros.clean(cand)}", f"{int(q.met_holm.sum())} of {len(q)}")
    # figure: key components for the four best (round 8 mask_cmc, mask_bal; round 9 mask_irm, mask_vrex)
    c8 = csv(R8 / "confirm" / "components.csv")
    panels = [("D1 all pairs", ("thyroid", "capsule", "isic_BCN", "isic_HAM", "isic_MSK", "isic2020"), "unaltered test sets, all pairs"),
              ("D2 hard pairs", ("thyroid", "isic_BCN", "isic_MSK", "isic2020"), "conflicting pairs"),
              ("D2 Trap A min(rev,corr)", ("thyroid", "capsule", "ovary", "isic"), "Trap A (artifact in the ROI)")]
    series = [(c8, "mask_cmc", "mask_cmc (round 8)", BLUE, "o"), (c8, "mask_bal", "mask_bal (round 8)", MUTED, "s"),
              (c, "mask_irm", "IRMv1 (round 9)", ORANGE, "D"), (c, "mask_vrex", "V-REx (round 9)", GREEN, "^")]
    plt = L.plot_style()
    fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.4), gridspec_kw={"width_ratios": [6, 4, 4]})
    short = {"thyroid": "thyroid", "capsule": "capsule", "isic_BCN": "BCN", "isic_HAM": "HAM", "isic_MSK": "MSK",
             "isic2020": "ISIC 2020", "ovary": "ovary*", "isic": "hair"}
    for a, (comp, cohs, title) in zip(ax, panels):
        for k, (src, cand, lab_, col, mk) in enumerate(series):
            for j, coh in enumerate(cohs):
                q = src[(src.scope == "all cohorts") & (src.candidate == cand) & (src.component == comp) & (src.cohort == coh)]
                if q.empty:
                    continue
                r = q.iloc[0]
                x = j + (k - 1.5) * 0.18
                a.errorbar([x], [r.estimate], yerr=[[r.estimate - r.ci95_lo], [r.ci95_hi - r.estimate]], fmt=mk, color=col, ms=5,
                           capsize=1.5, lw=1.0, label=lab_ if j == 0 else None)
        a.axhline(0, color=MUTED, lw=0.6)
        if comp == "D1 all pairs":
            a.axhline(-0.01, color=MUTED, lw=0.6, ls="--")
        a.set_xticks(range(len(cohs))); a.set_xticklabels([short[x] for x in cohs], fontsize=7, rotation=30)
        a.set_title(title, fontsize=8)
    ax[0].set_ylabel("candidate $-$ masking (AUROC)")
    h, lb = ax[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.legend(h, lb, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.savefig(F / "dominance.pdf"); plt.close(fig)


def main():
    experiments()
    round8()
    search()
    round9()
    M.write(T / "numbers.tex")
    L.report_missing("stage9")


if __name__ == "__main__":
    main()
