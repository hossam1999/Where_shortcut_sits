"""Stage 8 tables, macros and figures: artifact labels checked without a clinician (Part A) and a fine-tuned thyroid
model near the published benchmark (Part B); docs/PREREGISTRATION_ROUND6.md with Amendments 1-3.
Every number is read from results/round6/ (and, for the Stage 3 and Stage 5 references, results/rerun_2026-09-28/).
  PYTHONPATH=src python stages/stage8_labels_strong_model/make_stage8.py"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import stagelib as L  # noqa: E402

R6 = L.ROOT / "results" / "round6"
T, F = HERE / "tables", HERE / "figures"
T.mkdir(exist_ok=True); F.mkdir(exist_ok=True)
M = L.Macros("sEight")
COH = [("isic", "Dermoscopy hair (ISIC 2019)"), ("thyroid", "Thyroid calipers"), ("capsule", "Capsule debris"),
       ("ovary", "Ovary calipers")]
LAB = dict(COH)
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#898781"
OPS = [("OP1_maxBA", "OP1 max.\\ balanced accuracy", "OP1"), ("OP2_spec0.80", "OP2 specificity $\\ge0.80$", "OP2"),
       ("OP3_spec0.90", "OP3 specificity $\\ge0.90$", "OP3"), ("OP4_sens0.80", "OP4 sensitivity $\\ge0.80$", "OP4"),
       ("OP5_sens0.90", "OP5 sensitivity $\\ge0.90$", "OP5")]


def c6(rel):
    return L.csv(rel, R6)


def j6(rel):
    p = R6 / rel
    if not p.exists():
        L.MISSING.append(str(p.relative_to(L.ROOT)))
        return None
    return json.loads(p.read_text())  # round-6 JSON may hold NaN


def cir(d, est="estimate"):
    return L.ci(float(d[est]), float(d["ci95_lo"]), float(d["ci95_hi"]))


def cia(d, est="estimate"):
    """Agreement (kappa, Dice, share) with its interval, unsigned."""
    return f"{float(d[est]):.3f} [{float(d['ci95_lo']):.3f}, {float(d['ci95_hi']):.3f}]"


def pct(x, nd=0):
    return f"{100 * float(x):.{nd}f}\\%"


def first_commit(rel: str) -> str:
    out = subprocess.run(["git", "-C", str(L.ROOT), "log", "--diff-filter=A", "--format=%h|%ad", "--date=format:%Y-%m-%d %H:%M",
                          "--", rel], capture_output=True, text=True).stdout.split("\n")
    out = [o for o in out if o.strip()]
    if not out:
        L.MISSING.append(rel)
        return "\\na"
    h, d = out[-1].split("|")
    return f"\\texttt{{{h}}}, {d}"


def per_image(c):
    return c6(f"agreement/{c}/per_image.csv")


# ------------------------------------------------------------------------------------------------ registration
def experiments():
    doc = "docs/PREREGISTRATION_ROUND6.md"
    M.add("Reg", L.reg(doc))
    M.add("CommitCoverage", first_commit("results/round6/coverage.csv"))
    M.add("CommitRecipe", first_commit("results/round6/ft_recipe.json"))
    M.add("CommitAfourb", first_commit("results/round6/a4b_KEY_SHA256.txt"))
    rows = [["A0 sources, licences, coverage", "\\registered{round 6}; coverage committed before A1", L.reg(doc),
             "$\\ge200$ matched images: analysed; 50--199: descriptive; $<50$: not feasible"],
            ["A1 agreement with independent sources", "\\registered{round 6}", L.reg(doc),
             "descriptive; a model labeller is used only if $\\kappa\\ge0.40$ (Amendment 2: validity guard)"],
            ["A2 traps on uncontradicted labels (HA2)", "\\registered{round 6}; primary of Part A", L.reg(doc),
             "crossover $>0$ in every cohort that passes the count gate (Holm)"],
            ["A3 differential error, attenuation", "\\registered{round 6}", L.reg(doc),
             "flag: $|e_{Y=1}-e_{Y=0}|>0.10$, CI excluding 0; correction only for unflagged cohorts"],
            ["A4 blinded rating by the author (A4b optional)", "\\registered{round 6}; not yet done", L.reg(doc),
             "intra-rater $\\kappa$; error rates of our cells; judges a single model labeller (Amendment 2)"],
            ["B1 fine-tune recipe (validation only)", "\\registered{round 6}; choice committed before B2", L.reg(doc),
             "highest validation AUROC of 12 recipes"],
            ["B2--B3 natural split and traps (FT0--FT4)", "\\registered{round 6}", L.reg(doc),
             "FT0 descriptive ($\\ge0.773$); FT1--FT4 one-sided, Holm over four"],
            ["Round 7: thresholds matched on the test set (TM1--TM2)", "\\registered{round 7}",
             L.reg("docs/PREREGISTRATION_ROUND7.md"), "conflicting sensitivity at test specificity 0.80, mask $-$ ERM $<0$ (Holm over two)"],
            ["E1--E2 consensus labels", "\\posthoc{exploratory; decided after A1--A3 were known}", "---",
             "descriptive; not a test"]]
    L.table(T / "experiments.tex", ["Experiment", "Status", "First commit (local time)", "Support criterion"], rows,
            "Experiments of this stage. \\texttt{docs/PREREGISTRATION\\_ROUND6.md} and its three amendments were committed "
            "before any analysis of this round; the coverage report (\\sEightCommitCoverage), the fine-tune recipe "
            "(\\sEightCommitRecipe) and the hash of the A4b key (\\sEightCommitAfourb) were committed before the steps that "
            "use them. Commit times are set by the committing machine and are not independent proof of order.", "tab:exp",
            align="p{3.8cm}p{3.4cm}p{2.8cm}p{5.0cm}", size="\\scriptsize")


# ------------------------------------------------------------------------------------------------ A0
def coverage():
    src = j6("sources.json")
    cov = c6("coverage.csv")
    exc = j6("matches/capsule_exclusion.json")
    if src is None or cov is None:
        return
    s = {x["name"]: x for x in src["sources"]}
    top = cov[cov.cell == "all"]
    lic = {"DermArtifactDB": "DermArtifactDB", "IMA++": "IMA++", "Kabir": "Kabir hair masks", "BUSClean": "BUSClean",
           "MedGemma": "MedGemma 1.5 4B-it", "Mendeley_vmxhn95j8z": "Multicentre clear/contaminated capsule masks"}
    use = {"DermArtifactDB": "hair present / absent", "IMA++": "lesion masks, 1--5 annotators", "Kabir": "hair masks",
           "Mendeley_vmxhn95j8z": "contamination masks", "BUSClean": "annotation detector (\\texttt{detect\\_anno})",
           "MedGemma": "vision--language answers (presence, location)"}
    gate = {"analysed": "analysed", "descriptive": "descriptive", "not_feasible": "not feasible"}
    rows = []
    for r in top.itertuples():
        name = lic[r.source]
        rows.append([LAB[r.cohort].split(" (")[0], name.replace("++", "{+}{+}"), use[r.source],
                     L.tex_escape(str(s[name]["licence"])), f"{int(r.n):,}".replace(",", "{,}"), gate[r.gate]])
    L.table(T / "coverage.tex", ["Cohort", "Source", "Used for", "Licence", "Matched images", "Gate"], rows,
            "Independent label sources and their coverage of our cohorts after the exclusions of Amendment~1 "
            "(training images of our own labellers). No image of the thyroid or ovary cohorts left the GPU machine: "
            "BUSClean and MedGemma ran locally.", "tab:cov", align="p{2.2cm}p{3.2cm}p{3.4cm}p{2.4cm}p{1.6cm}p{1.6cm}",
            size="\\scriptsize")
    M.add("ImaExcl", f"{int(s['IMA++']['n_excluded_isic2018_train']):,}".replace(",", "{,}"))
    M.add("NIma", int(top[(top.cohort == "isic") & (top.source == "IMA++")].n.iloc[0]))
    M.add("NDerm", f"{int(top[(top.cohort == 'isic') & (top.source == 'DermArtifactDB')].n.iloc[0]):,}".replace(",", "{,}"))
    M.add("NKabir", int(top[(top.cohort == "isic") & (top.source == "Kabir")].n.iloc[0]))
    if exc is not None:
        M.add("CapBefore", exc["n_matched_before_exclusion"]); M.add("CapProbe", exc["n_excluded_probe_train"])
        M.add("CapExpert", exc["n_excluded_expert_labelled"])
    M.add("NCap", int(top[(top.cohort == "capsule")].n.iloc[0]))


# ------------------------------------------------------------------------------------------------ A1-A3 per cohort
def bias_rows(cohort):
    b = c6("bias.csv")
    return None if b is None else b[b.cohort == cohort]


def isic():
    a = j6("agreement/isic/agreement.json")
    per = per_image("isic")
    b = bias_rows("isic")
    if a is None or per is None:
        return
    M.add("DermKappa", cia(a["derm_kappa"]["all"]))
    M.add("DermKappaYZero", cia(a["derm_kappa"]["y0"])); M.add("DermKappaYOne", cia(a["derm_kappa"]["y1"]))
    d = per.dropna(subset=["hair_present"])
    free = d[d.cell == "artifact_free"]
    M.add("FreeOk", pct((free.hair_present == 0).mean()))
    M.add("NFreeDerm", len(free))
    neg = d[d.hair_present == 0]
    M.add("FlagNeg", pct(neg.present.astype(bool).mean()))
    M.add("NDermNeg", f"{len(neg):,}".replace(",", "{,}"))
    M.add("DermHairPrev", pct((d.hair_present == 1).mean()))
    M.add("OurHairPrev", pct(d.present.astype(bool).mean()))
    for cell in ("trapA", "trapB"):
        for y in (0, 1):
            q = d[(d.cell == cell) & (d.y == y)]
            M.add(f"DermNo{cell}Y{y}", pct((q.hair_present == 0).mean()))
        M.add(f"DermNo{cell}", pct((d[d.cell == cell].hair_present == 0).mean()))
    M.add("ImaDice", cia(a["ima_dice"]["all"])); M.add("ImaIou", cia(a["ima_iou"]["all"]))
    M.add("ImaTrap", cia(a["ima_trap_class"]["all"]))
    M.add("ImaTrapYZero", cia(a["ima_trap_class"]["y0"])); M.add("ImaTrapYOne", cia(a["ima_trap_class"]["y1"]))
    M.add("KabirDice", cia(a["kabir_dice"]["all"]))
    k = per.dropna(subset=["kabir_px"])
    M.add("KabirPresAgree", pct((k.present.astype(bool) == (k.kabir_present == 1)).mean(), 1))
    M.add("KabirNeg", int((k.kabir_present == 0).sum()))
    kk = k[k.cell.isin(["trapA", "trapB"])]
    M.add("KabirLocAgree", pct((kk.cell == kk.cell_kabir).mean()))
    rows = [["DermArtifactDB", "hair present (ours: $>30$ hair pixels)", f"{len(d):,}".replace(",", "{,}"),
             f"$\\kappa$ {cia(a['derm_kappa']['all'])}", f"$\\kappa$ {cia(a['derm_kappa']['y0'])}",
             f"$\\kappa$ {cia(a['derm_kappa']['y1'])}"],
            ["IMA++ (majority mask)", "lesion mask (Dice)", str(a["ima_dice"]["all"]["n"]), cia(a["ima_dice"]["all"]),
             cia(a["ima_dice"]["y0"]), cia(a["ima_dice"]["y1"])],
            ["IMA++", "trap class ($r$ recomputed)", str(a["ima_trap_class"]["all"]["n"]), cia(a["ima_trap_class"]["all"]),
             cia(a["ima_trap_class"]["y0"]), cia(a["ima_trap_class"]["y1"])],
            ["Kabir et al.", "hair mask (Dice)", str(a["kabir_dice"]["all"]["n"]), cia(a["kabir_dice"]["all"]),
             cia(a["kabir_dice"]["y0"]), cia(a["kabir_dice"]["y1"])],
            ["Kabir et al.", "hair present", str(len(k)), f"agreement \\sEightKabirPresAgree", "", ""]]
    L.table(T / "isic_agree.tex", ["Source", "Quantity", "Images", "All", "$Y=0$ (benign)", "$Y=1$ (melanoma)"], rows,
            "Dermoscopy: agreement of our labels with three independent sources (95\\% bootstrap CIs over images, 2{,}000 "
            "replicates). The Kabir images are all hair images and our masks mark hair on every one of them, so $\\kappa$ "
            "is not informative there; the raw agreement is given instead.", "tab:isic", size="\\scriptsize", resize=True)
    if b is not None:
        g = lambda s, c: b[(b.source == s) & (b.cell == c)].iloc[0]
        M.add("DermDiffA", L.ci(*[float(g("DermArtifactDB", "diff_y1_minus_y0_trapA")[k]) for k in ("error_rate", "ci95_lo", "ci95_hi")]))
        M.add("DermDiffB", L.ci(*[float(g("DermArtifactDB", "diff_y1_minus_y0_trapB")[k]) for k in ("error_rate", "ci95_lo", "ci95_hi")]))
        for s, key in (("IMA++", "Ima"), ("Kabir", "Kabir")):
            ea, eb = g(s, "e_A"), g(s, "e_B")
            M.add(f"{key}eA", f"{int(round(ea.error_rate * ea.n))}/{int(ea.n)}")
            M.add(f"{key}eB", f"{int(round(eb.error_rate * eb.n))}/{int(eb.n)}")
            M.add(f"{key}eSum", L.f3(ea.error_rate + eb.error_rate))


def thyroid():
    a = j6("agreement/thyroid/agreement.json")
    per = per_image("thyroid")
    b = bias_rows("thyroid")
    if a is None or per is None:
        return
    M.add("BusKappa", cia(a["busclean_kappa"]["all"])); M.add("MgKappa", cia(a["medgemma_kappa"]["all"]))
    M.add("BusLoc", cia(a["busclean_location"]["all"])); M.add("MgLoc", cia(a["medgemma_location"]["all"]))
    M.add("NBusLoc", a["busclean_location"]["all"]["n"]); M.add("NMgLoc", f"{a['medgemma_location']['all']['n']:,}".replace(",", "{,}"))
    M.add("BusFreePos", pct(a["busclean_positive_rate_on_caliper_free"], 1))
    inf = a["informative"]
    M.add("BusInf", L.f3(inf["BUSClean"]["kappa"]))
    M.add("MgInf", L.f3(inf["MedGemma"]["kappa"]))
    rows = []
    d = per[per.cell.isin(["artifact_free", "trapA", "trapB"])].copy()
    lab = {"artifact_free": "caliper-free", "trapA": "Trap~A (in the nodule)", "trapB": "Trap~B (outside)"}
    shares = c6("clean_traps/thyroid/shares.csv").set_index("cell")
    for cell in ("artifact_free", "trapA", "trapB"):
        q = d[d.cell == cell]
        bus_yes, mg_yes = q.bus_present.mean(), (q.mg_presence == "yes").mean()
        rows.append([lab[cell], f"{len(q):,}".replace(",", "{,}"), pct(bus_yes), pct(mg_yes),
                     pct(shares.loc[cell, "contradicted"])])
        k = L.Macros.clean(cell)
        M.add(f"BusYes{k}", pct(bus_yes)); M.add(f"MgYes{k}", pct(mg_yes))
        M.add(f"Contra{k}", pct(shares.loc[cell, "contradicted"]))
    L.table(T / "thyroid_labellers.tex", ["Our cell", "Images", "BUSClean: annotation found", "MedGemma: caliper ``yes''",
                                          "Contradicted (registered rule)"], rows,
            "Thyroid: how often each independent labeller reports a caliper in each of our cells, and the share of images "
            "the registered rule removes (an image is removed if \\emph{any} informative labeller places it in another "
            "cell).", "tab:thylab", size="\\scriptsize", align="p{3.4cm}p{1.4cm}p{2.8cm}p{2.8cm}p{2.8cm}")
    cn = c6("clean_traps/thyroid/counts.csv")
    if cn is not None:
        r = cn[cn.trap == "trapB"].iloc[0]
        M.add("GateBYZero", int(r.A1_Y0)); M.add("GateBYOne", int(r.A1_Y1))
    if b is not None:
        g = lambda s, c: b[(b.source == s) & (b.cell == c)].iloc[0]
        for s, key in (("BUSClean", "Bus"), ("MedGemma", "Mg")):
            M.add(f"{key}eA", L.f3(g(s, "e_A").error_rate)); M.add(f"{key}eB", L.f3(g(s, "e_B").error_rate))
            for t in ("A", "B"):
                r = g(s, f"diff_y1_minus_y0_trap{t}")
                M.add(f"{key}Diff{t}", L.ci(float(r.error_rate), float(r.ci95_lo), float(r.ci95_hi)))
    # exploratory E1 (descriptive): contradiction only where both independent labellers agree against our cell
    both = d[d.cell_bus == d.cell_mg]
    rows_c = []
    out = []
    for cell in ("artifact_free", "trapA", "trapB"):
        q = d[d.cell == cell]
        qb = both[both.cell == cell]
        con = (q.cell_bus == q.cell_mg) & (q.cell_bus != cell)
        for y in (0, 1):
            out.append({"cell": cell, "y": y, "n": int((q.y == y).sum()), "n_labellers_agree": int((qb.y == y).sum()),
                        "share_agree_with_ours": float((qb[qb.y == y].cell_bus == cell).mean()),
                        "share_contradicted_by_both": float(con[q.y == y].mean())})
        rows_c.append([lab[cell], f"{len(qb):,}".replace(",", "{,}"), pct((qb.cell_bus == cell).mean()), pct(con.mean()),
                       pct(con[q.y == 0].mean()), pct(con[q.y == 1].mean())])
        M.add(f"ConsContra{L.Macros.clean(cell)}", pct(con.mean()))
        M.add(f"ConsAgree{L.Macros.clean(cell)}", pct((qb.cell_bus == cell).mean()))
    L.derived("stage8_thyroid_consensus", pd.DataFrame(out))
    M.add("NBoth", f"{len(both):,}".replace(",", "{,}")); M.add("NCells", f"{len(d):,}".replace(",", "{,}"))
    M.add("BothAgree", pct((both.cell_bus == both.cell).mean()))
    for y in (0, 1):
        q = d[(d.cell == "trapB") & (d.y == y)]
        M.add(f"ConsContraBY{y}", pct(((q.cell_bus == q.cell_mg) & (q.cell_bus != "trapB")).mean()))
    L.table(T / "thyroid_consensus.tex", ["Our cell", "Both labellers agree with each other", "\\ldots\\ and with our cell",
                                          "Contradicted by both", "benign", "malignant"], rows_c,
            "Thyroid, \\textbf{exploratory}: the images on which BUSClean and MedGemma give the same cell. An image counts as "
            "contradicted only if both place it in the same cell and that cell differs from ours.", "tab:thycons",
            size="\\scriptsize", align="p{3.4cm}p{2.4cm}p{2.0cm}p{2.0cm}p{1.4cm}p{1.4cm}")
    # the registered FT3 subgroup (malignant test nodules with an in-ROI caliper): what the two labellers say
    sub = per[(per.image_id.str.startswith("te_")) & (per.y == 1) & (per.cell == "trapA")]
    M.add("SubN", len(sub))
    M.add("SubConfirmed", int(((sub.cell_bus == "trapA") & (sub.cell_mg == "trapA")).sum()))
    M.add("SubContraBoth", int(((sub.cell_bus == sub.cell_mg) & (sub.cell_bus != "trapA")).sum()))
    M.add("SubMgIn", int((sub.cell_mg == "trapA").sum())); M.add("SubBusIn", int((sub.cell_bus == "trapA").sum()))
    M.add("SubBusNone", int((sub.cell_bus == "artifact_free").sum())); M.add("SubMgNone", int((sub.cell_mg == "artifact_free").sum()))
    L.derived("stage8_ft3_subgroup_labels", pd.DataFrame([{
        "n": len(sub), "confirmed_by_both": int(((sub.cell_bus == "trapA") & (sub.cell_mg == "trapA")).sum()),
        "contradicted_by_both": int(((sub.cell_bus == sub.cell_mg) & (sub.cell_bus != "trapA")).sum()),
        "medgemma_inside": int((sub.cell_mg == "trapA").sum()), "busclean_inside": int((sub.cell_bus == "trapA").sum()),
        "busclean_none": int((sub.cell_bus == "artifact_free").sum()), "medgemma_none": int((sub.cell_mg == "artifact_free").sum())}]))
    ex = j6("exploratory/ft3_confirmed.json") if (R6 / "exploratory" / "ft3_confirmed.json").exists() else None
    M.add("EtwoDone", "yes" if ex else "no")


def ovary():
    a = j6("agreement/ovary/agreement.json")
    if a is None:
        return
    M.add("OvBusFreePos", pct(a["busclean_positive_rate_on_caliper_free"], 1))
    M.add("OvMgKappa", cia(a["medgemma_kappa"]["all"]))
    M.add("OvMgKappaYZero", cia(a["medgemma_kappa"]["y0"])); M.add("OvMgKappaYOne", cia(a["medgemma_kappa"]["y1"]))
    M.add("OvMgLoc", cia(a["medgemma_location"]["all"])); M.add("OvBusKappa", cia(a["busclean_kappa"]["all"]))


def capsule():
    per = per_image("capsule")
    if per is None:
        return
    x = per.dropna(subset=["iou_expert"])
    M.add("CapEmpty", int((x.contam_frac_expert == 0).sum()))
    M.add("CapTrapCells", int(x.cell.isin(["trapA", "trapB"]).sum()))


# ------------------------------------------------------------------------------------------------ A2
def cleaned():
    s = c6("clean_traps/SUMMARY.csv")
    if s is None:
        return
    rows, pts = [], []
    for c, lab in COH:
        x = j6(f"clean_traps/{c}/crossover.json")
        sh = c6(f"clean_traps/{c}/shares.csv").set_index("cell")
        r = s[s.cohort == c].iloc[0]
        o = x["original"]
        k = L.Macros.clean(c)
        M.add(f"Orig{k}", cir(o))
        removed = f"{pct(sh.loc['trapA', 'contradicted'])} / {pct(sh.loc['trapB', 'contradicted'])}"
        if x.get("ran"):
            M.add(f"Clean{k}", cir(x)); M.add(f"CleanP{k}", L.f3(float(r.p_holm)))
            note = {"isic": "tested", "thyroid": "", "capsule": "nothing removed (no source analysable)",
                    "ovary": "nothing removed (no informative source yet)"}[c]
            verdict = "supported" if r.verdict == "SUPPORTED" else "not supported"
            rows.append([lab, cir(o), cir(x), removed, f"{verdict}; {note}" if c != "isic" else verdict])
            pts.append((lab, o, x))
        else:
            M.add(f"Clean{k}", "\\na")
            rows.append([lab, cir(o), "---", removed, "count gate not met (Trap~B)"])
            pts.append((lab, o, None))
    L.table(T / "cleaned.tex", ["Cohort", "Stage 3 crossover", "Uncontradicted labels", "Removed (Trap~A / Trap~B)", "HA2"],
            rows, "Traps re-run on the images no informative source contradicts (A2; DINOv2 ViT-B/14, Stage~3 protocol, "
            "crossed 95\\% CIs, Holm over the cohorts that ran). For capsule and ovary no image was removed, so the re-run "
            "reproduces Stage~3 and adds no evidence about the labels.", "tab:clean", size="\\scriptsize",
            align="p{3.2cm}p{2.6cm}p{2.6cm}p{2.4cm}p{4.0cm}")
    plt = L.plot_style()
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    y = 0
    ticks = []
    for lab, o, x in pts:
        ax.errorbar([o["estimate"]], [y], xerr=[[o["estimate"] - o["ci95_lo"]], [o["ci95_hi"] - o["estimate"]]], fmt="o",
                    color=MUTED, mfc="white", ms=6, capsize=2, lw=1.2, label="Stage 3 labels" if y == 0 else None)
        if x is not None:
            ax.errorbar([x["estimate"]], [y + 0.35], xerr=[[x["estimate"] - x["ci95_lo"]], [x["ci95_hi"] - x["estimate"]]],
                        fmt="s", color=BLUE, ms=6, capsize=2, lw=1.2, label="uncontradicted labels" if y == 0 else None)
        else:
            ax.text(0.01, y + 0.35, "gate not met", fontsize=7, color=MUTED, va="center")
        ticks.append((y + 0.18, lab.split(" (")[0]))
        y += 1.2
    ax.set_yticks([t for t, _ in ticks]); ax.set_yticklabels([l for _, l in ticks], fontsize=8)
    ax.axvline(0, color=MUTED, lw=0.6); ax.invert_yaxis(); ax.set_xlabel("crossover (reversed AUROC)")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(F / "cleaned.pdf"); plt.close(fig)


# ------------------------------------------------------------------------------------------------ Part B
def part_b():
    rec = j6("ft_recipe.json")
    res = j6("ft_results.json")
    op = c6("ft_natural/operating_points.csv")
    if rec is None or res is None or op is None:
        return
    arch_lab = {"convnext_tiny.fb_in22k_ft_in1k": "ConvNeXt-T (IN-22k)", "resnet50": "ResNet-50",
                "vit_base_patch14_dinov2.lvd142m": "DINOv2 ViT-B/14"}
    ch = rec["choice"]
    rows = []
    for r in rec["runs"]:
        best = (r["arch"], r["lr"], r["epochs"]) == (ch["arch"], ch["lr"], ch["epochs"])
        v = L.f3(r["val_auroc"])
        rows.append([arch_lab[r["arch"]], f"{r['lr']:.0e}".replace("e-0", "e-"), r["epochs"], f"\\textbf{{{v}}}" if best else v])
    L.table(T / "recipe.tex", ["Architecture", "Learning rate", "Epochs", "Validation AUROC (ERM, seed 42)"], rows,
            "Recipe selection on validation images only (B1); no test image was scored. The chosen recipe (bold) was "
            "committed before any final model was trained.", "tab:recipe", size="\\scriptsize")
    M.add("Arch", arch_lab[ch["arch"]]); M.add("Lr", "$10^{-4}$" if ch["lr"] == 1e-4 else "$3\\cdot10^{-5}$")
    M.add("Epochs", ch["epochs"]); M.add("ValAuc", L.f3(ch["val_auroc"]))
    dino = max((r for r in rec["runs"] if r["arch"].startswith("vit_base")), key=lambda r: r["val_auroc"])
    M.add("DinoVal", L.f3(dino["val_auroc"]))
    f0 = res["FT0"]
    M.add("FTZero", L.f3(f0["estimate"])); M.add("FTZeroGap", L.f3(f0["benchmark"] - f0["estimate"]))
    lab = {"FT1": "hard-pair AUROC, mask $-$ ERM $<0$", "FT2": "sensitivity loss (CI $<0$) at $\\ge3$ of OP2--OP5",
           "FT3": "OP2 sensitivity, malignant with in-ROI caliper, mask $-$ ERM $<0$",
           "FT4": "thyroid trap crossover $>0$"}
    rows = [["FT0", "ERM test AUROC $\\ge0.773$ (descriptive)", L.f3(f0["estimate"]), "---", "---", "below benchmark"]]
    for k in ("FT1", "FT2", "FT3", "FT4"):
        h = res[k]
        est = f"{h['n_ops_ci_below_0']} of {h['n_ops']}" if k == "FT2" else cir(h)
        rows.append([k, lab[k], est, f"{h['p']:.4f}", f"{h['p_holm']:.4f}", h["verdict"].lower()])
        if k != "FT2":
            M.add(k, cir(h))
        M.add(f"{k}Holm", f"{h['p_holm']:.4f}")
    M.add("FTTwoN", res["FT2"]["n_ops_ci_below_0"]); M.add("FTTwoOf", res["FT2"]["n_ops"])
    L.table(T / "ft.tex", ["", "Hypothesis", "Estimate [95\\% CI]", "$p$ (one-sided)", "Holm $p$", "Verdict"], rows,
            "Part B: the chosen fine-tuned model on the official patient-disjoint thyroid split (5 seeds $\\times$ 4 arms) and "
            "in the thyroid traps (10 bootstrap clusters). Crossed bootstrap; Holm over FT1--FT4.", "tab:ft",
            size="\\scriptsize", align="lp{5.4cm}p{3.0cm}p{1.3cm}p{1.1cm}p{1.9cm}")
    for c in res["contrasts"]:
        M.add(f"Con{L.Macros.clean(c['subset'] + c['contrast'])}", cir(c))
    # operating points
    g = lambda o, m: op[(op.op == o) & (op.metric == m)].iloc[0]
    rows = []
    for o, olab, short in OPS:
        s, sp, sc = g(o, "sens"), g(o, "spec"), g(o, "sens_conflict")
        rows.append([olab, f"{L.f3(s.ref_value)} $\\to$ {L.f3(s.arm_value)}", L.ci(s.delta, s.ci95_lo, s.ci95_hi),
                     L.ci(sp.delta, sp.ci95_lo, sp.ci95_hi), f"{L.f3(sc.ref_value)} $\\to$ {L.f3(sc.arm_value)}",
                     L.ci(sc.delta, sc.ci95_lo, sc.ci95_hi)])
        M.add(f"Sens{short}", L.ci(s.delta, s.ci95_lo, s.ci95_hi)); M.add(f"Spec{short}", L.ci(sp.delta, sp.ci95_lo, sp.ci95_hi))
        M.add(f"SensErm{short}", L.f3(s.ref_value)); M.add(f"SensMask{short}", L.f3(s.arm_value))
        M.add(f"ConfErm{short}", L.f3(sc.ref_value)); M.add(f"ConfMask{short}", L.f3(sc.arm_value))
        M.add(f"SpecErm{short}", L.f3(sp.ref_value)); M.add(f"SpecMask{short}", L.f3(sp.arm_value))
    L.table(T / "op_ft.tex", ["Operating point (fixed on validation)", "Sensitivity ERM $\\to$ mask", "$\\Delta$ sensitivity",
                              "$\\Delta$ specificity", "Sensitivity, malignant with in-ROI caliper (ERM $\\to$ mask)",
                              "$\\Delta$ in that subgroup"], rows,
            "Fine-tuned model, official patient-disjoint thyroid test split: masking at five validation-fixed operating "
            "points (crossed bootstrap, thresholds re-estimated in every replicate).", "tab:opft", size="\\scriptsize",
            resize=True)
    # comparison with the frozen probe of Stage 5 (same split, same subgroup)
    N = L.NEW
    fop = L.csv("final_op/operating_points_crossed_all.csv")
    clin = L.csv("review2/clinical_metrics_repro.csv")
    nat = L.js("natural/thyroid_dino518_repro/natural_boot.json")
    ftc = L.csv("finetune/ft_crossovers.csv")
    t3 = L.js("thyroid/dino518_main/T3_crossover.json")
    if fop is None or clin is None or nat is None or t3 is None:
        return
    th = fop[fop.cohort == "thyroid"]
    fg = lambda o, m: th[(th.op == o) & (th.metric == m)].iloc[0]
    ce = clin[clin.test.str.startswith("Thyroid") & (clin.arm == "erm")].iloc[0]
    s2, sc2 = fg("OP2_spec0.80", "sens"), fg("OP2_spec0.80", "sens_conflict")
    h = nat["hard | mask-erm"]
    vits = ftc[ftc.run == "Thyroid, ViT-S"].iloc[0] if ftc is not None else None
    rows = [["ERM test AUROC", L.f3(ce.AUROC_mean), L.f3(f0["estimate"])],
            ["Hard pairs, mask $-$ ERM", L.ci(h[0], h[1], h[2]), cir(res["FT1"])],
            ["OP2 sensitivity, mask $-$ ERM", L.ci(s2.delta, s2.crossed_lo, s2.crossed_hi), M.items["sEightSensOPTwo"]],
            [f"OP2 sensitivity, {int(sc2.n_conflicting_positives)} malignant with in-ROI caliper",
             f"{L.f3(sc2.erm_value)} $\\to$ {L.f3(sc2.mask_value)}",
             f"{M.items['sEightConfErmOPTwo']} $\\to$ {M.items['sEightConfMaskOPTwo']}"],
            ["\\quad change", L.ci(sc2.delta, sc2.crossed_lo, sc2.crossed_hi), cir(res["FT3"])],
            ["Thyroid trap crossover", L.ci(t3["seed_delta_mean"] if "seed_delta_mean" in t3 else t3["estimate"],
                                            t3["ci95_lo"], t3["ci95_hi"]), cir(res["FT4"])]]
    L.table(T / "compare.tex", ["Quantity", "Frozen DINOv2 ViT-B/14 probe (Stage~5)", f"Fine-tuned {M.items['sEightArch']} (this stage)"],
            rows, "The same analyses for the frozen probe of Stage~5 and for the fine-tuned model of this stage. The subgroup "
            "was post hoc for the probe and registered in advance for the fine-tuned model.", "tab:cmp", size="\\scriptsize",
            align="p{5.6cm}p{4.2cm}p{4.2cm}")
    M.add("FrozenAuc", L.f3(ce.AUROC_mean)); M.add("FrozenHard", L.ci(h[0], h[1], h[2]))
    M.add("FrozenConf", L.ci(sc2.delta, sc2.crossed_lo, sc2.crossed_hi))
    M.add("FrozenCross", L.ci(t3.get("seed_delta_mean", t3.get("estimate")), t3["ci95_lo"], t3["ci95_hi"]))
    if vits is not None:
        M.add("ViTSCross", L.ci(vits.seed_delta_mean, vits.ci95_lo, vits.ci95_hi))
    M.add("NSub", int(sc2.n_conflicting_positives))
    # figure: sensitivity at each operating point, fine-tuned (filled) and frozen probe (open)
    plt = L.plot_style()
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 2.9), sharey=True)
    xs = np.arange(len(OPS))
    for a, (m, title) in zip(ax, (("sens", "all malignant nodules"),
                                  ("sens_conflict", f"{int(sc2.n_conflicting_positives)} malignant nodules with an in-ROI caliper"))):
        for dx, arm, col, mk, lab_ in ((-0.12, "erm", BLUE, "o", "ERM"), (0.12, "mask", ORANGE, "s", "mask")):
            ftv = [float(g(o, m)["ref_value" if arm == "erm" else "arm_value"]) for o, _, _ in OPS]
            frv = [float(fg(o, m)[f"{arm}_value"]) for o, _, _ in OPS]
            a.plot(xs + dx, ftv, mk, color=col, ms=7, label=f"{lab_}, fine-tuned")
            a.plot(xs + dx, frv, mk, color=col, mfc="white", ms=7, label=f"{lab_}, frozen probe (Stage 5)")
            for x0, v1, v2 in zip(xs + dx, ftv, frv):
                a.plot([x0, x0], [v1, v2], color=col, lw=0.8, alpha=0.5)
        a.set_xticks(xs); a.set_xticklabels([s for _, _, s in OPS]); a.set_title(title, fontsize=8); a.set_ylim(0, 1)
    ax[0].set_ylabel("test sensitivity")
    hh, ll = ax[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.legend(hh, ll, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.savefig(F / "op_ft.pdf"); plt.close(fig)


def exploratory():
    """E1/E2 results, when the exploratory script has been run on the GPU machine."""
    e1 = R6 / "exploratory" / "consensus_traps" / "thyroid" / "crossover.json"
    M.add("EoneDone", "yes" if e1.exists() else "no")
    if e1.exists():
        x = json.loads(e1.read_text())
        M.add("EoneCross", cir(x) if x.get("ran") else "\\na")
        cn = pd.read_csv(e1.parent / "counts.csv")
        r = cn[cn.trap == "trapB"].iloc[0]
        M.add("EoneBYZero", int(r.A1_Y0)); M.add("EoneBYOne", int(r.A1_Y1))
    e2 = R6 / "exploratory" / "ft3_confirmed.json"
    if e2.exists():
        x = json.loads(e2.read_text())
        for model, key in (("finetuned_round6", "Ft"), ("frozen_stage5", "Frozen")):
            for grp, gk in (("confirmed_by_both", "Conf"), ("not_confirmed_by_both", "Rest")):
                if model in x:
                    r = x[model][grp]
                    M.add(f"Etwo{key}{gk}", cir(r, "delta")); M.add(f"Etwo{key}{gk}N", r["n_nodules"])


def round7():
    """Round 7: each arm's threshold chosen on the test set itself (docs/PREREGISTRATION_ROUND7.md)."""
    R7 = L.ROOT / "results" / "round7"
    if not (R7 / "tm.json").exists():
        L.MISSING.append("results/round7/tm.json")
        return
    tm = json.loads((R7 / "tm.json").read_text())
    d = pd.read_csv(R7 / "matched_thresholds.csv")
    for k in ("TM1", "TM2"):
        r = tm[k]
        key = k.replace("TM", "TM")
        M.add(key, cir(r)); M.add(f"{key}Holm", f"{r['p_holm']:.3f}")
        M.add(f"{key}Erm", L.f3(r["erm_value"])); M.add(f"{key}Mask", L.f3(r["mask_value"]))
        M.add(f"{key}Verdict", r["verdict"].lower())
    res = j6("ft_results.json")
    if res is not None:
        M.add("TMShare", pct(tm["TM1"]["estimate"] / res["FT3"]["estimate"]))
    mlab = {"M-spec80": "specificity 0.80", "M-spec90": "specificity 0.90", "M-sens80": "sensitivity 0.80"}
    rows = []
    for model, mname in (("finetuned_round6", "fine-tuned"), ("frozen_stage5", "frozen probe")):
        for m, ml in mlab.items():
            g = lambda k: d[(d.model == model) & (d.cohort == "thyroid") & (d.match == m) & (d.metric == k)].iloc[0]
            cells = []
            for k in ("sens_conflict", "sens_aligned", "sens"):
                r = g(k)
                cells.append(f"{L.f3(r.erm_value)} $\\to$ {L.f3(r.mask_value)}, {L.ci(r.delta, r.ci95_lo, r.ci95_hi)}")
                M.add(f"M{L.Macros.clean(model + m + k)}", L.ci(r.delta, r.ci95_lo, r.ci95_hi))
            sp = g("spec")
            rows.append([mname if m == "M-spec80" else "", f"test {ml}", *cells, f"{L.f3(sp.erm_value)} / {L.f3(sp.mask_value)}"])
    L.table(T / "matched.tex", ["Model", "Both arms matched at", "Malignant, in-ROI caliper (78)", "Malignant, no in-ROI caliper",
                                "All malignant", "Specificity ERM / mask"], rows,
            "Round 7: sensitivity when each arm's threshold is chosen on the test set itself, so that both arms reach the same "
            "test specificity (or sensitivity); ERM $\\to$ mask, and mask $-$ ERM with crossed 95\\% CIs. The first row of each "
            "model is the registered test (TM1, TM2).", "tab:matched", size="\\scriptsize", resize=True)
    rows = []
    lab = {"isic_BCN": "Dermoscopy, BCN held out", "isic_HAM": "Dermoscopy, HAM held out", "isic_MSK": "Dermoscopy, MSK held out",
           "capsule": "Capsule (held-out frames)"}
    for c, cl in lab.items():
        g = lambda k: d[(d.model == "frozen_stage5") & (d.cohort == c) & (d.match == "M-spec80") & (d.metric == k)].iloc[0]
        rows.append([cl] + [L.ci(g(k).delta, g(k).ci95_lo, g(k).ci95_hi) for k in ("sens_conflict", "sens_aligned", "sens")])
        M.add(f"MOther{L.Macros.clean(c)}", L.ci(g("sens_conflict").delta, g("sens_conflict").ci95_lo, g("sens_conflict").ci95_hi))
    L.table(T / "matched_other.tex", ["Test set (frozen probe)", "Conflicting positives", "Aligned positives", "All positives"], rows,
            "Round 7, descriptive: the other natural test sets of Stage~5 at matched test specificity 0.80, mask $-$ ERM "
            "(crossed 95\\% CIs); conflicting = positives whose artifact status conflicts with the test set's own association.",
            "tab:matchedother", size="\\scriptsize")


def main():
    experiments()
    coverage()
    isic(); thyroid(); ovary(); capsule()
    cleaned()
    part_b()
    round7()
    exploratory()
    M.write(T / "numbers.tex")
    L.report_missing("stage8")


if __name__ == "__main__":
    main()
