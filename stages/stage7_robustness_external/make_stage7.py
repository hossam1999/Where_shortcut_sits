"""Stage 7 tables, macros and figures: masking implementations (R8), dose-response over the real overlap (R9), the
external natural test ISIC 2019 -> ISIC 2020 (R10) and the prospective theory test (R11); docs/PREREGISTRATION_ROUND4.md.
Every number is read from results/round4/ (crossed seed x image bootstrap).
  PYTHONPATH=src python stages/stage7_robustness_external/make_stage7.py"""
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

R4 = L.ROOT / "results" / "round4"
T, F = HERE / "tables", HERE / "figures"
T.mkdir(exist_ok=True); F.mkdir(exist_ok=True)
M = L.Macros("sSeven")
COH = [("isic", "Dermoscopy hair (ISIC 2019)"), ("thyroid", "Thyroid calipers"), ("capsule", "Capsule debris"),
       ("ovary", "Ovary calipers")]
LAB = dict(COH)
VIEWS = [("mask", "mean-colour fill"), ("mask_black", "black fill"), ("mask_blur", "blurred background"),
         ("crop_box", "box crop"), ("crop_mask", "box crop of the masked image")]
VLAB = dict(VIEWS)
VCOL = {"mask": "#2a78d6", "mask_black": "#eb6834", "mask_blur": "#1baf7a", "crop_box": "#eda100", "crop_mask": "#e87ba4"}
VMARK = {"mask": "o", "mask_black": "s", "mask_blur": "D", "crop_box": "^", "crop_mask": "v"}
TRAP_COL = {"trapA": "#eb6834", "trapB": "#2a78d6"}
INK, MUTED = "#0b0b0b", "#898781"


def c4(rel):
    return L.csv(rel, R4)


def j4(rel):
    return L.js(rel, R4)


def cir(r, est="estimate"):
    return L.ci(float(r[est]), float(r["ci95_lo"]), float(r["ci95_hi"]))


def first_commit(rel: str) -> str:
    out = subprocess.run(["git", "-C", str(L.ROOT), "log", "--diff-filter=A", "--format=%h|%ad", "--date=format:%Y-%m-%d %H:%M",
                          "--", rel], capture_output=True, text=True).stdout.split("\n")
    out = [o for o in out if o.strip()]
    if not out:
        L.MISSING.append(rel)
        return "\\na"
    h, d = out[-1].split("|")
    return f"\\texttt{{{h}}}, {d}"


# ------------------------------------------------------------------------------------------------ registration
def experiments():
    doc = "docs/PREREGISTRATION_ROUND4.md"
    M.add("Reg", L.reg(doc))
    rows = [["R8 masking implementations (H8a, H8b)", "\\registered{round 4}", L.reg(doc),
             "H8a: $C_v>0$ for black fill and box crop of the masked image in all four cohorts (Holm over 16)"],
            ["R9 dose-response over the real overlap (H9)", "\\registered{round 4}; bin counts committed before fitting", L.reg(doc),
             "slope of the mask gain on $r$ $<0$ in every cohort that passes the count gate (Holm)"],
            ["R10 ISIC 2019 $\\to$ ISIC 2020 natural test (H10a--c)", "\\registered{round 4}", L.reg(doc),
             "hard-pair AUROC mask $-$ ERM $<0$; OP5 sensitivity loss; balancing $>$ mask on hard pairs"],
            ["R11 prospective theory test (T1$'$--T3$'$)", "\\registered{round 4}", L.reg(doc),
             "all crossover signs, $r\\ge0.7$, MAE below ``reversed = clean''"]]
    L.table(T / "experiments.tex", ["Experiment", "Status", "First commit (local time)", "Support criterion"], rows,
            "Experiments of this stage. One registration document (\\texttt{docs/PREREGISTRATION\\_ROUND4.md}) was committed "
            "before any feature, count or model of this stage existed. Commit times are set by the committing machine and "
            "are not independent proof of order.", "tab:exp", align="p{3.6cm}p{3.2cm}p{2.8cm}p{5.4cm}", size="\\scriptsize")
    M.add("CommitCounts", first_commit("results/round4/dose_response/isic/counts.csv"))
    M.add("CommitGains", first_commit("results/round4/dose_response/isic/bin_gains.csv"))


# ------------------------------------------------------------------------------------------------ R8
def views_schematic():
    """The five views on a synthetic image (no patient image is shown)."""
    sys.path.insert(0, str(L.ROOT / "scripts" / "round4"))
    import cv2
    from masking_variants import variant_renderers
    from wtss.ops import apply_roi_mask
    from PIL import Image
    rng = np.random.default_rng(3)
    S = 256
    img = np.clip(np.array([214, 170, 140]) + rng.normal(0, 7, (S, S, 3)), 0, 255).astype(np.uint8)
    yy, xx = np.mgrid[:S, :S]
    roi = (((yy - 132) / 58.0) ** 2 + ((xx - 120) / 70.0) ** 2 <= 1).astype(np.uint8)
    img[roi > 0] = np.clip(np.array([120, 72, 52]) + rng.normal(0, 10, (int(roi.sum()), 3)), 0, 255).astype(np.uint8)
    for (x0, y0, x1, y1) in ((20, 10, 70, 240), (200, 20, 240, 250), (90, 110, 160, 160)):
        cv2.line(img, (x0, y0), (x1, y1), (30, 22, 18), 2)

    class C:
        def get(self, i):
            return img, roi, None
    R = variant_renderers(C(), size=S)
    panels = [("original", Image.fromarray(img))] + [(VLAB[v], (apply_roi_mask(Image.fromarray(img), roi) if v == "mask" else R[v](0)))
                                                     for v, _ in VIEWS]
    plt = L.plot_style()
    fig, ax = plt.subplots(1, 6, figsize=(10, 2.0))
    for a, (t, im) in zip(ax, panels):
        a.imshow(np.asarray(im)); a.set_title(t, fontsize=8); a.axis("off")
    fig.savefig(F / "views_schematic.pdf"); plt.close(fig)


def r8():
    s = c4("masking_variants/SUMMARY.csv")
    if s is None:
        return
    g = lambda c, v, q: s[(s.cohort == c) & (s.view == v) & (s.quantity == q)].iloc[0]
    prim = s[(s.quantity == "crossover") & s.view.isin([v for v, _ in VIEWS[1:]])]
    full = prim[prim.view.isin(["mask_black", "crop_mask"])]
    part = prim[prim.view.isin(["mask_blur", "crop_box"])]
    M.add("HEightaN", int(full.supported.sum())); M.add("HEightaOf", len(full))
    M.add("HEightbN", int(part.supported.sum())); M.add("HEightbOf", len(part))
    con = s[(s.quantity == "C_mask_minus_C_v") & s.view.isin(["mask_blur", "crop_box"])]
    M.add("HEightbConN", int((con.ci95_lo > 0).sum())); M.add("HEightbConOf", len(con))
    M.add("HolmMax", L.f3(float(prim.p_holm16.max())))
    M.add("NPrim", len(prim)); M.add("NPrimOK", int(prim.supported.sum()))
    rows, rows_c = [], []
    for c, lab in COH:
        rows.append([lab] + [cir(g(c, v, "crossover")) for v, _ in VIEWS])
        rows_c.append([lab] + [cir(g(c, v, "C_mask_minus_C_v")) for v, _ in VIEWS[1:]])
        for v, _ in VIEWS:
            k = L.Macros.clean(c + v)
            M.add(f"X{k}", cir(g(c, v, "crossover")))
            for t in ("trapA", "trapB"):
                for e in ("rev", "clean"):
                    M.add(f"D{k}{t}{e}", cir(g(c, v, f"{t}_delta_{e}")))
            if v != "mask":
                M.add(f"C{k}", cir(g(c, v, "C_mask_minus_C_v")))
    hdr = ["Cohort"] + [VLAB[v] + (" (reference)" if v == "mask" else "") for v, _ in VIEWS]
    L.table(T / "r8_crossover.tex", hdr, rows,
            "Location crossover $C_v$ for every masking implementation (DINOv2 ViT-B/14, crossed 95\\% CIs). All 16 "
            "non-reference crossovers are positive with Holm-adjusted $p\\le$ \\sSevenHolmMax{} (family of 16).",
            "tab:r8", size="\\scriptsize", resize=True)
    L.table(T / "r8_contrast.tex", ["Cohort"] + [VLAB[v] for v, _ in VIEWS[1:]], rows_c,
            "Secondary contrast $C_{\\text{mask}}-C_v$: how much smaller the crossover of an implementation is than that of "
            "mean-colour fill (positive = weaker location effect than the reference).", "tab:r8c", size="\\scriptsize",
            resize=True)
    sec = []
    for c, lab in COH:
        for v, _ in VIEWS:
            sec.append([lab if v == "mask" else "", VLAB[v], cir(g(c, v, "trapA_delta_rev")), cir(g(c, v, "trapB_delta_rev")),
                        cir(g(c, v, "trapA_delta_clean")), cir(g(c, v, "trapB_delta_clean"))])
    L.table(T / "r8_traps.tex", ["Cohort", "Implementation", "Trap~A reversed", "Trap~B reversed", "Trap~A clean", "Trap~B clean"],
            sec, "Each implementation minus ERM, per trap and test environment (AUROC, crossed 95\\% CIs; descriptive).",
            "tab:r8t", size="\\scriptsize", resize=True)
    # repro gate and counts
    rr = []
    for c, lab in COH:
        rc, cn = c4(f"masking_variants/{c}/repro_check.csv"), c4(f"masking_variants/{c}/counts.csv")
        if rc is None or cn is None:
            continue
        a, b = cn[cn.trap == "trapA"].iloc[0], cn[cn.trap == "trapB"].iloc[0]
        rr.append({"cohort": c, "max_abs_diff": float(rc.abs_diff.max()), "pool_images": int(a.pool_images),
                   "empty_roi_images": int(a.empty_roi_images)})
    rr = L.derived("stage7_repro_gate", pd.DataFrame(rr))
    M.add("ReproMax", L.f3(float(rr.max_abs_diff.max())))
    M.add("EmptyRoi", int(rr.empty_roi_images.sum()))
    # figures
    plt = L.plot_style()
    fig, ax = plt.subplots(1, 4, figsize=(10, 2.8), sharey=True)
    for a, (c, lab) in zip(ax, COH):
        for k, (v, _) in enumerate(VIEWS):
            r = g(c, v, "crossover")
            a.errorbar([k], [r.estimate], yerr=[[r.estimate - r.ci95_lo], [r.ci95_hi - r.estimate]], fmt=VMARK[v], color=VCOL[v],
                       ms=6, capsize=2, lw=1.2, label=VLAB[v])
        a.axhline(0, color=MUTED, lw=0.6); a.set_title(lab, fontsize=8); a.set_xticks([])
    ax[0].set_ylabel("crossover $C_v$ (reversed AUROC)")
    h, lb = ax[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.legend(h, lb, loc="lower center", ncol=5, fontsize=7, frameon=False)
    fig.savefig(F / "r8_crossover.pdf"); plt.close(fig)
    fig, ax = plt.subplots(1, 4, figsize=(10, 2.8), sharey=True)
    for a, (c, lab) in zip(ax, COH):
        for k, (v, _) in enumerate(VIEWS):
            for dx, t in ((-0.15, "trapA"), (0.15, "trapB")):
                r = g(c, v, f"{t}_delta_rev")
                a.errorbar([k + dx], [r.estimate], yerr=[[r.estimate - r.ci95_lo], [r.ci95_hi - r.estimate]],
                           fmt="o" if t == "trapA" else "s", color=TRAP_COL[t], ms=5, capsize=2, lw=1.1,
                           label=("Trap A (artifact in ROI)" if t == "trapA" else "Trap B (artifact outside)") if k == 0 else None)
        a.axhline(0, color=MUTED, lw=0.6); a.set_title(lab, fontsize=8)
        a.set_xticks(range(len(VIEWS))); a.set_xticklabels(["mean", "black", "blur", "box", "box+mask"], fontsize=6.5, rotation=30)
    ax[0].set_ylabel("implementation $-$ ERM, reversed AUROC")
    h, lb = ax[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.legend(h, lb, loc="lower center", ncol=2, fontsize=7, frameon=False)
    fig.savefig(F / "r8_traps.pdf"); plt.close(fig)


# ------------------------------------------------------------------------------------------------ R9
def r9():
    s = c4("dose_response/SUMMARY.csv")
    if s is None:
        return
    rows, cnt_rows, per_all = [], [], []
    for c, lab in COH:
        r = s[s.cohort == c].iloc[0]
        k = L.Macros.clean(c)
        M.add(f"Slope{k}", L.ci(r.slope, r.ci95_lo, r.ci95_hi))
        M.add(f"Bins{k}", int(r.n_bins))
        rows.append([lab, int(r.n_bins), L.ci(r.slope, r.ci95_lo, r.ci95_hi), L.f3(float(r.p_holm)) if float(r.p_holm) >= 0.001 else "$<0.001$",
                     f"{float(r.spearman_x_gain):.2f}", L.f3(float(r.zero_crossing_r)) if pd.notna(r.zero_crossing_r) else "none",
                     "\\ok" if bool(r.H9_supported) else "\\no"])
        bg = c4(f"dose_response/{c}/bin_gains.csv")
        cn = c4(f"dose_response/{c}/counts.csv")
        if bg is None or cn is None:
            continue
        per_all.append(bg)
        for q in bg.itertuples():
            n = cn[cn.bin == q.bin].iloc[0]
            cnt_rows.append([lab if q.Index == 0 else "", q.bin, f"{n.lo:.2f}--{min(n.hi, 1.0):.2f}", L.f3(q.x_mean_r),
                             f"{int(n.A1_Y0)} / {int(n.A1_Y1)}", f"{int(n.A0_Y0)} / {int(n.A0_Y1)}", int(n.rev_pos_seed42),
                             L.ci(q.gain, q.ci95_lo, q.ci95_hi)])
            M.add(f"G{k}{q.bin}", L.ci(q.gain, q.ci95_lo, q.ci95_hi))
    M.add("NHNine", int(s.H9_supported.sum())); M.add("NHNineOf", len(s))
    zi = s[s.cohort == "isic"].zero_crossing_r
    M.add("ZeroIsic", L.f3(float(zi.iloc[0])) if len(zi) and pd.notna(zi.iloc[0]) else "\\na")
    L.table(T / "r9_slopes.tex", ["Cohort", "Bins", "Slope of the mask gain on $r$", "Holm $p$", "Spearman", "Zero crossing $r$",
                                  "H9"], rows,
            "Dose-response: least-squares slope of $g_b=$ rev(mask) $-$ rev(ERM) on the bin mean overlap $r$ (crossed 95\\% "
            "CIs; the artifact-free images shared by all bins carry one bootstrap weight).", "tab:r9", size="\\scriptsize")
    L.table(T / "r9_bins.tex", ["Cohort", "Bin", "$r$ range", "Mean $r$", "$A{=}1$: $Y{=}0$ / $Y{=}1$", "$A{=}0$: $Y{=}0$ / $Y{=}1$",
                                "Rev.\\ positives", "Mask gain $g_b$"], cnt_rows,
            "Bins after the registered count gate ($\\ge25$ per matched cell, $\\ge40$ reversed-test positives over the folds "
            "of seed 42); merged bins are named by their parts. Cell counts are those of the matched pool.", "tab:r9b",
            size="\\scriptsize", resize=True)
    if not per_all:
        return
    per = pd.concat(per_all, ignore_index=True)
    tb = c4("theory/bin_gains.csv")
    plt = L.plot_style()
    fig, ax = plt.subplots(1, 4, figsize=(10, 2.8), sharey=True)
    for a, (c, lab) in zip(ax, COH):
        q = per[per.cohort == c].sort_values("x_mean_r")
        a.errorbar(q.x_mean_r, q.gain, yerr=[q.gain - q.ci95_lo, q.ci95_hi - q.gain], fmt="o-", color=VCOL["mask"], ms=5,
                   capsize=2, lw=1.2, label="observed (crossed 95% CI)")
        if tb is not None:
            p = tb[tb.cohort == c].set_index("trap").reindex(q.bin)
            a.plot(q.x_mean_r, p.pred.to_numpy(), "o", mfc="none", color=INK, ms=6, lw=0, label="theory, predicted")
        a.axhline(0, color=MUTED, lw=0.6); a.set_title(lab, fontsize=8); a.set_xlabel("mean overlap $r$ of the bin")
        a.set_xlim(-0.03, 1.0)
    ax[0].set_ylabel("rev. AUROC: mask $-$ ERM")
    h, lb = ax[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.legend(h, lb, loc="lower center", ncol=2, fontsize=7, frameon=False)
    fig.savefig(F / "r9_dose.pdf"); plt.close(fig)


# ------------------------------------------------------------------------------------------------ R10
OPN = {"OP1_maxBA": "OP1 max.\\ balanced accuracy", "OP2_spec0.80": "OP2 specificity $\\ge0.80$",
       "OP3_spec0.90": "OP3 specificity $\\ge0.90$", "OP4_sens0.80": "OP4 sensitivity $\\ge0.80$",
       "OP5_sens0.90": "OP5 sensitivity $\\ge0.90$"}
ARMN = {"erm": "ERM", "mask": "mask", "balanced": "balanced", "mask_balanced": "mask + balanced", "mte": "U-MtE",
        "mte_balanced": "U-MtE balanced", "mte_protect": "U-MtE protect", "mte_protect_balanced": "U-MtE protect balanced",
        "dfr": "DFR", "mask_dfr": "mask + DFR", "jtt": "JTT"}


def r10():
    cn = j4("natural_isic2020/counts.json")
    if cn is None:
        return
    for k in ("isic2020_with_lesion_mask", "dropped_near_duplicates", "test_images", "test_melanomas", "test_patients",
              "isic2019_train_val_images", "isic2019_melanomas", "tau_pred_hair_px"):
        M.add(L.Macros.clean(k.title().replace("_", "")), f"{cn[k]:,}".replace(",", "{,}"))
    for ds in ("isic2019", "isic2020"):
        h = cn[f"{ds}_in_lesion_hair"]
        M.add(f"Hair{ds}mel", f"{100 * h['P(a|y=1)']:.1f}\\%"); M.add(f"Hair{ds}ben", f"{100 * h['P(a|y=0)']:.1f}\\%")
    M.add("DropPct", f"{100 * cn['dropped_near_duplicates'] / cn['isic2020_with_lesion_mask']:.0f}\\%")
    # near-duplicate removal: distance distribution and label balance (derived; no interval)
    dd = c4("natural_isic2020/dedup.csv")
    coh = L.csv("external_isic2020/cohort.csv")
    if dd is not None and coh is not None:
        lab = dict(zip("i20_" + coh.image_name.astype(str), coh.target))
        dd["y"] = dd.image_id.map(lab)
        dr = dd[dd.dropped]
        bins = [(0, 0), (1, 2), (3, 5), (6, 8)]
        rows = [{"hamming": f"{a}-{b}", "n_dropped": int(dr.hamming.between(a, b).sum())} for a, b in bins]
        rows += [{"hamming": "melanoma drop rate", "n_dropped": float(dd[dd.y == 1].dropped.mean())},
                 {"hamming": "benign drop rate", "n_dropped": float(dd[dd.y == 0].dropped.mean())}]
        d = L.derived("stage7_isic2020_dedup", pd.DataFrame(rows))
        for (a, b), n in zip(bins, d.n_dropped[:4]):
            M.add(f"Dup{a}to{b}", f"{int(n):,}".replace(",", "{,}"))
        M.add("DropMel", f"{100 * float(d.n_dropped.iloc[4]):.0f}\\%"); M.add("DropBen", f"{100 * float(d.n_dropped.iloc[5]):.0f}\\%")
    nm = c4("natural_isic2020/natural_metrics.csv")
    if nm is not None:
        rows = []
        for m in ARMN:
            q = nm[nm.method == m]
            if q.empty:
                continue
            r = q.iloc[0]
            rows.append([ARMN[m], L.f3(r.auc), L.f3(r.cross_group), L.f3(r.auc_in_lesion_hair), L.f3(r.auc_hard), L.f3(r.auc_easy)])
            for col in ("auc", "cross_group", "auc_hard", "auc_easy"):
                M.add(f"Nm{m}{col}", L.f3(float(r[col])))
        L.table(T / "r10_metrics.tex", ["Arm", "AUROC", "Cross-group AUROC", "AUROC, in-lesion hair", "Hard pairs", "Easy pairs"], rows,
                "ISIC 2020 (external, natural prevalence), models trained on ISIC 2019: seed means. Hard pairs = melanomas "
                "without in-lesion hair vs benign lesions with it (they conflict with the training association).",
                "tab:r10m", size="\\scriptsize")
    nb = c4("natural_isic2020/natural_boot.csv")
    if nb is not None:
        rows = []
        for a1, a0 in (("mask", "erm"), ("balanced", "mask"), ("mte_balanced", "mask"), ("mte", "mask")):
            g = lambda sub: nb[(nb.subset == sub) & (nb.arm == a1) & (nb.ref == a0)].iloc[0]
            rows.append([f"{ARMN[a1]} $-$ {ARMN[a0]}", cir(g("all")), cir(g("hard")), cir(g("easy"))])
            for sub in ("all", "hard", "easy"):
                M.add(f"B{a1}{a0}{sub}", cir(g(sub)))
            M.add(f"Bhi{a1}{a0}", f"{float(g('hard').ci95_hi):+.4f}")
        L.table(T / "r10_boot.tex", ["Contrast", "All pairs", "Hard pairs", "Easy pairs"], rows,
                "ISIC 2020: AUROC contrasts with crossed 95\\% CIs (5 seeds; one Poisson weight per ISIC 2020 image).",
                "tab:r10b", size="\\scriptsize")
    op = c4("natural_isic2020/operating_points.csv")
    if op is not None:
        rows = []
        for o, lab in OPN.items():
            g = lambda arm, ref, met: op[(op.arm == arm) & (op.ref == ref) & (op.op == o) & (op.metric == met)].iloc[0]
            s, sp, sc = g("mask", "erm", "sens"), g("mask", "erm", "spec"), g("mask", "erm", "sens_conflict")
            rows.append([lab, f"{L.f3(s.ref_value)} $\\to$ {L.f3(s.arm_value)}", L.ci(s.delta, s.ci95_lo, s.ci95_hi),
                         L.ci(sp.delta, sp.ci95_lo, sp.ci95_hi), f"{L.f3(sc.ref_value)} $\\to$ {L.f3(sc.arm_value)}",
                         L.ci(sc.delta, sc.ci95_lo, sc.ci95_hi)])
            k = L.Macros.clean(o.split("_")[0])
            M.add(f"Op{k}sens", L.ci(s.delta, s.ci95_lo, s.ci95_hi)); M.add(f"Op{k}conf", L.ci(sc.delta, sc.ci95_lo, sc.ci95_hi))
            M.add(f"Op{k}spec", L.ci(sp.delta, sp.ci95_lo, sp.ci95_hi))
        L.table(T / "r10_op.tex", ["Operating point (threshold from ISIC 2019 validation)", "Sensitivity ERM $\\to$ mask",
                                   "$\\Delta$ sensitivity", "$\\Delta$ specificity", "Sensitivity, melanomas without in-lesion hair",
                                   "$\\Delta$"], rows,
                "ISIC 2020 operating points, mask $-$ ERM (crossed bootstrap with thresholds re-estimated on the weighted "
                "validation images in every replicate, 2{,}000 replicates).", "tab:r10op", size="\\scriptsize", resize=True)
    if nb is not None:
        plt = L.plot_style()
        fig, ax = plt.subplots(figsize=(6.2, 2.4))
        labs = []
        for i, (a1, a0) in enumerate((("mask", "erm"), ("balanced", "mask"), ("mte_balanced", "mask"))):
            for j, (sub, col, mk) in enumerate((("all", "#52514e", "o"), ("hard", "#eb6834", "s"), ("easy", "#2a78d6", "D"))):
                r = nb[(nb.subset == sub) & (nb.arm == a1) & (nb.ref == a0)].iloc[0]
                y = i * 4 + j
                ax.errorbar([r.estimate], [y], xerr=[[r.estimate - r.ci95_lo], [r.ci95_hi - r.estimate]], fmt=mk, color=col, ms=5,
                            capsize=2, lw=1.2, label={"all": "all pairs", "hard": "hard pairs", "easy": "easy pairs"}[sub] if i == 0 else None)
            labs.append((i * 4 + 1, f"{ARMN[a1]} $-$ {ARMN[a0]}"))
        ax.axvline(0, color=MUTED, lw=0.6); ax.set_yticks([p for p, _ in labs]); ax.set_yticklabels([t for _, t in labs], fontsize=8)
        ax.invert_yaxis(); ax.set_xlabel("AUROC difference on ISIC 2020"); ax.legend(fontsize=7, frameon=False, loc="lower right")
        fig.tight_layout(); fig.savefig(F / "r10_pairs.pdf"); plt.close(fig)


# ------------------------------------------------------------------------------------------------ R11
def r11():
    sm = j4("theory/summary.json")
    if sm is None:
        return
    t1, ra, rb = sm["T1"]["theory"], sm["T1"]["ref_a_rev_eq_clean"], sm["T1"]["ref_b_symmetry"]
    M.add("TN", t1["n"]); M.add("TMae", L.f3(t1["mae"])); M.add("TR", L.f3(t1["r"])); M.add("TWithin", f"{100 * t1['within_0.05']:.0f}\\%")
    M.add("TMaeClean", L.f3(ra["mae"])); M.add("TMaeSym", L.f3(rb["mae"]))
    t2, t3s, t3b = sm["T2"], sm["T3"]["slopes"], sm["T3"]["bin_gains"]
    M.add("TTwoN", t2["n"]); M.add("TTwoSign", t2["sign_agree"]); M.add("TTwoR", L.f3(t2["pearson_r"])); M.add("TTwoMae", L.f3(t2["mae"]))
    M.add("TThreeN", t3s["n"]); M.add("TThreeSign", t3s["sign_agree"]); M.add("TThreeMae", L.f3(t3s["mae"]))
    M.add("TBinN", t3b["n"]); M.add("TBinSign", t3b["sign_agree"]); M.add("TBinMae", L.f3(t3b["mae"]))
    M.add("TVerdict", sm["verdict"])
    rows = [["T1$'$ reversed AUROC, all new cells", t1["n"], f"MAE {L.f3(t1['mae'])}, $r=$ {L.f3(t1['r'])}",
             f"MAE {L.f3(ra['mae'])}", f"MAE {L.f3(rb['mae'])}"],
            ["T2$'$ R8 crossovers", t2["n"], f"signs {t2['sign_agree']}/{t2['n']}, $r=$ {L.f3(t2['pearson_r'])}, MAE {L.f3(t2['mae'])}", "", ""],
            ["T3$'$ R9 slopes", t3s["n"], f"signs {t3s['sign_agree']}/{t3s['n']}, MAE {L.f3(t3s['mae'])}", "", ""],
            ["T3$'$ R9 bin gains", t3b["n"], f"signs {t3b['sign_agree']}/{t3b['n']}, MAE {L.f3(t3b['mae'])}", "", ""]]
    L.table(T / "r11.tex", ["Endpoint", "Cells", "Theory", "``reversed = clean''", "Symmetric reversal"], rows,
            "Prospective theory test: fitted to each cell's clean and correlated AUROC only; the reversed AUROC and every "
            "quantity derived from it are held out. None of these cells existed when the rule was registered.", "tab:r11",
            size="\\scriptsize")
    cells, cr, sl = c4("theory/cells.csv"), c4("theory/crossovers.csv"), c4("theory/slopes.csv")
    if cells is None or cr is None:
        return
    plt = L.plot_style()
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.1))
    for kind, col, mk, lab in (("trap", "#2a78d6", "o", "R8 cells"), ("bins", "#eb6834", "s", "R9 cells")):
        q = cells[cells.kind == kind]
        ax[0].plot(q.pred_rev, q.rev, mk, color=col, ms=4, alpha=0.8, label=lab)
    ax[0].plot([0.2, 1], [0.2, 1], color=MUTED, lw=0.6); ax[0].set_xlabel("predicted reversed AUROC"); ax[0].set_ylabel("observed")
    ax[0].legend(fontsize=7, frameon=False)
    for v, _ in VIEWS:
        q = cr[cr.view == v]
        ax[1].plot(q.pred, q.obs, VMARK[v], color=VCOL[v], ms=5, label=VLAB[v])
    lim = [0, max(cr.obs.max(), cr.pred.max()) * 1.08]
    ax[1].plot(lim, lim, color=MUTED, lw=0.6); ax[1].set_xlabel("predicted crossover"); ax[1].set_ylabel("observed")
    ax[1].legend(fontsize=6, frameon=False)
    if sl is not None:
        ax[2].plot(sl.pred_slope, sl.obs_slope, "o", color="#1baf7a", ms=6)
        for r in sl.itertuples():
            ax[2].annotate(LAB[r.cohort].split(" (")[0], (r.pred_slope, r.obs_slope), fontsize=6.5, xytext=(4, -3),
                           textcoords="offset points")
        lo = min(sl.pred_slope.min(), sl.obs_slope.min()) * 1.1
        ax[2].plot([lo, 0], [lo, 0], color=MUTED, lw=0.6); ax[2].set_xlabel("predicted dose-response slope"); ax[2].set_ylabel("observed")
    fig.tight_layout(); fig.savefig(F / "r11_theory.pdf"); plt.close(fig)


def main():
    experiments()
    try:
        views_schematic()
    except Exception as e:  # noqa: BLE001 - the schematic needs opencv; the report builds without it
        print("[stage7] schematic not rebuilt:", e)
    r8(); r9(); r10(); r11()
    M.write(T / "numbers.tex")
    L.report_missing("stage7")


if __name__ == "__main__":
    main()
