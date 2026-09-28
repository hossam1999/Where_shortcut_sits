"""Task lists for scripts/final/orchestrate.py (docs/PREREGISTRATION_FINAL.md). Order = priority within a lane."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "results"
NEW = OLD / "rerun_2026-09-28"
PY = "python"
GEN = "erm mask balanced mask_balanced dfr mask_dfr mte mte_balanced mte_aug mte_protect mte_protect_balanced jtt mask_jtt umte_jtt".split()
A_MED = "erm mask balanced dfr mte mte_balanced mte_aug mte_protect mte_protect_balanced mte_dfr mask_dfr".split()
A_CNX = "erm mask balanced dfr mte mte_balanced mte_aug mte_protect mte_protect_balanced mask_dfr mte_dfr".split()
A_SENS = "erm mask balanced mte mte_protect mte_balanced".split()
A_SCALE = "erm mask balanced mte mte_balanced mte_protect mte_aug".split()


def T(name, *args, copy=(), **kw):
    return {"name": name, "cmd": [PY, *[str(a) for a in args]] if args else None,
            "copy": [(OLD / s, NEW / s) for s in copy], **kw}


def syn(name, cohort, bb, tag, arms, *extra):
    return T(name, "scripts/run_synthetic.py", "--cohort", cohort, "--backbone", bb, "--tag", tag, "--workers", "6",
             "--arms", *arms, *extra)


SWEEPS = [
    syn("syn_isic_main", "isic2018", "dino518", "main",
        "erm mask inpaint balanced groupdro dfr leace inpaint_consistency inpaint_consistency_lam0".split()),
    syn("syn_isic_proposed", "isic2018", "dino518", "proposed", ["erm"], "--proposed"),
    syn("syn_isic_occlusion", "isic2018", "dino518", "occlusion", "erm mask inpaint".split(), "--phase", "occlusion"),
    syn("syn_isic_dilation", "isic2018", "dino518", "dilation", "erm dilate0 dilate10 dilate25 dilate50".split(),
        "--overlaps", "0.5", "0.75", "1.0"),
    syn("syn_isic_stress", "isic2018", "dino518", "stress", "erm mask balanced leace".split(), "--artifact",
        "ruler_variable", "--overlaps", "0", "0.5", "1", "--proposed"),
    syn("syn_thy_dino", "thyroid", "dino518", "main", "erm mask inpaint balanced dfr leace".split(), "--artifact", "caliper", "--proposed"),
    syn("syn_cap_dino", "capsule", "dino518", "main", "erm mask inpaint balanced dfr leace".split(), "--artifact", "debris", "--proposed"),
    syn("syn_ov_dino", "ovary", "dino518", "main", "erm mask inpaint balanced dfr leace".split(), "--artifact", "caliper", "--proposed"),
    syn("syn_isic_dino224", "isic2018", "dino224", "main", "erm mask balanced dfr leace".split(), "--proposed"),
    syn("syn_isic_dermlip224", "isic2018", "dermlip224", "main", "erm mask balanced dfr leace".split(), "--proposed"),
    syn("syn_thy_med", "thyroid", "medsiglip448", "main", "erm mask inpaint balanced dfr leace".split(), "--artifact", "caliper", "--proposed"),
    syn("syn_cap_med", "capsule", "medsiglip448", "main", "erm mask inpaint balanced dfr leace".split(), "--artifact", "debris", "--proposed"),
    T("leakage", "scripts/run_leakage_experiment.py"),
    # chest tube: needs the NIH download and the CXR preparation
    T("cxr_prepare", "scripts/data/prepare_cxr.py", "--stage", "synthetic",
      wait_for="/root/data/cxr/nih/.download_complete", wait_hours=10),
    syn("syn_cxr_raddino", "nih_ptx", "raddino518", "main", "erm mask inpaint balanced dfr leace".split(), "--artifact", "tube", "--proposed"),
    syn("syn_cxr_dino", "nih_ptx", "dino518", "main", "erm mask inpaint balanced dfr leace".split(), "--artifact", "tube", "--proposed"),
]


def trap(name, cohort, bb, tag, arms=None, *extra, copy=()):
    a = ["scripts/run_thyroid_traps.py", "--cohort", cohort, "--backbone", bb]
    if tag != "main":
        a += ["--tag", tag]
    a += list(extra)
    if arms:
        a += ["--arms", *arms]
    return T(name, *a, copy=copy)


def spec(name, bb, tag, arms=None, *extra, copy=()):
    a = ["scripts/run_spec_e13.py", "--backbone", bb, "--tag", tag, *extra]
    if arms:
        a += ["--arms", *arms]
    return T(name, *a, copy=copy)


COPIES = []
for c in ("thyroid", "ovary", "capsule"):
    for tag, extra in (("repro", ()), ("matched", ("--match",)), ("matched05", ("--match", "--match_caliper", "0.05"))):
        COPIES.append(trap(f"cp_{c}_{tag}", c, "dino518", tag, ["erm", "mask"], *extra,
                           copy=[f"{c}/dino518_{tag}/predictions.csv.gz"]))
for tag, extra in (("repro", ()), ("matched", ("--match",)), ("matched05", ("--match", "--match_caliper", "0.05"))):
    COPIES.append(spec(f"cp_isic_{tag}", "dino518", tag, ["erm", "mask"], *extra, copy=[f"spec_e13/dino518_{tag}/predictions.csv.gz"]))
for b in ("dinos518", "dinol518"):
    COPIES.append(spec(f"cp_isic_scale_{b}", b, "spec_scale", "erm mask balanced dfr mte mte_balanced mte_protect mte_aug".split(),
                       "--generic", copy=[f"spec_e13/{b}_spec_scale/predictions.csv.gz"]))
for c in ("thyroid", "isic_BCN", "isic_HAM", "isic_MSK", "capsule"):
    COPIES.append(T(f"cp_nat_{c}", "scripts/run_natural.py", "--cohort", c, "--tag", "repro", "--save_val",
                    copy=[f"natural/{c}_dino518_repro/predictions.csv.gz"]))
for c in ("isic", "thyroid", "ovary", "capsule"):
    COPIES.append(T(f"cp_tr_{c}", "scripts/run_transplant.py", "--cohort", c, copy=[f"review2/transplant/{c}_dino518/fold{k}" for k in range(5)]))
    COPIES.append(T(f"cp_neu_{c}", "scripts/run_transplant.py", "--cohort", c, "--neutral", "--arms", "erm", "mask",
                    copy=[f"review2/transplant/{c}_dino518_neutral/fold{k}" for k in range(5)]))
COPIES += [
    T("cp_ft_isic", "scripts/run_finetune_spec.py", "--cohort", "isic", "--arms", "erm", "mask", "balanced", "mask_balanced",
      copy=["finetune/isic/resnet50/predictions.csv.gz"]),
    T("cp_ft_ovary_power", "scripts/run_finetune_spec.py", "--cohort", "ovary", "--tag", "power", "--env_seed", "456", "--arms", "erm", "mask",
      copy=["finetune/ovary/resnet50_power/predictions.csv.gz"]),
    T("cp_ft_ovary_g4", "scripts/run_finetune_spec.py", "--cohort", "ovary", "--tag", "power_g4", "--env_seed", "456", "--arms", "mask",
      "umte_cons_ft", "--traps", "trapA", copy=["finetune/ovary/resnet50_power_g4/predictions.csv.gz"]),
]

TRAPS = COPIES + [
    trap("u_thy_univ", "thyroid", "dino518", "universal", GEN, "--generic", "--save_val"),
    trap("u_cap_univ", "capsule", "dino518", "universal", GEN, "--generic", "--save_val"),
    trap("u_ov_univ", "ovary", "dino518", "universal", GEN, "--generic", "--save_val"),
    spec("u_isic_univ", "dino518", "spec_universal", GEN, "--generic", "--save_val"),
    trap("t_thy_main", "thyroid", "dino518", "main"),
    trap("t_cap_main", "capsule", "dino518", "main"),
    trap("t_ov_main", "ovary", "dino518", "main"),
    spec("t_isic_spec", "dino518", "spec"),
    spec("t_isic_spec_dermlip", "dermlip224", "spec"),
    spec("u_isic_univ_dermlip", "dermlip224", "spec_universal", "erm mask balanced dfr mte mte_balanced mte_protect".split(), "--generic"),
    trap("t_thy_med_main", "thyroid", "medsiglip448", "main"),
    trap("t_cap_med_main", "capsule", "medsiglip448", "main"),
    trap("u_thy_med", "thyroid", "medsiglip448", "universal", A_MED, "--generic"),
    trap("u_cap_med", "capsule", "medsiglip448", "universal", A_MED, "--generic"),
    trap("u_ov_med", "ovary", "medsiglip448", "universal", A_MED, "--generic"),
    trap("u_thy_cnx", "thyroid", "convnext384", "universal", A_CNX, "--generic"),
    trap("u_cap_cnx", "capsule", "convnext384", "universal", A_CNX, "--generic"),
    trap("abl_thy", "thyroid", "dino518", "ablation", "erm mask insert_aug mte mte_aug".split(), "--generic"),
    spec("abl_isic", "dino518", "spec_ablation", "erm mask insert_aug mte mte_aug".split(), "--generic"),
    *[trap(f"scale_{c}_{b}", c, b, "scale", A_SCALE, "--generic") for b in ("dinos518", "dinol518") for c in ("thyroid", "ovary", "capsule")],
    *[trap(f"sens_{c}_px50", c, "dino518", "sens_px50", A_SENS, "--generic", "--min_px", "50") for c in ("thyroid", "ovary")],
    *[trap(f"sens_{c}_strictloc", c, "dino518", "sens_strictloc", A_SENS, "--generic", "--rA", "0.7", "--rB", "0.05") for c in ("thyroid", "ovary")],
    trap("sens_capsule_strictB", "capsule", "dino518", "sens_strictB", A_SENS, "--generic", "--max_cover_B", "0.05"),
    T("emb_groups", "scripts/data/embedding_groups.py"),
    *[trap(f"emb_{c}", c, "dino518", "emb_groups", A_SCALE, "--generic", "--groups_csv",
           f"/root/data/{'us/tncd' if c == 'thyroid' else c}/groups_emb.csv") for c in ("thyroid", "ovary", "capsule")],
    trap("jtt_thy", "thyroid", "dino518", "jtt", "erm mask mte jtt mask_jtt umte_jtt".split(), "--generic"),
    trap("jtt_cap", "capsule", "dino518", "jtt", "erm mask mte jtt mask_jtt umte_jtt".split(), "--generic"),
    trap("splice_thy", "thyroid", "dino518", "splice_v2", "erm mask balanced splice mask_splice mte_protect mte_balanced".split(), "--generic"),
    trap("splice_cap", "capsule", "dino518", "splice_v2", "erm mask balanced splice mask_splice mte_protect mte_balanced".split(), "--generic"),
    spec("splice_isic", "dino518", "spec_splice_v2", "erm mask balanced splice mask_splice mte_protect mte_balanced".split(), "--generic"),
    # end-to-end fine-tuning (heavy; last)
    T("ft_thy_r50", "scripts/run_finetune_spec.py", "--cohort", "thyroid", "--arms", "erm", "mask", "balanced", "mask_balanced", "mte_ft",
      "umte_ft", "cons_ft", "umte_cons_ft"),
    T("ft_thy_r50_post", "scripts/run_finetune_spec.py", "--cohort", "thyroid", "--arms", "mte_post", "--traps", "trapA"),
    T("ft_cap_r50", "scripts/run_finetune_spec.py", "--cohort", "capsule", "--arms", "erm", "mask", "balanced", "mask_balanced", "mte_ft", "umte_cons_ft"),
    T("ft_thy_vit", "scripts/run_finetune_spec.py", "--cohort", "thyroid", "--arch", "vit_small_patch16_224.augreg_in21k_ft_in1k",
      "--lr", "3e-5", "--arms", "erm", "mask", "balanced", "mask_balanced", "mte_ft", "umte_ft", "cons_ft", "umte_cons_ft"),
]

LANES = {"sweeps": SWEEPS, "traps": TRAPS}
