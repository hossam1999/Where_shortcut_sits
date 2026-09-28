"""Round 8 (docs/ROUND8_DESIGN.md): the shared crossed bootstrap, the pair decomposition and the hard-pair rules."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "round8"))


def _load(name):
    spec = importlib.util.spec_from_file_location(f"r8_{name}", ROOT / "scripts" / "round8" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


C = _load("common")


def _preds(seed=0, n=120, seeds=(1, 2, 3)):
    rng = np.random.default_rng(seed)
    rows = []
    ids = [f"im{i}" for i in range(n)]
    y = rng.integers(0, 2, n)
    a = rng.integers(0, 2, n)
    for s in seeds:
        for arm, shift in (("mask", 0.0), ("new", 0.3)):
            p = y * 0.8 + a * 0.4 + shift * a * (1 - y) + rng.normal(0, 1, n)
            rows.append(pd.DataFrame({"seed": s, "arm": arm, "env": "clean", "image_id": ids, "y": y,
                                      "artifact_present": a, "prob": p}))
    return pd.concat(rows, ignore_index=True)


def test_joint_replicates_point_is_seed_mean_and_matches_crossed_estimator():
    from wtss.stats import binary_auc
    from wtss.stats_crossed import hierarchical_paired_bootstrap
    p = _preds()
    point, reps = C.joint_replicates(C.terms_from_predictions(p, ["arm", "env"]), 400, 7)
    exp = np.mean([binary_auc(q.y.to_numpy(), q.prob.to_numpy()) for _, q in p[p.arm == "new"].groupby("seed")])
    assert abs(point[("new", "clean")] - exp) < 1e-12
    ref = hierarchical_paired_bootstrap(p.rename(columns={"arm": "method"}), "new", "mask", "clean", 400, 7)
    assert abs((point[("new", "clean")] - point[("mask", "clean")]) - ref["seed_delta_mean"]) < 1e-12
    d = reps[("new", "clean")] - reps[("mask", "clean")]
    lo, hi = C.ci(d)
    assert lo < ref["seed_delta_mean"] < hi and len(d) == 400


def test_joint_replicates_shares_image_weights_across_keys():
    p = _preds()
    q = pd.concat([p[p.arm == "mask"], p[p.arm == "mask"].assign(arm="copy")])
    _, reps = C.joint_replicates(C.terms_from_predictions(q, ["arm", "env"]), 200, 3)
    assert np.allclose(reps[("copy", "clean")], reps[("mask", "clean")])  # identical scores -> identical replicates


def test_pair_decomposition_is_exact():
    from wtss.stats import binary_auc
    PD = _load("pair_decomposition")
    q = _preds(seed=5, seeds=(1,))
    q = q[q.arm == "new"]
    d = PD.decompose(q, marker_of_positive=1)
    tot = sum(n for _, n in d.values())
    mix = sum(a * n for a, n in d.values()) / tot
    assert abs(mix - binary_auc(q.y.to_numpy(), q.prob.to_numpy())) < 1e-12


def test_hard_pair_rules():
    SB = _load("scoreboard")
    p = pd.DataFrame({"image_id": list("abcd"), "y": [1, 1, 0, 0], "artifact_present": [1, 0, 1, 0]})
    hair = SB.natural_hard(p, "hair")          # melanoma without / benign with the artifact
    cal = SB.natural_hard(p, "caliper")        # malignant with / benign without the artifact
    assert hair.tolist() == [False, True, True, False]
    assert cal.tolist() == [True, False, False, True]
    # prevalence rule: artifact rarer in the positive class -> caliper-type pairing
    q = pd.DataFrame({"image_id": list("abcdef"), "y": [1, 1, 1, 0, 0, 0], "artifact_present": [1, 0, 0, 1, 1, 0]})
    assert SB.natural_hard(q, "prevalence").tolist() == SB.natural_hard(q, "caliper").tolist()


def test_arm_names_distinguish_template_and_universal_mte():
    SB = _load("scoreboard")
    assert SB.arm_name("dino518_main", "mte_balanced") == "mte_tpl_balanced"
    assert SB.arm_name("dino518_universal", "mte_protect") == "umte_protect"
    assert SB.arm_name("dino518_spec", "mte") == "mte_tpl"
    assert SB.arm_name("thyroid_dino518_repro", "mask_balanced") == "mask_balanced"


def test_holm_matches_round4():
    R4 = C.load_round4_common()
    p = [0.01, 0.04, 0.03, 0.2]
    assert np.allclose(C.holm(p), R4.holm(p))


# ----------------------------------------------------------------------------------------------- Phase 2
CA = _load("candidates")


def _trap_like_validation(n=4000, seed=0):
    """Validation split with the training association of a trap (P(A=1|Y=1) = 0.9, P(A=1|Y=0) = 0.1)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    a = (rng.random(n) < np.where(y == 1, 0.9, 0.1)).astype(int)
    d = y * 1.0 + rng.normal(0, 1, n)               # disease evidence
    return y, a, d


def test_same_artifact_guard_admits_shortcut_free_head_all_pairs_guard_does_not():
    y, a, d = _trap_like_validation()
    val = {"mask": (y, a, d + 2.0 * a),              # masking head: disease + the (in-ROI) shortcut
           "free": (y, a, d)}                        # shortcut-free head: disease only
    # the shortcut-free head is worse on all pairs of this validation split (the association helps the masking head)...
    assert C.all_pairs_auc(y, d) < C.all_pairs_auc(y, d + 2.0 * a) - 0.005
    # ...identical on same-artifact pairs (the artifact term cancels inside such a pair)...
    assert abs(C.same_artifact_auc(y, a, d) - C.same_artifact_auc(y, a, d + 2.0 * a)) < 1e-12
    # ...and better on the worst group
    assert C.worst_group_auc(y, a, d) > C.worst_group_auc(y, a, d + 2.0 * a)
    new, t_new = C.select_setting(val, reference="mask", guard="same_artifact")
    old, t_old = C.select_setting(val, reference="mask", guard="all_pairs")
    assert new == "free" and bool(t_new.set_index("setting").admissible["free"])
    assert old == "mask" and not bool(t_old.set_index("setting").admissible["free"])


def test_same_artifact_guard_rejects_a_head_that_loses_disease_signal():
    y, a, d = _trap_like_validation(seed=1)
    rng = np.random.default_rng(2)
    val = {"mask": (y, a, d + 2.0 * a), "noisy": (y, a, 0.3 * d + rng.normal(0, 1, len(y)))}
    choice, t = C.select_setting(val, reference="mask")
    assert choice == "mask" and not bool(t.set_index("setting").admissible["noisy"])


def test_selection_falls_back_to_masking_and_breaks_ties_towards_masking():
    y, a, d = _trap_like_validation(seed=3)
    same = (y, a, d)
    choice, _ = C.select_setting({"mask": same, "other": same}, reference="mask")
    assert choice == "mask"  # equal objective -> the setting listed first (closest to masking)


def test_mask_bal_weights_path():
    from wtss.heads import group_weights
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    a = (rng.random(500) < np.where(y == 1, 0.9, 0.1)).astype(int)
    w0, w1 = CA.mask_bal_weights(y, a, 0.0), CA.mask_bal_weights(y, a, 1.0)
    assert np.isclose(w0[y == 1].sum(), w0[y == 0].sum())                 # class-balanced
    assert np.allclose(w1 / w1.mean(), group_weights(y, a) / group_weights(y, a).mean())
    for g in range(4):  # lambda = 1: equal total weight per artifact x label group
        m = (2 * a + y) == g
        assert np.isclose(w1[m].sum(), w1.sum() / 4)


def test_cmc_projection_removes_within_class_mean_artifact_effect():
    rng = np.random.default_rng(0)
    n, dim = 800, 16
    y = rng.integers(0, 2, n)
    a = (rng.random(n) < np.where(y == 1, 0.8, 0.2)).astype(int)
    X = rng.normal(0, 1, (n, dim)) + np.outer(y, np.eye(dim)[0]) + np.outer(a, 2 * np.eye(dim)[1] + np.eye(dim)[2])
    for mode in CA.CMC_MODES:
        P = CA.Projector(CA.cmc_directions(X, y, a, mode))
        Z = P(X)
        for c in (0, 1):
            d = Z[(y == c) & (a == 1)].mean(0) - Z[(y == c) & (a == 0)].mean(0)
            if mode == "per_class":
                assert np.abs(d).max() < 1e-8
        assert np.abs(Z @ CA.cmc_directions(X, y, a, mode)).max() < 1e-6


def test_paste_plan_equalises_artifact_rate_and_splits_locations():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.integers(0, 2, n)
    a = (rng.random(n) < np.where(y == 1, 0.9, 0.1)).astype(int)
    tr = pd.DataFrame({"image_id": [f"i{k}" for k in range(n)], "y": y, "a_any": a, "recipient": a == 0,
                       "a_in": a, "a_out": 0})
    plan = CA.paste_plan(tr, np.random.default_rng(1), "locrand")
    assert all(tr.set_index("image_id").loc[list(plan), "recipient"])   # only artifact-free images
    b = CA.plan_balance(tr, plan)
    assert abs(b["p_a_y1_after"] - b["p_a_y0_after"]) < 0.01
    locs = np.array(list(plan.values()))
    assert 0.45 < (locs == "in").mean() < 0.55
    lm = CA.paste_plan(tr, np.random.default_rng(2), "loc_matched")
    got = tr.assign(p_in=tr.image_id.map(lambda i: lm.get(i) == "in"))
    rate = [(got.a_in | got.p_in)[got.y == c].mean() for c in (0, 1)]
    assert abs(rate[0] - rate[1]) < 0.01


def test_cfs_transform_is_identity_without_penalty_and_shrinks_overlay_directions():
    rng = np.random.default_rng(0)
    X0 = rng.normal(0, 1, (300, 8))
    X1 = X0 + np.outer(rng.normal(0, 1, 300), np.eye(8)[0])
    assert np.allclose(CA.cfs_transform(X0, X1, 0.0)(X0), X0)
    Z = CA.cfs_transform(X0, X1, 100.0)(np.eye(8))
    assert abs(Z[0, 0]) < 0.2 and abs(Z[1, 1] - 1) < 1e-6


def test_smoke_envs_never_contain_test_images():
    E = {"val_groups": pd.DataFrame({"image_id": ["v"]}), "test_rev": pd.DataFrame({"image_id": ["t"]}),
         "clean": pd.DataFrame({"image_id": ["t2"]})}
    S = C.smoke_envs_from_validation(E)
    assert S["test_rev"].image_id.tolist() == ["v"] and S["clean"].image_id.tolist() == ["v"]
