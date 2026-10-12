"""Patient-grouped splits for chest X-rays: no NIH patient ID may appear in two splits or two folds."""
import numpy as np
import pandas as pd

from wtss.data.cxr import patient_split
from wtss.data.isic2019_spec import build_spec_envs


def _toy_cohort(n_pat=400, seed=0):
    rng = np.random.default_rng(seed)
    pats = np.repeat(np.arange(n_pat), rng.integers(1, 6, n_pat))  # 1-5 images per patient
    c = pd.DataFrame({"image_id": [f"img{i}" for i in range(len(pats))], "patient": pats.astype(str)})
    c["y"] = rng.integers(0, 2, len(c))
    c["A0"] = rng.random(len(c)) < 0.5
    c["trapA_A1"] = ~c.A0 & (rng.random(len(c)) < 0.5)
    c["trapB_A1"] = ~c.A0 & ~c.trapA_A1
    c["source"] = "nih"
    return c


def test_patient_split_puts_each_patient_in_one_split():
    c = _toy_cohort()
    c["split"] = patient_split(c)
    assert (c.groupby("patient").split.nunique() == 1).all()
    assert set(c.split) == {"train", "val", "test"}


def test_grouped_folds_never_share_a_patient():
    c = _toy_cohort()
    envs = build_spec_envs(c, seeds=(42, 123), group_col="patient")
    pat = c.set_index("image_id").patient
    for trap in ("trapA", "trapB"):
        for seed in (42, 123):
            test_pats = []
            for k in range(5):
                tr = set(pat.loc[envs[(trap, seed, k, "train_all")].image_id])
                va = set(pat.loc[envs[(trap, seed, k, "val_groups")].image_id])
                te = set(pat.loc[envs[(trap, seed, k, "test_rev")].image_id]) | set(
                    pat.loc[envs[(trap, seed, k, "clean")].image_id])
                assert not (tr & va) and not (tr & te) and not (va & te)
                test_pats.append(te)
            for i in range(5):
                for j in range(i + 1, 5):
                    assert not (test_pats[i] & test_pats[j])
