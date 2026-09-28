"""Round-5 threshold and decision rules (docs/PREREGISTRATION_ROUND5.md). No data and no GPU."""
import importlib.util
from pathlib import Path

import numpy as np

_p = Path(__file__).resolve().parents[1] / "scripts" / "round5" / "isic2020_robust.py"
_spec = importlib.util.spec_from_file_location("isic2020_robust", _p)
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


def test_t_star_is_the_largest_t_under_one_percent():
    d0 = np.array([0, 1, 2, 3, 8, 9, 10, 12] + [20] * 92)  # n = 100
    # F(0)=0.01, F(1)=0.02, so the largest t with F<=0.01 is 0
    ts, F = R.t_star(d0)
    assert ts == 0 and abs(F["0"] - 0.01) < 1e-12 and F["1"] > 0.01
    d0 = np.array([4] * 1 + [6] * 2 + [20] * 97)
    ts, F = R.t_star(d0)
    assert ts == 5 and F["5"] <= 0.01 and F["6"] > 0.01


def test_c_star_is_the_smallest_c_under_one_percent():
    s0 = np.array([0.99] * 5 + [0.95] * 15 + [0.5] * 980)  # n = 1000
    cs, G = R.c_star(s0)
    assert abs(cs - 0.951) < 1e-12
    assert G["0.950"] > 0.01 and G["0.951"] <= 0.01


def test_tier_flags_match_the_registered_rules():
    d19 = np.array([0, 2, 3, 8, 9])
    s19 = np.array([0.1, 0.1, 0.99, 0.1, 0.1])
    same = np.array([False, False, False, False, True])
    f = R.tier_flags(d19, s19, same, t_star_v=1, c_star_v=0.98)
    assert list(f["drop_exact"]) == [True, False, False, False, True]
    assert list(f["drop_d19_le2"]) == [True, True, False, False, False]
    assert list(f["drop_calibrated"]) == [True, False, True, False, True]
    assert list(f["drop_d19_le8"]) == [True, True, True, True, True]


def test_h12_label_and_h13_gate():
    assert R.h12_label(-0.02, -0.04, -0.001) == "robust"
    assert R.h12_label(-0.02, -0.05, 0.01) == "direction-consistent, inconclusive"
    assert R.h12_label(0.01, -0.01, 0.03) == "not robust"
    assert R.h12_label(0.0, -0.01, 0.01) == "not robust"
    assert R.h13_gate(30, 100) and not R.h13_gate(29, 100) and not R.h13_gate(30, 99)


def test_hair_groups_are_disjoint():
    g = R.hair_groups([100, 200, 201, 600, 600, 600], [0.0, 0.9, 0.2, 0.5, 0.09, 0.2], tau=200)
    assert list(g) == ["H0", "H0", "ambiguous", "Hin", "Hout", "ambiguous"]


def test_nearest_hamming_skips_the_same_patient():
    h = np.array([0, 0, 7], np.uint64)  # first two identical, third differs by 3 bits
    d0 = R.nearest_hamming_other(h, np.array([1, 1, 2]))
    assert d0[0] == 3 and d0[1] == 3 and d0[2] == 3
    d, j = R.nearest_hamming(h[:1], h)
    assert int(d[0]) == 0 and int(j[0]) == 0
