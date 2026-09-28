"""Round-6 decision rules (docs/PREREGISTRATION_ROUND6.md, Amendment 1). No data and no GPU."""
import importlib.util
from pathlib import Path

import numpy as np

_p = Path(__file__).resolve().parents[1] / "scripts" / "round6" / "common.py"
_spec = importlib.util.spec_from_file_location("round6_common", _p)
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


def test_coverage_gates():
    assert C.coverage_gate(200) == "analysed"
    assert C.coverage_gate(199) == "descriptive"
    assert C.coverage_gate(50) == "descriptive"
    assert C.coverage_gate(49) == "not_feasible"


def test_cells_and_contradiction():
    assert C.our_cell(True, False, np.nan) == "artifact_free"
    assert C.our_cell(False, True, 0.8) == "trapA"
    assert C.our_cell(False, True, 0.05) == "trapB"
    assert C.our_cell(False, True, 0.3) == "mid"
    assert C.our_cell(False, False, np.nan) == "gap"
    assert C.contradicts("artifact_free", "present")
    assert C.contradicts("trapA", "trapB")
    assert not C.contradicts("trapA", "mid")
    assert not C.contradicts("trapA", "missing")
    assert not C.contradicts("artifact_free", "artifact_free")


def test_isic_id_strips_suffix():
    assert C.isic_key("ISIC_0001234_downsampled.jpg") == "ISIC_0001234"
    assert C.isic_key("ISIC_7") == "ISIC_0000007"


def test_kappa_and_one_sided_p():
    assert abs(C.kappa_binary([0, 0, 1, 1], [0, 0, 1, 1]) - 1) < 1e-9
    p = C.one_sided_p(np.array([-1.0, -0.2, 0.1]), "less")
    assert abs(p - 1 / 3) < 1e-9
    p2 = C.one_sided_p(np.full(100, -1.0), "less")
    assert abs(p2 - 0.01) < 1e-9


def test_holm_keeps_family_order():
    rows = C.holm_table([{"id": "a", "p": 0.04}, {"id": "b", "p": 0.01}], "p")
    by = {r["id"]: r["p_holm"] for r in rows}
    assert by["b"] <= by["a"]
    assert abs(by["b"] - 0.02) < 1e-9


def test_model_kwargs_only_for_dinov2():
    ft = Path(__file__).resolve().parents[1] / "scripts" / "round6" / "ft_thyroid.py"
    spec = importlib.util.spec_from_file_location("ft_thyroid_r6", ft)
    # importing ft_thyroid pulls torch; the pure helper is re-implemented here to avoid GPU init
    # The registered architectures and the img_size exception are the contract under test.
    text = ft.read_text()
    assert "vit_base_patch14_dinov2.lvd142m" in text
    assert "img_size" in text
    assert "convnext_tiny.fb_in22k_ft_in1k" in text


def test_draw_contour_marks_only_the_boundary():
    rgb = np.zeros((20, 20, 3), np.uint8)
    roi = np.zeros((20, 20), np.uint8)
    roi[5:15, 5:15] = 1
    out = np.asarray(C.draw_contour(rgb, roi))
    green = (out[..., 1] == 180) & (out[..., 0] == 0)
    assert green[5, 5] and green[14, 10] and not green[10, 10] and not green[0, 0]


def test_rater_cell_and_multiclass_kappa():
    ar = importlib.util.spec_from_file_location("ar6", Path(__file__).resolve().parents[1] / "scripts" / "round6" / "analyse_rating.py")
    mod = importlib.util.module_from_spec(ar)
    ar.loader.exec_module(mod)
    assert mod.rater_cell("no", "") == "artifact_free"
    assert mod.rater_cell("yes", "inside") == "trapA"
    assert mod.rater_cell("yes", "outside") == "trapB"
    assert mod.rater_cell("yes", "both") == "mid"
    assert mod.rater_cell("unsure", "inside") == "missing"
    cats = ("inside", "outside", "both", "absent")
    assert abs(mod.kappa_multi(["inside", "outside", "absent"], ["inside", "outside", "absent"], cats) - 1) < 1e-9


def test_presence_only_source_has_no_location_error():
    import pandas as pd
    ba = importlib.util.spec_from_file_location("ba6", Path(__file__).resolve().parents[1] / "scripts" / "round6" / "bias_analysis.py")
    mod = importlib.util.module_from_spec(ba)
    ba.loader.exec_module(mod)
    ours = pd.Series(["trapA", "trapA", "trapB"])
    e, n = mod._opposite_rate(ours, pd.Series(["present", "artifact_free", "present"]), "trapA", "trapB")
    assert np.isnan(e) and n == 0  # DermArtifactDB-like source: e_A undefined, not 0
    e, n = mod._opposite_rate(ours, pd.Series(["trapB", "trapA", "trapB"]), "trapA", "trapB")
    assert e == 0.5 and n == 2


def test_consensus_rule_needs_both_labellers():
    import sys
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "round6"))
    ct = importlib.util.spec_from_file_location("ct6", Path(__file__).resolve().parents[1] / "scripts" / "round6" / "clean_traps.py")
    mod = importlib.util.module_from_spec(ct)
    saved = sys.modules.get("common")
    sys.modules["common"] = C  # round 4's scripts also have a module named common
    try:
        ct.loader.exec_module(mod)
    finally:
        if saved is not None:
            sys.modules["common"] = saved
    per = pd.DataFrame({"cell": ["trapA", "trapA", "trapB", "artifact_free"],
                        "cell_bus": ["artifact_free", "trapB", "trapB", "trapA"],
                        "cell_mg": ["trapB", "trapB", "artifact_free", "trapA"],
                        "contradicted": [True, True, True, True]})
    assert mod.contradicted(per, "consensus").tolist() == [False, True, False, True]
    assert mod.contradicted(per, "registered").tolist() == [True, True, True, True]
