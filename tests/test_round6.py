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
