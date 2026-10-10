"""Long-ruler sweep (docs/PREREGISTRATION_LONG_RULER.md): drawing, frozen placements and the image subset."""
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image

from wtss import paths
from wtss.synthetic import LONG_RULER_GEOMETRY, draw_ruler_long, feasible_ids

ROOT = Path(__file__).resolve().parents[1]


def test_draw_ruler_long_mask_is_its_box():
    img = Image.new("RGB", (518, 518), (120, 80, 60))
    w, h = LONG_RULER_GEOMETRY[518]
    out, m = draw_ruler_long(img, 100, 200, w, h)
    assert m.sum() == w * h and m[200:210, 100:300].all()
    a = np.asarray(out)
    assert (a[:200] == np.array([120, 80, 60])).all() and not (a[200:210, 100:300] == np.array([120, 80, 60])).all()


def test_subset_is_exactly_the_feasible_long_ruler_images():
    pl = json.loads((paths.FROZEN / "placements_518_200x10.json").read_text())
    spec = importlib.util.spec_from_file_location("rs", ROOT / "scripts" / "run_synthetic.py")
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)
    cohort, placements, _, geom = rs.cohort_and_placements("isic2018_longruler", 518, "ruler_long")
    ids = set(cohort.df.image_id)
    assert len(ids) == 1314 and ids == feasible_ids(pl) & set(pl)
    assert geom == {518: (200, 10)} and cohort.name == "isic2018_longruler"
    std, _, _, g2 = rs.cohort_and_placements("isic2018_longruler", 518, "ruler_fixed")
    assert set(std.df.image_id) == ids and g2 is None
    assert (cohort.df[cohort.df.split == "test"].y.sum()) == 48
