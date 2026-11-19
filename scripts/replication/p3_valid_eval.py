"""A24 SECONDARY in-distribution evaluation of a P3 ISNet run: P3's cohort-validation images rendered in the biased /
standard / deceiving conditions of the run's design (its mark position; biased = its presence rule with own-class shapes,
standard = no mark, deceiving = P3's swapped shapes on every image), macro-F1 with an image-only bootstrap CI.
SECONDARY: these images were used for epoch/model selection. CPU (inference only; logits with heat=False).

    python scripts/replication/p3_valid_eval.py --name isnet_corner_all_seed0 --position corner --presence all
Writes results/replication/isnet/valid_secondary_<name>.{md,csv}.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "replication"), "/workspace/third_party/isnet/backup"]
D = os.environ.get("WTSS_DATA", "/workspace/data")


def boot_f1(y, p, n=2000, seed=20260918):
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        w = rng.poisson(1.0, len(y)); k = np.repeat(np.arange(len(y)), w)
        v.append(f1_score(y[k], p[k], average="macro"))
    return np.quantile(v, [0.025, 0.975])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--position", choices=["corner", "inside"], required=True)
    ap.add_argument("--presence", choices=["all", "9010"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("EVAL_THREADS", "32")))
    torch.backends.cudnn.deterministic = True
    import p3_train as T
    import ISNetLightning as L
    net = L.IsDense121Lightning(multiLabel=False, multiMask=False, classes=3, heat=False, LR=1e-3, P=0.7, Ea=1, A=1, B=3, d=0.9, momentum=0.99)
    net.load_state_dict(torch.load(f"{D}/replication/isnet/runs/{a.name}/best.pt", map_location="cpu")); net.eval()
    co = pd.read_csv(f"{D}/replication/isnet/cohort.csv").set_index("image_id")
    gray, roi, ids = T.load("cohort")
    pl = T.placements(ids, roi, a.position)
    y = co.loc[ids, "y"].to_numpy(); split = co.loc[ids, "split"].to_numpy()
    vidx = np.where(split == "valid")[0]
    vids = [ids[k] for k in vidx]; yv = y[vidx]
    conds = {"biased": [T.SHAPE[int(c)] if T.present(i, a.seed, a.presence) else None for i, c in zip(vids, yv)],
             "standard": [None] * len(vids), "deceiving": [T.DECEIVE[int(c)] for c in yv]}
    rows = []
    for cond, marks in conds.items():
        ds = T.Data(gray[vidx], roi[vidx], vids, yv, pl, marks, False, a.seed)
        L_ = []
        with torch.no_grad():
            for b in torch.utils.data.DataLoader(ds, batch_size=32, num_workers=8):
                L_.append(net(b[0]).float())
        p = torch.cat(L_).numpy().argmax(1)
        lo, hi = boot_f1(yv, p)
        rows.append({"run": a.name, "condition": cond, "n": len(yv), "macro_f1": f1_score(yv, p, average="macro"), "ci95_lo": lo, "ci95_hi": hi})
        print(rows[-1], flush=True)
    r = pd.DataFrame(rows); out = ROOT / "results/replication/isnet"; out.mkdir(parents=True, exist_ok=True)
    r.to_csv(out / f"valid_secondary_{a.name}.csv", index=False)
    md = [f"# SECONDARY (A24): {a.name} on P3's cohort-validation images, in-distribution", "",
          "**Secondary:** these images were used for epoch/model selection (P3's validation split). Image-only bootstrap 95% CIs "
          "(2,000 replicates); one seed. The rebuilt external test stays the registered primary test.", "",
          f"Design: marks at position **{a.position}**, presence **{a.presence}** (biased = own-class shape per the presence rule; "
          "standard = no mark; deceiving = P3's swapped shapes on every image).", "", "| condition | n | macro-F1 [95% CI] |", "|---|---|---|"]
    md += [f"| {x.condition} | {x.n} | {x.macro_f1:.3f} [{x.ci95_lo:.3f}, {x.ci95_hi:.3f}] |" for _, x in r.iterrows()]
    (out / f"valid_secondary_{a.name}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
