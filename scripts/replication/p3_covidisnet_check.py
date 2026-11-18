"""A9 equivalence check: P3's released COVIDIsNet.pt (Bassi et al. 2024, Table 2 ISNet macro-F1 0.773 on the o.o.d.
COVID-19 test) scored on our rebuilt test (primary: 370 / 1,319 / 2,501; size-matched: 370 / 1,295 / 1,515, A9/A10) with
P3's preprocessing (p3_prepare.py cache: histogram-equalized 224-px grayscale, [0, 1], repeated to 3 channels), no marks.
Logits from IsDense with heat=False (its forward returns the classifier logits; the LRP branch is not needed for scoring). Reports macro-F1 per
variant against 0.773 (within 0.03 -> P3 verdicts allowed on that variant, A9) and the confusion matrix against P3's
Table 4 (ISNet: normal 210/157/3, pneumonia 81/1210/4, COVID-19 93/156/1266). Pahde env (their model code).

    python scripts/replication/p3_covidisnet_check.py
Writes results/replication/isnet/covidisnet_check.{md,csv} and per-image logits to $WTSS_DATA/replication/isnet/.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, "/workspace/third_party/isnet/backup")
D = os.environ.get("WTSS_DATA", "/workspace/data")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/replication/isnet"
TARGET, TOL = 0.773, 0.03
TABLE4 = np.array([[210, 157, 3], [81, 1210, 4], [93, 156, 1266]])

c = Path(f"{D}/replication/isnet/cache_test")
gray = np.load(c / "gray.npy", mmap_mode="r"); ids = (c / "ids.txt").read_text().split()
t = pd.read_csv(f"{D}/replication/isnet/test_cohort.csv").drop_duplicates("image_id").set_index("image_id").loc[ids]
DEV = os.environ.get("CHECK_DEVICE", "cpu")   # inference only; their LRP DenseNet keeps all activations (GPU OOM beside ISNet)
torch.set_num_threads(int(os.environ.get("CHECK_THREADS", "32")))
net = torch.load(f"{D}/replication/../isnet_weights/COVIDIsNet.pt", map_location=DEV, weights_only=False).eval()
net.heat = False   # IsDense.forward then returns the logits and clears the LRP activation lists (globals.X / XI) each batch
L = []
with torch.no_grad():
    for s in range(0, len(ids), 32):
        x = torch.from_numpy(np.asarray(gray[s:s + 32])).float().div(255.0)[:, None].repeat(1, 3, 1, 1).to(DEV)
        L.append(net(x).float().cpu())
lg = torch.cat(L).numpy()
pd.DataFrame({"image_id": ids, "y": t.y.to_numpy(), "size_matched": t.size_matched.to_numpy(), "logit0": lg[:, 0],
              "logit1": lg[:, 1], "logit2": lg[:, 2]}).to_csv(f"{D}/replication/isnet/covidisnet_test_logits.csv.gz", index=False)
rows, md = [], ["# A9 equivalence check: P3's released COVIDIsNet.pt on our rebuilt o.o.d. test", "",
                f"Target: P3 Table 2, ISNet macro-F1 {TARGET} (within {TOL} → P3 verdicts allowed on that variant, A9). "
                "Rows = true class (normal, pneumonia, COVID-19), columns = predicted.", ""]
for name, keep in (("primary (370 / 1,319 / 2,501)", np.ones(len(ids), bool)), ("size-matched (370 / 1,295 / 1,515)", t.size_matched.to_numpy().astype(bool))):
    y, p = t.y.to_numpy()[keep], lg[keep].argmax(1)
    f1 = f1_score(y, p, average="macro"); per = f1_score(y, p, average=None)
    cm = confusion_matrix(y, p, labels=[0, 1, 2])
    ok = abs(f1 - TARGET) <= TOL
    rows.append({"variant": name, "n": int(keep.sum()), "macro_f1": f1, "f1_normal": per[0], "f1_pneumonia": per[1], "f1_covid": per[2],
                 "within_0.03_of_0.773": ok, "confusion": cm.tolist()})
    md += [f"## {name}: n = {keep.sum()}, macro-F1 **{f1:.3f}** (per class {per[0]:.3f} / {per[1]:.3f} / {per[2]:.3f}) → "
           f"{'within 0.03: verdicts allowed' if ok else 'not within 0.03: descriptive only'}", "",
           "| true \\ predicted | normal | pneumonia | COVID-19 | P3 Table 4 (ISNet) |", "|---|---|---|---|---|"]
    md += [f"| {n} | {cm[i, 0]} | {cm[i, 1]} | {cm[i, 2]} | {' / '.join(map(str, TABLE4[i]))} |" for i, n in enumerate(("normal", "pneumonia", "COVID-19"))]
    md.append("")
OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT / "covidisnet_check.csv", index=False)
(OUT / "covidisnet_check.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
