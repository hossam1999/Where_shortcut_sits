"""P2 consistency check (author's request, 2026-11-17): is the artifact each image carries the same in Vanilla training, in
their class-0 activation step (CAV) and in the RR-ClArC fine-tune, after the port fix random.seed(int(idx))?
Uses their dataset code only (CPU). Their microscope artifact (utils/artificial_artifact.py:insert_microscope) is a
centred black ring whose radius is np.random.randint(...) after np.random.seed(idx) in add_artifact; idx is the index
inside the dataset object that is rendered. Prints a JSON report.

    PYTHONPATH=<their repo> python scripts/replication/p2_artifact_check.py <vanilla config.yaml> <mitigation config.yaml>
"""
import copy
import json
import random
import sys

import numpy as np
import torch
import yaml

from datasets import load_dataset

van_cfg, mit_cfg = (yaml.safe_load(open(p)) for p in sys.argv[1:3])
van, mit = load_dataset(van_cfg), load_dataset(mit_cfg)
rep = {"same_artifact_flags": bool((van.artifact_labels == mit.artifact_labels).all()),
       "same_splits": all((np.asarray(getattr(van, k)) == np.asarray(getattr(mit, k))).all() for k in ("idxs_train", "idxs_val", "idxs_test")),
       "n_artifact_all": int(van.artifact_labels.sum())}

# 1) the fix reproduces Python <= 3.10 exactly: random.seed(x) for a non-int x used hash(x); hash(np.int64(i)) == i
ok = True
for g in range(len(van)):
    random.seed(int(np.int64(g))); a = random.random()
    random.seed(hash(np.int64(g))); b = random.random()
    ok &= a == b
rep["fix_equals_py310_seed_all_indices"] = bool(ok)


def ring(ds, idx):
    """the artifact mask add_artifact draws for index idx of this dataset object (image content does not matter)"""
    from PIL import Image
    _, m = ds.add_artifact(Image.new("RGB", (ds.image_size, ds.image_size)), idx)
    return m.numpy().astype(bool)


tr_v = van.get_subset_by_idxs(van.idxs_train)        # Vanilla training (DataLoader -> Python int local index)
tr_m = mit.get_subset_by_idxs(mit.idxs_train)        # RR-ClArC fine-tune (same code path)
loc = np.where(tr_v.artifact_labels)[0]
glob = np.asarray(van.idxs_train)[loc]
assert (van.artifact_labels[glob]).all()
same_ft = same_cav = 0
radii_v, radii_c = [], []
for j, g in zip(loc, glob):
    mv = ring(tr_v, int(j))                           # what Vanilla saw
    mf = ring(tr_m, int(j))                           # what the fine-tune sees
    mc = ring(van, np.int64(g))                       # what the activation step (split "all", numpy index) sees
    same_ft += (mv == mf).all(); same_cav += (mv == mc).all()
    radii_v.append(int((~mv).sum())); radii_c.append(int((~mc).sum()))
rep.update(n_artifact_train=len(loc), finetune_identical_to_vanilla=int(same_ft), cav_step_identical_to_vanilla=int(same_cav),
           kept_area_px_vanilla_mean=float(np.mean(radii_v)), kept_area_px_cav_mean=float(np.mean(radii_c)))
# 2) the activation step with Python ints (as Python 3.10 would have seeded) gives the same rings as with numpy ints + fix
rep["cav_numpy_index_equals_int_index"] = bool(all((ring(van, np.int64(g)) == ring(van, int(g))).all() for g in glob[:500]))
print(json.dumps(rep, indent=1))
