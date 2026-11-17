"""Check that activation checkpointing (p4_train.py --grad_ckpt) is the same computation as the plain model:
one batch of real images, identical initial weights; compares loss, every parameter gradient and the BatchNorm running
statistics after the step (BNGuard). Then measures the peak memory of one batch-16 training step with checkpointing.

    python scripts/replication/p4_ckpt_check.py
"""
import copy
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import p4_train as T  # noqa: E402

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False


def batch(n):
    rows = pd.read_csv(f"{T.B}/isic_annotated_train1.csv").rename(columns={"label": "target"})
    rows["id"] = rows.image.str.split(".").str[0]
    img, msk, ids = T.cache(list(rows.id))
    ds = T.Data(img, msk, rows, {i: j for j, i in enumerate(ids)}, True, aug=False)
    xs, sg, _ = zip(*[ds[k] for k in range(n)])
    return torch.stack(xs).cuda(), torch.stack(sg).cuda()


def step(model, fc, x, sg, guard=None):
    model.train(); fc.train()
    model.zero_grad(set_to_none=True); fc.zero_grad(set_to_none=True)
    _, o = T.image_scores(model, fc, x)
    loss = torch.nn.functional.cross_entropy(o, sg)
    if guard:
        guard.snapshot()
    loss.backward()
    if guard:
        guard.restore()
    return float(loss)


def main():
    torch.manual_seed(0)
    m0, fc0 = T.build("seg")
    m1, fc1 = copy.deepcopy(m0), copy.deepcopy(fc0)
    T.add_checkpointing(m1)
    m0.cuda(); fc0.cuda(); m1.cuda(); fc1.cuda()
    x, sg = batch(4)
    l0 = step(m0, fc0, x, sg)
    l1 = step(m1, fc1, x, sg, T.BNGuard(m1))
    g0 = dict(m0.named_parameters()); g1 = {k.replace(".m.", "."): v for k, v in m1.named_parameters()}
    diffs = [float((g0[k].grad - g1[k].grad).abs().max() / (g0[k].grad.abs().max() + 1e-12)) for k in g0 if g0[k].grad is not None]
    b0 = {k: v for k, v in m0.named_buffers() if "running" in k}
    b1 = {k.replace(".m.", "."): v for k, v in m1.named_buffers() if "running" in k}
    bd = max(float((b0[k] - b1[k]).abs().max()) for k in b0)
    print(f"batch 4: loss plain {l0:.8f} checkpointed {l1:.8f} |diff| {abs(l0 - l1):.2e}")
    print(f"batch 4: gradients of {len(diffs)} parameters, max relative difference {max(diffs):.2e}")
    print(f"batch 4: BatchNorm running stats max |difference| {bd:.2e}")
    del m0, fc0
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    opt = torch.optim.Adam(list(m1.parameters()) + list(fc1.parameters()), lr=5e-4)
    x, sg = batch(16)
    step(m1, fc1, x, sg, T.BNGuard(m1)); opt.step()
    peak = torch.cuda.max_memory_allocated() / 2**30
    free, total = torch.cuda.mem_get_info()
    print(f"batch 16 with checkpointing: peak allocated {peak:.2f} GiB of {total / 2**30:.2f} GiB total")


if __name__ == "__main__":
    main()
