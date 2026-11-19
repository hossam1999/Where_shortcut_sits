"""P6 audit (2026-11-19): activation checkpointing vs plain for one full EPVT training update (DoPrompt_group_decompose: the
domain-mixup pass with the prompt pre-hook, backward; the eval/train toggles; the prompt-adapter pass, backward), on CPU.
Their code hard-codes device="cuda" in torch.full / torch.zeros; in this test process only, those calls are redirected to the
CPU. Same seeds before each run (np.random.beta for the mixup lambda, torch.randperm for the domain pairs, initial weights);
the three optimizer steps are replaced by no-ops so the accumulated gradients of both backward passes can be compared.
Real bias-1 training images, 2 per domain x 5 domains. Run: /workspace/envs/epvt/bin/python <this>"""
import os, sys, random
import numpy as np, torch
sys.path.insert(0, "/workspace/third_party/epvt")
torch.set_num_threads(8)
_full, _zeros = torch.full, torch.zeros
torch.full = lambda *a, **k: _full(*a, **{**k, "device": "cpu"} if "device" in k else k)
torch.zeros = lambda *a, **k: _zeros(*a, **{**k, "device": "cpu"} if "device" in k else k)
from domainbed import algorithms, hparams_registry, datasets

D = "/workspace/data/epvt/bias1_1/"
hp = hparams_registry.default_hparams("DoPrompt_group_decompose", "SKIN")
hp.update({"lr": 5e-6, "lr_classifier": 5e-5, "batch_size": 26, "wd_classifier": 1e-5, "prompt_dim": 10})
ds = datasets.SKIN(D, [0], hp)
torch.manual_seed(1)
mb = [(torch.stack([ds[e][i][0] for i in (0, 1)]), torch.tensor([ds[e][i][1] for i in (0, 1)])) for e in range(5)]


def run(ck):
    os.environ["EPVT_GRAD_CKPT"] = "1" if ck else "0"
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    a = algorithms.get_algorithm_class("DoPrompt_group_decompose")((3, 224, 224), 2, 5, hp)
    a.train()
    for o in (a.prompt_opt, a.optimizer, a.project_opt):
        o.step = lambda *x, **k: None
    np.random.seed(0); torch.manual_seed(0)
    out = a.update(mb)
    g = {n: p.grad.detach().clone() for n, p in a.named_parameters() if p.grad is not None}
    return out, g


(o0, g0), (o1, g1) = run(False), run(True)
md = max(((g0[n] - g1[n]).abs().max() / (g0[n].abs().max() + 1e-12)).item() for n in g0)
print("losses plain", o0, "\nlosses ckpt ", o1)
print(f"gradients: {len(g0)} tensors, same names {set(g0) == set(g1)}, max relative diff {md:.2e}")
