"""P6 port check: activation checkpointing of EPVT's ViT-B/16 blocks gives the same loss and gradients as the plain model,
and peak GPU memory at their batch (26 per domain x 5 = 130) for ERM and EPVT. Real training images (bias 1 split 1)."""
import importlib, os, sys, torch
sys.path.insert(0, "/workspace/third_party/epvt")
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False; torch.use_deterministic_algorithms(True, warn_only=True)
from domainbed import algorithms, hparams_registry, datasets
import domainbed.networks as N
def build(alg, ck, hp):
    os.environ["EPVT_GRAD_CKPT"] = "1" if ck else "0"
    torch.manual_seed(0)
    return algorithms.get_algorithm_class(alg)((3, 224, 224), 2, 5, hp).cuda()
D = "/workspace/data/epvt/bias1_1/"
ds = datasets.SKIN(D, [0], hparams_registry.default_hparams("ERM", "SKIN"))
x = torch.stack([ds[1][i][0] for i in range(8)]).cuda(); y = torch.tensor([ds[1][i][1] for i in range(8)]).cuda()
# 1) identical loss and gradients (ERM featurizer + classifier), 8 real images
hp = hparams_registry.default_hparams("ERM", "SKIN"); hp.update({"lr": 5e-6, "lr_classifier": 5e-5, "batch_size": 26, "wd_classifier": 1e-5})
res = []
for ck in (False, True):
    a = build("ERM", ck, hp); a.train()
    loss = torch.nn.functional.cross_entropy(a.predict(x), y); loss.backward()
    res.append((loss.item(), {n: p.grad.detach().clone() for n, p in a.named_parameters() if p.grad is not None}))
    del a; torch.cuda.empty_cache()
(l0, g0), (l1, g1) = res
md = max(((g0[n] - g1[n]).abs().max() / (g0[n].abs().max() + 1e-12)).item() for n in g0)
print(f"loss plain {l0:.8f} ckpt {l1:.8f} |d| {abs(l0-l1):.2e}; gradients: {len(g0)} tensors, same names {set(g0)==set(g1)}, max relative diff {md:.2e}")
# 2) peak memory of one training step at batch 26 x 5 domains, ERM and EPVT (DoPrompt_group_decompose, prompt_dim 10)
for alg, extra in (("ERM", {}), ("DoPrompt_group_decompose", {"prompt_dim": 10})):
    for ck in (True,):
        hp = hparams_registry.default_hparams(alg, "SKIN"); hp.update({"lr": 5e-6, "lr_classifier": 5e-5, "batch_size": 26, "wd_classifier": 1e-5, **extra})
        a = build(alg, ck, hp); torch.cuda.reset_peak_memory_stats()
        mb = [(torch.randn(26, 3, 224, 224).cuda(), torch.randint(0, 2, (26,)).cuda()) for _ in range(5)]
        a.update(mb, None)
        print(f"{alg} ckpt={ck}: peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB of {torch.cuda.get_device_properties(0).total_memory/2**30:.2f}")
        del a; torch.cuda.empty_cache()
