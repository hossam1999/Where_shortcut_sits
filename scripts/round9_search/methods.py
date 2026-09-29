"""Round-9 search candidates: linear heads on frozen features (docs/ROUND9_SEARCH_LEDGER.md).

One solver (`fit_linear`, L-BFGS in float64 on the CPU) minimises
    Σ_i w_i ℓ(y_i, x_i·β + b + o_i + u·f_i) / Σ_i w_i  +  ||β||² / (2 C n)  +  penalty(scores)
with class-balanced weights w (the masking head's weighting), an optional fixed offset o (product of experts, logit
adjustment), an optional unpenalised covariate f (backdoor adjustment) and an optional group penalty on the scores
(V-REx, IRMv1, conditional score-moment matching). The L2 scaling matches liblinear's C Σ ℓ + ½||β||². At prediction
the offset and the covariate term are dropped, which for AUROC is the same as scoring at a fixed value of A.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

from typing import Callable, Optional

import numpy as np


class LinearScore:
    def __init__(self, beta, b, transform: Optional[Callable] = None):
        self.beta, self.b, self.transform = beta, b, transform

    def prob(self, X):
        Z = self.transform(X) if self.transform is not None else np.asarray(X, np.float64)
        z = Z @ self.beta + self.b
        return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def class_weights(y):
    y = np.asarray(y).astype(int)
    n = np.bincount(y, minlength=2).astype(float)
    w = 1.0 / n[y]
    return w / w.mean()


def fit_linear(X, y, C=1.0, w=None, offset=None, free=None, penalty: Optional[Callable] = None, max_iter=200,
               transform: Optional[Callable] = None) -> LinearScore:
    import torch
    torch.set_num_threads(1)
    Z = transform(X) if transform is not None else np.asarray(X, np.float64)
    n, d = Z.shape
    Xt = torch.as_tensor(Z, dtype=torch.float64)
    yt = torch.as_tensor(np.asarray(y, np.float64))
    wt = torch.as_tensor(class_weights(y) if w is None else np.asarray(w, np.float64))
    ot = torch.zeros(n, dtype=torch.float64) if offset is None else torch.as_tensor(np.asarray(offset, np.float64))
    ft = None if free is None else torch.as_tensor(np.asarray(free, np.float64).reshape(n, -1))
    beta = torch.zeros(d, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    u = torch.zeros(0 if ft is None else ft.shape[1], dtype=torch.float64, requires_grad=True)
    params = [beta, b] + ([u] if ft is not None else [])
    opt = torch.optim.LBFGS(params, max_iter=max_iter, line_search_fn="strong_wolfe", tolerance_grad=1e-9,
                            tolerance_change=1e-12, history_size=20)
    lam = 1.0 / (2.0 * C * n)

    def closure():
        opt.zero_grad()
        s = Xt @ beta + b
        z = s + ot + (ft @ u if ft is not None else 0.0)
        loss_i = torch.nn.functional.binary_cross_entropy_with_logits(z, yt, reduction="none")
        loss = (wt * loss_i).sum() / wt.sum() + lam * (beta ** 2).sum()
        if penalty is not None:
            loss = loss + penalty(s, z, loss_i, wt)
        loss.backward()
        return loss

    opt.step(closure)
    return LinearScore(beta.detach().numpy().copy(), float(b.detach()), transform)


def select_C(fit: Callable[[float], LinearScore], Xv, yv, Cs=(0.01, 0.1, 1.0, 10.0)):
    """C by validation AUROC, as wtss.heads.select_C."""
    from sklearn.metrics import roc_auc_score
    best = None
    for Cc in Cs:
        h = fit(Cc)
        s = roc_auc_score(yv, h.prob(Xv)) if len(np.unique(yv)) == 2 else -1
        if best is None or s > best[0]:
            best = (s, Cc, h)
    return best[2], best[1], best[0]


# ------------------------------------------------------------------------------------------------ penalties
def groups(y, a):
    return 2 * np.asarray(a).astype(int) + np.asarray(y).astype(int)


def vrex_penalty(g: np.ndarray, beta_v: float):
    """V-REx (Krueger et al. 2021): variance of the group risks, groups = artifact x label."""
    import torch
    gs = [torch.as_tensor(g == k) for k in range(4) if (g == k).any()]

    def pen(s, z, loss_i, w):
        r = torch.stack([(w[m] * loss_i[m]).sum() / w[m].sum() for m in gs])
        return beta_v * r.var(unbiased=False)
    return pen


def irm_penalty(g: np.ndarray, y: np.ndarray, lam: float):
    """IRMv1 (Arjovsky et al. 2019): squared gradient of each group's risk w.r.t. a dummy scale on the logits."""
    import torch
    gs = [torch.as_tensor(g == k) for k in range(4) if (g == k).any()]
    yt = torch.as_tensor(np.asarray(y, np.float64))

    def pen(s, z, loss_i, w):
        tot = 0.0
        for m in gs:
            scale = torch.ones(1, dtype=torch.float64, requires_grad=True)
            l = torch.nn.functional.binary_cross_entropy_with_logits(z[m] * scale, yt[m])
            gr = torch.autograd.grad(l, [scale], create_graph=True)[0]
            tot = tot + (gr ** 2).sum()
        return lam * tot
    return pen


def moment_penalty(y: np.ndarray, a: np.ndarray, gamma: float):
    """Conditional score-moment matching (CORAL-style, Sun & Saenko 2016, on the scores): within each class, the mean and
    the variance of the score must not differ between images with and without the artifact."""
    import torch
    y, a = np.asarray(y).astype(int), np.asarray(a).astype(int)
    pairs = [(torch.as_tensor((y == c) & (a == 1)), torch.as_tensor((y == c) & (a == 0))) for c in (0, 1)
             if ((y == c) & (a == 1)).sum() > 1 and ((y == c) & (a == 0)).sum() > 1]

    def pen(s, z, loss_i, w):
        tot = 0.0
        for m1, m0 in pairs:
            tot = tot + (s[m1].mean() - s[m0].mean()) ** 2 + (s[m1].var() - s[m0].var()) ** 2
        return gamma * tot
    return pen


# ------------------------------------------------------------------------------------------------ offsets
def group_log_odds(y, a, w=None) -> np.ndarray:
    """Bias-only model on A (product of experts): the weighted log-odds of Y within each artifact group, per image."""
    y, a = np.asarray(y).astype(int), np.asarray(a).astype(int)
    w = class_weights(y) if w is None else w
    out = np.zeros(len(y))
    for v in (0, 1):
        m = a == v
        if m.any():
            p1, p0 = w[m & (y == 1)].sum(), w[m & (y == 0)].sum()
            out[m] = np.log((p1 + 1e-6) / (p0 + 1e-6))
    return out


# ------------------------------------------------------------------------------------------------ learned projections
class Projection:
    """Frozen MLP projection x -> h (float64 numpy in, numpy out)."""

    def __init__(self, net):
        self.net = net

    def __call__(self, X):
        import torch
        with torch.no_grad():
            return self.net(torch.as_tensor(np.asarray(X, np.float32))).double().numpy()


def _mlp(d, hid=256, out=128):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(d, hid), torch.nn.ReLU(), torch.nn.Linear(hid, out))


class _GradReverse:
    @staticmethod
    def apply(x, lam):
        import torch

        class F(torch.autograd.Function):
            @staticmethod
            def forward(ctx, t):
                return t.view_as(t)

            @staticmethod
            def backward(ctx, g):
                return -lam * g
        return F.apply(x)


def train_conditional_adversary(X, y, a, lam: float, seed: int, steps: int = 400, lr: float = 1e-3) -> Projection:
    """g. Conditional adversarial debiasing: projection + classifier trained on the label; an adversary predicts the
    artifact label from the projection GIVEN the diagnosis (input [h, one-hot y]), through a gradient-reversal layer of
    strength lam (Ganin et al. 2016; Zhang et al. 2018, which conditions the adversary on the label for equality of
    odds; Kim et al. 2019). Conditioning on Y keeps disease information that is correlated with the artifact."""
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    Xt = torch.as_tensor(np.asarray(X, np.float32))
    yt = torch.as_tensor(np.asarray(y, np.float32))
    at = torch.as_tensor(np.asarray(a, np.float32))
    w = torch.as_tensor(class_weights(y).astype(np.float32))
    wa = torch.as_tensor(class_weights(a).astype(np.float32))
    net = _mlp(X.shape[1])
    clf = torch.nn.Linear(128, 1)
    adv = torch.nn.Sequential(torch.nn.Linear(130, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(list(net.parameters()) + list(clf.parameters()) + list(adv.parameters()), lr=lr,
                           weight_decay=1e-4)
    Y1 = torch.nn.functional.one_hot(yt.long(), 2).float()
    n = len(Xt)
    bs = min(512, n)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,), generator=g)
        h = net(Xt[idx])
        lc = (w[idx] * torch.nn.functional.binary_cross_entropy_with_logits(clf(h).squeeze(1), yt[idx], reduction="none")).mean()
        hr = _GradReverse.apply(h, lam)
        la = (wa[idx] * torch.nn.functional.binary_cross_entropy_with_logits(adv(torch.cat([hr, Y1[idx]], 1)).squeeze(1),
                                                                             at[idx], reduction="none")).mean()
        opt.zero_grad()
        (lc + la).backward()
        opt.step()
    net.eval()
    return Projection(net)


def _supcon(z, pos_mask, tau):
    """Contrastive loss: for each anchor, -log sum_pos exp(s/tau) / sum_{not self} exp(s/tau)."""
    import torch
    z = torch.nn.functional.normalize(z, dim=1)
    s = z @ z.T / tau
    eye = torch.eye(len(z), dtype=torch.bool)
    s = s.masked_fill(eye, -1e9)
    logden = torch.logsumexp(s, 1)
    pos = pos_mask & ~eye
    has = pos.any(1)
    lognum = torch.logsumexp(s.masked_fill(~pos, -1e9), 1)
    return (logden - lognum)[has].mean() if has.any() else s.sum() * 0


def train_cnc(X, y, a, weight: float, seed: int, steps: int = 400, lr: float = 1e-3, tau: float = 0.1) -> Projection:
    """h1. Correct-n-Contrast (Zhang et al. 2022) with the artifact x label groups given (no ERM-inferred groups):
    positives = same label, different artifact status; negatives in the denominator include same artifact, other label;
    plus the classification loss on the projection."""
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    Xt = torch.as_tensor(np.asarray(X, np.float32))
    yt = torch.as_tensor(np.asarray(y, np.int64))
    at = torch.as_tensor(np.asarray(a, np.int64))
    w = torch.as_tensor(class_weights(y).astype(np.float32))
    net, clf = _mlp(X.shape[1]), torch.nn.Linear(128, 1)
    opt = torch.optim.Adam(list(net.parameters()) + list(clf.parameters()), lr=lr, weight_decay=1e-4)
    n = len(Xt)
    bs = min(512, n)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,), generator=g)
        h = net(Xt[idx])
        yy, aa = yt[idx], at[idx]
        pos = (yy[:, None] == yy[None, :]) & (aa[:, None] != aa[None, :])
        lc = (w[idx] * torch.nn.functional.binary_cross_entropy_with_logits(clf(h).squeeze(1), yy.float(),
                                                                             reduction="none")).mean()
        loss = lc + weight * _supcon(h, pos, tau)
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    return Projection(net)


def train_counterfactual_contrastive(X, y, Xcf, cf_index, weight: float, seed: int, steps: int = 400, lr: float = 1e-3,
                                     tau: float = 0.1) -> Projection:
    """h2. Counterfactual contrastive learning (Roschewitz et al. 2024) on frozen features: each training image and its
    counterfactual renders (artifact inserted inside the ROI and masked: generic overlays, and real pasted instances for
    artifact-free images) are positives; other images are negatives; plus the classification loss.
    Xcf: counterfactual feature rows; cf_index: for each row of Xcf the index of its source image in X."""
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    Xt = torch.as_tensor(np.asarray(X, np.float32))
    Ct = torch.as_tensor(np.asarray(Xcf, np.float32))
    src = np.asarray(cf_index)
    by_src = {}
    for j, i in enumerate(src):
        by_src.setdefault(int(i), []).append(j)
    has_cf = np.array(sorted(by_src))
    yt = torch.as_tensor(np.asarray(y, np.float32))
    w = torch.as_tensor(class_weights(y).astype(np.float32))
    net, clf = _mlp(X.shape[1]), torch.nn.Linear(128, 1)
    opt = torch.optim.Adam(list(net.parameters()) + list(clf.parameters()), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    bs = min(256, len(has_cf))
    for _ in range(steps):
        pick = rng.choice(has_cf, bs, replace=False)
        cf = np.array([rng.choice(by_src[int(i)]) for i in pick])
        h = net(torch.cat([Xt[pick], Ct[cf]]))
        ids = torch.as_tensor(np.r_[np.arange(bs), np.arange(bs)])
        pos = ids[:, None] == ids[None, :]
        lc = (w[pick] * torch.nn.functional.binary_cross_entropy_with_logits(clf(h[:bs]).squeeze(1), yt[pick],
                                                                              reduction="none")).mean()
        loss = lc + weight * _supcon(h, pos, tau)
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    return Projection(net)
