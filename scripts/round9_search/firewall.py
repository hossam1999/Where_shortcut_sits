"""Data firewall of the round-9 search (docs/ROUND9_SEARCH_LEDGER.md). Every data request of the search goes through
`check`; it raises if a forbidden seed, cohort or test set is asked for.

Allowed: the development seeds of round 8 (42, 123, 456, 789, 2026), their trap test environments (reversed, correlated,
clean) for thyroid, capsule and ISIC hair, and the VALIDATION folds of the natural training sets (thyroid, capsule,
ISIC BCN / HAM / MSK hold-out designs).
Forbidden: the round-8 confirmation seeds (8101–8505), the reserved round-9 confirmation seeds (9101–9505), every natural
test set (thyroid official test split, capsule test split, the BCN / HAM / MSK hold-out sources), ISIC 2020, and ovary
(held out).
"""
from __future__ import annotations

ALLOWED_SEEDS = (42, 123, 456, 789, 2026)
ROUND8_CONFIRMATION_SEEDS = (8101, 8202, 8303, 8404, 8505)
ROUND9_RESERVED_SEEDS = (9101, 9202, 9303, 9404, 9505)
FORBIDDEN_SEEDS = set(ROUND8_CONFIRMATION_SEEDS) | set(ROUND9_RESERVED_SEEDS)
TRAP_COHORTS = ("thyroid", "capsule", "isic")
NATURAL_COHORTS = ("thyroid", "capsule", "isic_BCN", "isic_HAM", "isic_MSK")
FORBIDDEN_COHORTS = ("ovary", "isic2020")
TRAP_ENVS = ("train_corr", "val_clean", "val_groups", "train_all", "test_rev", "test_corr", "clean")
NATURAL_ENVS = ("train_corr", "val_clean", "val_groups", "train_all", "val_eval")  # never the natural test set


class FirewallError(RuntimeError):
    pass


SMOKE_SEEDS = (99991,)  # smoke runs only (never a confirmation or reserved seed)


def check(kind: str, cohort: str, seeds=(), envs=(), smoke: bool = False) -> None:
    seeds = tuple(int(s) for s in seeds)
    ok = set(ALLOWED_SEEDS) | (set(SMOKE_SEEDS) if smoke else set())
    bad = [s for s in seeds if s in FORBIDDEN_SEEDS or s not in ok]
    if bad:
        raise FirewallError(f"forbidden seed(s) requested: {bad}")
    if cohort in FORBIDDEN_COHORTS:
        raise FirewallError(f"forbidden cohort requested: {cohort}")
    if kind == "trap":
        if cohort not in TRAP_COHORTS:
            raise FirewallError(f"trap cohort not allowed: {cohort}")
        extra = [e for e in envs if e not in TRAP_ENVS]
    elif kind == "natural":
        if cohort not in NATURAL_COHORTS:
            raise FirewallError(f"natural cohort not allowed: {cohort}")
        extra = [e for e in envs if e not in NATURAL_ENVS]
    else:
        raise FirewallError(f"unknown kind {kind}")
    if extra:
        raise FirewallError(f"forbidden environment(s) for {kind}/{cohort}: {extra}")
