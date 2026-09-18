"""Uncertainty quantification for session-level results.

Everything is seeded (``np.random.default_rng``) so intervals are reproducible.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

Stat = Callable[[np.ndarray], float]


def bootstrap_ci(x: Sequence[float], stat: Stat = np.mean, n_boot: int = 4000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of ``stat`` over i.i.d. sessions."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    if stat is np.mean:
        boots = x[idx].mean(axis=1)
    else:
        boots = np.apply_along_axis(stat, 1, x[idx])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def summarize(x: Sequence[float], n_boot: int = 4000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Mean/median/std (ddof=1), bootstrap CI of the mean, and n."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    lo, hi = bootstrap_ci(x, np.mean, n_boot, alpha, seed)
    return {
        "n": int(len(x)),
        "mean": float(x.mean()) if len(x) else float("nan"),
        "median": float(np.median(x)) if len(x) else float("nan"),
        "std": float(x.std(ddof=1)) if len(x) > 1 else float("nan"),
        "ci_lo": lo, "ci_hi": hi,
    }


def paired_diff_ci(a: Sequence[float], b: Sequence[float], n_boot: int = 4000,
                   alpha: float = 0.05, seed: int = 0) -> dict:
    """Bootstrap CI for ``mean(a - b)`` over *paired* sessions (same seeds, common random
    numbers). Pairing removes between-seed variance from the comparison. ``significant`` is
    true iff the interval excludes 0; it is not a licence to claim superiority when the effect
    is economically trivial - always read it with the point estimate."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired samples must have equal length")
    d = a - b
    d = d[~np.isnan(d)]
    lo, hi = bootstrap_ci(d, np.mean, n_boot, alpha, seed)
    return {"n": int(len(d)), "mean_diff": float(np.mean(d)) if len(d) else float("nan"), "ci_lo": lo, "ci_hi": hi,
            "significant": bool(lo > 0 or hi < 0), "p_value": bootstrap_p(d, n_boot, seed),
            "effect_size_dz": float(np.mean(d) / np.std(d, ddof=1)) if len(d) > 1 and np.std(d, ddof=1) > 0 else float("nan")}


def bootstrap_p(d: Sequence[float], n_boot: int = 4000, seed: int = 0) -> float:
    """Two-sided bootstrap p-value for ``mean(d) = 0`` (the smaller tail of the bootstrap distribution of the mean,
    doubled; floored at 1/n_boot). A *resampling* p-value: it assumes sessions are i.i.d. and is only as good as the sample."""
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    p = 2 * min(np.mean(boots <= 0), np.mean(boots >= 0))
    return float(min(1.0, max(p, 1.0 / n_boot)))


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate over a family of comparisons)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()
