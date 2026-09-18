"""Does order-book imbalance predict short-horizon mid-price changes?

``OBI_t = (V_b - V_a) / (V_b + V_a)`` over the best ``L`` levels (L = 1: the touch; L = 5: recorded depth).
For each horizon ``h`` we relate ``OBI_t`` to the forward mid change ``dm_t(h) = m_{t+h} - m_t`` (ticks), on the
simulator's fixed sample grid. Per *session* we report

* Pearson correlation and Spearman rank correlation,
* OLS slope ``beta`` (expected ticks of mid change per unit OBI),
* directional accuracy ``P(sign dm = sign OBI | dm != 0, OBI != 0) - 1/2``,
* the **backward** correlation of ``OBI_t`` with ``m_t - m_{t-h}``: imbalance is also *caused* by recent price
  moves (reverse causality), so a strong backward correlation warns that a forward correlation may partly
  reflect persistence rather than information,
* a **shuffle null** (OBI permuted within the session), which must give ~0.

Observations within a session overlap (a 1 s window contains 100 samples of 10 ms), so sample-level standard
errors are wrong; uncertainty is quantified by bootstrapping *sessions* (see the experiment). Statistical
significance is not tradability: ``beta`` is compared with the half-spread the maker/taker would pay.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats as sps

from research.common import ffill, obi_from_samples

# OBI bins (fixed edges so sessions are poolable): strongly ask-heavy ... strongly bid-heavy
OBI_EDGES = (-1.0001, -0.8, -0.4, -0.1, 0.1, 0.4, 0.8, 1.0001)


def forward_changes(mid: np.ndarray, k: int) -> np.ndarray:
    """``m[i+k] - m[i]`` aligned to ``i`` (NaN for the last k points)."""
    out = np.full(len(mid), np.nan)
    out[:-k] = mid[k:] - mid[:-k]
    return out


def backward_changes(mid: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(mid), np.nan)
    out[k:] = mid[k:] - mid[:-k]
    return out


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 30 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def session_stats(samples: dict[str, np.ndarray], dt_s: float, horizons_s: Sequence[float], levels: int = 1,
                  start_s: float = 5.0, rng: np.random.Generator | None = None, only_spread: int | None = None) -> dict:
    """Per-horizon predictiveness of OBI in one session. Returns ``{horizon: {stat: value}}`` plus bin means.

    ``only_spread`` restricts to samples with that exact spread (ticks), e.g. 1 for the tight-spread regime.
    """
    t = samples["t_ns"] / 1e9
    keep = t >= start_s
    mid = ffill(samples["mid"])
    obi = obi_from_samples(samples, levels)
    valid_now = keep & ~np.isnan(obi) & ~np.isnan(samples["mid"])
    if only_spread is not None:
        valid_now &= samples["spread"] == only_spread
    out: dict = {}
    rng = rng or np.random.default_rng(0)
    for h in horizons_s:
        k = max(1, int(round(h / dt_s)))
        fwd, bwd = forward_changes(mid, k), backward_changes(mid, k)
        m = valid_now & ~np.isnan(fwd)
        x, y = obi[m], fwd[m]
        r: dict = {"n": int(m.sum())}
        if len(x) < 2:  # degenerate selection (e.g. empty spread filter): report NaNs, not warnings
            out[h] = {**{k: float("nan") for k in ("corr", "spearman", "slope", "dir_acc_excess", "frac_moved",
                                                    "corr_backward", "corr_shuffled")},
                      "n": len(x), "bin_mean": [float("nan")] * (len(OBI_EDGES) - 1), "bin_n": [0] * (len(OBI_EDGES) - 1)}
            continue
        r["corr"] = _corr(x, y)
        r["spearman"] = float(sps.spearmanr(x, y)[0]) if len(x) > 30 and np.std(y) > 0 else float("nan")
        var = np.var(x)
        r["slope"] = float(np.cov(x, y, bias=True)[0, 1] / var) if var > 0 and len(x) > 30 else float("nan")
        nz = (y != 0) & (x != 0)
        r["dir_acc_excess"] = float(np.mean(np.sign(y[nz]) == np.sign(x[nz])) - 0.5) if nz.sum() > 30 else float("nan")
        r["frac_moved"] = float(np.mean(y != 0)) if len(y) else float("nan")
        mb = valid_now & ~np.isnan(bwd)
        r["corr_backward"] = _corr(obi[mb], bwd[mb])
        r["corr_shuffled"] = _corr(rng.permutation(x), y)
        # conditional mean of the forward change by OBI bin (ticks)
        b = np.digitize(x, OBI_EDGES) - 1
        r["bin_mean"] = [float(y[b == i].mean()) if np.any(b == i) else float("nan") for i in range(len(OBI_EDGES) - 1)]
        r["bin_n"] = [int(np.sum(b == i)) for i in range(len(OBI_EDGES) - 1)]
        out[h] = r
    return out
