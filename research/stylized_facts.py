"""Estimators for the stylized facts of limit-order markets (spec section 14).

Each estimator is validated in ``tests/unit/test_stylized_facts.py`` against processes with a known answer. None of
them embeds an expected empirical value: comparing a simulator (or real data) with the literature is done in the
experiment/report, where the reference ranges are written down explicitly and labelled as literature-typical.

* returns: moments (skew, excess kurtosis), tail index (Hill estimator) of ``|r|``;
* volatility clustering: autocorrelation of ``|r|`` and ``r^2`` (positive and slowly decaying = clustering);
* return autocorrelation (bid-ask bounce / mean reversion appears as a negative lag-1 value);
* order-sign long memory: ACF of aggressor signs and the exponent ``gamma`` in ``C(tau) ~ tau^-gamma``;
* arrival clustering: Fano factor (variance/mean of counts per window; 1 for Poisson) and inter-arrival coefficient of variation;
* spread and order-size distributions.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from research.common import mid_at
from simulator.simulator import NS, SimResult


# ------------------------------------------------------------------------------------------ sampling
def mid_on_grid(res: SimResult, dt_s: float, start_s: float = 0.0) -> np.ndarray:
    """True mid (ticks) sampled every ``dt_s`` seconds from ``start_s`` to the end of the session."""
    t = np.arange(int(start_s * NS), int(res.config.horizon_s * NS) + 1, int(dt_s * NS))
    return mid_at(res, t)


def log_returns(price: np.ndarray) -> np.ndarray:
    p = np.asarray(price, dtype=float)
    return np.diff(np.log(p))


# ------------------------------------------------------------------------------------------ distributions
def moments(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    m, s = x.mean(), x.std()
    if s == 0:
        return {"n": len(x), "mean": float(m), "std": 0.0, "skew": float("nan"), "excess_kurtosis": float("nan")}
    z = (x - m) / s
    return {"n": int(len(x)), "mean": float(m), "std": float(s), "skew": float((z ** 3).mean()),
            "excess_kurtosis": float((z ** 4).mean() - 3.0)}


def hill_tail_index(x: np.ndarray, k: int | None = None, frac: float = 0.05) -> dict:
    """Hill estimator of the tail exponent ``alpha`` (``P(|X| > x) ~ x^-alpha``) from the top ``k`` order statistics of
    ``|x|`` (default: the top ``frac`` of the non-zero sample). Reports ``se = alpha / sqrt(k)`` (asymptotic).
    Sensitive to ``k``; report it for several ``frac`` before believing a single value."""
    a = np.abs(np.asarray(x, dtype=float))
    a = np.sort(a[a > 0])
    n = len(a)
    if n < 50:
        return {"alpha": float("nan"), "se": float("nan"), "k": 0}
    k = max(10, int(frac * n)) if k is None else k
    k = min(k, n - 1)
    tail, thresh = a[-k:], a[-k - 1]
    alpha = 1.0 / np.mean(np.log(tail / thresh))
    return {"alpha": float(alpha), "se": float(alpha / math.sqrt(k)), "k": int(k)}


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Sample autocorrelation at lags 1..nlags."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)] - np.nanmean(x)
    d = float(np.dot(x, x))
    if d == 0:
        return np.full(nlags, np.nan)
    return np.array([np.dot(x[:-k], x[k:]) / d for k in range(1, nlags + 1)])


def power_law_exponent(acf_vals: np.ndarray, lags: Sequence[int] | None = None, lo: int = 1, hi: int | None = None) -> dict:
    """Fit ``C(tau) ~ tau^-gamma`` by OLS of ``ln C`` on ``ln tau`` over lags ``lo..hi`` where ``C > 0``.
    Returns ``gamma``, ``r2`` and the number of usable lags (non-positive ACF values cannot be logged and are dropped)."""
    c = np.asarray(acf_vals, dtype=float)
    lag = np.arange(1, len(c) + 1) if lags is None else np.asarray(lags)
    m = (lag >= lo) & (lag <= (hi if hi is not None else lag.max())) & (c > 0)
    if m.sum() < 4:
        return {"gamma": float("nan"), "r2": float("nan"), "n_lags": int(m.sum())}
    x, y = np.log(lag[m]), np.log(c[m])
    A = np.column_stack([np.ones_like(x), x])
    beta, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    ss_res = float(np.sum((y - A @ beta) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"gamma": float(-beta[1]), "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"), "n_lags": int(m.sum())}


# ------------------------------------------------------------------------------------------ arrivals
def fano_factor(times_ns: np.ndarray, window_s: float, start_ns: int, end_ns: int) -> float:
    """Variance-to-mean ratio of event counts in consecutive windows (1 for a Poisson process, >1 = clustering)."""
    edges = np.arange(start_ns, end_ns + 1, int(window_s * NS))
    counts, _ = np.histogram(times_ns, bins=edges)
    return float(counts.var(ddof=1) / counts.mean()) if len(counts) > 2 and counts.mean() > 0 else float("nan")


def interarrival_stats(times_ns: np.ndarray) -> dict:
    """Coefficient of variation (1 for Poisson) and lag-1 autocorrelation of inter-arrival times."""
    gaps = np.diff(np.sort(np.asarray(times_ns, dtype=float)))
    gaps = gaps[gaps > 0]
    if len(gaps) < 30:
        return {"cv": float("nan"), "acf1": float("nan"), "n": len(gaps)}
    return {"cv": float(gaps.std() / gaps.mean()), "acf1": float(acf(gaps, 1)[0]), "n": int(len(gaps))}


def spread_stats(spread: np.ndarray) -> dict:
    s = np.asarray(spread, dtype=float)
    s = s[~np.isnan(s)]
    return {"mean": float(s.mean()), "median": float(np.median(s)), "std": float(s.std()), "q90": float(np.quantile(s, 0.9)),
            "q99": float(np.quantile(s, 0.99)), "frac_at_min": float(np.mean(s == s.min())),
            "acf_lag1": float(acf(s, 1)[0]) if s.std() > 0 else float("nan")}


# ------------------------------------------------------------------------------------------ one-call summary
def summarize_session(res: SimResult, horizons_s: Sequence[float] = (0.1, 1.0), sign_lags: int = 100, acf_lags: int = 50) -> dict:
    """All stylized-fact statistics of one simulated session (post warm-up)."""
    w = res.config.warmup_s
    out: dict = {}
    for h in horizons_s:
        r = log_returns(mid_on_grid(res, h, w))
        mo = moments(r)
        out[f"ret_{h:g}s"] = {**mo, "hill_5pct": hill_tail_index(r, frac=0.05)["alpha"], "hill_1pct": hill_tail_index(r, frac=0.01)["alpha"],
                              "zero_share": float(np.mean(r == 0)), "acf_ret": acf(r, min(acf_lags, len(r) // 4)).tolist(),
                              "acf_abs": acf(np.abs(r), min(acf_lags, len(r) // 4)).tolist()}
    tp = res.trades
    m = tp["t_ns"] >= int(w * NS)
    t, sgn = tp["t_ns"][m], tp["aggressor"][m].astype(float)
    # aggregate fills of one aggressive order into one sign observation
    first = np.r_[True, tp["taker_id"][m][1:] != tp["taker_id"][m][:-1]]
    sg = sgn[first]
    sacf = acf(sg, sign_lags)
    out["sign_acf"] = sacf.tolist()
    out["sign_power_law"] = power_law_exponent(sacf, lo=1, hi=sign_lags)
    out["sign_acf_lag1"] = float(sacf[0])
    arr = t[first]
    out["fano_1s"] = fano_factor(arr, 1.0, int(w * NS), int(res.config.horizon_s * NS))
    out["fano_0.1s"] = fano_factor(arr, 0.1, int(w * NS), int(res.config.horizon_s * NS))
    out["interarrival"] = interarrival_stats(arr)
    s = res.steady()[0]["spread"]
    out["spread"] = spread_stats(s)
    qty = tp["qty"][m][first]
    out["trade_qty"] = {**moments(qty), "hill_5pct": hill_tail_index(qty, frac=0.05)["alpha"]}
    return out


def summarize_window(res: SimResult, t0_ns: int, t1_ns: int, horizons_s: Sequence[float] = (0.1, 1.0), sign_lags: int = 50,
                     acf_lags: int = 30, fano_windows_s: Sequence[float] = (1.0, 10.0), merge_bursts: bool = False) -> dict:
    """Stylized-fact statistics of the time window ``[t0_ns, t1_ns)`` of one run - the block estimator used on real data
    (one contiguous recording = one run; blocks play the role of sessions). Same definitions as ``summarize_session``.

    ``merge_bursts``: treat prints sharing a timestamp and aggressor side as ONE parent order (needed for feeds that publish one trade per
    fill: a single order sweeping several levels would otherwise inflate order-sign autocorrelation, arrival clustering and size statistics)."""
    out: dict = {}
    for h in horizons_s:
        t = np.arange(t0_ns, t1_ns + 1, int(h * NS))
        r = log_returns(mid_at(res, t))
        mo = moments(r)
        lab = f"{h:g}s"
        out[f"kurt_{lab}"] = mo["excess_kurtosis"]
        out[f"zero_share_{lab}"] = float(np.mean(r == 0))
        out[f"hill5_{lab}"] = hill_tail_index(r, frac=0.05)["alpha"]
        out[f"hill1_{lab}"] = hill_tail_index(r, frac=0.01)["alpha"]
        nl = min(acf_lags, max(len(r) // 4, 2))
        a_abs, a_ret = acf(np.abs(r), nl), acf(r, min(3, nl))
        out[f"acf_ret1_{lab}"] = float(a_ret[0])
        out[f"acf_abs1_{lab}"] = float(a_abs[0])
        out[f"acf_abs10_{lab}"] = float(a_abs[9]) if nl >= 10 else float("nan")
        out[f"_acf_abs_{lab}"] = a_abs.tolist()
        out[f"n_returns_{lab}"] = int(len(r))
    tp = res.trades
    m = (tp["t_ns"] >= t0_ns) & (tp["t_ns"] < t1_ns)
    if m.sum() < 30:
        return out
    ids = tp["taker_id"][m]
    if merge_bursts:
        tt, ss = tp["t_ns"][m], tp["aggressor"][m]
        first = np.r_[True, (tt[1:] != tt[:-1]) | (ss[1:] != ss[:-1])]
        group = np.cumsum(first) - 1
        parent_qty = np.bincount(group, weights=tp["qty"][m].astype(float))
    else:
        first = np.r_[True, ids[1:] != ids[:-1]]
        parent_qty = tp["qty"][m][first].astype(float)
    sg = tp["aggressor"][m][first].astype(float)
    arr = tp["t_ns"][m][first]
    sacf = acf(sg, min(sign_lags, max(len(sg) // 4, 2)))
    fit = power_law_exponent(sacf, lo=1, hi=len(sacf))
    out.update(sign_acf1=float(sacf[0]), sign_gamma=fit["gamma"], sign_r2=fit["r2"], _sign_acf=sacf.tolist(), n_aggressive=int(len(arr)),
               trade_rate_per_s=float(len(arr) / ((t1_ns - t0_ns) / NS)))
    for w in fano_windows_s:
        out[f"fano_{w:g}s"] = fano_factor(arr, w, t0_ns, t1_ns)
    gaps = np.diff(arr.astype(float)); gaps = gaps[gaps > 0]
    out["arrival_cv"] = float(gaps.std() / gaps.mean()) if len(gaps) > 5 else float("nan")
    ms = (res.samples["t_ns"] >= t0_ns) & (res.samples["t_ns"] < t1_ns)
    sp = res.samples["spread"][ms]
    if np.isfinite(sp).any():
        st = spread_stats(sp)
        out.update(spread_mean=st["mean"], spread_median=st["median"], spread_q99=st["q99"])
    out["size_hill5"] = hill_tail_index(parent_qty, frac=0.05)["alpha"]
    return out
