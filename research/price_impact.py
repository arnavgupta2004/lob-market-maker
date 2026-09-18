"""Price impact of aggressive orders: ``I(Q) ~ c Q^alpha``, estimated rather than assumed.

An *aggressive order* is all trades sharing a taker order id. For signed size ``s Q`` with ``s = +1`` for a
buy, and ``m^-`` the mid before the order::

    immediate impact   I_0    = s (m_after - m^-)        mid change right after the order (book walked)
    lagged impact      I_h    = s (m_{t+h} - m^-)        after ``h`` seconds (permanent + resilience)
    slippage           S      = s (VWAP - m^-)           what the aggressor paid vs the mid

Estimation: orders are binned by size on a log grid; per-bin mean impact (session-bootstrap CIs) and a
weighted log-log regression ``ln E[I] = ln c + alpha ln Q``. Bins whose mean impact is not positive are
excluded from the log fit (and reported), since the log of a non-positive mean is undefined.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from research.common import mid_at
from simulator.simulator import NS, SimResult


@dataclass
class AggressiveOrders:
    t: np.ndarray
    side: np.ndarray  # +1 buy
    qty: np.ndarray  # total executed quantity
    mid_before: np.ndarray
    vwap: np.ndarray
    taker_owner: np.ndarray


def aggressive_orders(res: SimResult, taker_owner: int | None = None) -> AggressiveOrders:
    tp = res.trades
    if len(tp["t_ns"]) == 0:
        e = np.array([])
        return AggressiveOrders(e, e, e, e, e, e)
    ids = tp["taker_id"]
    order = np.argsort(ids, kind="stable")
    ids_s = ids[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ids_s)) + 1]
    q = tp["qty"][order].astype(float)
    notional = tp["price"][order] * q
    qty = np.add.reduceat(q, starts)
    first = order[starts]
    ao = AggressiveOrders(tp["t_ns"][first], tp["aggressor"][first].astype(float), qty, tp["mid_before"][first],
                          np.add.reduceat(notional, starts) / qty, tp["taker_owner"][first])
    keep = ~np.isnan(ao.mid_before)
    if taker_owner is not None:
        keep &= ao.taker_owner == taker_owner
    return AggressiveOrders(*[a[keep] for a in (ao.t, ao.side, ao.qty, ao.mid_before, ao.vwap, ao.taker_owner)])


def impacts(res: SimResult, ao: AggressiveOrders, horizons_s: Sequence[float] = (0.1, 1.0, 5.0)) -> dict[str, np.ndarray]:
    """Per-order impact arrays: ``I_0`` (immediate), ``I_<h>s`` (lagged) and ``slippage``, in ticks."""
    end = res.config.horizon_s * NS
    out = {"I_0": ao.side * (mid_at(res, ao.t) - ao.mid_before), "slippage": ao.side * (ao.vwap - ao.mid_before)}
    for h in horizons_s:
        hn = int(h * NS)
        v = ao.side * (mid_at(res, ao.t + hn) - ao.mid_before)
        out[f"I_{h:g}s"] = np.where(ao.t + hn <= end, v, np.nan)
    return out


def size_bin_stats(qty: np.ndarray, impact: np.ndarray, edges: Sequence[float]) -> dict[int, np.ndarray]:
    """Per size-bin sufficient statistics ``[n, sum_I, sum_I^2, sum_Q]`` (additive across sessions)."""
    stats: dict[int, np.ndarray] = {}
    ok = ~np.isnan(impact)
    b = np.digitize(qty[ok], edges) - 1
    for i in np.unique(b):
        m = b == i
        x, q = impact[ok][m], qty[ok][m]
        stats[int(i)] = np.array([m.sum(), x.sum(), (x * x).sum(), q.sum()], dtype=float)
    return stats


def add_stats(a: dict[int, np.ndarray], b: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    out = {k: v.copy() for k, v in a.items()}
    for k, v in b.items():
        out[k] = out[k] + v if k in out else v.copy()
    return out


def fit_power_law(stats: dict[int, np.ndarray], min_n: int = 30) -> dict:
    """Weighted log-log fit of ``E[I]`` against mean bin size. Weights ``n * mean^2 / var`` (inverse of the
    delta-method variance of ``ln mean``), so noisy bins count less."""
    xs, ys, ws, used = [], [], [], []
    for i in sorted(stats):
        n, s1, s2, sq = stats[i]
        if n < min_n:
            continue
        mean = s1 / n
        var = max(s2 / n - mean ** 2, 1e-12)
        if mean <= 0:
            continue
        xs.append(math.log(sq / n)); ys.append(math.log(mean)); ws.append(n * mean ** 2 / var); used.append(i)
    if len(xs) < 3:
        return dict(alpha=float("nan"), c=float("nan"), n_bins=len(xs), r2=float("nan"))
    x, y, w = np.array(xs), np.array(ys), np.array(ws)
    X = np.column_stack([np.ones_like(x), x])
    cov = np.linalg.inv(X.T @ (w[:, None] * X))
    beta = cov @ X.T @ (w * y)
    res_ = y - X @ beta
    ss_res = float(np.sum(w * res_ ** 2)); ss_tot = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    return dict(alpha=float(beta[1]), c=float(math.exp(beta[0])), n_bins=len(xs),
                r2=1 - ss_res / ss_tot if ss_tot > 0 else float("nan"), bins=used)


def bin_table(stats: dict[int, np.ndarray], edges: Sequence[float]) -> list[dict]:
    rows = []
    for i in sorted(stats):
        n, s1, s2, sq = stats[i]
        mean = s1 / n
        se = math.sqrt(max(s2 / n - mean ** 2, 0) / n)
        rows.append(dict(bin=i, q_lo=edges[i], q_hi=edges[i + 1] if i + 1 < len(edges) else float("inf"),
                         n=int(n), mean_q=sq / n, mean_impact=mean, se=se))
    return rows


def merge_bursts(ao: AggressiveOrders) -> AggressiveOrders:
    """Merge aggressive orders that share a timestamp, side and owner into one parent order.

    An exchange that prints one trade per fill reports a single order sweeping several price levels as a *burst* of prints at the same
    instant; treating each print as an order understates the size of the parent order. The merged order has the summed quantity, the
    pre-trade mid of its first print and the volume-weighted average price."""
    if len(ao.t) == 0:
        return ao
    order = np.lexsort((np.arange(len(ao.t)), ao.side, ao.taker_owner, ao.t))
    t, s, o = ao.t[order], ao.side[order], ao.taker_owner[order]
    new = np.r_[True, (t[1:] != t[:-1]) | (s[1:] != s[:-1]) | (o[1:] != o[:-1])]
    starts = np.flatnonzero(new)
    q = ao.qty[order]
    notional = ao.vwap[order] * q
    qty = np.add.reduceat(q, starts)
    return AggressiveOrders(t[starts], s[starts], qty, ao.mid_before[order][starts], np.add.reduceat(notional, starts) / qty, o[starts])
