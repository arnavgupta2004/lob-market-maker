"""Book resilience: how fast do spread, depth and mid return to their pre-shock state after a controlled shock?

A :class:`ShockInjector` sends a single aggressive market order of known size at known times. For every shock at time ``t``
with aggressor sign ``s`` (+1 = buy, consuming the ask side), pre-shock reference values are the means over the ``pre_s``
seconds before ``t``, and for every lag ``tau`` after it::

    mid impact        I(tau)   = s (m_{t+tau} - m_pre)                      ticks (0 = fully recovered)
    spread excess     X(tau)   = spread_{t+tau} / spread_pre - 1            (0 = recovered)
    depth deficit     D(tau)   = 1 - depth_consumed_side_{t+tau} / depth_pre  (5-level depth of the side that was hit; 0 = recovered)

Curves are averaged over the shocks of one session; uncertainty comes from bootstrapping sessions (in the experiment).
``half_life`` is the first lag at which a curve falls to half of its value right after the shock; ``None`` means it did not
within the window. ``residual`` is the curve at the end of the window as a fraction of its initial value: for the mid this is
the *permanent* share of the impact.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from engine.commands import Command, NewMarket
from engine.common import Side
from simulator.order_flow.arrivals import NS
from simulator.participant import Participant


class ShockInjector(Participant):
    """Sends a market order of ``size`` lots at each time in ``times_s``; sides alternate starting with ``first_side``."""

    feed = "none"

    def __init__(self, times_s: Sequence[float], size: int, first_side: int = 1, name: str = "shock"):
        self.name = name
        self.times = [int(t * NS) for t in times_s]
        self.size = size
        self.first_side = first_side
        self.i = 0
        self.log: list[tuple[int, int]] = []  # (sim time ns, aggressor sign)

    def next_time(self) -> Optional[int]:
        return self.times[self.i] if self.i < len(self.times) else None

    def on_wakeup(self, sim, now: int) -> list[Command]:
        side = self.first_side if self.i % 2 == 0 else -self.first_side
        self.i += 1
        self.log.append((now, side))
        return [NewMarket(sim.new_id(), Side(side), self.size, owner=self.owner_id)]


def _ffill(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    ok = ~np.isnan(x)
    if not ok.any():
        return x
    idx = np.where(ok, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = x[idx]
    out[: np.argmax(ok)] = out[np.argmax(ok)]
    return out


def resilience_curves(samples: dict[str, np.ndarray], dt_s: float, shocks: Sequence[tuple[int, int]],
                      horizon_s: float = 10.0, pre_s: float = 1.0) -> dict[str, np.ndarray]:
    """Per-session mean recovery curves at lags ``dt_s, 2 dt_s, ..., horizon_s`` after the shocks.

    ``samples`` is ``SimResult.samples`` (must be sampled at ``dt_s``). Returns arrays ``tau, mid, spread, depth`` (length ``n_lags``)."""
    t = samples["t_ns"]
    mid, spr = _ffill(samples["mid"]), _ffill(samples["spread"])
    k_pre, n_lag = int(round(pre_s / dt_s)), int(round(horizon_s / dt_s))
    I, X, D = [], [], []
    for t_shock, s in shocks:
        i0 = int(np.searchsorted(t, t_shock, side="right"))  # first sample strictly after the shock
        if i0 - k_pre < 0 or i0 + n_lag > len(t):
            continue
        pre = slice(i0 - k_pre - 1, i0 - 1)
        depth = samples["ask_depth5"] if s > 0 else samples["bid_depth5"]
        m_pre, s_pre, d_pre = mid[pre].mean(), spr[pre].mean(), depth[pre].mean()
        if s_pre <= 0 or d_pre <= 0:
            continue
        post = slice(i0, i0 + n_lag)
        I.append(s * (mid[post] - m_pre))
        X.append(spr[post] / s_pre - 1.0)
        D.append(1.0 - depth[post] / d_pre)
    tau = dt_s * np.arange(1, n_lag + 1)
    if not I:
        nan = np.full(n_lag, np.nan)
        return {"tau": tau, "mid": nan, "spread": nan, "depth": nan, "n_shocks": 0}
    return {"tau": tau, "mid": np.mean(I, 0), "spread": np.mean(X, 0), "depth": np.mean(D, 0), "n_shocks": len(I)}


def half_life(curve: np.ndarray, tau: np.ndarray, smooth: int = 3) -> dict:
    """Time for the (lightly smoothed) curve to *halve from its first observed level*, with the crossing linearly
    interpolated; ``None`` if it does not halve within the window. ``initial`` is that first smoothed level and ``residual`` is the
    last smoothed level divided by it (for the mid: the permanent share of the impact). For a pure exponential ``exp(-t/T)``
    this returns ``T ln 2`` regardless of the smoothing window."""
    c = np.asarray(curve, dtype=float)
    tau = np.asarray(tau, dtype=float)
    if len(c) < smooth + 1 or not np.isfinite(c).all():
        return {"half_life_s": None, "initial": float("nan"), "residual": float("nan")}
    cs = np.convolve(c, np.ones(smooth) / smooth, mode="valid")
    tc = np.convolve(tau, np.ones(smooth) / smooth, mode="valid")  # window centres
    v0 = cs[0]
    if v0 <= 0:
        return {"half_life_s": None, "initial": float(v0), "residual": float("nan")}
    below = np.flatnonzero(cs <= 0.5 * v0)
    if len(below) == 0:
        return {"half_life_s": None, "initial": float(v0), "residual": float(cs[-1] / v0)}
    j = int(below[0])
    t_cross = tc[j] if j == 0 else tc[j - 1] + (cs[j - 1] - 0.5 * v0) / (cs[j - 1] - cs[j]) * (tc[j] - tc[j - 1])
    return {"half_life_s": float(t_cross - tc[0]), "initial": float(v0), "residual": float(cs[-1] / v0)}
