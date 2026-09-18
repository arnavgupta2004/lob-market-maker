"""Latent efficient ("fundamental") value process.

Simulated once, up front, on a fixed grid from its own RNG stream, so the path is a pure
function of ``(seed, config)`` and independent of what participants do. ``value_at`` is
piecewise constant on the grid.

* Brownian (``kappa == 0``):  ``dV = sigma dW``
* Ornstein-Uhlenbeck:         ``dV = kappa (mu - V) dt + sigma dW`` simulated with the *exact*
  transition ``V' = mu + (V - mu) e^{-kappa dt} + sigma sqrt((1 - e^{-2 kappa dt}) / (2 kappa)) Z``.

Units: price units (currency), ``sigma`` in price-units per sqrt(second), ``kappa`` in 1/s.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

NS = 1_000_000_000


@dataclass(frozen=True)
class FundamentalConfig:
    sigma: float = 0.02  # price units per sqrt(second)
    kappa: float = 0.0  # mean-reversion speed (1/s); 0 => Brownian motion
    mu: Optional[float] = None  # OU long-run mean; defaults to the starting value
    dt_s: float = 0.01  # simulation grid step


class FundamentalProcess:
    def __init__(self, cfg: FundamentalConfig, v0: float, horizon_ns: int, rng: np.random.Generator):
        self.cfg = cfg
        self.dt_ns = max(1, int(round(cfg.dt_s * NS)))
        n = horizon_ns // self.dt_ns + 2
        dt = self.dt_ns / NS
        z = rng.standard_normal(n - 1)
        path = np.empty(n)
        path[0] = v0
        if cfg.kappa <= 0.0:
            path[1:] = v0 + cfg.sigma * math.sqrt(dt) * np.cumsum(z)
        else:
            mu = v0 if cfg.mu is None else cfg.mu
            a = math.exp(-cfg.kappa * dt)
            s = cfg.sigma * math.sqrt((1.0 - a * a) / (2.0 * cfg.kappa))
            v = v0
            for i in range(n - 1):  # AR(1) recursion; n is small enough (grid, not events)
                v = mu + (v - mu) * a + s * z[i]
                path[i + 1] = v
        self.path = path

    def value_at(self, t_ns: int) -> float:
        i = t_ns // self.dt_ns
        return float(self.path[min(i, len(self.path) - 1)])
