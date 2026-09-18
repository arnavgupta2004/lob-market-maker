"""Baseline inventory-aware market maker (Avellaneda & Stoikov, 2008).

Theoretical model (all quantities in *tick* units; ``q`` in lots)
-----------------------------------------------------------------
**Assumptions**

A1. The reference price is arithmetic Brownian motion, ``dS = sigma dW`` (no drift).
A2. Limit orders posted at distance ``delta`` from ``S`` are hit by market orders as a Poisson
    process with intensity ``lambda(delta) = A exp(-k delta)``.
A3. The agent has CARA utility over terminal wealth, risk aversion ``gamma``, horizon ``T``:
    ``max E[-exp(-gamma (X_T + q_T S_T))]``.
A4. Continuous quoting, no latency, no fees, unit-size orders, no queue priority, no
    price impact of our own trades, unbounded inventory.

**Derivation sketch.** Let ``u(x, s, q, t)`` be the value function. Dynamic programming gives
the HJB equation::

    u_t + (sigma^2 / 2) u_ss
        + max_{delta^b} lambda(delta^b) [u(x - s + delta^b, s, q + 1, t) - u]
        + max_{delta^a} lambda(delta^a) [u(x + s + delta^a, s, q - 1, t) - u] = 0,

with ``u(x, s, q, T) = -exp(-gamma (x + q s))``. The ansatz ``u = -exp(-gamma x) exp(-gamma
theta(s, q, t))`` reduces it to an equation for ``theta``; first-order conditions with the
exponential intensity give optimal distances of the form ``(1/gamma) ln(1 + gamma/k) +/- ...``
around the **reservation price** ``r``. Solving for ``theta`` with a second-order expansion in
``q`` (the approximation made in the original paper, accurate for small inventories; exact
solutions exist - Gueant, Lehalle & Fernandez-Tapia 2013 - but are not used here) yields::

    r(s, q, t)        = s - q * gamma * sigma^2 * (T - t)
    delta^a + delta^b = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / k)

    bid = r - (delta^a + delta^b) / 2,     ask = r + (delta^a + delta^b) / 2

Interpretation: long inventory (``q > 0``) lowers ``r`` (skews both quotes down => more likely
to sell); the ``gamma sigma^2 (T-t)`` term widens the spread with risk; the second term is the
spread a *risk-neutral* maker would set given fill-intensity decay ``k``; as ``gamma -> 0`` the
total spread tends to ``2 / k``.

**Implementation choices that are NOT part of the theory** (documented so they can be ablated):

* ``sigma`` is an EWMA estimate of realised mid volatility unless a fixed value is supplied.
* ``k`` must be supplied; ``experiments.baseline.as_assumptions`` estimates it from the
  simulator (probe orders) rather than assuming a value.
* The horizon is finite but trading is continuous: ``tau_mode='restart'`` uses
  ``tau = T - (t mod T)`` (time-to-go restarts every ``T``, causing a periodic jump in the quotes);
  ``'constant'`` uses ``tau = T`` (a stationary approximation).
* Prices are rounded outward to integer ticks: ``bid = floor(bid*)``, ``ask = ceil(ask*)``.

Whether A1-A4 hold in this simulator is an *empirical question* (see the experiment above);
nothing here assumes they do.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from strategies.base import NS, MMConfig, MarketMaker, Quote


@dataclass(frozen=True)
class ASConfig(MMConfig):
    gamma: float = 0.1  # risk aversion (1 / tick)
    k: float = 0.5  # fill-intensity decay (1 / tick)
    horizon_s: float = 10.0  # T
    tau_mode: str = "restart"  # "restart" | "constant"
    sigma: Optional[float] = None  # fixed volatility (ticks/sqrt(s)); None => EWMA estimate

    def build(self, name: str = "mm") -> "AvellanedaStoikovMM":
        return AvellanedaStoikovMM(self, name)


def avellaneda_stoikov(mid: float, q: float, gamma: float, sigma: float, tau: float,
                       k: float) -> tuple[float, float]:
    """Return ``(reservation_price, total_spread)`` in ticks for the closed-form AS quotes."""
    var_tau = sigma * sigma * tau
    reservation = mid - q * gamma * var_tau
    if gamma > 0:
        spread = gamma * var_tau + (2.0 / gamma) * math.log1p(gamma / k)
    else:
        spread = 2.0 / k  # gamma -> 0 limit
    return reservation, spread


class AvellanedaStoikovMM(MarketMaker):
    cfg: ASConfig

    def time_to_go(self, now: int) -> float:
        T = self.cfg.horizon_s
        if self.cfg.tau_mode == "constant":
            return T
        elapsed = ((now - self._start_ns) / NS) % T
        return max(T - elapsed, 1e-3 * T)

    def desired_quotes(self, now: int, mid: float) -> tuple[Optional[Quote], Optional[Quote]]:
        c = self.cfg
        sigma = c.sigma if c.sigma is not None else self.sigma
        r, spread = avellaneda_stoikov(mid, self.inventory, c.gamma, sigma, self.time_to_go(now), c.k)
        bid = math.floor(r - spread / 2)
        ask = math.ceil(r + spread / 2)
        if ask <= bid:
            ask = bid + 1
        return (bid, c.quote_size), (ask, c.quote_size)
