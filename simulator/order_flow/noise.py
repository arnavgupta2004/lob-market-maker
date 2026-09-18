"""Zero-intelligence noise traders (Poisson order flow).

Three independent stochastic ingredients (all configurable):

* **Arrivals** - limit and market orders arrive as a superposition of Poisson processes with
  rates ``limit_rate`` and ``market_rate`` (per second). Any :class:`ArrivalProcess` can be
  substituted for the combined process.
* **Cancellations** - every resting noise order draws an independent Exp(``cancel_rate``)
  lifetime when placed (per-order hazard, as in Cont-Stoikov-Talreja); if still live when the
  timer fires it is cancelled.
* **Order attributes** - side ~ Bernoulli(``p_buy``); size and price offset from the mid are
  drawn from user-supplied distributions.

Price rule for limit orders with offset ``d >= 0`` ticks around mid ``m`` (ticks)::

    BUY  price = floor(m) - d        SELL price = ceil(m) + d

so ``d = 0`` joins/improves the touch, and a wide spread lets ``d = 0`` orders cross - all
emergent, nothing hard-coded. These are *assumptions*, not empirical claims.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from engine.commands import Cancel, Command, NewLimit, NewMarket
from engine.common import Side
from simulator.order_flow.arrivals import NS, ArrivalProcess, PoissonArrival
from simulator.order_flow.distributions import Dist, Exponential, LogNormal, sample_int
from simulator.participant import Participant


@dataclass(frozen=True)
class NoiseTraderConfig:
    limit_rate: float = 50.0  # limit orders per second
    market_rate: float = 15.0  # market orders per second
    cancel_rate: float = 1.0  # per-order cancellation hazard (1/s); 0 disables cancels
    p_buy: float = 0.5
    offset: Dist = Exponential(3.0)  # ticks from mid, floored to an integer >= 0
    size: Dist = LogNormal(median=5.0, sigma=0.8)  # lots, floored to an integer >= 1
    market_size: Dist = LogNormal(median=5.0, sigma=0.8)
    max_size: int = 1000


class NoiseTrader(Participant):
    feed = "none"

    def __init__(self, cfg: NoiseTraderConfig = NoiseTraderConfig(), name: str = "noise",
                 arrivals: Optional[ArrivalProcess] = None):
        self.cfg = cfg
        self.name = name
        total = cfg.limit_rate + cfg.market_rate
        self.arrivals = arrivals if arrivals is not None else PoissonArrival(total)
        self._p_limit = cfg.limit_rate / total
        self._next_arrival = 0
        self._timers: list[tuple[int, int]] = []  # (cancel_time_ns, order_id)
        self.rng: np.random.Generator

    def on_start(self, sim) -> None:
        self.rng = sim.rng_for(self.name)
        self._next_arrival = self.arrivals.next_gap_ns(self.rng)

    def next_time(self) -> Optional[int]:
        if self._timers:
            return min(self._next_arrival, self._timers[0][0])
        return self._next_arrival

    def on_wakeup(self, sim, now: int) -> list[Command]:
        cmds: list[Command] = []
        while self._timers and self._timers[0][0] <= now:
            _, oid = heapq.heappop(self._timers)
            if oid in sim.book:
                cmds.append(Cancel(oid))
        if self._next_arrival <= now:
            cmds.append(self._new_order(sim, now))
            self._next_arrival = now + self.arrivals.next_gap_ns(self.rng)
        return cmds

    def _new_order(self, sim, now: int) -> Command:
        c, rng = self.cfg, self.rng
        side = Side.BUY if rng.random() < c.p_buy else Side.SELL
        if rng.random() < self._p_limit:
            m = sim.reference_mid()
            d = sample_int(c.offset, rng, lo=0)
            price = max(1, math.floor(m) - d if side is Side.BUY else math.ceil(m) + d)
            oid = sim.new_id()
            qty = sample_int(c.size, rng, lo=1, hi=c.max_size)
            if c.cancel_rate > 0:
                life = int(rng.exponential(1.0 / c.cancel_rate) * NS)
                heapq.heappush(self._timers, (now + max(1, life), oid))
            return NewLimit(oid, side, price, qty, owner=self.owner_id)
        return NewMarket(sim.new_id(), side, sample_int(c.market_size, rng, lo=1, hi=c.max_size),
                         owner=self.owner_id)
