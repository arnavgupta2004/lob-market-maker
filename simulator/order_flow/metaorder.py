"""Order-splitting ("metaorder") traders.

Large parent orders are executed as many small child market orders in the same direction. This is the standard mechanism
proposed for the *long memory of order signs* observed in real markets: if the number of children of a metaorder has a
power-law tail ``P(N > n) ~ n^-alpha`` with ``1 < alpha < 2``, the autocorrelation of the sign series decays as
``tau^-gamma`` with ``gamma = alpha - 1`` (Lillo, Mike & Farmer 2005). That is a *prediction*, not an assumption: the
experiment measures ``gamma`` in the simulator and compares it with ``alpha - 1``.

Model: parent orders arrive as a Poisson process (``rate`` per second); each has a random side and a total size drawn from
``size``; children of size ``child_size`` are sent as market orders at Poisson times (``child_rate`` per second per active
parent) until the parent is exhausted. Sizes are integers (lots); the final child carries the remainder.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Optional

import numpy as np

from engine.commands import Command, NewMarket
from engine.common import Side
from simulator.order_flow.arrivals import NS
from simulator.order_flow.distributions import Dist, LogNormal, Pareto, sample_int
from simulator.participant import Participant


@dataclass(frozen=True)
class MetaOrderConfig:
    rate: float = 0.3  # parent orders per second
    total: Dist = Pareto(xm=10.0, alpha=1.5)  # parent size in lots (heavy tailed)
    child: Dist = LogNormal(median=3.0, sigma=0.4)  # child size in lots
    child_rate: float = 4.0  # child orders per second per active parent
    p_buy: float = 0.5
    max_total: int = 5000


class MetaOrderTrader(Participant):
    feed = "none"

    def __init__(self, cfg: MetaOrderConfig = MetaOrderConfig(), name: str = "meta"):
        self.cfg = cfg
        self.name = name
        self._next_parent = 0
        self._active: list[tuple[int, int, int, int]] = []  # heap of (next_child_time_ns, seq, side, remaining)
        self._seq = 0
        self.parents_started = 0
        self.children_sent = 0
        self.rng: np.random.Generator

    def on_start(self, sim) -> None:
        self.rng = sim.rng_for(self.name)
        self._next_parent = int(self.rng.exponential(1.0 / self.cfg.rate) * NS)

    def next_time(self) -> Optional[int]:
        t = self._next_parent
        return min(t, self._active[0][0]) if self._active else t

    def _gap(self) -> int:
        return max(1, int(self.rng.exponential(1.0 / self.cfg.child_rate) * NS))

    def on_wakeup(self, sim, now: int) -> list[Command]:
        cmds: list[Command] = []
        c, rng = self.cfg, self.rng
        if self._next_parent <= now:
            side = 1 if rng.random() < c.p_buy else -1
            total = sample_int(c.total, rng, lo=1, hi=c.max_total)
            self._seq += 1
            heapq.heappush(self._active, (now + self._gap(), self._seq, side, total))
            self.parents_started += 1
            self._next_parent = now + max(1, int(rng.exponential(1.0 / c.rate) * NS))
        while self._active and self._active[0][0] <= now:
            _, seq, side, remaining = heapq.heappop(self._active)
            qty = min(remaining, sample_int(c.child, rng, lo=1))
            cmds.append(NewMarket(sim.new_id(), Side(side), qty, owner=self.owner_id))
            self.children_sent += 1
            if remaining - qty > 0:
                heapq.heappush(self._active, (now + self._gap(), seq, side, remaining - qty))
        return cmds
