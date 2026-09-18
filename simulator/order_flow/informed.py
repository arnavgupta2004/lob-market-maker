"""Informed / momentum traders: a controlled source of adverse selection and price discovery.

At Poisson wake-ups the trader forms a linear score (in ticks)::

    s = w_value * (V_hat - m)  +  w_momentum * (m - m_lag)  +  w_imbalance * OBI * imbalance_scale

with ``V_hat = V + eps`` the (noisy) latent fundamental in ticks, ``m`` the current mid,
``m_lag`` the mid ``lookback`` seconds ago and OBI the top-``levels`` order-book imbalance.
If ``|s| >= threshold`` it sends a market order of sign ``s``. With only ``w_value > 0`` the
trader is a pure fundamental (informed) trader: its flow forces the mid toward ``V`` and its
counterparties are adversely selected. With only ``w_momentum > 0`` it is a trend follower.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from engine.commands import Command, NewMarket
from engine.common import Side
from simulator.order_flow.arrivals import ArrivalProcess, PoissonArrival, NS
from simulator.order_flow.distributions import Dist, LogNormal, sample_int
from simulator.participant import Participant


@dataclass(frozen=True)
class InformedTraderConfig:
    rate: float = 5.0  # wake-ups per second
    w_value: float = 1.0
    w_momentum: float = 0.0
    w_imbalance: float = 0.0
    threshold: float = 1.0  # ticks
    signal_noise: float = 0.0  # std of eps (ticks); 0 = perfectly informed
    lookback_s: float = 1.0
    imbalance_levels: int = 3
    imbalance_scale: float = 5.0  # ticks per unit OBI
    size: Dist = LogNormal(median=5.0, sigma=0.8)
    max_size: int = 1000


class InformedTrader(Participant):
    feed = "none"

    def __init__(self, cfg: InformedTraderConfig = InformedTraderConfig(), name: str = "informed",
                 arrivals: Optional[ArrivalProcess] = None):
        self.cfg = cfg
        self.name = name
        self.arrivals = arrivals if arrivals is not None else PoissonArrival(cfg.rate)
        self._next = 0
        self.rng: np.random.Generator

    def on_start(self, sim) -> None:
        self.rng = sim.rng_for(self.name)
        self._next = self.arrivals.next_gap_ns(self.rng)

    def next_time(self) -> Optional[int]:
        return self._next

    def score(self, sim, now: int) -> float:
        c = self.cfg
        m = sim.reference_mid()
        s = 0.0
        if c.w_value:
            v = sim.fundamental_ticks(now)
            if c.signal_noise:
                v += c.signal_noise * self.rng.standard_normal()
            s += c.w_value * (v - m)
        if c.w_momentum:
            lag = sim.mid_history.at(now - int(c.lookback_s * NS))
            s += c.w_momentum * (0.0 if lag is None else m - lag)
        if c.w_imbalance:
            obi = sim.book.imbalance(c.imbalance_levels)
            s += c.w_imbalance * c.imbalance_scale * (0.0 if obi is None else obi)
        return s

    def on_wakeup(self, sim, now: int) -> list[Command]:
        self._next = now + self.arrivals.next_gap_ns(self.rng)
        s = self.score(sim, now)
        if abs(s) < self.cfg.threshold:
            return []
        side = Side.BUY if s > 0 else Side.SELL
        return [NewMarket(sim.new_id(), side, sample_int(self.cfg.size, self.rng, 1, self.cfg.max_size),
                          owner=self.owner_id)]
