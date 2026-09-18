"""Point-process arrival models: how long until the next event?"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

NS = 1_000_000_000


class ArrivalProcess(ABC):
    @abstractmethod
    def next_gap_ns(self, rng: np.random.Generator) -> int:
        """Nanoseconds until the next arrival (>= 1)."""


@dataclass(frozen=True)
class PoissonArrival(ArrivalProcess):
    """Homogeneous Poisson process with ``rate`` events per second."""

    rate: float

    def next_gap_ns(self, rng: np.random.Generator) -> int:
        return max(1, int(rng.exponential(1.0 / self.rate) * NS))


class HawkesArrival(ArrivalProcess):
    """Self-exciting (Hawkes) arrivals with an exponential kernel, sampled exactly by Ogata thinning::

        lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta (t - t_i))

    Each arrival raises the intensity by ``alpha`` (1/s), decaying at rate ``beta`` (1/s). The branching ratio
    ``n = alpha / beta`` must be < 1 for stationarity; the long-run rate is ``mu / (1 - n)``. Arrivals cluster (Fano factor
    > 1) instead of arriving independently. The object is stateful and is asked for one gap at a time, in order, by a single
    participant.
    """

    def __init__(self, mu: float, alpha: float, beta: float):
        if not 0 <= alpha < beta:
            raise ValueError("Hawkes process needs 0 <= alpha < beta (branching ratio < 1)")
        self.mu, self.alpha, self.beta = mu, alpha, beta
        self._exc = 0.0  # excitation above baseline at time ``_t``
        self._t = 0.0  # seconds

    @property
    def branching_ratio(self) -> float:
        return self.alpha / self.beta

    @property
    def mean_rate(self) -> float:
        return self.mu / (1.0 - self.branching_ratio)

    @staticmethod
    def with_mean_rate(rate: float, branching_ratio: float, beta: float) -> "HawkesArrival":
        """Hawkes process with long-run rate ``rate`` events/s, given branching ratio and decay ``beta`` (1/s)."""
        return HawkesArrival(mu=rate * (1.0 - branching_ratio), alpha=branching_ratio * beta, beta=beta)

    def next_gap_ns(self, rng: np.random.Generator) -> int:
        import math
        t_prev = self._t
        while True:
            lam_bar = self.mu + self._exc  # the intensity only decays until the next event: a valid upper bound
            w = rng.exponential(1.0 / lam_bar)
            self._t += w
            self._exc *= math.exp(-self.beta * w)
            if rng.random() * lam_bar <= self.mu + self._exc:
                self._exc += self.alpha
                return max(1, int((self._t - t_prev) * NS))
