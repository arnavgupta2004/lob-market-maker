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
