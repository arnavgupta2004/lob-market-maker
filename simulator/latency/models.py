"""Latency models. All latencies are integer nanoseconds.

Two one-way latencies matter to a participant:

* ``order``: participant -> exchange (delay before a command reaches the matching engine)
* ``feed``:  exchange -> participant (delay before it observes a public event)

Delivery is FIFO per participant (the simulator clamps arrival times to be non-decreasing),
mirroring a sequenced session; jitter therefore changes *delays*, never message order.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

MS = 1_000_000


class LatencyModel(ABC):
    @abstractmethod
    def sample(self, rng: np.random.Generator) -> int:
        """One-way delay in ns (>= 0)."""


@dataclass(frozen=True)
class ConstantLatency(LatencyModel):
    ns: int = 0

    def sample(self, rng: np.random.Generator) -> int:
        return self.ns


@dataclass(frozen=True)
class UniformLatency(LatencyModel):
    """Base delay plus uniform jitter in ``[0, jitter_ns]``."""

    base_ns: int
    jitter_ns: int

    def sample(self, rng: np.random.Generator) -> int:
        return self.base_ns + int(rng.integers(0, self.jitter_ns + 1))


@dataclass(frozen=True)
class LogNormalLatency(LatencyModel):
    """Heavy-ish right tail: ``median_ns * exp(sigma * Z)``."""

    median_ns: int
    sigma: float = 0.3

    def sample(self, rng: np.random.Generator) -> int:
        return int(self.median_ns * math.exp(self.sigma * rng.standard_normal()))


@dataclass(frozen=True)
class LatencyConfig:
    order: LatencyModel = ConstantLatency(0)
    feed: LatencyModel = ConstantLatency(0)

    @staticmethod
    def symmetric(one_way_ms: float) -> "LatencyConfig":
        """Constant one-way latency ``one_way_ms`` on both legs (round trip = 2x)."""
        ns = int(round(one_way_ms * MS))
        return LatencyConfig(ConstantLatency(ns), ConstantLatency(ns))


ZERO_LATENCY = LatencyConfig()
