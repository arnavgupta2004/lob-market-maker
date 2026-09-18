"""Small univariate distributions for order sizes and price offsets.

Each spec is a frozen dataclass with ``sample(rng) -> float`` and (where finite) ``mean()``.
``sample_int`` discretises to ``max(lo, floor(x))`` (optionally capped) - the form used for
sizes (lots) and offsets (ticks). None of these is assumed realistic; validating them against
data is the job of ``research.stylized_facts``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


class Dist(Protocol):
    def sample(self, rng: np.random.Generator) -> float: ...
    def mean(self) -> float: ...


@dataclass(frozen=True)
class Const:
    value: float

    def sample(self, rng: np.random.Generator) -> float:
        return self.value

    def mean(self) -> float:
        return self.value


@dataclass(frozen=True)
class Exponential:
    """Exponential with the given mean."""

    scale: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.exponential(self.scale))

    def mean(self) -> float:
        return self.scale


@dataclass(frozen=True)
class LogNormal:
    """``exp(N(mu, sigma^2))`` parameterised by its median (``exp(mu)``) and ``sigma``."""

    median: float
    sigma: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(self.median * math.exp(self.sigma * rng.standard_normal()))

    def mean(self) -> float:
        return self.median * math.exp(0.5 * self.sigma ** 2)


@dataclass(frozen=True)
class Pareto:
    """Pareto type I: ``P(X > x) = (xm / x)^alpha`` for ``x >= xm`` (heavy tail, mean finite iff alpha > 1)."""

    xm: float
    alpha: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(self.xm * (1.0 - rng.random()) ** (-1.0 / self.alpha))

    def mean(self) -> float:
        return self.xm * self.alpha / (self.alpha - 1) if self.alpha > 1 else math.inf


@dataclass(frozen=True)
class Uniform:
    lo: float
    hi: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.lo, self.hi))

    def mean(self) -> float:
        return 0.5 * (self.lo + self.hi)


def sample_int(dist: Dist, rng: np.random.Generator, lo: int = 0, hi: Optional[int] = None) -> int:
    """``floor`` of a draw, clipped to ``[lo, hi]``."""
    x = int(dist.sample(rng))
    if x < lo:
        x = lo
    if hi is not None and x > hi:
        x = hi
    return x
