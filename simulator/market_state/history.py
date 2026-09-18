"""Compact history of the true mid-price, for return/momentum look-backs."""
from __future__ import annotations

from bisect import bisect_right
from typing import Optional


class MidHistory:
    """Step-function record of the mid-price (ticks). Appends only when the mid changes."""

    def __init__(self) -> None:
        self.t: list[int] = []
        self.mid: list[float] = []

    def update(self, t: int, mid: Optional[float]) -> None:
        if mid is None:
            return
        if not self.mid or mid != self.mid[-1]:
            self.t.append(t)
            self.mid.append(mid)

    def at(self, t: int) -> Optional[float]:
        """Mid in force at time ``t`` (None before the first observation)."""
        i = bisect_right(self.t, t) - 1
        return self.mid[i] if i >= 0 else None

    def last(self) -> Optional[float]:
        return self.mid[-1] if self.mid else None
