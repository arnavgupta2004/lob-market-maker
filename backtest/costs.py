"""Transaction-cost model."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeeModel:
    """Proportional fees in basis points of notional. Negative ``maker_bps`` is a rebate.

    Slippage is *not* a fee: it is realised through the book (a taker sweeping levels pays the
    walk-the-book price) and shows up in the tape prices.
    """

    maker_bps: float = 0.0
    taker_bps: float = 0.0

    def fee(self, notional: float, is_maker: bool) -> float:
        return notional * (self.maker_bps if is_maker else self.taker_bps) * 1e-4
