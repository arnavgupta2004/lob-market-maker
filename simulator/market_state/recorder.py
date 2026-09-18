"""Market-state recording: periodic book samples and the trade tape."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from engine.common import EventType, Side
from engine.events import Event
from engine.python.order_book import OrderBook

_NAN = float("nan")

SAMPLE_COLUMNS = (
    "t_ns", "best_bid", "best_ask", "mid", "spread", "bid_qty1", "ask_qty1",
    "bid_depth5", "ask_depth5", "n_orders", "fundamental",
)


@dataclass
class MarketRecorder:
    """Accumulates fixed-interval book samples and every trade.

    Prices are in ticks; ``nan`` marks an empty side. ``fundamental`` is in ticks too (nan when
    the simulation has no latent-value process).
    """

    depth_levels: int = 5
    _rows: list = field(default_factory=list)
    _trades: list = field(default_factory=list)

    def on_event(self, ev: Event) -> None:
        if ev.type is EventType.TRADE:
            # (t_ns, price_ticks, qty, aggressor_sign, taker_owner, maker_owner)
            self._trades.append((ev.ts, ev.price, ev.qty, int(ev.side), ev.owner, ev.maker_owner))

    def sample(self, t: int, book: OrderBook, fundamental_ticks: float = _NAN) -> None:
        b, a = book.best_bid(), book.best_ask()
        bids, asks = book.depth(self.depth_levels)
        self._rows.append((
            t,
            _NAN if b is None else b,
            _NAN if a is None else a,
            _NAN if b is None or a is None else (a + b) / 2,
            _NAN if b is None or a is None else a - b,
            bids[0][1] if bids else 0,
            asks[0][1] if asks else 0,
            sum(x[1] for x in bids),
            sum(x[1] for x in asks),
            len(book),
            fundamental_ticks,
        ))

    def samples(self) -> dict[str, np.ndarray]:
        arr = np.array(self._rows, dtype=float).reshape(-1, len(SAMPLE_COLUMNS))
        return {c: arr[:, i] for i, c in enumerate(SAMPLE_COLUMNS)}

    def trades(self) -> dict[str, np.ndarray]:
        cols = ("t_ns", "price", "qty", "aggressor", "taker_owner", "maker_owner")
        arr = np.array(self._trades, dtype=np.int64).reshape(-1, len(cols))
        return {c: arr[:, i] for i, c in enumerate(cols)}
