"""Price-time-priority matching.

The matcher walks the opposite side from the best price inwards. Within a level it consumes
the FIFO queue from the head. Each execution happens at the *maker's* (resting) price.
Cost is O(k) for k consumed orders plus O(1) per fully consumed level.
"""
from __future__ import annotations

from typing import Callable, Optional

from engine.common import Side
from engine.python.price_levels import BookSide

# emit_trade(maker_node, executed_qty, maker_remaining_after)
TradeEmit = Callable[["object", int, int], None]


def match(
    opposite: BookSide,
    taker_side: Side,
    limit_price: Optional[int],
    qty: int,
    index: dict,
    emit_trade: TradeEmit,
) -> int:
    """Execute up to ``qty`` against ``opposite``; return the unfilled remainder.

    ``limit_price=None`` means a market order (no price bound). Filled makers are removed from
    ``index`` and emptied levels are removed from ``opposite``; partially filled makers keep
    their queue position.
    """
    while qty > 0:
        level = opposite.best_level()
        if level is None:
            break
        if limit_price is not None:
            if taker_side is Side.BUY and level.price > limit_price:
                break
            if taker_side is Side.SELL and level.price < limit_price:
                break
        while qty > 0 and level.head is not None:
            maker = level.head
            fill = qty if qty < maker.qty else maker.qty
            remaining = maker.qty - fill
            emit_trade(maker, fill, remaining)  # emitted before mutation so maker.qty is pre-fill
            qty -= fill
            if remaining == 0:
                level.remove(maker)
                del index[maker.order_id]
            else:
                level.reduce(maker, fill)
        if level.count == 0:
            opposite.remove_level(level.price)
    return qty
