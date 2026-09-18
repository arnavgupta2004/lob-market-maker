"""Order node: the unit stored in a price level's FIFO queue and in the order-id index."""
from __future__ import annotations

from typing import Optional

from engine.common import Side


class OrderNode:
    """A resting order. It is simultaneously a node of its level's intrusive doubly linked
    list (``prev``/``next``) and the value in the order-id index, so an arbitrary order can be
    unlinked in O(1) once looked up by id."""

    __slots__ = ("order_id", "side", "price", "qty", "owner", "post_only", "prev", "next", "level")

    def __init__(self, order_id: int, side: Side, price: int, qty: int, owner: int, post_only: bool):
        self.order_id = order_id
        self.side = side
        self.price = price
        self.qty = qty  # remaining (unfilled) quantity
        self.owner = owner
        self.post_only = post_only
        self.prev: Optional["OrderNode"] = None
        self.next: Optional["OrderNode"] = None
        self.level = None  # owning PriceLevel while resting, else None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Order(id={self.order_id}, {self.side.name}, {self.qty}@{self.price})"
