"""Price levels and one side of the book.

``BookSide`` maps ``price -> PriceLevel`` (dict, O(1) lookup) and keeps the active prices in a
sorted list so the best price is always the *last* element:

* bids are stored under key ``price`` (ascending => best = highest price),
* asks are stored under key ``-price`` (ascending => best = lowest price).

Complexity (P = number of active price levels on the side):

* best price / best level ........ O(1)
* insert a new level ............. O(log P) search + O(P) ``list.insert`` memmove
* remove the *best* level ........ O(1) (pop from the tail)  <- the matching hot path
* remove an interior level ....... O(log P) search + O(P) memmove
* append/unlink an order ......... O(1)

The memmove term is a very small constant (contiguous pointer copy) and is negligible for
realistic P; the C++ engine replaces the sorted list with ``std::map`` for a true O(log P).
"""
from __future__ import annotations

from bisect import bisect_left, insort
from typing import Iterator, Optional

from engine.common import Side
from engine.python.orders import OrderNode


class PriceLevel:
    """FIFO queue of resting orders at one price (intrusive doubly linked list)."""

    __slots__ = ("price", "head", "tail", "total_qty", "count")

    def __init__(self, price: int):
        self.price = price
        self.head: Optional[OrderNode] = None
        self.tail: Optional[OrderNode] = None
        self.total_qty = 0
        self.count = 0

    def push_back(self, node: OrderNode) -> None:
        """Append to the back of the queue (lowest time priority). O(1)."""
        node.level = self
        node.prev = self.tail
        node.next = None
        if self.tail is not None:
            self.tail.next = node
        else:
            self.head = node
        self.tail = node
        self.total_qty += node.qty
        self.count += 1

    def remove(self, node: OrderNode) -> None:
        """Unlink ``node`` wherever it is in the queue. O(1)."""
        if node.prev is not None:
            node.prev.next = node.next
        else:
            self.head = node.next
        if node.next is not None:
            node.next.prev = node.prev
        else:
            self.tail = node.prev
        self.total_qty -= node.qty
        self.count -= 1
        node.prev = node.next = None
        node.level = None

    def reduce(self, node: OrderNode, delta: int) -> None:
        """Decrease ``node.qty`` by ``delta`` in place (keeps queue position). O(1)."""
        node.qty -= delta
        self.total_qty -= delta

    def __iter__(self) -> Iterator[OrderNode]:
        n = self.head
        while n is not None:
            yield n
            n = n.next


class BookSide:
    """All price levels of one side of the book."""

    __slots__ = ("side", "levels", "_keys", "_sign")

    def __init__(self, side: Side):
        self.side = side
        self.levels: dict[int, PriceLevel] = {}
        self._keys: list[int] = []  # ascending priority keys; best is last
        self._sign = 1 if side is Side.BUY else -1

    def __len__(self) -> int:
        return len(self.levels)

    def best_price(self) -> Optional[int]:
        return self._sign * self._keys[-1] if self._keys else None

    def best_level(self) -> Optional[PriceLevel]:
        return self.levels[self._sign * self._keys[-1]] if self._keys else None

    def get_or_create(self, price: int) -> PriceLevel:
        lvl = self.levels.get(price)
        if lvl is None:
            lvl = PriceLevel(price)
            self.levels[price] = lvl
            insort(self._keys, self._sign * price)
        return lvl

    def remove_level(self, price: int) -> None:
        """Drop an (empty) level."""
        lvl = self.levels.pop(price)
        assert lvl.count == 0, "removing non-empty price level"
        key = self._sign * price
        keys = self._keys
        if keys and keys[-1] == key:
            keys.pop()
        else:
            del keys[bisect_left(keys, key)]

    def iter_levels(self) -> Iterator[PriceLevel]:
        """Levels from best to worst."""
        for key in reversed(self._keys):
            yield self.levels[self._sign * key]
