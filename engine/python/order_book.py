"""Price-time-priority limit order book (Python reference implementation).

Architecture::

    BookSide.levels : price    -> PriceLevel -> FIFO (intrusive linked list of OrderNode)
    OrderBook._index: order_id -> OrderNode   (O(1) cancel / modify / queue-position lookup)

Semantics (identical in the C++ engine; differential tests enforce this):

* **Limit order** – marketable part executes immediately at the *resting* price (price
  improvement goes to the taker); any remainder rests at the back of its level (GTC), or is
  dropped with an ``EXPIRE`` event if ``ioc``. ``post_only`` orders that would cross are
  rejected. Because marketable orders always match before resting, the book can never be
  crossed.
* **Market order** – sweeps levels best-to-worst; the unfilled remainder is discarded
  (``EXPIRE``/``MARKET_EXHAUSTED``).
* **Cancel** – removes a resting order; unknown/filled ids are rejected.
* **Modify (cancel-replace)** – a same-price size *decrease* is applied in place and keeps
  queue priority. Anything else (price change or size increase) loses priority: the order
  moves to the back of the (new) level. If the new price is marketable, the modify is executed
  as ``CANCEL(REPLACE)`` followed by a fresh aggressive arrival with the same id; a post-only
  order whose modify would cross is rejected and left untouched.
* **Ids** – unique over the lifetime of the book (a consumed id is never reusable, even after
  the order is filled/cancelled/rejected). This keeps event streams unambiguous.
* **No self-trade prevention** – owners are recorded but never block a match.

Every state transition emits :class:`~engine.events.Event` objects; see ``apply_event`` for
the inverse (event -> state) used by replay.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional

from engine.commands import (
    Cancel, Command, Modify, NewLimit, NewMarket, SnapshotRequest,
)
from engine.common import EventType, Reason, Side
from engine.events import Event
from engine.python.matching import match
from engine.python.orders import OrderNode
from engine.python.price_levels import BookSide


class ReplayError(Exception):
    """An event is inconsistent with the book state it is being applied to."""


class OrderBook:
    """Single-instrument limit order book. Not thread-safe."""

    def __init__(self, record_events: bool = False):
        self.bids = BookSide(Side.BUY)
        self.asks = BookSide(Side.SELL)
        self._index: dict[int, OrderNode] = {}
        self._seen: set[int] = set()
        self._seq = 0
        self._record = record_events
        self.events: list[Event] = []  # populated only when record_events=True
        self._listeners: list[Callable[[Event], None]] = []
        self._out: list[Event] = []

    # ------------------------------------------------------------------ events
    def subscribe(self, fn: Callable[[Event], None]) -> None:
        """Register a callback invoked synchronously for every emitted event."""
        self._listeners.append(fn)

    def _emit(self, ts: int, etype: EventType, **kw) -> Event:
        self._seq += 1
        ev = Event(self._seq, ts, etype, **kw)
        self._out.append(ev)
        if self._record:
            self.events.append(ev)
        for fn in self._listeners:
            fn(ev)
        return ev

    @property
    def seq(self) -> int:
        """Sequence number of the last emitted event (0 if none)."""
        return self._seq

    # --------------------------------------------------------------- commands
    def process(self, cmd: Command) -> list[Event]:
        """Dispatch a command object; returns the events it produced."""
        t = type(cmd)
        if t is NewLimit:
            return self.submit_limit(cmd.order_id, cmd.side, cmd.price, cmd.qty, cmd.ts,
                                     cmd.owner, cmd.post_only, cmd.ioc)
        if t is NewMarket:
            return self.submit_market(cmd.order_id, cmd.side, cmd.qty, cmd.ts, cmd.owner)
        if t is Cancel:
            return self.cancel(cmd.order_id, cmd.ts)
        if t is Modify:
            return self.modify(cmd.order_id, cmd.new_price, cmd.new_qty, cmd.ts)
        if t is SnapshotRequest:
            return [self.snapshot(cmd.ts)]
        raise TypeError(f"unknown command {cmd!r}")

    def process_all(self, cmds: Iterable[Command]) -> None:
        for c in cmds:
            self.process(c)

    def submit_limit(self, order_id: int, side: Side, price: int, qty: int, ts: int = 0,
                     owner: int = 0, post_only: bool = False, ioc: bool = False) -> list[Event]:
        """Submit a limit order. Returns the events emitted (possibly a single REJECT)."""
        self._out = []
        if order_id in self._seen:
            self._reject(ts, order_id, Reason.DUPLICATE_ID, side, price, qty, owner)
            return self._out
        self._seen.add(order_id)
        if qty < 1 or price < 1 or (post_only and ioc):
            self._reject(ts, order_id, Reason.INVALID, side, price, qty, owner)
            return self._out
        if post_only and self._would_cross(side, price):
            self._reject(ts, order_id, Reason.POST_ONLY_WOULD_CROSS, side, price, qty, owner)
            return self._out
        remaining = self._execute(side, price, qty, order_id, owner, ts)
        if remaining > 0:
            if ioc:
                self._emit(ts, EventType.EXPIRE, order_id=order_id, side=side, price=price,
                           qty=remaining, owner=owner, reason=Reason.IOC)
            else:
                self._rest(OrderNode(order_id, side, price, remaining, owner, post_only))
                self._emit(ts, EventType.ADD, order_id=order_id, side=side, price=price,
                           qty=remaining, owner=owner, post_only=post_only)
        return self._out

    def submit_market(self, order_id: int, side: Side, qty: int, ts: int = 0,
                      owner: int = 0) -> list[Event]:
        """Submit a market order; any unfilled remainder is discarded."""
        self._out = []
        if order_id in self._seen:
            self._reject(ts, order_id, Reason.DUPLICATE_ID, side, 0, qty, owner)
            return self._out
        self._seen.add(order_id)
        if qty < 1:
            self._reject(ts, order_id, Reason.INVALID, side, 0, qty, owner)
            return self._out
        remaining = self._execute(side, None, qty, order_id, owner, ts)
        if remaining > 0:
            self._emit(ts, EventType.EXPIRE, order_id=order_id, side=side, qty=remaining,
                       owner=owner, reason=Reason.MARKET_EXHAUSTED)
        return self._out

    def cancel(self, order_id: int, ts: int = 0) -> list[Event]:
        """Cancel a resting order."""
        self._out = []
        node = self._index.get(order_id)
        if node is None:
            self._emit(ts, EventType.REJECT, order_id=order_id, reason=Reason.UNKNOWN_ORDER)
            return self._out
        self._unrest(node)
        self._emit(ts, EventType.CANCEL, order_id=order_id, side=node.side, price=node.price,
                   qty=node.qty, owner=node.owner, reason=Reason.USER)
        return self._out

    def modify(self, order_id: int, new_price: int, new_qty: int, ts: int = 0) -> list[Event]:
        """Cancel-replace a resting order (see module docstring for priority rules)."""
        self._out = []
        node = self._index.get(order_id)
        if node is None:
            self._emit(ts, EventType.REJECT, order_id=order_id, price=new_price, qty=new_qty,
                       reason=Reason.UNKNOWN_ORDER)
            return self._out
        side, owner = node.side, node.owner
        if new_qty < 1 or new_price < 1:
            self._reject(ts, order_id, Reason.INVALID, side, new_price, new_qty, owner)
            return self._out
        if new_price == node.price and new_qty == node.qty:
            self._reject(ts, order_id, Reason.NO_CHANGE, side, new_price, new_qty, owner)
            return self._out
        old_price, old_qty = node.price, node.qty
        if new_price == old_price and new_qty < old_qty:
            node.level.reduce(node, old_qty - new_qty)  # in place: keeps queue priority
            self._emit(ts, EventType.MODIFY, order_id=order_id, side=side, price=old_price,
                       qty=old_qty, owner=owner, new_price=new_price, new_qty=new_qty,
                       keeps_priority=True, post_only=node.post_only)
            return self._out
        if self._would_cross(side, new_price):
            if node.post_only:
                self._reject(ts, order_id, Reason.POST_ONLY_WOULD_CROSS, side, new_price,
                             new_qty, owner)
                return self._out
            self._unrest(node)
            self._emit(ts, EventType.CANCEL, order_id=order_id, side=side, price=old_price,
                       qty=old_qty, owner=owner, reason=Reason.REPLACE)
            remaining = self._execute(side, new_price, new_qty, order_id, owner, ts)
            if remaining > 0:
                self._rest(OrderNode(order_id, side, new_price, remaining, owner, False))
                self._emit(ts, EventType.ADD, order_id=order_id, side=side, price=new_price,
                           qty=remaining, owner=owner)
            return self._out
        post_only = node.post_only
        self._unrest(node)
        self._rest(OrderNode(order_id, side, new_price, new_qty, owner, post_only))
        self._emit(ts, EventType.MODIFY, order_id=order_id, side=side, price=old_price,
                   qty=old_qty, owner=owner, new_price=new_price, new_qty=new_qty,
                   keeps_priority=False, post_only=post_only)
        return self._out

    def snapshot(self, ts: int = 0) -> Event:
        """Emit a SNAPSHOT event carrying the full L3 state."""
        self._out = []
        return self._emit(ts, EventType.SNAPSHOT, book=self.state())

    # -------------------------------------------------------------- internals
    def _reject(self, ts: int, order_id: int, reason: str, side: Optional[Side], price: int,
                qty: int, owner: int) -> None:
        self._emit(ts, EventType.REJECT, order_id=order_id, side=side, price=price, qty=qty,
                   owner=owner, reason=reason)

    def _would_cross(self, side: Side, price: int) -> bool:
        if side is Side.BUY:
            ask = self.asks.best_price()
            return ask is not None and price >= ask
        bid = self.bids.best_price()
        return bid is not None and price <= bid

    def _execute(self, side: Side, limit_price: Optional[int], qty: int, taker_id: int,
                 taker_owner: int, ts: int) -> int:
        def emit_trade(maker: OrderNode, fill: int, remaining: int) -> None:
            self._emit(ts, EventType.TRADE, order_id=taker_id, side=side, price=maker.price,
                       qty=fill, owner=taker_owner, maker_id=maker.order_id,
                       maker_owner=maker.owner, maker_remaining=remaining)

        opposite = self.asks if side is Side.BUY else self.bids
        return match(opposite, side, limit_price, qty, self._index, emit_trade)

    def _side(self, side: Side) -> BookSide:
        return self.bids if side is Side.BUY else self.asks

    def _rest(self, node: OrderNode) -> None:
        self._side(node.side).get_or_create(node.price).push_back(node)
        self._index[node.order_id] = node

    def _unrest(self, node: OrderNode) -> None:
        level = node.level
        level.remove(node)
        if level.count == 0:
            self._side(node.side).remove_level(level.price)
        del self._index[node.order_id]

    # ---------------------------------------------------------------- queries
    def __len__(self) -> int:
        """Number of resting orders."""
        return len(self._index)

    def __contains__(self, order_id: int) -> bool:
        return order_id in self._index

    def get_order(self, order_id: int) -> Optional[OrderNode]:
        return self._index.get(order_id)

    def best_bid(self) -> Optional[int]:
        return self.bids.best_price()

    def best_ask(self) -> Optional[int]:
        return self.asks.best_price()

    def spread(self) -> Optional[int]:
        """Best ask minus best bid in ticks (None if either side is empty)."""
        b, a = self.bids.best_price(), self.asks.best_price()
        return None if b is None or a is None else a - b

    def mid_price(self) -> Optional[float]:
        """(best bid + best ask) / 2 in ticks (None if either side is empty)."""
        b, a = self.bids.best_price(), self.asks.best_price()
        return None if b is None or a is None else (a + b) / 2

    def depth(self, levels: Optional[int] = None) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
        """L2 view: ``(bids, asks)``, each a best-first list of ``(price, total_qty, n_orders)``."""
        def side_view(bs: BookSide) -> list[tuple[int, int, int]]:
            out = []
            for lvl in bs.iter_levels():
                if levels is not None and len(out) >= levels:
                    break
                out.append((lvl.price, lvl.total_qty, lvl.count))
            return out
        return side_view(self.bids), side_view(self.asks)

    def volume(self, side: Side, levels: int) -> int:
        """Total resting quantity in the best ``levels`` price levels of ``side``."""
        total = 0
        for i, lvl in enumerate(self._side(side).iter_levels()):
            if i >= levels:
                break
            total += lvl.total_qty
        return total

    def imbalance(self, levels: int = 1) -> Optional[float]:
        """Order-book imbalance ``(V_b - V_a) / (V_b + V_a)`` over the best ``levels`` levels."""
        vb, va = self.volume(Side.BUY, levels), self.volume(Side.SELL, levels)
        return None if vb + va == 0 else (vb - va) / (vb + va)

    def level_qty(self, side: Side, price: int) -> int:
        """Total resting quantity at ``price`` on ``side`` (0 if no such level)."""
        lvl = self._side(side).levels.get(price)
        return lvl.total_qty if lvl is not None else 0

    def queue_position(self, order_id: int) -> Optional[tuple[int, int]]:
        """``(quantity_ahead, orders_ahead)`` of a resting order in its level's FIFO. O(position)."""
        node = self._index.get(order_id)
        if node is None:
            return None
        qty_ahead = orders_ahead = 0
        n = node.prev
        while n is not None:
            qty_ahead += n.qty
            orders_ahead += 1
            n = n.prev
        return qty_ahead, orders_ahead

    def state(self) -> dict:
        """Canonical L3 state. Levels are best-first; orders are in FIFO order::

            {"bids": [[price, [[order_id, qty, owner, post_only], ...]], ...], "asks": [...]}
        """
        def side_state(bs: BookSide) -> list:
            return [[lvl.price, [[n.order_id, n.qty, n.owner, int(n.post_only)] for n in lvl]]
                    for lvl in bs.iter_levels()]
        return {"bids": side_state(self.bids), "asks": side_state(self.asks)}

    def load_state(self, state: dict) -> None:
        """Replace the book contents with a canonical ``state`` (e.g. from a SNAPSHOT)."""
        self.__init__(self._record)  # type: ignore[misc]
        for side, key in ((Side.BUY, "bids"), (Side.SELL, "asks")):
            for price, orders in state[key]:
                for oid, qty, owner, post_only in orders:
                    self._rest(OrderNode(oid, side, price, qty, owner, bool(post_only)))
                    self._seen.add(oid)

    # ----------------------------------------------------------------- replay
    def apply_event(self, ev: Event) -> None:
        """Apply a recorded event as a pure state delta (no matching is performed).

        Every consistency condition that the event asserts about the pre-state (maker price,
        remaining quantity, cancelled quantity, ...) is verified; a mismatch raises
        :class:`ReplayError`. Successful replay of a full stream therefore *proves* the stream
        is sufficient to reproduce, and is consistent with, the book.
        """
        if ev.seq <= self._seq:
            raise ReplayError(f"non-monotonic seq {ev.seq} after {self._seq}")
        self._seq = ev.seq
        t = ev.type
        if t is EventType.ADD:
            if ev.order_id in self._index:
                raise ReplayError(f"ADD of live order {ev.order_id}")
            self._rest(OrderNode(ev.order_id, ev.side, ev.price, ev.qty, ev.owner, ev.post_only))
            self._seen.add(ev.order_id)
        elif t is EventType.TRADE:
            maker = self._index.get(ev.maker_id)
            if maker is None:
                raise ReplayError(f"TRADE against unknown maker {ev.maker_id}")
            if (maker.price != ev.price or maker.side is ev.side
                    or maker.qty - ev.qty != ev.maker_remaining or ev.qty < 1):
                raise ReplayError(f"TRADE inconsistent with maker state: {ev} vs {maker}")
            if ev.maker_remaining == 0:
                self._unrest(maker)
            else:
                maker.level.reduce(maker, ev.qty)
            self._seen.add(ev.order_id)
        elif t is EventType.CANCEL:
            node = self._index.get(ev.order_id)
            if node is None or node.price != ev.price or node.qty != ev.qty:
                raise ReplayError(f"CANCEL inconsistent with state: {ev} vs {node}")
            self._unrest(node)
        elif t is EventType.MODIFY:
            node = self._index.get(ev.order_id)
            if node is None or node.price != ev.price or node.qty != ev.qty:
                raise ReplayError(f"MODIFY inconsistent with state: {ev} vs {node}")
            if ev.keeps_priority:
                if ev.new_price != node.price or ev.new_qty >= node.qty:
                    raise ReplayError(f"bad in-place MODIFY: {ev}")
                node.level.reduce(node, node.qty - ev.new_qty)
            else:
                self._unrest(node)
                self._rest(OrderNode(ev.order_id, ev.side, ev.new_price, ev.new_qty, ev.owner,
                                     ev.post_only))
        elif t is EventType.EXPIRE:
            self._seen.add(ev.order_id)
        elif t is EventType.REJECT:
            if ev.reason in (Reason.INVALID, Reason.POST_ONLY_WOULD_CROSS):
                self._seen.add(ev.order_id)
        elif t is EventType.SNAPSHOT:
            if self.state() != ev.book:
                raise ReplayError(f"SNAPSHOT at seq {ev.seq} does not match replayed state")

    # -------------------------------------------------------------- invariants
    def validate(self) -> None:
        """Check every structural invariant; raises ``AssertionError`` on the first violation."""
        counted = 0
        for bs in (self.bids, self.asks):
            keys = bs._keys
            assert all(keys[i] < keys[i + 1] for i in range(len(keys) - 1)), "keys not sorted"
            assert {bs._sign * k for k in keys} == set(bs.levels), "keys/levels mismatch"
            for price, lvl in bs.levels.items():
                assert lvl.price == price and lvl.count > 0, "empty or mislabeled level"
                total = n_orders = 0
                prev = None
                for n in lvl:
                    assert n.level is lvl and n.price == price and n.side is bs.side
                    assert n.qty > 0, "non-positive resting qty"
                    assert n.prev is prev, "broken prev link"
                    assert self._index.get(n.order_id) is n, "index mismatch"
                    total += n.qty
                    n_orders += 1
                    prev = n
                assert lvl.tail is prev, "bad tail"
                assert total == lvl.total_qty and n_orders == lvl.count, "level aggregates stale"
                counted += n_orders
        assert counted == len(self._index), "index has orphans"
        assert self._index.keys() <= self._seen, "live id missing from seen-set"
        b, a = self.bids.best_price(), self.asks.best_price()
        assert b is None or a is None or b < a, f"crossed book: bid {b} >= ask {a}"
