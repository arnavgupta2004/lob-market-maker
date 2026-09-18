"""Python-facing wrapper of the C++ order book (``engine._lob_cpp``).

:class:`CppOrderBook` exposes the same public API as :class:`engine.python.order_book.OrderBook`
(commands, events, queries, ``state()``), so any code written against the reference engine - the
simulator, market makers, research modules, tests - runs unchanged on the C++ engine. Events are
converted back into :class:`engine.events.Event` objects, which is what makes *field-by-field* differential
testing against the Python engine possible.

Batch interface (``run_batch``) executes a packed command array entirely in C++; the array/event layouts
below are shared with ``engine/cpp/bindings.cpp``.
"""
from __future__ import annotations

from typing import Callable, Iterable, NamedTuple, Optional

import numpy as np

from engine.commands import Cancel, Command, Modify, NewLimit, NewMarket, SnapshotRequest
from engine.common import EventType, Reason, Side
from engine.events import Event

try:  # the extension is built by scripts/build_cpp.sh
    from engine import _lob_cpp
except ImportError:  # pragma: no cover - exercised only when the extension is absent
    _lob_cpp = None


def available() -> bool:
    """True if the compiled extension can be imported."""
    return _lob_cpp is not None


# ---- wire-format codes (must match engine/cpp/types.hpp) -----------------------------------------------
TYPE_BY_CODE = [EventType.ADD, EventType.TRADE, EventType.CANCEL, EventType.MODIFY, EventType.EXPIRE,
                EventType.REJECT, EventType.SNAPSHOT]
TYPE_CODE = {t: i for i, t in enumerate(TYPE_BY_CODE)}
REASON_BY_CODE = ["", Reason.USER, Reason.REPLACE, Reason.IOC, Reason.MARKET_EXHAUSTED, Reason.DUPLICATE_ID,
                  Reason.UNKNOWN_ORDER, Reason.INVALID, Reason.POST_ONLY_WOULD_CROSS, Reason.NO_CHANGE]
REASON_CODE = {r: i for i, r in enumerate(REASON_BY_CODE)}
OP_LIMIT, OP_MARKET, OP_CANCEL, OP_MODIFY, OP_SNAPSHOT = range(5)
COMMAND_COLS = ("op", "order_id", "side", "price", "qty", "ts", "owner", "flags")
EVENT_COLS = ("seq", "ts", "type", "order_id", "side", "price", "qty", "owner", "maker_id", "maker_owner",
              "maker_remaining", "new_price", "new_qty", "flags", "reason")


class OrderView(NamedTuple):
    """Read-only view of a resting order (the Python engine returns its live node instead)."""

    order_id: int
    side: Side
    price: int
    qty: int
    owner: int
    post_only: bool


def _norm_state(d: dict) -> dict:
    """C++ tuples -> the reference engine's nested lists, so ``state()`` values compare equal."""
    return {k: [[p, [list(o) for o in orders]] for p, orders in d[k]] for k in ("bids", "asks")}


def _to_event(t: tuple) -> Event:
    return Event(
        seq=t[0], ts=t[1], type=TYPE_BY_CODE[t[2]], order_id=t[3], side=Side(t[4]) if t[4] else None, price=t[5],
        qty=t[6], owner=t[7], maker_id=t[8], maker_owner=t[9], maker_remaining=t[10], new_price=t[11], new_qty=t[12],
        keeps_priority=bool(t[13] & 1), post_only=bool(t[13] & 2), reason=REASON_BY_CODE[t[14]],
        book=_norm_state(t[15]) if t[15] is not None else None)


class CppOrderBook:
    """Drop-in replacement for the Python ``OrderBook`` backed by the C++ engine."""

    def __init__(self, record_events: bool = False, reserve_orders: int = 0):
        if _lob_cpp is None:
            raise RuntimeError("C++ extension not built: run scripts/build_cpp.sh")
        self._b = _lob_cpp.OrderBook(reserve_orders)
        self._record = record_events
        self.events: list[Event] = []
        self._listeners: list[Callable[[Event], None]] = []

    # ---------------------------------------------------------------- events
    def subscribe(self, fn: Callable[[Event], None]) -> None:
        self._listeners.append(fn)

    @property
    def seq(self) -> int:
        return self._b.seq

    def _run(self, op: int, order_id: int = 0, side: int = 0, price: int = 0, qty: int = 0, ts: int = 0,
             owner: int = 0, flags: int = 0) -> list[Event]:
        events = [_to_event(t) for t in self._b.process_raw(op, order_id, side, price, qty, ts, owner, flags)]
        if self._record:
            self.events.extend(events)
        for e in events:
            for fn in self._listeners:
                fn(e)
        return events

    # -------------------------------------------------------------- commands
    def process(self, cmd: Command) -> list[Event]:
        t = type(cmd)
        if t is NewLimit:
            return self.submit_limit(cmd.order_id, cmd.side, cmd.price, cmd.qty, cmd.ts, cmd.owner, cmd.post_only, cmd.ioc)
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

    def submit_limit(self, order_id: int, side: Side, price: int, qty: int, ts: int = 0, owner: int = 0,
                     post_only: bool = False, ioc: bool = False) -> list[Event]:
        return self._run(OP_LIMIT, order_id, int(side), price, qty, ts, owner, int(post_only) | (int(ioc) << 1))

    def submit_market(self, order_id: int, side: Side, qty: int, ts: int = 0, owner: int = 0) -> list[Event]:
        return self._run(OP_MARKET, order_id, int(side), 0, qty, ts, owner)

    def cancel(self, order_id: int, ts: int = 0) -> list[Event]:
        return self._run(OP_CANCEL, order_id, ts=ts)

    def modify(self, order_id: int, new_price: int, new_qty: int, ts: int = 0) -> list[Event]:
        return self._run(OP_MODIFY, order_id, 0, new_price, new_qty, ts)

    def snapshot(self, ts: int = 0) -> Event:
        return self._run(OP_SNAPSHOT, ts=ts)[0]

    # ---------------------------------------------------------------- queries
    def __len__(self) -> int:
        return len(self._b)

    def __contains__(self, order_id: int) -> bool:
        return self._b.contains(order_id)

    def get_order(self, order_id: int) -> Optional[OrderView]:
        o = self._b.order_info(order_id)
        return None if o is None else OrderView(order_id, Side(o[0]), o[1], o[2], o[3], bool(o[4]))

    def best_bid(self) -> Optional[int]:
        return self._b.best_bid()

    def best_ask(self) -> Optional[int]:
        return self._b.best_ask()

    def spread(self) -> Optional[int]:
        b, a = self._b.best_bid(), self._b.best_ask()
        return None if b is None or a is None else a - b

    def mid_price(self) -> Optional[float]:
        b, a = self._b.best_bid(), self._b.best_ask()
        return None if b is None or a is None else (a + b) / 2

    def depth(self, levels: Optional[int] = None):
        n = 0 if levels is None else levels
        return self._b.depth(int(Side.BUY), n), self._b.depth(int(Side.SELL), n)

    def volume(self, side: Side, levels: int) -> int:
        return self._b.volume(int(side), levels)

    def imbalance(self, levels: int = 1) -> Optional[float]:
        vb, va = self.volume(Side.BUY, levels), self.volume(Side.SELL, levels)
        return None if vb + va == 0 else (vb - va) / (vb + va)

    def level_qty(self, side: Side, price: int) -> int:
        return self._b.level_qty(int(side), price)

    def owner_orders_at(self, side: Side, price: int, owner: int) -> list[tuple[int, int]]:
        return self._b.owner_orders_at(int(side), price, owner)

    def queue_position(self, order_id: int) -> Optional[tuple[int, int]]:
        return self._b.queue_position(order_id)

    def state(self) -> dict:
        return _norm_state(self._b.state())

    def validate(self) -> None:
        self._b.validate()

    # ------------------------------------------------------------------ batch
    def run_batch(self, commands: np.ndarray, collect_events: bool = False, checksum: bool = False,
                  timed: bool = False) -> dict:
        """Execute a packed ``(n, 8)`` int64 command array in C++ (see :func:`commands_to_array`)."""
        return self._b.run_batch(np.ascontiguousarray(commands, dtype=np.int64), collect_events, checksum, timed)


# ------------------------------------------------------------------- array codecs & checksum
def commands_to_array(cmds: Iterable[Command]) -> np.ndarray:
    """Pack commands into the ``(n, 8)`` int64 layout of :data:`COMMAND_COLS`."""
    rows = []
    for c in cmds:
        t = type(c)
        if t is NewLimit:
            rows.append((OP_LIMIT, c.order_id, int(c.side), c.price, c.qty, c.ts, c.owner, int(c.post_only) | (int(c.ioc) << 1)))
        elif t is NewMarket:
            rows.append((OP_MARKET, c.order_id, int(c.side), 0, c.qty, c.ts, c.owner, 0))
        elif t is Cancel:
            rows.append((OP_CANCEL, c.order_id, 0, 0, 0, c.ts, 0, 0))
        elif t is Modify:
            rows.append((OP_MODIFY, c.order_id, 0, c.new_price, c.new_qty, c.ts, 0, 0))
        elif t is SnapshotRequest:
            rows.append((OP_SNAPSHOT, 0, 0, 0, 0, c.ts, 0, 0))
        else:
            raise TypeError(f"unknown command {c!r}")
    return np.array(rows, dtype=np.int64).reshape(-1, len(COMMAND_COLS))


def events_to_array(events: Iterable[Event]) -> np.ndarray:
    """Pack Python events into the ``(n, 15)`` layout of :data:`EVENT_COLS` (SNAPSHOT payload dropped)."""
    rows = [(e.seq, e.ts, TYPE_CODE[e.type], e.order_id, int(e.side or 0), e.price, e.qty, e.owner, e.maker_id,
             e.maker_owner, e.maker_remaining, e.new_price, e.new_qty, int(e.keeps_priority) | (int(e.post_only) << 1),
             REASON_CODE[e.reason]) for e in events]
    return np.array(rows, dtype=np.int64).reshape(-1, len(EVENT_COLS))


_FNV_OFFSET = 1469598103934665603
_FNV_PRIME = 0x100000001B3
_MASK = (1 << 64) - 1


def events_checksum(arr: np.ndarray) -> int:
    """Order-sensitive Horner hash ``h = h * P + w (mod 2^64)`` over every word of an event array, identical to
    the running checksum the C++ engine maintains (``OrderBook::hash_event``). Vectorised with modular powers."""
    w = np.ascontiguousarray(arr, dtype=np.int64).reshape(-1).view(np.uint64)
    n = len(w)
    if n == 0:
        return _FNV_OFFSET
    # h_n = offset * P^n + sum_i w_i * P^(n-1-i)   (mod 2^64)
    pw = np.empty(n, dtype=np.uint64)
    pw[0] = 1
    if n > 1:
        pw[1:] = np.uint64(_FNV_PRIME)
        with np.errstate(over="ignore"):
            pw = np.cumprod(pw, dtype=np.uint64)
    with np.errstate(over="ignore"):
        tail = int(np.sum(w * pw[::-1], dtype=np.uint64))
        head = (_FNV_OFFSET * int(pw[-1]) * _FNV_PRIME) & _MASK
    return (head + tail) & _MASK
