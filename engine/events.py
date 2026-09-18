"""Structured, serializable order-book events and the (de)serialization of event streams.

An :class:`Event` is a *state delta*: applying the events of a stream in order to an empty
book (see ``OrderBook.apply_event``) reproduces the book without re-running matching. The
field meaning per event type is:

============ ========================================================================
``ADD``      ``order_id, side, price, qty`` (= resting qty), ``owner, post_only``
``TRADE``    ``order_id``/``side``/``owner`` = taker (aggressor); ``maker_id``,
             ``maker_owner``, ``maker_remaining`` (after the fill); ``price`` = maker's
             price (execution price); ``qty`` = executed quantity
``CANCEL``   ``order_id, side, price, qty`` (= qty removed), ``owner``, ``reason``
``MODIFY``   ``price, qty`` = *old* values; ``new_price, new_qty``; ``keeps_priority``
``EXPIRE``   ``order_id, side, price`` (0 for market), ``qty`` = unfilled, ``reason``
``REJECT``   ``order_id``, ``reason`` (+ submitted ``side/price/qty`` where available)
``SNAPSHOT`` ``book`` = canonical L3 state (see ``OrderBook.state``)
============ ========================================================================
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from engine.common import EventType, Side


@dataclass(frozen=True, slots=True)
class Event:
    """One order-book state transition. See module docstring for per-type field semantics."""

    seq: int
    ts: int
    type: EventType
    order_id: int = 0
    side: Optional[Side] = None
    price: int = 0
    qty: int = 0
    owner: int = 0
    maker_id: int = 0
    maker_owner: int = 0
    maker_remaining: int = 0
    new_price: int = 0
    new_qty: int = 0
    keeps_priority: bool = False
    post_only: bool = False
    reason: str = ""
    book: Optional[dict] = None

    def to_dict(self) -> dict[str, Any]:
        """Sparse dict: default-valued fields are omitted to keep files small."""
        d: dict[str, Any] = {"seq": self.seq, "ts": self.ts, "type": self.type.value}
        for name in _OPTIONAL_FIELDS:
            v = getattr(self, name)
            if v == _DEFAULTS[name]:
                continue
            d[name] = v.code if name == "side" else v
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Event":
        kw: dict[str, Any] = {"seq": d["seq"], "ts": d["ts"], "type": EventType(d["type"])}
        for name in _OPTIONAL_FIELDS:
            if name in d:
                kw[name] = Side.from_code(d[name]) if name == "side" else d[name]
        return Event(**kw)

    def core(self) -> tuple:
        """Comparison key ignoring ``seq``/``ts`` (used when comparing against oracles)."""
        return (
            self.type.value, self.order_id, int(self.side or 0), self.price, self.qty,
            self.owner, self.maker_id, self.maker_owner, self.maker_remaining,
            self.new_price, self.new_qty, self.keeps_priority, self.post_only, self.reason,
        )


_OPTIONAL_FIELDS = [f.name for f in fields(Event) if f.name not in ("seq", "ts", "type")]
_DEFAULTS = {f.name: f.default for f in fields(Event) if f.name in _OPTIONAL_FIELDS}


def dumps(event: Event) -> str:
    """Canonical single-line JSON (sorted keys, no spaces): byte-stable across runs."""
    return json.dumps(event.to_dict(), sort_keys=True, separators=(",", ":"))


def write_jsonl(path: str | Path, events: Iterable[Event]) -> None:
    """Write an event stream as JSON Lines."""
    with open(path, "w", encoding="utf-8") as fh:
        for e in events:
            fh.write(dumps(e))
            fh.write("\n")


def read_jsonl(path: str | Path) -> Iterator[Event]:
    """Lazily read an event stream written by :func:`write_jsonl`."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Event.from_dict(json.loads(line))


def stream_hash(events: Iterable[Event]) -> str:
    """SHA-256 of the canonical serialization: two runs are identical iff hashes match."""
    h = hashlib.sha256()
    for e in events:
        h.update(dumps(e).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()
