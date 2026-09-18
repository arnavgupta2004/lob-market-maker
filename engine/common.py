"""Shared primitives for every order-book implementation (Python reference and C++).

Conventions
-----------
* Prices are **integer ticks** (``price >= 1``); the tick size in currency units is a
  simulator concern, never an engine concern. This removes floating-point ambiguity from
  price-level identity and makes Python/C++ comparison exact.
* Quantities are **integer lots** (``qty >= 1``).
* Timestamps are **integer nanoseconds** supplied by the caller. The engine never reads a
  wall clock; time priority is decided by queue order, not by timestamp values.
* Order ids are caller-supplied integers and are unique over the *lifetime* of a book.
"""
from __future__ import annotations

from enum import Enum, IntEnum


class Side(IntEnum):
    """Order side. The integer value is the signed direction of the resulting position."""

    BUY = 1
    SELL = -1

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY

    @property
    def code(self) -> str:
        return "B" if self is Side.BUY else "S"

    @staticmethod
    def from_code(code: str) -> "Side":
        if code == "B":
            return Side.BUY
        if code == "S":
            return Side.SELL
        raise ValueError(f"bad side code {code!r}")


class EventType(str, Enum):
    """State-transition event kinds emitted by the engine.

    ``ADD``      a new order came to rest in the book (after any aggressive fills)
    ``TRADE``    an incoming (taker) order executed against a resting (maker) order
    ``CANCEL``   a resting order was removed
    ``MODIFY``   a resting order was amended (in-place reduce, or requeue at a new price/size)
    ``EXPIRE``   the unfilled remainder of a market/IOC order was discarded (no book effect)
    ``REJECT``   a command was refused (no book effect)
    ``SNAPSHOT`` full L3 state dump (no book effect; used as a replay checkpoint)
    """

    ADD = "ADD"
    TRADE = "TRADE"
    CANCEL = "CANCEL"
    MODIFY = "MODIFY"
    EXPIRE = "EXPIRE"
    REJECT = "REJECT"
    SNAPSHOT = "SNAPSHOT"


# Reason strings are part of the wire format; both engines must emit exactly these.
class Reason:
    USER = "USER"  # CANCEL: explicit cancel command
    REPLACE = "REPLACE"  # CANCEL: modify that crosses is executed as cancel + fresh arrival
    IOC = "IOC"  # EXPIRE: unfilled IOC remainder
    MARKET_EXHAUSTED = "MARKET_EXHAUSTED"  # EXPIRE: market order ran out of opposing liquidity
    DUPLICATE_ID = "DUPLICATE_ID"  # REJECT
    UNKNOWN_ORDER = "UNKNOWN_ORDER"  # REJECT (also: already filled / already cancelled)
    INVALID = "INVALID"  # REJECT: non-positive qty/price, bad flag combination
    POST_ONLY_WOULD_CROSS = "POST_ONLY_WOULD_CROSS"  # REJECT
    NO_CHANGE = "NO_CHANGE"  # REJECT: modify identical to current state
