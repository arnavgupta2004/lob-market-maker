"""Deterministic replay of command streams and event streams.

Two distinct guarantees are provided:

* :func:`replay_commands` re-executes *inputs* (matching included). Same commands => same
  event stream (byte-identical, see ``events.stream_hash``).
* :func:`replay_events` re-applies *outputs* as state deltas without matching. It
  reconstructs the book purely from the audit trail and verifies every consistency condition
  the events assert.
"""
from __future__ import annotations

from typing import Callable, Iterable

from engine.commands import Command
from engine.events import Event
from engine.python.order_book import OrderBook


def replay_commands(cmds: Iterable[Command], book_factory: Callable[[], OrderBook] = lambda: OrderBook(record_events=True)) -> OrderBook:
    """Feed ``cmds`` to a fresh book (event recording on by default) and return it."""
    book = book_factory()
    book.process_all(cmds)
    return book


def replay_events(events: Iterable[Event], book: OrderBook | None = None) -> OrderBook:
    """Rebuild a book by applying ``events`` in order. Raises ``ReplayError`` on inconsistency."""
    book = book if book is not None else OrderBook()
    for ev in events:
        book.apply_event(ev)
    return book
