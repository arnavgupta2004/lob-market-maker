"""Participant interface: the single plug-in point for order flow and strategies.

A participant is anything that reacts to time (its own scheduled wake-ups) and/or to market
events, and emits :mod:`engine.commands`. Exogenous synthetic flow, historical replay and
market-making agents all implement this same interface, so the same strategy can be run
against any of them (spec section 6).

Contract with the simulator:

* ``next_time()`` returns the absolute time (ns) of the next self-scheduled wake-up, or ``None``.
  The simulator re-queries it after every callback.
* Commands returned from callbacks are sent to the exchange and arrive after the participant's
  ``latency.order`` delay.
* ``feed`` selects what the participant is notified about (after ``latency.feed``):
  ``"none"``, ``"own"`` (events where it is taker or maker) or ``"all"`` (full public feed).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from engine.commands import Command
from engine.events import Event
from simulator.latency.models import ZERO_LATENCY, LatencyConfig

if TYPE_CHECKING:  # pragma: no cover
    from simulator.simulator import Simulator


class Participant(ABC):
    name: str = "participant"
    owner_id: int = 0  # assigned by the simulator at registration
    latency: LatencyConfig = ZERO_LATENCY
    feed: str = "none"  # "none" | "own" | "all"

    def on_start(self, sim: "Simulator") -> None:
        """Called once before the run starts (obtain RNG streams, schedule first wake-up)."""

    @abstractmethod
    def next_time(self) -> Optional[int]:
        """Absolute ns of the next self-scheduled wake-up, or ``None``."""

    @abstractmethod
    def on_wakeup(self, sim: "Simulator", now: int) -> list[Command]:
        """Scheduled wake-up at ``now``; return commands to send."""

    def on_events(self, sim: "Simulator", now: int, events: list[Event]) -> list[Command]:
        """Delayed notification of exchange events (see ``feed``); may return commands."""
        return []
