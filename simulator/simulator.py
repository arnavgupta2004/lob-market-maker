"""Discrete-event market simulator around the limit order book.

Time is an integer number of nanoseconds. The simulator owns the *true* exchange book and a
future-event queue with four kinds of entries:

``WAKE``    a participant's self-scheduled wake-up
``ARRIVE``  a command reaching the matching engine (after the sender's order latency)
``FEED``    delayed delivery of exchange events to a subscribed participant
``SAMPLE``  fixed-interval market-state recording

Determinism: a run is a pure function of ``(SimConfig, participants)``. Every stochastic
component draws from its own ``numpy`` stream keyed by ``(seed, name)``, so adding a
participant does not perturb the random numbers of existing ones (crucial for paired
comparisons and ablations). Queue ties are broken FIFO.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from engine.commands import Command, NewLimit, NewMarket
from engine.common import Side
from engine.events import Event
from engine.python.order_book import OrderBook
from simulator.events.queue import EventQueue
from simulator.market_state.fundamental import FundamentalConfig, FundamentalProcess
from simulator.market_state.history import MidHistory
from simulator.market_state.recorder import MarketRecorder
from simulator.participant import Participant

NS = 1_000_000_000
_WAKE, _ARRIVE, _FEED, _SAMPLE = 0, 1, 2, 3
SEEDER_OWNER = 0  # owner id of the initial-liquidity orders


@dataclass(frozen=True)
class SimConfig:
    seed: int = 0
    horizon_s: float = 60.0
    tick_size: float = 0.01  # currency units per tick
    mid0_ticks: int = 10_000  # initial reference price (ticks)
    sample_interval_s: float = 0.1
    warmup_s: float = 5.0  # initial transient excluded by ``SimResult.steady``
    seed_levels: int = 3  # initial static depth: this many levels per side...
    seed_size: int = 10  # ...with this many lots each
    seed_gap_ticks: int = 1  # first seeded level is mid0 -/+ seed_gap_ticks
    fundamental: Optional[FundamentalConfig] = None
    record_events: bool = True
    record_commands: bool = True


@dataclass
class SimResult:
    config: SimConfig
    book: OrderBook
    events: list[Event]
    commands: list[Command]
    samples: dict[str, np.ndarray]
    trades: dict[str, np.ndarray]
    mid_history: MidHistory
    n_commands: int
    n_events: int
    participants: dict[str, Participant] = field(default_factory=dict)

    def steady(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """``(samples, trades)`` restricted to ``t >= warmup`` (drops the start-up transient)."""
        w = int(round(self.config.warmup_s * NS))
        ms = self.samples["t_ns"] >= w
        mt = self.trades["t_ns"] >= w
        return ({k: v[ms] for k, v in self.samples.items()},
                {k: v[mt] for k, v in self.trades.items()})


class Simulator:
    """Event-driven simulation of one instrument. Also the ``ctx`` handed to participants."""

    def __init__(self, config: SimConfig, participants: list[Participant]):
        names = [p.name for p in participants]
        if len(set(names)) != len(names):
            raise ValueError(f"participant names must be unique: {names}")
        self.cfg = config
        self.participants = participants
        self.horizon_ns = int(round(config.horizon_s * NS))
        self.book = OrderBook(record_events=config.record_events)
        self.queue = EventQueue()
        self.now = 0
        self.mid_history = MidHistory()
        self.recorder = MarketRecorder()
        self.book.subscribe(self.recorder.on_event)
        self._next_id = 1
        self._commands: list[Command] = []
        self._n_commands = 0
        self._tokens = [0] * len(participants)
        self._sched: list[Optional[int]] = [None] * len(participants)
        self._last_arrival = [0] * len(participants)
        self._last_feed = [0] * len(participants)
        self._latency_rng = self.rng_for("__latency__")
        for i, p in enumerate(participants):
            p.owner_id = i + 1
        self._by_owner = {p.owner_id: i for i, p in enumerate(participants)}
        self._all_feed = [i for i, p in enumerate(participants) if p.feed == "all"]
        self._own_feed = [i for i, p in enumerate(participants) if p.feed == "own"]
        self.fundamental: Optional[FundamentalProcess] = None
        if config.fundamental is not None:
            self.fundamental = FundamentalProcess(
                config.fundamental, config.mid0_ticks * config.tick_size, self.horizon_ns,
                self.rng_for("__fundamental__"))

    # ------------------------------------------------------------ services for participants
    def rng_for(self, name: str) -> np.random.Generator:
        """Independent, reproducible RNG stream for a named component."""
        ss = np.random.SeedSequence(self.cfg.seed, spawn_key=(zlib.crc32(name.encode()),))
        return np.random.Generator(np.random.PCG64(ss))

    def new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def fundamental_ticks(self, t: Optional[int] = None) -> float:
        """Latent value in (fractional) ticks at time ``t`` (default: now); nan if none."""
        if self.fundamental is None:
            return float("nan")
        return self.fundamental.value_at(self.now if t is None else t) / self.cfg.tick_size

    def reference_mid(self) -> float:
        """Current mid in ticks; last valid mid if a side is empty; ``mid0`` before any."""
        m = self.book.mid_price()
        if m is not None:
            return m
        last = self.mid_history.last()
        return float(self.cfg.mid0_ticks) if last is None else last

    # ------------------------------------------------------------------------------- run
    def run(self) -> SimResult:
        self._seed_book()
        for i, p in enumerate(self.participants):
            p.on_start(self)
            self._reschedule(i)
        self.queue.push(0, _SAMPLE)
        horizon = self.horizon_ns
        while len(self.queue):
            t, _, kind, a, b = self.queue.pop()
            if t > horizon:
                break
            self.now = t
            if kind == _ARRIVE:
                self._exchange(a, t)
            elif kind == _WAKE:
                if b != self._tokens[a]:
                    continue  # superseded wake-up
                p = self.participants[a]
                self._sched[a] = None
                self._send(a, p.on_wakeup(self, t), t)
                self._reschedule(a)
            elif kind == _FEED:
                p = self.participants[a]
                self._send(a, p.on_events(self, t, b), t)
                self._reschedule(a)
            else:  # _SAMPLE
                self.recorder.sample(t, self.book, self.fundamental_ticks(t))
                nxt = t + int(round(self.cfg.sample_interval_s * NS))
                if nxt <= horizon:
                    self.queue.push(nxt, _SAMPLE)
        return SimResult(
            config=self.cfg, book=self.book, events=self.book.events, commands=self._commands,
            samples=self.recorder.samples(), trades=self.recorder.trades(),
            mid_history=self.mid_history, n_commands=self._n_commands,
            n_events=self.book.seq, participants={p.name: p for p in self.participants})

    # -------------------------------------------------------------------------- internals
    def _seed_book(self) -> None:
        c = self.cfg
        for lvl in range(c.seed_levels):
            off = c.seed_gap_ticks + lvl
            self._exchange(NewLimit(self.new_id(), Side.BUY, c.mid0_ticks - off, c.seed_size, 0, SEEDER_OWNER), 0)
            self._exchange(NewLimit(self.new_id(), Side.SELL, c.mid0_ticks + off, c.seed_size, 0, SEEDER_OWNER), 0)

    def _reschedule(self, i: int) -> None:
        nt = self.participants[i].next_time()
        if nt is None:
            self._tokens[i] += 1
            self._sched[i] = None
            return
        nt = max(nt, self.now)
        if nt != self._sched[i]:
            self._tokens[i] += 1
            self._sched[i] = nt
            self.queue.push(nt, _WAKE, i, self._tokens[i])

    def _send(self, i: int, cmds: list[Command], t: int) -> None:
        if not cmds:
            return
        lat = self.participants[i].latency.order
        for cmd in cmds:
            arrive = max(t + lat.sample(self._latency_rng), self._last_arrival[i])
            self._last_arrival[i] = arrive
            self.queue.push(arrive, _ARRIVE, cmd)

    def _exchange(self, cmd: Command, t: int) -> None:
        """A command reaches the matching engine at time ``t``."""
        cmd = replace(cmd, ts=t)
        if self.cfg.record_commands:
            self._commands.append(cmd)
        self._n_commands += 1
        m = self.book.mid_price()
        self.recorder.pre_mid = float("nan") if m is None else m
        events = self.book.process(cmd)
        self.mid_history.update(t, self.book.mid_price())
        if events and (self._all_feed or self._own_feed):
            self._dispatch(events, t)

    def _dispatch(self, events: list[Event], t: int) -> None:
        for i in self._all_feed:
            self._deliver(i, events, t)
        if self._own_feed:
            per_owner: dict[int, list[Event]] = {}
            for ev in events:
                for owner in {ev.owner, ev.maker_owner}:
                    if owner in self._by_owner:
                        per_owner.setdefault(owner, []).append(ev)
            for owner, evs in per_owner.items():
                i = self._by_owner[owner]
                if self.participants[i].feed == "own":
                    self._deliver(i, evs, t)

    def _deliver(self, i: int, events: list[Event], t: int) -> None:
        lat = self.participants[i].latency.feed
        at = max(t + lat.sample(self._latency_rng), self._last_feed[i])
        self._last_feed[i] = at
        self.queue.push(at, _FEED, i, list(events))
