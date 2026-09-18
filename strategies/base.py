"""Market-maker scaffolding shared by every strategy.

A strategy subclasses :class:`MarketMaker` and implements only :meth:`desired_quotes`. The base
class supplies everything that is *not* a modelling choice:

* **Delayed market view** - the maker subscribes to the full public feed and maintains a
  *replica* order book by applying events (``OrderBook.apply_event``). With feed latency L the
  replica is exactly the exchange book as of L ago, so stale-quote effects emerge naturally.
* **Own-order bookkeeping** - believed live quotes, inventory, cash and fees are updated from
  the feed. Beliefs lag reality by the feed latency; *reported* results are recomputed from the
  exchange tape (``backtest.metrics``), never from these beliefs.
* **Quote management** - a quote whose price is unchanged is left alone (keeps queue priority);
  otherwise it is cancelled and re-posted as a post-only limit order (never crosses the book).
* **Risk** - soft inventory limit (stop quoting the side that would extend the position),
  hard inventory limit and drawdown limit that trip a kill-switch: cancel all quotes, optionally
  flatten with a market order, and stop quoting for the rest of the session.
* **Volatility estimate** - EWMA of squared mid changes per second, in ticks^2/s.
* **Safety guards** (engineering, not modelling; identical for every strategy, configurable and
  switchable off): bounds on the volatility estimate and a cap on how far a quote may sit from
  the mid. The volatility estimator is fed by mid moves that the maker's *own* quotes cause; in a
  reflexive single-venue market a large risk-aversion can therefore make quotes, volatility
  estimate and mid chase each other without bound (demonstrated in ``experiments.inventory``).
  The counters ``n_sigma_clamped`` / ``n_offset_clamped`` report how often the guards bind.
"""
from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from backtest.costs import FeeModel
from engine.commands import Cancel, Command, NewLimit, NewMarket
from engine.common import EventType as ET, Side
from engine.events import Event
from engine.python.order_book import OrderBook
from simulator.latency.models import ZERO_LATENCY, LatencyConfig
from simulator.participant import Participant

NS = 1_000_000_000

Quote = tuple[int, int]  # (price_ticks, qty)


@dataclass(frozen=True)
class MMConfig:
    """Parameters common to all market makers."""

    quote_size: int = 5
    requote_interval_s: float = 0.1
    start_s: float = 5.0  # begin quoting after the market warm-up
    inventory_limit: int = 50  # soft: never post a side that could push |q| past this
    kill_inventory: int = 100  # hard: kill-switch if |q| reaches this
    max_drawdown: float = math.inf  # currency units, from running peak of believed MtM equity
    flatten_on_kill: bool = True
    requote_on_fill: bool = True
    vol_halflife_s: float = 10.0
    sigma0: float = 2.0  # prior for volatility, ticks / sqrt(s)
    sigma_bounds: Optional[tuple[float, float]] = (0.05, 10.0)  # clamp on the EWMA estimate; None = off
    max_quote_offset_ticks: Optional[int] = 20  # quotes stay within +-this of the mid; None = off
    fees: FeeModel = FeeModel()
    latency: LatencyConfig = ZERO_LATENCY


@dataclass
class _Live:
    order_id: int
    price: int
    qty: int  # believed remaining quantity
    sent_ns: int


class MarketMaker(Participant):
    """Base class; see module docstring. Subclasses implement :meth:`desired_quotes`."""

    feed = "all"

    def __init__(self, cfg: MMConfig, name: str = "mm"):
        self.cfg = cfg
        self.name = name
        self.latency = cfg.latency
        self.replica = OrderBook()
        self.tick = 0.01
        # believed state (feed-lagged)
        self.inventory = 0
        self.cash = 0.0
        self.fees_paid = 0.0
        self.peak_equity = 0.0
        self.killed = False
        self.kill_time: Optional[int] = None
        self.kill_reason: Optional[str] = None
        self.n_new = 0
        self.n_cancel = 0
        self.n_flatten = 0
        self.n_sigma_clamped = 0
        self.n_offset_clamped = 0
        self._live: dict[Side, Optional[_Live]] = {Side.BUY: None, Side.SELL: None}
        self._var = cfg.sigma0 ** 2  # ticks^2 / s
        self._vol_last: Optional[tuple[int, float]] = None
        self._next: Optional[int] = None
        self._start_ns = 0
        self._flat_id: Optional[int] = None
        self._flat_left = 0

    # ---------------------------------------------------------------- strategy hook
    @abstractmethod
    def desired_quotes(self, now: int, mid: float) -> tuple[Optional[Quote], Optional[Quote]]:
        """Return the desired ``(bid, ask)``, each ``(price_ticks, qty)`` or ``None``.

        Called with the replica mid. The base class afterwards clamps sizes to the inventory
        limits and prices so that a post-only order cannot cross the replica book.
        """

    # ---------------------------------------------------------------------- state
    @property
    def sigma(self) -> float:
        """EWMA volatility estimate in ticks / sqrt(second)."""
        return math.sqrt(self._var)

    def mid(self) -> Optional[float]:
        return self.replica.mid_price()

    def equity(self) -> Optional[float]:
        """Believed mark-to-market equity in currency (cash + inventory at replica mid)."""
        m = self.replica.mid_price()
        return None if m is None else self.cash + self.inventory * m * self.tick

    def live_quote(self, side: Side) -> Optional[_Live]:
        return self._live[side]

    # ------------------------------------------------------------- participant API
    def on_start(self, sim) -> None:
        self.tick = sim.cfg.tick_size
        self._start_ns = int(self.cfg.start_s * NS)
        self._next = self._start_ns

    def next_time(self) -> Optional[int]:
        return self._next

    def on_wakeup(self, sim, now: int) -> list[Command]:
        interval = int(self.cfg.requote_interval_s * NS)
        self._update_vol(now)
        cmds = self._risk_check(sim, now)
        if self.killed:
            cmds += self._flatten(sim)
            self._next = now + interval if (self.inventory != 0 or self._flat_id is not None) else None
            return cmds
        self._next = now + interval
        return cmds + self._requote(sim, now)

    def on_events(self, sim, now: int, events: list[Event]) -> list[Command]:
        filled = False
        for ev in events:
            self.replica.apply_event(ev)
            filled |= self._observe(ev)
        cmds = self._risk_check(sim, now)
        if self.killed:
            return cmds + self._flatten(sim)
        if filled and self.cfg.requote_on_fill and now >= self._start_ns:
            cmds += self._requote(sim, now)
        return cmds

    # --------------------------------------------------------------------- internals
    def _update_vol(self, now: int) -> None:
        m = self.replica.mid_price()
        if m is None:
            return
        if self._vol_last is not None:
            t0, m0 = self._vol_last
            dt = (now - t0) / NS
            if dt > 0:
                alpha = 1.0 - 0.5 ** (dt / self.cfg.vol_halflife_s)
                self._var = (1 - alpha) * self._var + alpha * (m - m0) ** 2 / dt
                if self.cfg.sigma_bounds is not None:
                    lo, hi = self.cfg.sigma_bounds
                    if self._var > hi * hi:
                        self._var = hi * hi  # clamp the state itself: prevents wind-up
                        self.n_sigma_clamped += 1
                    elif self._var < lo * lo:
                        self._var = lo * lo
                        self.n_sigma_clamped += 1
        self._vol_last = (now, m)

    def _observe(self, ev: Event) -> bool:
        """Update beliefs from one exchange event; return True if it was an own fill."""
        me = self.owner_id
        fee = self.cfg.fees
        filled = False
        if ev.type is ET.TRADE:
            if ev.maker_owner == me:  # our resting order was hit: aggressor side is opposite ours
                sgn = -int(ev.side)
                self._apply_fill(sgn, ev.price, ev.qty, is_maker=True)
                filled = True
                side = Side.BUY if sgn > 0 else Side.SELL
                lv = self._live[side]
                if lv is not None and lv.order_id == ev.maker_id:
                    if ev.maker_remaining == 0:
                        self._live[side] = None
                    else:
                        lv.qty = ev.maker_remaining
            if ev.owner == me:  # we were the aggressor
                self._apply_fill(int(ev.side), ev.price, ev.qty, is_maker=False)
                filled = True
                if ev.order_id == self._flat_id:
                    self._flat_left -= ev.qty
                    if self._flat_left <= 0:
                        self._flat_id = None
        elif ev.owner == me:
            if ev.type in (ET.CANCEL, ET.REJECT):
                for side, lv in self._live.items():
                    if lv is not None and lv.order_id == ev.order_id:
                        self._live[side] = None
            elif ev.type is ET.EXPIRE and ev.order_id == self._flat_id:
                self._flat_id = None
        return filled

    def _apply_fill(self, sgn: int, price: int, qty: int, is_maker: bool) -> None:
        notional = price * self.tick * qty
        self.inventory += sgn * qty
        self.cash -= sgn * notional
        f = self.cfg.fees.fee(notional, is_maker)
        self.cash -= f
        self.fees_paid += f

    def _risk_check(self, sim, now: int) -> list[Command]:
        eq = self.equity()
        if eq is not None and now >= self._start_ns and not self.killed:
            self.peak_equity = max(self.peak_equity, eq)
            dd = self.peak_equity - eq
            if abs(self.inventory) >= self.cfg.kill_inventory:
                return self._kill(now, "inventory")
            if dd >= self.cfg.max_drawdown:
                return self._kill(now, "drawdown")
        return []

    def _kill(self, now: int, reason: str) -> list[Command]:
        self.killed = True
        self.kill_time = now
        self.kill_reason = reason
        cmds: list[Command] = []
        for side, lv in self._live.items():
            if lv is not None:
                cmds.append(Cancel(lv.order_id))
                self.n_cancel += 1
                self._live[side] = None
        return cmds

    def _flatten(self, sim) -> list[Command]:
        if not self.cfg.flatten_on_kill or self.inventory == 0 or self._flat_id is not None:
            return []
        side = Side.SELL if self.inventory > 0 else Side.BUY
        qty = abs(self.inventory)
        self._flat_id = sim.new_id()
        self._flat_left = qty
        self.n_flatten += 1
        return [NewMarket(self._flat_id, side, qty, owner=self.owner_id)]

    def _requote(self, sim, now: int) -> list[Command]:
        mid = self.replica.mid_price()
        if mid is None:
            return []
        bid, ask = self.desired_quotes(now, mid)
        bid = self._clamp(Side.BUY, bid, mid)
        ask = self._clamp(Side.SELL, ask, mid)
        cmds: list[Command] = []
        for side, want in ((Side.BUY, bid), (Side.SELL, ask)):
            cur = self._live[side]
            if cur is not None:
                if want is not None and cur.price == want[0] and 0 < cur.qty <= want[1]:
                    continue  # unchanged price: keep the order (and its queue priority)
                cmds.append(Cancel(cur.order_id))
                self.n_cancel += 1
                self._live[side] = None
            if want is not None:
                oid = sim.new_id()
                cmds.append(NewLimit(oid, side, want[0], want[1], owner=self.owner_id, post_only=True))
                self._live[side] = _Live(oid, want[0], want[1], now)
                self.n_new += 1
        return cmds

    def _clamp(self, side: Side, q: Optional[Quote], mid: float) -> Optional[Quote]:
        """Enforce quote-offset cap, soft inventory limit and post-only feasibility."""
        if q is None:
            return None
        px, qty = q
        off = self.cfg.max_quote_offset_ticks
        if off is not None:
            capped = min(max(px, math.ceil(mid - off)), math.floor(mid + off))
            if capped != px:
                self.n_offset_clamped += 1
                px = capped
        lim = self.cfg.inventory_limit
        room = lim - self.inventory if side is Side.BUY else lim + self.inventory
        qty = min(qty, room)
        if qty < 1:
            return None
        if side is Side.BUY:
            ask = self.replica.best_ask()
            if ask is not None:
                px = min(px, ask - 1)
        else:
            bid = self.replica.best_bid()
            if bid is not None:
                px = max(px, bid + 1)
        return (px, qty) if px >= 1 else None
