"""Historical-data replay through the same ``Participant`` interface as synthetic order flow.

L2 feeds carry no order identities, so the replay *reconstructs order-level flow* with a documented convention
(:class:`L2Reconciler`): the replay participant owns a set of synthetic resting orders whose per-price total is driven to
equal the feed's level quantity after every ``LEVEL`` update:

* quantity increase  -> a new limit order of the difference (joins the back of the queue),
* quantity decrease  -> the difference is removed from the **back** of the replay's own orders at that price (cancel, or an
  in-place size reduction that keeps queue priority),
* ``TRADE``          -> a market order of the printed quantity (it consumes the front of the queue, as trades do),
* ``RESET``          -> all of the replay's orders are cancelled.

Because trades are replayed as real market orders and the level update that follows is *reconciled against the engine's
actual state*, the trade's own depletion of a level is never double counted. Assumptions (all testable): decreases that are
not trades are treated as back-of-queue cancellations (real cancels can be anywhere in the queue, so the queue position of a
strategy's order relative to exogenous orders is an approximation); aggregated feed levels lose order granularity.

Other participants (a market maker) share the engine with the replay. Their orders displace exogenous liquidity, so the
replay is *not* counterfactual-exact: when the strategy trades, the reconstructed book departs from history and the
reconciler keeps pulling the exogenous part back toward the feed. ``track_fidelity`` measures agreement with the feed's
top of book (meaningful when no strategy is present).

One feed event is processed per wake-up so each reconciliation sees the engine state after the previous event's commands.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from engine.commands import Cancel, Command, Modify, NewLimit, NewMarket
from engine.common import Side
from simulator.order_flow.market_data import MD
from simulator.participant import Participant


class L2Reconciler:
    """Stateless translation of one market-data row into engine commands (see module docstring)."""

    def __init__(self) -> None:
        self.n_trade_price_mismatch = 0  # trade printed at a price the book's touch does not show
        self.n_crossing_adds = 0  # a level update that would cross the opposite touch (feed/book desynchronised)

    def commands_for(self, book, row, owner: int, new_id) -> list[Command]:
        _, kind, side, price, qty = (int(x) for x in row)
        if kind == MD.TRADE:
            best = book.best_ask() if side == 1 else book.best_bid()
            if best != price:
                self.n_trade_price_mismatch += 1
            return [NewMarket(new_id(), Side(side), qty, owner=owner)]
        if kind == MD.RESET:
            st = book.state()
            return [Cancel(o[0]) for key in ("bids", "asks") for _, orders in st[key] for o in orders if o[2] == owner]
        s = Side(side)
        mine = book.owner_orders_at(s, price, owner)
        delta = qty - sum(q for _, q in mine)
        if delta > 0:
            opp = book.best_ask() if s is Side.BUY else book.best_bid()
            if opp is not None and (price >= opp if s is Side.BUY else price <= opp):
                self.n_crossing_adds += 1
            return [NewLimit(new_id(), s, price, delta, owner=owner)]
        cmds: list[Command] = []
        need = -delta
        for oid, q in reversed(mine):  # remove from the back of our own queue
            if need <= 0:
                break
            if need >= q:
                cmds.append(Cancel(oid))
                need -= q
            else:
                cmds.append(Modify(oid, price, q - need))
                need = 0
        return cmds


class HistoricalReplay(Participant):
    """Replays a normalised market-data array (``simulator.order_flow.market_data``) into the simulator.

    Event times are shifted so the first event occurs at ``start_ns`` of simulation time. ``feed = "none"``: the replay needs no
    notifications; it reads the engine's actual state when reconciling.
    """

    feed = "none"

    def __init__(self, data: np.ndarray, start_ns: int = 0, name: str = "replay", track_fidelity: bool = False):
        self.name = name
        self.data = np.asarray(data, dtype=np.int64)
        self.i = 0
        self.offset = start_ns - int(self.data[0, 0]) if len(self.data) else 0
        self.rec = L2Reconciler()
        self.track = track_fidelity
        self._src: dict[int, dict[int, int]] = {1: {}, -1: {}}  # the feed's own view of the book (for fidelity)
        self._checked = self._matched = 0
        self._pending_check = False
        self.n_trades = self.n_levels = 0

    def next_time(self) -> Optional[int]:
        return int(self.data[self.i, 0]) + self.offset if self.i < len(self.data) else None

    def on_wakeup(self, sim, now: int) -> list[Command]:
        if self.track and self._pending_check:
            self._compare(sim.book)
        row = self.data[self.i]
        self.i += 1
        kind = int(row[1])
        if kind == MD.TRADE:
            self.n_trades += 1
        elif kind == MD.LEVEL:
            self.n_levels += 1
        if self.track:
            self._update_source(row)
            # Compare only after LEVEL/RESET: between a trade print and the level update it causes, the feed's own
            # book is transiently inconsistent (the trade has depleted a level the feed has not yet reported).
            self._pending_check = kind != MD.TRADE
        return self.rec.commands_for(sim.book, row, self.owner_id, sim.new_id)

    # ----------------------------------------------------------- fidelity vs the feed's own top of book
    def _update_source(self, row) -> None:
        _, kind, side, price, qty = (int(x) for x in row)
        if kind == MD.RESET:
            self._src = {1: {}, -1: {}}
        elif kind == MD.LEVEL:
            if qty > 0:
                self._src[side][price] = qty
            else:
                self._src[side].pop(price, None)

    def _compare(self, book) -> None:
        sb = max(self._src[1]) if self._src[1] else None
        sa = min(self._src[-1]) if self._src[-1] else None
        self._checked += 1
        self._matched += int(book.best_bid() == sb and book.best_ask() == sa)
        self._pending_check = False

    def finalize(self, book) -> None:
        if self.track and self._pending_check:
            self._compare(book)

    @property
    def fidelity(self) -> float:
        """Fraction of events after which the engine's touch equalled the feed's touch."""
        return self._matched / self._checked if self._checked else float("nan")
