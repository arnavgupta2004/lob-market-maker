"""Session metrics computed from exchange ground truth (trade tape + event log).

Nothing here trusts a strategy's own bookkeeping: fills come from the tape, marks come from the
true book mid, so results are correct under latency (where the strategy's beliefs lag).

P&L identity (currency units; ``m^-`` = mid immediately before the command that caused a fill)::

    MtM_gross(T) = cash_gross(T) + q_T m_T
                 = sum_i s_i Q_i (m_i^- - p_i)          "edge": spread capture vs pre-trade mid
                   + sum_j q_j (m_j - m_{j-1})          "inventory P&L": inventory x mid moves
    net P&L      = MtM_gross - fees

The first term is what a maker earns by trading away from the mid; the second is what it earns
(or loses) from holding inventory while the mid moves, *including* the mid jump caused by the
fill itself - so systematic adverse selection appears as negative inventory P&L. Inventory P&L is
computed as the residual ``MtM_gross - edge``; ``tests/unit/test_metrics.py`` verifies the
identity against the integral form on an explicit scenario.

Realised P&L uses average-cost accounting (round trips closed against the running average
entry price); ``unrealised = MtM_gross - realised``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from backtest.costs import FeeModel
from engine.common import EventType as ET
from simulator.simulator import NS, SimResult

SECONDS_PER_YEAR = 365.0 * 24 * 3600


@dataclass
class Legs:
    """One row per participant execution leg (a self-trade yields two legs)."""

    t: np.ndarray  # ns
    price: np.ndarray  # ticks
    signed_qty: np.ndarray  # +bought / -sold (lots)
    is_maker: np.ndarray  # bool
    mid_before: np.ndarray  # ticks (nan if book one-sided)
    fee: np.ndarray  # currency, >= 0 cost


def extract_legs(res: SimResult, owner: int, fees: FeeModel) -> Legs:
    tape, tick = res.trades, res.config.tick_size
    parts = []
    for role, col, sgn in ((True, "maker_owner", -1), (False, "taker_owner", +1)):
        m = tape[col] == owner
        if not m.any():
            continue
        q = sgn * tape["aggressor"][m] * tape["qty"][m]
        fee = np.array([fees.fee(p * tick * abs(x), role) for p, x in zip(tape["price"][m], q)])
        parts.append((tape["t_ns"][m], tape["price"][m], q, np.full(m.sum(), role),
                      tape["mid_before"][m], fee))
    if not parts:
        e = np.array([])
        return Legs(e.astype(np.int64), e, e.astype(np.int64), e.astype(bool), e, e)
    cat = [np.concatenate(c) for c in zip(*parts)]
    order = np.argsort(cat[0], kind="stable")
    return Legs(*[c[order] for c in cat])


def equity_curve(res: SimResult, legs: Legs, start_ns: int = 0):
    """Net equity, inventory and true mid on the simulator's sample grid for ``t >= start_ns``.

    Returns ``(t_ns, equity_net, inventory, mid_ticks)``; mid is forward-filled through empty-side
    samples. Equity is zero at ``start_ns`` if the participant has no fills before it.
    """
    s = res.samples
    keep = s["t_ns"] >= start_ns
    t = s["t_ns"][keep]
    mid = _ffill(s["mid"])[keep]
    idx = np.searchsorted(legs.t, t, side="right")
    cq = np.concatenate([[0], np.cumsum(legs.signed_qty)])[idx]
    cash = np.concatenate([[0.0], np.cumsum(-legs.signed_qty * legs.price * res.config.tick_size - legs.fee)])[idx]
    return t, cash + cq * mid * res.config.tick_size, cq, mid


def _ffill(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    valid = ~np.isnan(x)
    if not valid.any():
        return x
    idx = np.where(valid, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    x = x[idx]
    first = np.argmax(valid)
    x[:first] = x[first]
    return x


def realized_pnl(legs: Legs, tick: float) -> float:
    """Average-cost realised P&L in currency."""
    pos, avg, realized = 0, 0.0, 0.0
    for p, q in zip(legs.price.tolist(), legs.signed_qty.tolist()):  # python floats: no numpy overflow warnings
        q = int(q)
        if pos == 0 or (pos > 0) == (q > 0):
            avg = (avg * abs(pos) + p * abs(q)) / (abs(pos) + abs(q))
            pos += q
        else:
            closing = min(abs(q), abs(pos))
            realized += closing * (p - avg) * (1 if pos > 0 else -1) * tick
            rem = abs(q) - closing
            pos += q
            if rem > 0:  # position flipped through zero
                avg = float(p)
    return realized


def session_metrics(res: SimResult, owner: int, fees: FeeModel = FeeModel(), start_s: float = 0.0) -> dict:
    """Flat dict of P&L, risk, inventory, execution and microstructure metrics for one owner."""
    tick = res.config.tick_size
    start_ns = int(start_s * NS)
    legs = extract_legs(res, owner, fees)
    t, eq_net, inv, mid = equity_curve(res, legs, start_ns)
    out: dict = {}

    # ------------------------------------------------------------------ P&L
    fees_paid = float(legs.fee.sum())
    net = float(eq_net[-1]) if len(eq_net) else 0.0
    gross = net + fees_paid
    valid = ~np.isnan(legs.mid_before)
    edge_ticks = float(np.sum(legs.signed_qty[valid] * (legs.mid_before[valid] - legs.price[valid])))
    edge = edge_ticks * tick
    notional = legs.price * tick * np.abs(legs.signed_qty)
    out.update(notional_maker=float(notional[legs.is_maker].sum()), notional_taker=float(notional[~legs.is_maker].sum()))
    out.update(pnl_net=net, pnl_gross=gross, fees=fees_paid, pnl_edge=edge,
               pnl_inventory=gross - edge, pnl_realized=realized_pnl(legs, tick))
    out["pnl_unrealized"] = gross - out["pnl_realized"]

    # ----------------------------------------------------------------- risk
    stride = max(1, int(round(1.0 / res.config.sample_interval_s)))
    d1 = np.diff(eq_net[::stride]) if len(eq_net) > stride else np.array([0.0])
    sd = float(d1.std(ddof=1)) if len(d1) > 1 else float("nan")
    out["pnl_vol_1s"] = sd
    out["sharpe_1s"] = float(d1.mean() / sd) if sd and sd > 0 else float("nan")
    out["sharpe_ann_247"] = out["sharpe_1s"] * math.sqrt(SECONDS_PER_YEAR) if sd and sd > 0 else float("nan")
    out["max_drawdown"] = float(np.max(np.maximum.accumulate(eq_net) - eq_net)) if len(eq_net) else 0.0
    q05 = float(np.quantile(d1, 0.05))
    out["pnl_1s_q05"] = q05
    out["pnl_1s_cvar05"] = float(d1[d1 <= q05].mean())
    out["pnl_1s_worst"] = float(d1.min())

    # ------------------------------------------------------------ inventory
    out.update(inv_mean=float(inv.mean()), inv_std=float(inv.std()), inv_abs_mean=float(np.abs(inv).mean()),
               inv_max_abs=float(np.abs(inv).max()) if len(inv) else 0.0,
               inv_q05=float(np.quantile(inv, 0.05)), inv_q95=float(np.quantile(inv, 0.95)),
               inv_final=float(inv[-1]) if len(inv) else 0.0)

    # ------------------------------------------------------------ execution
    placed: dict[int, tuple[int, int]] = {}
    lifetimes: list[float] = []
    filled_orders: set[int] = set()
    n_add = n_cancel = n_reject = n_maker_fills = 0
    added_qty = filled_qty = 0
    for e in res.events:
        if e.type is ET.ADD and e.owner == owner:
            n_add += 1
            added_qty += e.qty
            placed[e.order_id] = (e.ts, e.qty)
        elif e.type is ET.CANCEL and e.owner == owner:
            n_cancel += 1
            if e.order_id in placed:
                lifetimes.append((e.ts - placed[e.order_id][0]) / NS)
        elif e.type is ET.REJECT and e.owner == owner:
            n_reject += 1
        elif e.type is ET.TRADE and e.maker_owner == owner:
            n_maker_fills += 1
            filled_qty += e.qty
            filled_orders.add(e.maker_id)
            if e.maker_remaining == 0 and e.maker_id in placed:
                lifetimes.append((e.ts - placed[e.maker_id][0]) / NS)
    n_taker = int(np.sum((res.trades["taker_owner"] == owner)))
    n_fills = n_maker_fills + n_taker
    total_qty = float(np.abs(legs.signed_qty).sum())
    out.update(
        n_fills=n_fills, n_maker_fills=n_maker_fills, n_taker_fills=n_taker,
        filled_qty=float(total_qty), n_quotes=n_add, n_cancels=n_cancel, n_rejects=n_reject,
        fill_rate_orders=len(filled_orders) / n_add if n_add else float("nan"),
        fill_ratio_qty=filled_qty / added_qty if added_qty else float("nan"),
        cancel_rate=n_cancel / n_add if n_add else float("nan"),
        quote_to_trade=(n_add + n_cancel + n_reject) / n_fills if n_fills else float("nan"),
        quote_lifetime_mean_s=float(np.mean(lifetimes)) if lifetimes else float("nan"),
        quote_lifetime_median_s=float(np.median(lifetimes)) if lifetimes else float("nan"),
        quote_surv_250ms=float(np.mean(np.array(lifetimes) >= 0.25)) if lifetimes else float("nan"),
        quote_surv_1s=float(np.mean(np.array(lifetimes) >= 1.0)) if lifetimes else float("nan"),
        avg_edge_ticks=edge_ticks / total_qty if total_qty else float("nan"),
        pnl_per_lot=net / total_qty if total_qty else float("nan"),
    )
    return out
