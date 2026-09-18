"""Seeded random command-stream generator for testing and benchmarking the engines.

This is *not* an economic order-flow model (see ``simulator.order_flow``); it is a stress
generator whose knobs (number of price levels, cancel rate, marketable fraction, ...) map
directly onto the benchmark workloads. Generation is driven by a Python reference book so that
cancels/modifies mostly target live orders. The returned list is a plain command sequence and
can be fed to any engine.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from engine.commands import Cancel, Command, Modify, NewLimit, NewMarket, SnapshotRequest
from engine.common import Side
from engine.python.order_book import OrderBook


@dataclass(frozen=True)
class WorkloadParams:
    n_commands: int = 1000
    mid0: int = 10_000  # initial reference price (ticks)
    price_range: int = 20  # passive limit orders are placed 1..price_range ticks from the reference
    p_limit: float = 0.55  # command-type mix (normalised internally)
    p_market: float = 0.05
    p_cancel: float = 0.30
    p_modify: float = 0.10
    p_aggressive: float = 0.05  # fraction of limit orders priced to cross the spread
    max_qty: int = 10
    p_post_only: float = 0.05
    p_ioc: float = 0.03
    p_invalid: float = 0.0  # fraction of commands that are deliberately erroneous
    p_snapshot: float = 0.0
    n_owners: int = 4
    mean_dt_ns: int = 1_000


def generate_workload(seed: int, params: WorkloadParams = WorkloadParams()) -> list[Command]:
    """Deterministically generate ``params.n_commands`` commands from ``seed``."""
    p = params
    rng = random.Random(seed)
    book = OrderBook()
    cmds: list[Command] = []
    live: list[int] = []  # candidate ids for cancel/modify; lazily purged of stale entries
    all_ids: list[int] = []
    next_id = 1
    ts = 0
    ref = p.mid0
    weights = [p.p_limit, p.p_market, p.p_cancel, p.p_modify]
    kinds = ["limit", "market", "cancel", "modify"]

    def pick_live() -> int | None:
        while live:
            i = rng.randrange(len(live))
            oid = live[i]
            if oid in book:
                return oid
            live[i] = live[-1]
            live.pop()
        return None

    for _ in range(p.n_commands):
        ts += rng.randint(1, 2 * p.mean_dt_ns)
        m = book.mid_price()
        if m is not None:
            ref = int(m)
        cmd: Command
        if p.p_snapshot and rng.random() < p.p_snapshot:
            cmd = SnapshotRequest(ts)
        elif p.p_invalid and rng.random() < p.p_invalid:
            cmd = _invalid_command(rng, all_ids, next_id, ts, p)
            if isinstance(cmd, (NewLimit, NewMarket)) and cmd.order_id == next_id:
                next_id += 1
        else:
            kind = rng.choices(kinds, weights)[0]
            oid = pick_live() if kind in ("cancel", "modify") else None
            if kind in ("cancel", "modify") and oid is None:
                kind = "limit"
            owner = rng.randrange(p.n_owners)
            if kind == "limit":
                side = Side.BUY if rng.random() < 0.5 else Side.SELL
                cmd = NewLimit(next_id, side, _limit_price(rng, side, ref, p),
                               rng.randint(1, p.max_qty), ts, owner,
                               post_only=rng.random() < p.p_post_only,
                               ioc=False)
                if not cmd.post_only and rng.random() < p.p_ioc:
                    cmd = NewLimit(cmd.order_id, cmd.side, cmd.price, cmd.qty, ts, owner, ioc=True)
                all_ids.append(next_id)
                next_id += 1
            elif kind == "market":
                cmd = NewMarket(next_id, Side.BUY if rng.random() < 0.5 else Side.SELL,
                                rng.randint(1, p.max_qty * 3), ts, owner)
                all_ids.append(next_id)
                next_id += 1
            elif kind == "cancel":
                cmd = Cancel(oid, ts)  # type: ignore[arg-type]
            else:
                node = book.get_order(oid)  # type: ignore[arg-type]
                cmd = _modify_command(rng, node, ref, ts, p)
        book.process(cmd)
        if isinstance(cmd, NewLimit) and cmd.order_id in book:
            live.append(cmd.order_id)
        cmds.append(cmd)
    return cmds


def _limit_price(rng: random.Random, side: Side, ref: int, p: WorkloadParams) -> int:
    if rng.random() < p.p_aggressive:
        off = rng.randint(0, 3)  # priced through the mid => usually marketable
        px = ref + off if side is Side.BUY else ref - off
    else:
        off = rng.randint(1, max(1, p.price_range))
        px = ref - off if side is Side.BUY else ref + off
    return max(1, px)


def _modify_command(rng: random.Random, node, ref: int, ts: int, p: WorkloadParams) -> Modify:
    r = rng.random()
    if r < 0.35 and node.qty > 1:  # in-place reduce (keeps priority)
        return Modify(node.order_id, node.price, rng.randint(1, node.qty - 1), ts)
    if r < 0.55:  # size increase (loses priority)
        return Modify(node.order_id, node.price, node.qty + rng.randint(1, p.max_qty), ts)
    off = rng.randint(-2, p.price_range)  # re-price; negative offsets may cross
    px = ref - off if node.side is Side.BUY else ref + off
    return Modify(node.order_id, max(1, px), rng.randint(1, p.max_qty), ts)


def _invalid_command(rng: random.Random, all_ids: list[int], next_id: int, ts: int,
                     p: WorkloadParams) -> Command:
    r = rng.randrange(5)
    if r == 0 and all_ids:  # duplicate id
        return NewLimit(rng.choice(all_ids), Side.BUY, p.mid0, 1, ts)
    if r == 1:  # zero quantity
        return NewLimit(next_id, Side.SELL, p.mid0, 0, ts)
    if r == 2:  # unknown cancel
        return Cancel(10**12 + rng.randrange(1000), ts)
    if r == 3:  # unknown modify
        return Modify(10**12 + rng.randrange(1000), p.mid0, 5, ts)
    return NewLimit(next_id, Side.BUY, 0, 1, ts)  # non-positive price
