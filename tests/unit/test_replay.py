"""Historical-replay machinery: reconciler semantics, round trip against a simulated exchange, I/O."""
import numpy as np
import pytest

from engine.common import Side
from engine.python.order_book import OrderBook
from simulator.market_state.fundamental import FundamentalConfig
from simulator.order_flow.historical import HistoricalReplay, L2Reconciler
from simulator.order_flow.informed import InformedTrader
from simulator.order_flow.market_data import (
    MD, events_to_market_data, read_market_data, to_lots, to_ticks, validate_market_data, write_market_data,
)
from simulator.order_flow.noise import NoiseTrader
from simulator.simulator import NS, SimConfig, Simulator

B, S = Side.BUY, Side.SELL


def rows(*r):
    return np.array(r, dtype=np.int64)


def apply(book, rec, row, owner=1, counter=[10_000]):
    def new_id():
        counter[0] += 1
        return counter[0]
    cmds = rec.commands_for(book, row, owner, new_id)
    book.process_all(cmds)
    return cmds


# ------------------------------------------------------------------------------ reconciler
def test_level_increase_adds_order_and_decrease_removes_from_the_back():
    b, rec = OrderBook(), L2Reconciler()
    apply(b, rec, (0, MD.LEVEL, 1, 99, 10))
    apply(b, rec, (1, MD.LEVEL, 1, 99, 15))  # +5 joins the back
    assert b.owner_orders_at(B, 99, 1)[0][1] == 10 and b.level_qty(B, 99) == 15
    cmds = apply(b, rec, (2, MD.LEVEL, 1, 99, 12))  # -3: shrinks the BACK order (5 -> 2), front untouched
    assert b.level_qty(B, 99) == 12
    orders = b.owner_orders_at(B, 99, 1)
    assert orders[0][1] == 10 and orders[1][1] == 2 and type(cmds[0]).__name__ == "Modify"
    apply(b, rec, (3, MD.LEVEL, 1, 99, 10))  # -2: the back order disappears entirely
    assert len(b.owner_orders_at(B, 99, 1)) == 1
    apply(b, rec, (4, MD.LEVEL, 1, 99, 0))
    assert b.best_bid() is None
    assert apply(b, rec, (5, MD.LEVEL, 1, 99, 0)) == []  # idempotent


def test_reconciliation_ignores_other_owners_liquidity():
    b, rec = OrderBook(), L2Reconciler()
    b.submit_limit(1, B, 99, 4, owner=2)  # a strategy's own order at the same price
    apply(b, rec, (0, MD.LEVEL, 1, 99, 10))  # the FEED's level of 10 is exogenous only
    assert b.level_qty(B, 99) == 14 and sum(q for _, q in b.owner_orders_at(B, 99, 1)) == 10


def test_trade_is_replayed_as_market_order_and_following_level_update_is_a_noop():
    b, rec = OrderBook(), L2Reconciler()
    apply(b, rec, (0, MD.LEVEL, -1, 101, 8))
    apply(b, rec, (1, MD.LEVEL, 1, 99, 8))
    cmds = apply(b, rec, (2, MD.TRADE, 1, 101, 3))  # buyer-initiated trade of 3
    assert b.level_qty(S, 101) == 5 and rec.n_trade_price_mismatch == 0
    assert apply(b, rec, (2, MD.LEVEL, -1, 101, 5)) == []  # the feed's post-trade level already matches
    apply(b, rec, (3, MD.TRADE, 1, 105, 1))  # print at a price the touch does not show
    assert rec.n_trade_price_mismatch == 1


def test_reset_cancels_only_the_replays_orders():
    b, rec = OrderBook(), L2Reconciler()
    b.submit_limit(1, S, 105, 2, owner=2)
    apply(b, rec, (0, MD.LEVEL, 1, 99, 5)); apply(b, rec, (0, MD.LEVEL, -1, 101, 5))
    apply(b, rec, (1, MD.RESET, 0, 0, 0))
    assert b.best_bid() is None and b.best_ask() == 105 and len(b) == 1


def test_crossing_level_update_is_counted_and_trades_against_the_book():
    b, rec = OrderBook(), L2Reconciler()
    b.submit_limit(1, S, 100, 2, owner=2)  # a strategy ask sits at 100
    apply(b, rec, (0, MD.LEVEL, 1, 100, 5))  # feed says a bid of 5 at 100: crosses the strategy's ask
    assert rec.n_crossing_adds == 1 and 1 not in b  # the strategy order was hit (as it would have been)


# ------------------------------------------------------------------------------ format & I/O
def test_conversions_and_validation():
    assert to_ticks(np.array([100.004, 100.006]), 0.01).tolist() == [10000, 10001]
    assert to_lots(np.array([0.0, 0.0004, 1.0, 2.5]), 0.001).tolist() == [0, 1, 1000, 2500]
    ok = rows((0, MD.LEVEL, 1, 99, 5), (1, MD.TRADE, -1, 99, 2))
    assert validate_market_data(ok)["n_trades"] == 1
    for bad in (rows((5, MD.LEVEL, 1, 99, 5), (4, MD.LEVEL, 1, 99, 5)), rows((0, MD.LEVEL, 0, 99, 5)),
                rows((0, MD.TRADE, 1, 99, 0)), rows((0, MD.LEVEL, 1, 0, 5)), rows((0, 9, 1, 99, 5))):
        with pytest.raises(ValueError):
            validate_market_data(bad)


def test_parquet_roundtrip_keeps_data_and_provenance(tmp_path):
    arr = rows((0, MD.LEVEL, 1, 99, 5), (1, MD.TRADE, -1, 99, 2), (2, MD.RESET, 0, 0, 0))
    p = tmp_path / "md.parquet"
    write_market_data(p, arr, {"tick_size": 0.01, "source": "unit-test", "license": "n/a"})
    back, meta = read_market_data(p)
    assert np.array_equal(arr, back) and meta["source"] == "unit-test" and meta["tick_size"] == 0.01


# ------------------------------------------------------------------ round trip against a simulated exchange
def simulate(seed=5, horizon=25.0):
    cfg = SimConfig(seed=seed, horizon_s=horizon, fundamental=FundamentalConfig(sigma=0.03), sample_interval_s=0.05)
    return Simulator(cfg, [NoiseTrader(), InformedTrader()]).run()


def replay(md, horizon, sample=0.05, track=True):
    rp = HistoricalReplay(md, track_fidelity=track)
    cfg = SimConfig(seed=999, horizon_s=horizon, seed_levels=0, sample_interval_s=sample, warmup_s=0.0)
    res = Simulator(cfg, [rp]).run()
    rp.finalize(res.book)
    return res, rp


def test_round_trip_reconstructs_the_book_and_tape_exactly():
    """simulate -> export as an L2+trades feed -> replay: the reconstructed exchange must equal the original."""
    src = simulate()
    md = events_to_market_data(src.events)
    validate_market_data(md)
    res, rp = replay(md, 25.0)
    # (1) fidelity: after EVERY feed event the replayed touch equals the feed's touch
    assert rp.fidelity == 1.0 and rp.rec.n_trade_price_mismatch == 0 and rp.rec.n_crossing_adds == 0
    # (2) final book: identical (price, quantity) per level. The number of orders per level differs BY CONSTRUCTION: an L2
    #     feed carries aggregates only, so order granularity is lost and reconstructed as one order per increase.
    def levels(b):
        bids, asks = b.depth()
        return [(p, q) for p, q, _ in bids], [(p, q) for p, q, _ in asks]
    assert levels(src.book) == levels(res.book) and len(levels(src.book)[0]) >= 5
    # (3) trade tape: identical (time, price, aggressor), and identical total quantity per (time, price)
    def tape(r):
        t = r.trades
        agg: dict = {}
        for ts, px, q, s in zip(t["t_ns"], t["price"], t["qty"], t["aggressor"]):
            agg[(int(ts), int(px), int(s))] = agg.get((int(ts), int(px), int(s)), 0) + int(q)
        return agg
    assert tape(src) == tape(res) and len(tape(src)) > 500
    # (4) sampled top of book / depth agree at every grid time after t = 0
    a, b = src.samples, res.samples
    for k in ("best_bid", "best_ask", "bid_depth5", "ask_depth5", "bid_qty1", "ask_qty1"):
        assert np.array_equal(a[k][1:], b[k][1:], equal_nan=True), k
    res.book.validate()


def test_replay_time_offset_and_termination():
    md = rows((1000, MD.LEVEL, 1, 99, 5), (2000, MD.LEVEL, -1, 101, 5))
    rp = HistoricalReplay(md, start_ns=NS)
    assert rp.next_time() == NS
    res = Simulator(SimConfig(horizon_s=2.0, seed_levels=0, warmup_s=0), [rp]).run()
    assert rp.next_time() is None and res.book.best_bid() == 99 and res.book.best_ask() == 101
    assert [e.ts for e in res.events] == [NS, NS + 1000]


def test_strategy_shares_the_book_with_replay_and_replay_pulls_exogenous_liquidity_back():
    from strategies.baseline_mm import ASConfig
    src = simulate(seed=8, horizon=20.0)
    md = events_to_market_data(src.events)
    rp = HistoricalReplay(md, track_fidelity=True)
    mm = ASConfig(gamma=0.01, k=0.9, start_s=5.0).build()
    cfg = SimConfig(seed=1, horizon_s=20.0, seed_levels=0, warmup_s=5.0)
    res = Simulator(cfg, [rp, mm]).run()
    res.book.validate()
    assert res.n_events > 1000 and len(res.trades["price"]) > 100
    assert np.any(res.trades["maker_owner"] == mm.owner_id)  # the strategy traded against replayed flow
