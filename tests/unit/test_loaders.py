"""Loaders against hand-written fixtures in the documented layouts (NOT real vendor files - see loaders.py docstring)."""
import numpy as np

from simulator.order_flow.historical import HistoricalReplay
from simulator.order_flow.loaders import (
    BINANCE_AGGTRADES, TARDIS_L2, TARDIS_TRADES, describe_file, drop_crossed_and_dedupe, load_l2_updates, load_trades, merge_streams,
)
from simulator.order_flow.market_data import MD, validate_market_data
from simulator.simulator import SimConfig, Simulator

L2 = """exchange,symbol,timestamp,local_timestamp,is_snapshot,side,price,amount
x,BTCUSDT,1000000,1000001,true,bid,100.00,2.0
x,BTCUSDT,1000000,1000001,true,ask,100.02,1.5
x,BTCUSDT,1000500,1000501,false,bid,100.01,0.5
x,BTCUSDT,1000900,1000901,false,ask,100.02,0.0
"""
TRADES = """exchange,symbol,timestamp,local_timestamp,id,side,price,amount
x,BTCUSDT,1000500,1000501,1,sell,100.00,0.25
x,BTCUSDT,1000900,1000901,2,buy,100.02,0.0004
"""
AGG = """1,100.01,0.30,1,1,1700000000123,True,True
2,100.02,0.10,2,2,1700000000456,False,True
"""


def test_l2_loader_converts_units_sides_and_inserts_reset_before_snapshot(tmp_path):
    p = tmp_path / "l2.csv"; p.write_text(L2)
    a = load_l2_updates(p, TARDIS_L2, tick_size=0.01, lot_size=0.1)
    assert a[0].tolist() == [1_000_000_000, MD.RESET, 0, 0, 0]  # reset precedes the snapshot block
    assert a[1].tolist() == [1_000_000_000, MD.LEVEL, 1, 10000, 20]  # us -> ns, price -> ticks, 2.0 -> 20 lots
    assert a[2].tolist() == [1_000_000_000, MD.LEVEL, -1, 10002, 15]
    assert a[3, 4] == 5 and a[4, 4] == 0  # 0.5 -> 5 lots; removal stays 0
    validate_market_data(a)


def test_trade_loaders_use_the_vendors_aggressor_convention(tmp_path):
    p = tmp_path / "t.csv"; p.write_text(TRADES)
    t = load_trades(p, TARDIS_TRADES, 0.01, 0.1)
    assert t[:, 2].tolist() == [-1, 1] and t[1, 4] == 1  # 0.0004 rounds to 0 lots but is kept as 1
    q = tmp_path / "agg.csv"; q.write_text(AGG)
    g = load_trades(q, BINANCE_AGGTRADES, 0.01, 0.1)
    assert g[:, 2].tolist() == [-1, 1]  # is_buyer_maker=True => aggressor SOLD
    assert g[0, 0] == 1700000000123 * 1_000_000 and g[0, 3] == 10001


def test_merge_puts_trades_before_level_updates_at_equal_time_and_replay_runs(tmp_path):
    (tmp_path / "l2.csv").write_text(L2); (tmp_path / "t.csv").write_text(TRADES)
    md = merge_streams(load_l2_updates(tmp_path / "l2.csv", TARDIS_L2, 0.01, 0.1), load_trades(tmp_path / "t.csv", TARDIS_TRADES, 0.01, 0.1))
    validate_market_data(md)
    at_900 = md[md[:, 0] == 1_000_900_000]
    assert at_900[0, 1] == MD.TRADE  # trade first, then the level update it caused
    rp = HistoricalReplay(md)
    res = Simulator(SimConfig(horizon_s=0.01, seed_levels=0, warmup_s=0), [rp]).run()
    assert res.book.best_bid() == 10001  # 100.01 bid joined the 100.00 bid (bid book: 10001 and 10000 minus the 0.25 trade)
    assert describe_file(tmp_path / "l2.csv", 1).startswith("exchange,symbol")


def test_dedupe_removes_noop_level_updates():
    a = np.array([[0, MD.LEVEL, 1, 99, 5], [1, MD.LEVEL, 1, 99, 5], [2, MD.LEVEL, 1, 99, 6], [3, MD.RESET, 0, 0, 0], [4, MD.LEVEL, 1, 99, 6]])
    assert drop_crossed_and_dedupe(a)[:, 0].tolist() == [0, 2, 3, 4]  # the level memory is cleared by a RESET
