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


# ------------------------------------------------------------------------------- Kraken v2 raw JSONL (schema as observed live)
def _kraken_lines():
    import json
    def rec(t, msg): return json.dumps({"recv_ns": t, "msg": msg})
    snap = {"channel": "book", "type": "snapshot", "data": [{"symbol": "BTC/USD", "bids": [{"price": 100.0, "qty": 2.0}, {"price": 99.9, "qty": 1.0}],
                                                              "asks": [{"price": 100.1, "qty": 1.5}, {"price": 100.2, "qty": 3.0}], "checksum": 0}]}
    upd1 = {"channel": "book", "type": "update", "data": [{"symbol": "BTC/USD", "bids": [{"price": 100.0, "qty": 2.5}], "asks": [], "checksum": 0,
                                                            "timestamp": "2026-09-18T22:41:19.335086Z"}]}
    trd = {"channel": "trade", "type": "update", "data": [{"symbol": "BTC/USD", "side": "buy", "price": 100.1, "qty": 0.5, "ord_type": "limit", "trade_id": 1,
                                                            "timestamp": "2026-09-18T22:41:19.400000Z"}]}
    upd2 = {"channel": "book", "type": "update", "data": [{"symbol": "BTC/USD", "bids": [], "asks": [{"price": 100.1, "qty": 1.0}], "checksum": 0,
                                                            "timestamp": "2026-09-18T22:41:19.400000Z"}]}
    return "\n".join([rec(1, {"channel": "status", "type": "update", "data": []}), rec(2, snap), rec(3, upd1), rec(4, trd), rec(5, upd2),
                      rec(6, {"channel": "_collector", "type": "reconnect", "reason": "x"}), "{not json"]) + "\n"


def test_kraken_loader_maps_snapshot_updates_trades_reconnects_and_units(tmp_path):
    from simulator.order_flow.loaders import _iso_ns, load_kraken_jsonl
    p = tmp_path / "raw.jsonl"; p.write_text(_kraken_lines())
    a, rep = load_kraken_jsonl(p, tick_size=0.1, lot_size=0.1, verify_checksum=False)
    assert rep["snapshots"] == 1 and rep["updates"] == 2 and rep["trades"] == 1 and rep["reconnects"] == 1 and rep["bad_lines"] == 1
    kinds = a[:, 1].tolist()
    assert kinds[0] == MD.RESET and kinds[1:5] == [MD.LEVEL] * 4  # snapshot: reset + 4 levels
    assert a[1].tolist()[2:] == [1, 1000, 20] and a[3].tolist()[2:] == [-1, 1001, 15]  # 100.0 -> 1000 ticks, 2.0 BTC -> 20 lots
    t_trade = _iso_ns("2026-09-18T22:41:19.400000Z")
    at_t = a[a[:, 0] == t_trade]
    assert at_t[0, 1] == MD.TRADE and at_t[0, 2] == 1 and at_t[0, 4] == 5  # trade first at equal timestamp; buyer-initiated; 0.5 BTC = 5 lots
    assert a[-1, 1] == MD.RESET  # the collector reconnect voids the book
    assert _iso_ns("2026-09-18T22:41:19.335086Z") % 1_000_000_000 == 335_086_000  # microsecond resolution preserved


def test_kraken_checksum_matches_the_documented_formatting():
    from simulator.order_flow.loaders import kraken_checksum
    import zlib
    bids = {100.0: 2.0, 99.9: 0.00005}
    asks = {100.1: 1.5}
    # asks ascending then bids descending; price/qty rendered with the decimal point and leading zeros removed
    s = "1001" + "150000000" + "1000" + "200000000" + "999" + "5000"
    assert kraken_checksum(bids, asks, 1, 8) == zlib.crc32(s.encode()) & 0xFFFFFFFF


def test_kraken_depth_truncation_emits_removals_for_levels_pushed_out(tmp_path):
    import json
    from simulator.order_flow.loaders import load_kraken_jsonl
    snap = {"channel": "book", "type": "snapshot", "data": [{"bids": [{"price": 100.0 - 0.1 * i, "qty": 1.0} for i in range(5)], "asks": [{"price": 100.1 + 0.1 * i, "qty": 1.0} for i in range(5)]}]}
    upd = {"channel": "book", "type": "update", "data": [{"bids": [{"price": 100.05, "qty": 1.0}], "asks": [], "timestamp": "2026-09-18T22:41:19.000000Z"}]}
    p = tmp_path / "r.jsonl"; p.write_text("\n".join(json.dumps({"recv_ns": i, "msg": m}) for i, m in enumerate([snap, upd])) + "\n")
    a, rep = load_kraken_jsonl(p, 0.1, 0.1, depth=5, verify_checksum=False)
    assert rep["truncated_levels_emitted"] == 1  # a 6th bid entered the 5-level window: the worst one is removed
    assert a[-1].tolist()[1:] == [int(MD.LEVEL), 1, 996, 0]  # the worst bid (99.6 = 996 ticks) is removed
