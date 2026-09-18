"""Adverse-selection and price-impact estimators against hand-computed and simulated ground truth."""
import numpy as np
import pytest

from engine.commands import NewLimit, NewMarket
from engine.common import Side
from research.adverse_selection import (
    HORIZONS_NS, by_group, passive_fills, random_time_markouts, markouts, summarize_fills, wmean,
)
from research.common import ffill, mid_at, obi_at, obi_from_samples
from research.price_impact import (
    add_stats, aggressive_orders, bin_table, fit_power_law, impacts, size_bin_stats,
)
from simulator.order_flow.informed import InformedTrader, InformedTraderConfig
from simulator.order_flow.noise import NoiseTrader
from simulator.simulator import SimConfig, Simulator
from tests.helpers import Scripted
from tests.unit.test_metrics import round_trip_result

B, S = Side.BUY, Side.SELL
MS = 1_000_000


# ------------------------------------------------------------------ markouts
def test_markout_decomposition_hand_computed():
    """Fill 1: maker buys 4@99, m^-=100.5, mid 1 ms later = 100  -> E=1.5, M=-0.5, R=1.0
    Fill 2: maker sells 4@101, m^-=100, mid 1 ms later = 100     -> E=1.0, M= 0.0, R=1.0"""
    res = round_trip_result()
    f = passive_fills(res, maker_owner=2)
    assert f.side.tolist() == [1, -1] and f.qty.tolist() == [4, 4]
    mo = markouts(res, f, {"1ms": MS})
    assert mo["E"].tolist() == pytest.approx([1.5, 1.0])
    assert mo["M_1ms"].tolist() == pytest.approx([-0.5, 0.0])
    assert mo["R_1ms"].tolist() == pytest.approx([1.0, 1.0])
    s = summarize_fills(res, f, {"1ms": MS})
    assert (s["E"], s["M_1ms"], s["R_1ms"]) == pytest.approx((1.25, -0.25, 1.0))
    # identity R = E + M = s (m_{t+h} - p)
    assert mo["R_1ms"] == pytest.approx(f.side * (mid_at(res, f.t + MS) - f.price))


def test_lookahead_beyond_session_end_is_excluded():
    res = round_trip_result()  # horizon 10 ms; second fill at 4 ms
    mo = markouts(res, passive_fills(res, 2), {"7ms": 7 * MS})
    assert not np.isnan(mo["M_7ms"][0]) and np.isnan(mo["M_7ms"][1])


def test_weighted_mean_ignores_nan_and_by_group_splits():
    assert wmean(np.array([1.0, np.nan, 3.0]), np.array([1.0, 5.0, 3.0])) == pytest.approx(2.5)
    res = round_trip_result()
    f = passive_fills(res, 2)
    g = by_group(res, f, {"buys": f.side > 0, "sells": f.side < 0}, horizon="10ms")
    assert g["buys"]["n"] == 1 and g["buys"]["E"] == pytest.approx(1.5) and g["sells"]["E"] == pytest.approx(1.0)


def test_mid_at_and_obi_helpers():
    res = round_trip_result()
    assert mid_at(res, np.array([0, 1 * MS, 2 * MS, 3 * MS])).tolist() == pytest.approx([100.0, 100.5, 100.5, 100.0])
    x = ffill(np.array([np.nan, 1.0, np.nan, 3.0, np.nan]))
    assert x.tolist() == [1.0, 1.0, 1.0, 3.0, 3.0]
    s = {"bid_qty1": np.array([3.0, 0.0, 1.0]), "ask_qty1": np.array([1.0, 0.0, 3.0]),
         "bid_depth5": np.array([3.0, 0.0, 1.0]), "ask_depth5": np.array([1.0, 0.0, 3.0])}
    o = obi_from_samples(s)
    assert o[0] == 0.5 and np.isnan(o[1]) and o[2] == -0.5


def test_fills_are_adverse_on_average_only_when_informed_flow_exists():
    """Passive fills against informed takers lose more to subsequent price moves than fills against noise;
    the same statistic at random times is ~0 (null control)."""
    cfg = SimConfig(seed=5, horizon_s=120)
    from simulator.market_state.fundamental import FundamentalConfig
    cfg = SimConfig(seed=5, horizon_s=120, fundamental=FundamentalConfig(sigma=0.03))
    res = Simulator(cfg, [NoiseTrader(), InformedTrader(InformedTraderConfig(rate=10.0))]).run()
    f = passive_fills(res)
    f = f.select(f.t >= 5 * 1_000_000_000)
    informed = f.taker_owner == 2
    g = by_group(res, f, {"informed": informed, "noise": ~informed}, horizon="500ms")
    assert g["informed"]["n"] > 300 and g["noise"]["n"] > 1000
    assert g["informed"]["M"] < g["noise"]["M"] - 0.3  # informed takers => worse post-fill drift for the maker
    assert g["informed"]["M"] < -0.3
    null = random_time_markouts(res, 20_000, np.random.default_rng(1))
    assert abs(null["M_500ms"]) < 0.1 and abs(null["M_1s"]) < 0.15


# ------------------------------------------------------------------- price impact
def sweep_result():
    bk = Scripted([(0, NewLimit(1, B, 99, 10, owner=1)), (0, NewLimit(2, S, 101, 3, owner=1)),
                   (0, NewLimit(3, S, 103, 3, owner=1))], "book", feed="none")
    tk = Scripted([(1 * MS, NewMarket(4, B, 5, owner=2))], "taker", feed="none")
    return Simulator(SimConfig(horizon_s=0.01, seed_levels=0, warmup_s=0), [bk, tk]).run()


def test_aggressive_order_grouping_vwap_slippage_and_immediate_impact():
    res = sweep_result()
    ao = aggressive_orders(res)
    assert len(ao.t) == 1 and ao.qty[0] == 5 and ao.side[0] == 1 and ao.mid_before[0] == 100.0
    assert ao.vwap[0] == pytest.approx((3 * 101 + 2 * 103) / 5)
    im = impacts(res, ao, horizons_s=(0.001,))
    assert im["slippage"][0] == pytest.approx(1.8)
    assert im["I_0"][0] == pytest.approx(1.0)  # mid 100 -> (99 + 103)/2 = 101
    assert aggressive_orders(res, taker_owner=99).qty.size == 0


def test_power_law_recovery_on_synthetic_impact():
    rng = np.random.default_rng(0)
    q = np.round(np.exp(rng.uniform(0, 4.5, 200_000)))
    I = 0.8 * q ** 0.5 + rng.normal(0, 1.5, len(q))
    edges = [1, 2, 4, 8, 16, 32, 64, 128]
    fit = fit_power_law(size_bin_stats(q, I, edges))
    assert fit["alpha"] == pytest.approx(0.5, abs=0.05) and fit["c"] == pytest.approx(0.8, rel=0.1)
    assert fit["r2"] > 0.98
    lin = fit_power_law(size_bin_stats(q, 0.3 * q + rng.normal(0, 1.5, len(q)), edges))
    assert lin["alpha"] == pytest.approx(1.0, abs=0.05)


def test_bin_stats_are_additive_and_table_reports_means():
    rng = np.random.default_rng(1)
    q1, q2 = rng.integers(1, 20, 500).astype(float), rng.integers(1, 20, 500).astype(float)
    i1, i2 = q1 * 0.1, q2 * 0.1
    edges = [1, 5, 10, 20]
    merged = add_stats(size_bin_stats(q1, i1, edges), size_bin_stats(q2, i2, edges))
    both = size_bin_stats(np.r_[q1, q2], np.r_[i1, i2], edges)
    assert all(np.allclose(merged[k], both[k]) for k in both)
    for row in bin_table(both, edges):
        assert row["mean_impact"] == pytest.approx(0.1 * row["mean_q"])


def test_no_positive_bins_gives_nan_fit():
    stats = {i: np.array([100, -5.0, 10.0, 100.0 * (i + 1)]) for i in range(4)}
    assert np.isnan(fit_power_law(stats)["alpha"])


def test_merge_bursts_combines_same_instant_prints_into_one_parent_order():
    from research.price_impact import AggressiveOrders, merge_bursts
    ao = AggressiveOrders(t=np.array([5, 5, 5, 9, 9]), side=np.array([1.0, 1.0, 1.0, 1.0, -1.0]), qty=np.array([2.0, 3.0, 5.0, 4.0, 4.0]),
                          mid_before=np.array([100.0, 100.5, 101.0, 100.0, 100.0]), vwap=np.array([101.0, 102.0, 103.0, 101.0, 99.0]),
                          taker_owner=np.array([1, 1, 1, 1, 1]))
    m = merge_bursts(ao)
    assert m.t.tolist() == [5, 9, 9] and m.qty.tolist() == [10.0, 4.0, 4.0]
    assert m.mid_before[0] == 100.0  # the first print's pre-trade mid
    assert m.vwap[0] == pytest.approx((2 * 101 + 3 * 102 + 5 * 103) / 10)
    assert sorted(m.side[1:].tolist()) == [-1.0, 1.0]  # opposite sides at the same instant stay separate
    assert merge_bursts(AggressiveOrders(*[np.array([])] * 6)).t.size == 0
