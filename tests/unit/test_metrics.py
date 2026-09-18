"""P&L accounting and metrics against hand-computed scenarios."""
import numpy as np
import pytest

from backtest.costs import FeeModel
from backtest.metrics import extract_legs, realized_pnl, session_metrics
from engine.commands import Cancel, NewLimit, NewMarket
from engine.common import Side
from simulator.simulator import SimConfig, Simulator
from tests.helpers import Scripted

B, S = Side.BUY, Side.SELL
MS = 1_000_000


def round_trip_result():
    """Owner 2 (maker) buys 4@99 then sells 4@101; hand-computed in ticks:

    mid before buy = 100.5 -> edge 4*(100.5-99) = 6;  mid before sell = 100 -> edge 4*(101-100) = 4
    mid moves 100.5 -> 100 while long 4  -> inventory P&L = 4*(-0.5) = -2
    total = 10 - 2 = 8 = realised (bought 99, sold 101, 4 lots)
    """
    bk = Scripted([(0, NewLimit(1, S, 102, 100, owner=1)), (0, NewLimit(2, B, 98, 100, owner=1))], "book", feed="none")
    mm = Scripted([(1 * MS, NewLimit(3, B, 99, 10, owner=2)), (3 * MS, NewLimit(5, S, 101, 10, owner=2))], "maker", feed="none")
    tk = Scripted([(2 * MS, NewMarket(4, S, 4, owner=3)), (4 * MS, NewMarket(6, B, 4, owner=3))], "taker", feed="none")
    cfg = SimConfig(horizon_s=0.01, seed_levels=0, warmup_s=0.0, sample_interval_s=0.001)
    return Simulator(cfg, [bk, mm, tk]).run()


def test_pnl_decomposition_matches_hand_computation():
    res = round_trip_result()
    m = session_metrics(res, owner=2)
    tick = 0.01
    assert m["pnl_edge"] == pytest.approx(10 * tick)
    assert m["pnl_inventory"] == pytest.approx(-2 * tick)
    assert m["pnl_gross"] == pytest.approx(8 * tick)
    assert m["pnl_realized"] == pytest.approx(8 * tick)
    assert m["pnl_unrealized"] == pytest.approx(0.0, abs=1e-12)
    assert m["pnl_net"] == pytest.approx(m["pnl_gross"] - m["fees"])
    assert m["n_fills"] == 2 and m["n_maker_fills"] == 2 and m["filled_qty"] == 8
    assert m["avg_edge_ticks"] == pytest.approx(10 / 8)
    assert m["inv_final"] == 0


def test_fees_and_rebates():
    res = round_trip_result()
    notional = (4 * 99 + 4 * 101) * 0.01
    paid = session_metrics(res, 2, FeeModel(maker_bps=10))
    assert paid["fees"] == pytest.approx(notional * 10e-4)
    assert paid["pnl_net"] == pytest.approx(paid["pnl_gross"] - paid["fees"])
    rebate = session_metrics(res, 2, FeeModel(maker_bps=-5))
    assert rebate["fees"] < 0 and rebate["pnl_net"] > rebate["pnl_gross"]


def test_taker_leg_uses_taker_fee_and_signs():
    res = round_trip_result()
    legs = extract_legs(res, owner=3, fees=FeeModel(maker_bps=0, taker_bps=20))
    assert legs.signed_qty.tolist() == [-4, 4] and not legs.is_maker.any()
    assert legs.fee.sum() == pytest.approx((4 * 99 + 4 * 101) * 0.01 * 20e-4)


def test_execution_metrics():
    m = session_metrics(round_trip_result(), owner=2)
    assert m["n_quotes"] == 2 and m["n_cancels"] == 0
    assert m["fill_rate_orders"] == 1.0
    assert m["fill_ratio_qty"] == pytest.approx(8 / 20)


def test_realized_pnl_average_cost_and_position_flip():
    from backtest.metrics import Legs
    def legs(prices, qtys):
        n = len(prices)
        return Legs(np.arange(n), np.array(prices, float), np.array(qtys), np.zeros(n, bool), np.zeros(n), np.zeros(n))
    # buy 2@10, buy 2@12 (avg 11), sell 3@13 -> +6 ; remaining long 1@11
    assert realized_pnl(legs([10, 12, 13], [2, 2, -3]), 1.0) == pytest.approx(6.0)
    # short 2@10, buy 5@9: closes 2 (+2), flips long 3@9, then sell 3@10 (+3)
    assert realized_pnl(legs([10, 9, 10], [-2, 5, -3]), 1.0) == pytest.approx(5.0)


def test_no_fills_gives_zero_pnl_and_nan_rates():
    cfg = SimConfig(horizon_s=2.0, seed_levels=2, warmup_s=0.0)
    res = Simulator(cfg, [Scripted([], "idle")]).run()
    m = session_metrics(res, owner=1)
    assert m["pnl_net"] == 0 and m["n_fills"] == 0 and np.isnan(m["fill_rate_orders"])
    assert m["inv_max_abs"] == 0 and m["max_drawdown"] == 0


def test_max_drawdown_and_inventory_stats_from_equity_curve():
    m = session_metrics(round_trip_result(), owner=2)
    assert m["max_drawdown"] >= 0
    assert m["inv_max_abs"] == 4  # held 4 lots between t=2ms and t=4ms
    assert m["inv_q95"] <= 4


def integral_inventory_pnl(res, owner):
    """sum_j q_j (m_j - m_{j-1}) over mid-change instants, q_j = inventory after fills at t <= t_j."""
    legs = extract_legs(res, owner, FeeModel())
    mh = res.mid_history
    cum = np.cumsum(legs.signed_qty)
    total = 0.0
    for j in range(1, len(mh.t)):
        k = np.searchsorted(legs.t, mh.t[j], side="right")
        q = cum[k - 1] if k else 0
        total += q * (mh.mid[j] - mh.mid[j - 1])
    return total * res.config.tick_size


def test_inventory_pnl_residual_equals_integral_form_scripted():
    res = round_trip_result()
    assert session_metrics(res, 2)["pnl_inventory"] == pytest.approx(integral_inventory_pnl(res, 2))


def test_inventory_pnl_residual_equals_integral_form_random_session():
    from backtest.runner import Scenario, run_session
    from strategies.baseline_mm import ASConfig
    s = run_session(Scenario("t", SimConfig(horizon_s=60), mm=ASConfig()), seed=3)
    m, res = s.metrics, s.result
    assert m["n_fills"] > 20
    # the start-up mid (before the first fill) contributes nothing since q = 0 there
    assert m["pnl_inventory"] == pytest.approx(integral_inventory_pnl(res, s.mm.owner_id), abs=1e-9)
