"""Adaptive market maker: each component in isolation, the tracker's maths, queue decision, latency measurement."""
import math

import numpy as np
import pytest

from backtest.runner import Scenario, run_session
from engine.commands import Cancel, Modify, NewLimit
from engine.common import Side
from research.queue_position import FillModel, fit_fill_model
from simulator.latency.models import LatencyConfig
from simulator.simulator import NS, SimConfig, Simulator
from strategies.adaptive_mm import AdaptiveConfig, AdaptiveMM, AdverseSelectionTracker
from strategies.base import _Live
from tests.helpers import Scripted

B, S = Side.BUY, Side.SELL
MS = 1_000_000
OFF = dict(use_inventory_skew=False, use_inventory_size=False, use_obi=False, use_adverse=False,
           use_queue=False, use_latency=False, use_returns=False)
FM = FillModel(intercept=0.0, b_q=-1.0, b_d=-1.0)


def mk(**kw) -> AdaptiveMM:
    cfg = AdaptiveConfig(**{**OFF, "k": 0.5, "start_s": 0.0, "inventory_limit": 50, "quote_size": 5, **kw})
    mm = cfg.build()
    mm.tick, mm.owner_id = 0.01, 1
    return mm


def with_book(mm, bids=((99, 30),), asks=((101, 10),)):
    oid = 1000
    for px, q in bids:
        mm.replica.submit_limit(oid, B, px, q); oid += 1
    for px, q in asks:
        mm.replica.submit_limit(oid, S, px, q); oid += 1
    return mm


# --------------------------------------------------------------------------- baseline (all off)
def test_all_components_off_is_symmetric_quoting_at_mid_plus_minus_one_over_k():
    mm = with_book(mk(), bids=((99, 5),), asks=((101, 5),))  # mid 100
    assert mm.h0() == 2.0
    assert mm.desired_quotes(0, 100.0) == ((98, 5), (102, 5))
    mm.inventory = 25  # inventory ignored when the components are off
    assert mm.desired_quotes(0, 100.0) == ((98, 5), (102, 5))
    assert mk(base_half_spread=1.4).h0() == 1.4


# ------------------------------------------------------------------------- one component at a time
def test_inventory_skew_shifts_the_centre_against_the_position():
    mm = with_book(mk(use_inventory_skew=True, lambda_inv=0.1))
    mm.inventory = 10  # long 10 lots: centre moves down by 0.1 * 10 = 1 tick
    assert mm.components(0, 100.0)["inv"] == pytest.approx(-1.0)
    (bid, _), (ask, _) = mm.desired_quotes(0, 100.0)
    assert (bid, ask) == (math.floor(99 - 2), math.ceil(99 + 2))
    mm.inventory = -10
    assert mm.components(0, 100.0)["inv"] == pytest.approx(+1.0)


def test_inventory_size_scaling_reduces_the_extending_side():
    mm = with_book(mk(use_inventory_size=True, kappa_size=1.0))
    mm.inventory = 10  # I/Imax = 0.2: bid (extends the long) 5*(1-.2)=4, ask 5*(1+.2)=6
    (_, qb), (_, qa) = mm.desired_quotes(0, 100.0)
    assert (qb, qa) == (4, 6)
    mm.inventory = 50
    (_, qb), _ = mm.desired_quotes(0, 100.0)
    assert qb == 0  # no bid at the position limit


def test_obi_shifts_the_centre_toward_the_heavy_side():
    mm = with_book(mk(use_obi=True, beta_obi=0.4), bids=((99, 30),), asks=((101, 10),))  # OBI = 0.5
    assert mm.components(0, 100.0)["obi"] == pytest.approx(0.2)
    mm2 = with_book(mk(use_obi=True, beta_obi=0.4), bids=((99, 10),), asks=((101, 30),))
    assert mm2.components(0, 100.0)["obi"] == pytest.approx(-0.2)
    empty = mk(use_obi=True)
    assert empty.components(0, 100.0)["obi"] == 0.0  # no book -> no signal


def test_returns_component_uses_the_lookback_mid():
    mm = mk(use_returns=True, beta_ret=-0.5, ret_horizon_s=1.0)  # fade the last second's move
    mm._mid_hist.extend([(0, 100.0), (int(0.5 * NS), 100.5), (int(1.0 * NS), 101.0)])
    assert mm.components(2 * NS, 102.0)["ret"] == pytest.approx(-0.5 * (102.0 - 101.0))  # mid 1 s ago = 101.0
    assert mk(use_returns=False, beta_ret=-0.5).components(0, 102.0)["ret"] == 0.0


def test_latency_component_is_one_sigma_of_the_unobserved_price_move():
    mm = mk(use_latency=True, c_latency=1.0)
    mm._var = 4.0  # sigma = 2 ticks/sqrt(s)
    assert mm.components(0, 100.0)["lat"] == 0.0  # nothing measured yet
    mm.ack_latency_s = 0.04
    assert mm.components(0, 100.0)["lat"] == pytest.approx(2.0 * math.sqrt(0.04))
    (bid, _), (ask, _) = with_book(mm).desired_quotes(0, 100.0)
    assert (bid, ask) == (math.floor(100 - 2.4), math.ceil(100 + 2.4))  # both sides widen equally


def test_adverse_component_widens_only_the_affected_side():
    mm = mk(use_adverse=True, c_adverse=1.0)
    mm.tracker._est[1] = 1.5  # bid fills have been costing 1.5 ticks
    k = mm.components(0, 100.0)
    assert (k["adv_bid"], k["adv_ask"]) == (1.5, 0.0)
    (bid, _), (ask, _) = with_book(mm).desired_quotes(0, 100.0)
    assert (bid, ask) == (math.floor(100 - 3.5), math.ceil(100 + 2.0))


# ------------------------------------------------------------------- adverse-selection tracker
def test_tracker_measures_cost_after_the_horizon_only():
    t = AdverseSelectionTracker(horizon_s=0.5, alpha=0.1, halflife_s=0)
    t.on_fill(0, +1, 1, pre_mid=100.0)  # maker bought at a mid of 100
    t.evaluate(int(0.4 * NS), 99.0)
    assert t.estimate(B) == 0.0 and t.n_measured == 0  # horizon not reached
    t.evaluate(int(0.5 * NS), 99.0)  # mid fell by 1: cost = -(+1)(99-100) = +1
    assert t.n_measured == 1 and t.estimate(B) == pytest.approx(0.1)
    assert t.estimate(S) == 0.0


def test_tracker_sign_convention_for_sells_and_favourable_moves():
    t = AdverseSelectionTracker(horizon_s=0.0, alpha=1.0, halflife_s=0)
    t.on_fill(0, -1, 1, 100.0)  # maker sold
    t.evaluate(1, 101.0)  # price rose after the sale: adverse, cost +1
    assert t.estimate(S) == pytest.approx(1.0)
    t.on_fill(2, +1, 1, 100.0)  # maker bought, price rose: favourable
    t.evaluate(3, 101.0)
    assert t.estimate(B) == 0.0  # negative cost is clipped at 0 in the estimate
    assert t._est[1] == pytest.approx(-1.0)  # ...but the raw EWMA remembers it


def test_tracker_quantity_weighting_cap_and_decay():
    t = AdverseSelectionTracker(horizon_s=0.0, alpha=0.1, halflife_s=0)
    t.on_fill(0, 1, 3, 100.0)
    t.evaluate(1, 99.0)
    assert t.estimate(B) == pytest.approx((1 - 0.9 ** 3) * 1.0)
    capped = AdverseSelectionTracker(horizon_s=0.0, alpha=1.0, halflife_s=0, cap=2.0)
    capped.on_fill(0, 1, 1, 100.0)
    capped.evaluate(1, 90.0)  # cost 10
    assert capped.estimate(B) == 2.0
    d = AdverseSelectionTracker(horizon_s=0.0, alpha=1.0, halflife_s=10.0)
    d.on_fill(0, 1, 1, 100.0)
    d.evaluate(0, 99.0)
    assert d.estimate(B) == pytest.approx(1.0)
    d.evaluate(10 * NS, None)  # one half-life later, no new information
    assert d.estimate(B) == pytest.approx(0.5)


def test_pre_apply_captures_the_pre_trade_mid_and_feeds_the_tracker():
    mm = with_book(mk(use_adverse=True), bids=((99, 10),), asks=((101, 10),))  # mid 100
    from engine.common import EventType
    from engine.events import Event
    trade = Event(seq=1, ts=5, type=EventType.TRADE, order_id=9, side=S, price=99, qty=2, owner=7, maker_id=1000, maker_owner=1,
                  maker_remaining=8)
    mm._pre_apply(trade)  # replica still holds the pre-trade book
    assert list(mm.tracker._pending) == [(5, +1, 2, 100.0)]  # taker sold => maker bought, pre-trade mid 100


# ------------------------------------------------------------------------------ queue decision
def _queue_setup(cur_price, want_price, center, bids, own_qty_ahead=0):
    mm = mk(use_queue=True, fill_model=FM)
    # own resting bid (id 7) behind `own_qty_ahead` lots at cur_price; other levels from `bids`
    if own_qty_ahead:
        mm.replica.submit_limit(500, B, cur_price, own_qty_ahead)
    mm.replica.submit_limit(7, B, cur_price, 1, owner=1)
    for i, (px, q) in enumerate(bids):
        mm.replica.submit_limit(600 + i, B, px, q)
    mm.replica.submit_limit(900, S, 110, 5)
    mm._center = center
    return mm, _Live(7, cur_price, 1, 0)


def test_queue_rule_keeps_a_front_of_queue_quote_when_moving_gains_too_little():
    # cur: at the touch (99), Q=0 -> p=sigmoid(0)=.5, edge 1.5 => EV .75;  want 100 (improves touch: alone, p=.5), edge .5 => EV .25
    mm, cur = _queue_setup(99, 100, 100.5, bids=())
    assert mm.keep_quote(B, cur, (100, 1)) is True and mm.n_kept_by_queue == 1


def test_queue_rule_moves_when_the_current_position_is_hopeless():
    # cur: 2 ticks behind a 30-lot touch, Q=20 ahead: p = sigmoid(-ln 21 - 2) ~ 0.006;  want the touch (98? use 100): alone => .5
    mm, cur = _queue_setup(97, 100, 100.5, bids=((99, 30),), own_qty_ahead=20)
    assert mm.keep_quote(B, cur, (100, 1)) is False


def test_queue_rule_disabled_or_unknown_order_defaults_to_moving():
    mm, cur = _queue_setup(99, 100, 100.5, bids=())
    off = mk(use_queue=False)
    off.replica = mm.replica; off._center = 100.5
    assert off.keep_quote(B, cur, (100, 1)) is False
    ghost = _Live(12345, 99, 1, 0)  # not visible in the replica yet
    assert mm.keep_quote(B, ghost, (100, 1)) is False
    with pytest.raises(ValueError):
        AdaptiveConfig(use_queue=True).build()  # a fill model is mandatory


def test_fill_model_probabilities_and_fit_recovers_known_coefficients():
    assert FM.p_fill(0, 0) == 0.5 and FM.p_fill(20, 0) < FM.p_fill(0, 0) and FM.p_fill(0, 3) < FM.p_fill(0, 0)
    rng = np.random.default_rng(0)
    n = 40_000
    Q, d = rng.integers(0, 40, n).astype(float), rng.integers(0, 3, n).astype(float)
    true = FillModel(0.8, -0.5, -1.2)
    p = 1 / (1 + np.exp(-(true.intercept + true.b_q * np.log1p(Q) + true.b_d * d)))
    hit = rng.random(n) < p
    ds = {"Q": Q, "d_touch": d, "age_s": np.zeros(n), "t_check_ns": np.zeros(n),
          "fill_ns": np.where(hit, 0.5 * NS, np.nan)}
    fit = fit_fill_model([ds], delta_s=1.0, dwell_s=3.0)
    assert (fit.intercept, fit.b_q, fit.b_d) == pytest.approx((0.8, -0.5, -1.2), abs=0.06)


# ---------------------------------------------------------- base-class hooks used by the adaptive strategy
class _Sim:
    def __init__(self): self.n = 100
    def new_id(self): self.n += 1; return self.n


def test_modify_to_shrink_keeps_priority_instead_of_cancel_and_repost():
    for shrink, expect_cancel in ((True, False), (False, True)):
        mm = with_book(mk(use_inventory_size=True, modify_to_shrink=shrink), bids=((98, 5),), asks=((102, 5),))
        mm.replica.submit_limit(7, B, 98, 5, owner=1)
        mm._live[B] = _Live(7, 98, 5, 0)
        mm.inventory = 40  # bid size -> 5 * (1 - .8) = 1
        cmds = mm._requote(_Sim(), 1)
        bid_cmds = [c for c in cmds if isinstance(c, (Modify, Cancel)) and c.order_id == 7]
        assert isinstance(bid_cmds[0], Cancel) == expect_cancel and isinstance(bid_cmds[0], Modify) == (not expect_cancel)
        if shrink:
            assert (bid_cmds[0].new_price, bid_cmds[0].new_qty) == (98, 1) and mm._live[B].qty == 1


def test_ack_latency_is_measured_from_own_order_acknowledgements():
    cfg = AdaptiveConfig(**{**OFF, "k": 0.5, "start_s": 0.0, "latency": LatencyConfig.symmetric(0.0)})
    lat = LatencyConfig(order=__import__("simulator.latency.models", fromlist=["x"]).ConstantLatency(5 * MS),
                        feed=__import__("simulator.latency.models", fromlist=["x"]).ConstantLatency(10 * MS))
    mm = AdaptiveConfig(**{**OFF, "k": 0.5, "start_s": 0.0, "latency": lat}).build()
    book = Scripted([(0, NewLimit(10**6, S, 110, 1000, owner=1)), (0, NewLimit(10**6 + 1, B, 90, 1000, owner=1))], "book", feed="none")
    res = Simulator(SimConfig(horizon_s=1.0, seed_levels=0, warmup_s=0.0), [book, mm]).run()
    assert mm.ack_latency_s == pytest.approx(0.015, abs=1e-9)  # order 5 ms + feed 10 ms


# ------------------------------------------------------------------------------------ integration
LADDER = [
    dict(**OFF),
    dict(**{**OFF, "use_inventory_skew": True}),
    dict(**{**OFF, "use_inventory_skew": True, "use_inventory_size": True}),
    dict(**{**OFF, "use_inventory_skew": True, "use_obi": True}),
    dict(**{**OFF, "use_adverse": True}),
    dict(**{**OFF, "use_queue": True}),
    dict(**{**OFF, "use_latency": True}),
    dict(use_inventory_skew=True, use_inventory_size=True, use_obi=True, use_adverse=True, use_queue=True, use_latency=True, use_returns=True,
         beta_ret=-0.2),
]


@pytest.mark.parametrize("flags", LADDER)
def test_every_ladder_configuration_runs_and_is_deterministic(flags):
    cfg = AdaptiveConfig(k=0.94, fill_model=FM, latency=LatencyConfig.symmetric(2.0), **flags)
    sc = Scenario("t", SimConfig(horizon_s=25), mm=cfg)
    a, b = run_session(sc, 3), run_session(sc, 3)
    assert a.metrics["n_fills"] > 0 and a.metrics["diverged"] == 0.0
    import json
    assert json.dumps(a.metrics, sort_keys=True) == json.dumps(b.metrics, sort_keys=True)
    a.result.book.validate()


def test_adverse_tracker_learns_something_in_a_market_with_informed_flow():
    from simulator.order_flow.informed import InformedTraderConfig
    cfg = AdaptiveConfig(k=0.565, **{**OFF, "use_adverse": True})
    sc = Scenario("t", SimConfig(horizon_s=60), informed=InformedTraderConfig(rate=10.0), mm=cfg)
    s = run_session(sc, 1)
    assert s.metrics["n_adverse_measured"] > 50
    assert s.metrics["mean_abs_adverse"] > 0.1  # the estimated adverse cost widened quotes on average
