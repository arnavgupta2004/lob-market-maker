"""Avellaneda-Stoikov maths, market-maker mechanics, limits and kill-switch."""
import json
import math

import numpy as np
import pytest
from hypothesis import given, strategies as st

from backtest.metrics import session_metrics
from backtest.runner import Scenario, run_many, run_session
from engine.commands import NewLimit, NewMarket
from engine.common import EventType as ET, Side
from simulator.latency.models import LatencyConfig
from simulator.simulator import SimConfig, Simulator
from strategies.baseline_mm import ASConfig, avellaneda_stoikov
from tests.helpers import Scripted

B, S = Side.BUY, Side.SELL
MS = 1_000_000
ID0 = 10**6


# ------------------------------------------------------------------- closed form
def test_as_closed_form_known_values():
    r, spread = avellaneda_stoikov(mid=100.0, q=0, gamma=0.1, sigma=2.0, tau=1.0, k=1.5)
    assert r == 100.0
    assert spread == pytest.approx(0.1 * 4 * 1 + 20 * math.log(1 + 0.1 / 1.5))
    r, _ = avellaneda_stoikov(100.0, q=5, gamma=0.1, sigma=2.0, tau=1.0, k=1.5)
    assert r == pytest.approx(100 - 5 * 0.1 * 4)


def test_as_zero_gamma_limit_is_two_over_k():
    _, spread = avellaneda_stoikov(100.0, 3, 0.0, 2.0, 1.0, 0.5)
    assert spread == pytest.approx(4.0)
    _, tiny = avellaneda_stoikov(100.0, 3, 1e-9, 0.0, 1.0, 0.5)
    assert tiny == pytest.approx(4.0, rel=1e-6)


@given(st.floats(-50, 50), st.floats(1e-3, 2), st.floats(0.1, 5), st.floats(0.01, 20), st.floats(0.05, 5))
def test_as_properties(q, gamma, sigma, tau, k):
    r0, s0 = avellaneda_stoikov(100.0, 0.0, gamma, sigma, tau, k)
    r1, s1 = avellaneda_stoikov(100.0, q, gamma, sigma, tau, k)
    assert s1 == s0 and s0 > 0  # spread does not depend on inventory
    assert (r1 - 100.0) * q <= 1e-9  # reservation price skews against the position
    # more risk aversion x variance x time-to-go => wider spread beyond the risk-neutral floor
    assert s0 >= (2.0 / gamma) * math.log1p(gamma / k) - 1e-9


# ------------------------------------------------------------------- mechanics
def quiet_sim(script_taker, mm_cfg, horizon=1.0, latency=LatencyConfig()):
    """Static wide book (bid 90 / ask 110, mid 100) + a maker + a scripted taker.

    Scripted order ids live in the 10**6 range so they never collide with ``sim.new_id()``.
    """
    book = Scripted([(0, NewLimit(ID0 + 1, S, 110, 10_000, owner=1)), (0, NewLimit(ID0 + 2, B, 90, 10_000, owner=1))], "book", feed="none")
    mm = mm_cfg.build()
    tk = Scripted(script_taker, "taker", feed="none")
    cfg = SimConfig(horizon_s=horizon, seed_levels=0, warmup_s=0.0, sample_interval_s=0.01)
    res = Simulator(cfg, [book, mm, tk]).run()
    return res, mm


def mm_events(res, mm, etype):
    return [e for e in res.events if e.type is etype and e.owner == mm.owner_id]


BASE = dict(start_s=0.0, quote_size=5, requote_interval_s=0.1)


def test_maker_quotes_around_mid_and_skews_after_being_hit():
    cfg = ASConfig(gamma=1.0, k=1.0, horizon_s=1.0, tau_mode="constant", sigma=1.0, **BASE)
    res, mm = quiet_sim([(300 * MS, NewMarket(ID0 + 50, B, 3, owner=3))], cfg)
    adds = mm_events(res, mm, ET.ADD)
    before = [e for e in adds if e.ts < 300 * MS]
    after = [e for e in adds if e.ts >= 300 * MS]
    bid0 = next(e.price for e in before if e.side is B)
    ask0 = next(e.price for e in before if e.side is S)
    # spread = 1 + 2 ln 2 = 2.386 => half 1.193: floor(98.807) = 98, ceil(101.193) = 102
    assert (bid0, ask0) == (98, 102)
    # hit on the ask: now short 3 => reservation price above mid => both quotes move up
    assert mm.inventory == -3
    # first requote after the fill (later ones chase the maker's own quotes: the book is only the maker)
    bid1 = next(e.price for e in after if e.side is B)
    ask1 = next(e.price for e in after if e.side is S)
    # r = 100 + 3*1*1*1 = 103 => bid floor(101.8) = 101, ask ceil(104.19) = 105
    assert (bid1, ask1) == (101, 105)


def test_inventory_skew_is_symmetric_for_long_position():
    cfg = ASConfig(gamma=1.0, k=1.0, horizon_s=1.0, tau_mode="constant", sigma=1.0, **BASE)
    res, mm = quiet_sim([(300 * MS, NewMarket(ID0 + 50, S, 3, owner=3))], cfg)
    assert mm.inventory == 3
    adds = mm_events(res, mm, ET.ADD)
    after = [e for e in adds if e.ts >= 300 * MS]
    bid1 = next(e.price for e in after if e.side is B)
    ask1 = next(e.price for e in after if e.side is S)
    # r = 97 => bid floor(95.8) = 95; ask ceil(98.19) = 99 (post-only clamp: > best bid 98)
    assert (bid1, ask1) == (95, 99)


def test_unchanged_quote_keeps_queue_priority_no_churn_on_static_book():
    cfg = ASConfig(gamma=0.5, k=1.0, horizon_s=1.0, tau_mode="constant", sigma=1.0, **BASE)
    res, mm = quiet_sim([], cfg, horizon=1.0)
    assert len(mm_events(res, mm, ET.ADD)) == 2 and mm.n_cancel == 0


def test_believed_state_equals_exchange_truth_at_zero_latency():
    sc = Scenario("t", SimConfig(horizon_s=60), mm=ASConfig())
    s = run_session(sc, seed=2)
    assert s.metrics["n_fills"] > 10
    assert s.mm.inventory == s.metrics["inv_final"]
    assert s.mm.cash == pytest.approx(_true_cash(s))


def _true_cash(s):
    from backtest.metrics import extract_legs
    legs = extract_legs(s.result, s.mm.owner_id, s.mm.cfg.fees)
    return float(np.sum(-legs.signed_qty * legs.price * s.result.config.tick_size - legs.fee))


def test_latency_delays_quotes_and_beliefs_lag_bounded():
    lat = LatencyConfig.symmetric(10.0)
    sc = Scenario("t", SimConfig(horizon_s=60), mm=ASConfig(latency=lat))
    s = run_session(sc, seed=2)
    first_add = next(e for e in s.result.events if e.type is ET.ADD and e.owner == s.mm.owner_id)
    # decision at start (from an already-delivered feed), then one-way order latency
    assert first_add.ts == int(5e9) + 10 * MS
    assert abs(s.mm.inventory - s.metrics["inv_final"]) <= 3 * s.mm.cfg.quote_size


# ---------------------------------------------------------------------- limits
def test_soft_inventory_limit_stops_quoting_the_extending_side():
    cfg = ASConfig(gamma=0.1, k=1.0, sigma=0.1, inventory_limit=10, kill_inventory=1000, **BASE)
    hits = [((300 + 100 * i) * MS, NewMarket(ID0 + 50 + i, B, 5, owner=3)) for i in range(6)]
    res, mm = quiet_sim(hits, cfg, horizon=1.5)
    m = session_metrics(res, mm.owner_id, start_s=0)
    assert m["inv_q05"] >= -10 and m["inv_max_abs"] <= 10
    assert mm.inventory == -10  # filled the limit, then only the bid stays
    assert mm.live_quote(S) is None


def test_kill_switch_on_inventory_cancels_flattens_and_stops_quoting():
    cfg = ASConfig(gamma=0.1, k=1.0, kill_inventory=5, inventory_limit=100, **BASE)
    res, mm = quiet_sim([(300 * MS, NewMarket(ID0 + 50, B, 5, owner=3))], cfg, horizon=1.5)
    assert mm.killed and mm.kill_reason == "inventory" and mm.n_flatten == 1
    assert mm.inventory == 0  # flattened
    late_adds = [e for e in mm_events(res, mm, ET.ADD) if e.ts > mm.kill_time]
    assert late_adds == []  # no quoting after the kill
    assert not any(lv is not None for lv in (mm.live_quote(B), mm.live_quote(S)))
    # flatten was a taker buy of 5 lots at the touch
    fl = [e for e in res.events if e.type is ET.TRADE and e.owner == mm.owner_id]
    assert sum(e.qty for e in fl) == 5 and all(e.side is B for e in fl)


def test_kill_switch_without_flatten_leaves_inventory():
    cfg = ASConfig(gamma=0.1, k=1.0, kill_inventory=5, flatten_on_kill=False, **BASE)
    res, mm = quiet_sim([(300 * MS, NewMarket(ID0 + 50, B, 5, owner=3))], cfg, horizon=1.0)
    assert mm.killed and mm.inventory == -5 and mm.n_flatten == 0


def test_drawdown_kill_switch_direct():
    cfg = ASConfig(max_drawdown=0.5, **BASE)
    mm = cfg.build()
    mm.tick, mm.owner_id = 0.01, 1
    mm.replica.submit_limit(1, B, 9999, 10)
    mm.replica.submit_limit(2, S, 10001, 10)  # mid 10000
    mm.inventory, mm.cash = 40, -40 * 10000 * 0.01  # bought 40 lots at the mid
    assert mm._risk_check(None, 1) == [] and not mm.killed
    mm.replica.cancel(1); mm.replica.cancel(2)
    mm.replica.submit_limit(3, B, 9979, 10)
    mm.replica.submit_limit(4, S, 9981, 10)  # mid 9980: 20 ticks * 40 lots * .01 = -8 => dd 8 >= .5
    assert mm._risk_check(None, 2) == [] or mm.killed
    assert mm.killed and mm.kill_reason == "drawdown"


def test_fees_reduce_believed_cash():
    from backtest.costs import FeeModel
    cfg = ASConfig(fees=FeeModel(maker_bps=10), k=0.5, sigma=0.1, **BASE)
    res, mm = quiet_sim([(300 * MS, NewMarket(ID0 + 50, B, 3, owner=3))], cfg)
    assert mm.fees_paid == pytest.approx(3 * mm.tick * 102 * 10e-4)


# ---------------------------------------------------------------- runner / experiment glue
def test_run_session_is_deterministic_and_parallel_matches_sequential():
    sc = Scenario("t", SimConfig(horizon_s=30), mm=ASConfig())
    a, b = run_session(sc, 4).metrics, run_session(sc, 4).metrics
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)  # nan-safe equality
    seq = run_many(sc, [1, 2, 3], workers=1)
    par = run_many(sc, [1, 2, 3], workers=2)
    assert json.dumps(seq, sort_keys=True) == json.dumps(par, sort_keys=True)


def test_mm_does_not_perturb_noise_traders_random_streams():
    sc0 = Scenario("n", SimConfig(horizon_s=10))
    sc1 = Scenario("m", SimConfig(horizon_s=10), mm=ASConfig())
    r0, r1 = run_session(sc0, 1).result, run_session(sc1, 1).result
    n0 = [c for c in r0.commands if isinstance(c, NewMarket) and c.owner == 1]
    n1 = [c for c in r1.commands if isinstance(c, NewMarket) and c.owner == 1]
    # the same random stream drives the noise trader; identical prefix until book-state dependence
    assert n0[0].qty == n1[0].qty and n0[0].side == n1[0].side


# ------------------------------------------------------------------ safety guards
def _mm_with_replica(**kw):
    mm = ASConfig(**{**BASE, **kw}).build()
    mm.tick, mm.owner_id = 0.01, 1
    return mm


def test_sigma_bounds_clamp_the_estimator_state():
    mm = _mm_with_replica(sigma_bounds=(0.5, 3.0), vol_halflife_s=0.1)
    for i, px in enumerate([100, 10_000, 100, 10_000, 100]):  # wild mid jumps every second
        mm.replica = type(mm.replica)()
        mm.replica.submit_limit(10 * i + 1, B, px - 1, 1)
        mm.replica.submit_limit(10 * i + 2, S, px + 1, 1)
        mm._update_vol((i + 1) * 1_000_000_000)
    assert mm.sigma == pytest.approx(3.0) and mm.n_sigma_clamped >= 1
    quiet = _mm_with_replica(sigma_bounds=(0.5, 3.0), vol_halflife_s=0.1, sigma0=1.0)
    for i in range(5):  # perfectly still market => estimate decays to the lower bound
        quiet.replica = type(quiet.replica)()
        quiet.replica.submit_limit(1, B, 99, 1); quiet.replica.submit_limit(2, S, 101, 1)
        quiet._update_vol((i + 1) * 1_000_000_000)
    assert quiet.sigma == pytest.approx(0.5)


def test_sigma_bounds_can_be_disabled():
    mm = _mm_with_replica(sigma_bounds=None, vol_halflife_s=0.1)
    for i, px in enumerate([100, 10_000]):
        mm.replica = type(mm.replica)()
        mm.replica.submit_limit(10 * i + 1, B, px - 1, 1); mm.replica.submit_limit(10 * i + 2, S, px + 1, 1)
        mm._update_vol((i + 1) * 1_000_000_000)
    assert mm.sigma > 100 and mm.n_sigma_clamped == 0


def test_quote_offset_cap():
    mm = _mm_with_replica(max_quote_offset_ticks=20)
    assert mm._clamp(B, (50, 5), 100.0) == (80, 5) and mm.n_offset_clamped == 1
    assert mm._clamp(S, (500, 5), 100.0) == (120, 5)
    assert mm._clamp(B, (150, 5), 100.0) == (120, 5)  # a bid far *above* the mid is also capped
    assert mm._clamp(B, (95, 5), 100.0) == (95, 5) and mm.n_offset_clamped == 3
    free = _mm_with_replica(max_quote_offset_ticks=None)
    assert free._clamp(B, (50, 5), 100.0) == (50, 5)


def test_guarded_high_gamma_session_does_not_diverge_regression():
    """Seed 4200018, gamma=0.2 used to run away (sigma-hat ~1e45) before the guards existed."""
    from experiments.inventory.gamma_sweep import scenario
    s = run_session(scenario("noise", 0.2, 0.939, 180.0), 4_200_018, keep_result=False)
    assert s.metrics["diverged"] == 0.0 and s.metrics["mid_max_dev_ticks"] < 200
    assert s.metrics["n_sigma_clamped"] >= 0
