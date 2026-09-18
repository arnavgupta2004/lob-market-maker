import numpy as np
import pytest

from engine.commands import NewLimit
from engine.common import Side
from research.resilience import ShockInjector, half_life, resilience_curves
from simulator.order_flow.noise import NoiseTrader
from simulator.simulator import NS, SimConfig, Simulator
from tests.helpers import Scripted


def test_half_life_on_exponential_decay_and_permanent_impact():
    tau = np.arange(1, 201) * 0.05
    curve = 2.0 * np.exp(-tau / 1.0)  # time constant 1 s => half-life ln2 = 0.69 s
    h = half_life(curve, tau)
    assert h["half_life_s"] == pytest.approx(np.log(2), abs=0.02) and h["initial"] == pytest.approx(2.0, rel=0.1)
    for T in (0.3, 2.0):  # any time constant, and independent of the smoothing window
        tt = np.arange(1, 401) * 0.05
        for sm in (1, 3, 5):
            assert half_life(np.exp(-tt / T), tt, smooth=sm)["half_life_s"] == pytest.approx(T * np.log(2), rel=0.06)
    assert h["residual"] < 0.01
    perm = half_life(np.full(200, 1.5), tau)
    assert perm["half_life_s"] is None and perm["residual"] == pytest.approx(1.0)
    assert half_life(np.full(200, np.nan), tau)["half_life_s"] is None
    assert half_life(np.zeros(200), tau)["half_life_s"] is None  # nothing to recover from


def test_shock_injector_sends_alternating_sides_at_the_requested_times():
    inj = ShockInjector([1.0, 2.0, 3.0], size=5)
    book = Scripted([(0, NewLimit(10**6 + i, Side.SELL if i % 2 else Side.BUY, 100 + (5 if i % 2 else -5), 1000, owner=1)) for i in range(2)], "b", feed="none")
    res = Simulator(SimConfig(horizon_s=4.0, seed_levels=0, warmup_s=0), [book, inj]).run()
    assert inj.log == [(NS, 1), (2 * NS, -1), (3 * NS, 1)]
    assert [e.ts for e in res.events if e.type.value == "TRADE"] == [NS, 2 * NS, 3 * NS]


def test_curves_on_a_hand_built_book_with_known_recovery():
    dt = 0.1
    t = (np.arange(0, 400) * dt * NS).astype(np.int64)
    mid = np.full(400, 100.0); spread = np.full(400, 2.0); ask_d = np.full(400, 50.0)
    shock_i = 100  # shock at sample 100 (t = 10 s): depth halves then recovers linearly over 5 s, mid +1 permanent, spread x2 then recovers
    rec = np.clip((np.arange(400) - shock_i) / 50, 0, 1)
    ask_d[shock_i:] = 25 + 25 * rec[shock_i:]
    spread[shock_i:] = 4 - 2 * rec[shock_i:]
    mid[shock_i:] = 101.0
    samples = {"t_ns": t, "mid": mid, "spread": spread, "ask_depth5": ask_d, "bid_depth5": np.full(400, 50.0)}
    c = resilience_curves(samples, dt, [(int(shock_i * dt * NS) - 1, 1)], horizon_s=8.0, pre_s=1.0)
    assert c["n_shocks"] == 1
    assert c["mid"][0] == pytest.approx(1.0) and c["mid"][-1] == pytest.approx(1.0)  # permanent
    assert c["spread"][0] == pytest.approx(0.96, abs=0.05) and c["spread"][-1] == pytest.approx(0.0, abs=1e-9)
    assert c["depth"][0] == pytest.approx(0.49, abs=0.03) and c["depth"][-1] == pytest.approx(0.0, abs=1e-9)
    # a sell shock reads the bid side and flips the mid sign: mid moved UP, so a sell's impact is negative
    cs = resilience_curves(samples, dt, [(int(shock_i * dt * NS) - 1, -1)], horizon_s=8.0)
    assert cs["mid"][0] == pytest.approx(-1.0) and cs["depth"][0] == pytest.approx(0.0)


def test_shocks_too_close_to_the_edges_are_skipped():
    res = Simulator(SimConfig(seed=1, horizon_s=30, sample_interval_s=0.05), [NoiseTrader()]).run()
    c = resilience_curves(res.samples, 0.05, [(int(0.2 * NS), 1), (int(29.9 * NS), 1), (int(15 * NS), 1)], horizon_s=5.0)
    assert c["n_shocks"] == 1
