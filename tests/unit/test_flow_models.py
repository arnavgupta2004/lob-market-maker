"""Hawkes arrivals and metaorder traders."""
import math

import numpy as np
import pytest

from engine.commands import NewMarket
from research.stylized_facts import acf, fano_factor, interarrival_stats, power_law_exponent
from simulator.order_flow.arrivals import NS, HawkesArrival, PoissonArrival
from simulator.order_flow.distributions import Const, Pareto
from simulator.order_flow.metaorder import MetaOrderConfig, MetaOrderTrader
from simulator.order_flow.noise import NoiseTrader, NoiseTraderConfig
from simulator.simulator import SimConfig, Simulator


def arrival_times(proc, n, seed=0):
    rng = np.random.default_rng(seed)
    t, out = 0, []
    for _ in range(n):
        t += proc.next_gap_ns(rng)
        out.append(t)
    return np.array(out, dtype=float)


def test_hawkes_long_run_rate_and_validation():
    h = HawkesArrival(mu=10.0, alpha=8.0, beta=20.0)  # n = 0.4 => rate = 10 / 0.6
    assert h.branching_ratio == pytest.approx(0.4) and h.mean_rate == pytest.approx(16.667, rel=1e-3)
    t = arrival_times(h, 60_000)
    assert len(t) / (t[-1] / NS) == pytest.approx(h.mean_rate, rel=0.05)
    with pytest.raises(ValueError):
        HawkesArrival(1.0, 5.0, 5.0)  # non-stationary
    w = HawkesArrival.with_mean_rate(65.0, 0.5, 10.0)
    assert w.mean_rate == pytest.approx(65.0)


def test_hawkes_clusters_while_poisson_does_not():
    T = 3000 * NS
    hk = arrival_times(HawkesArrival.with_mean_rate(20.0, 0.7, 10.0), 60_000)
    po = arrival_times(PoissonArrival(20.0), 60_000)
    end = int(min(hk[-1], po[-1]))
    f_h, f_p = fano_factor(hk, 1.0, 0, end), fano_factor(po, 1.0, 0, end)
    assert f_p == pytest.approx(1.0, abs=0.15)
    # theory for a stationary Hawkes process with branching ratio n: Fano -> 1/(1-n)^2 at long windows (= 11.1 here)
    assert 4.0 < f_h < 14.0
    assert interarrival_stats(hk)["cv"] > 1.3 and interarrival_stats(po)["cv"] == pytest.approx(1.0, abs=0.05)


def test_hawkes_is_deterministic_given_rng():
    a = arrival_times(HawkesArrival(5, 3, 10), 500, seed=4)
    b = arrival_times(HawkesArrival(5, 3, 10), 500, seed=4)
    assert np.array_equal(a, b) and np.all(np.diff(a) > 0)


def test_noise_trader_accepts_hawkes_arrivals_and_keeps_its_mean_rate():
    cfg = NoiseTraderConfig(limit_rate=50.0, market_rate=15.0)
    nt = NoiseTrader(cfg, arrivals=HawkesArrival.with_mean_rate(65.0, 0.6, 8.0))
    res = Simulator(SimConfig(seed=2, horizon_s=120), [nt]).run()
    n = sum(1 for c in res.commands if getattr(c, "owner", None) == 1 and type(c).__name__ in ("NewLimit", "NewMarket"))
    assert n / 120 == pytest.approx(65.0, rel=0.12)
    res.book.validate()


# ---------------------------------------------------------------------------- metaorders
def run_meta(cfg, seed=1, horizon=200.0):
    m = MetaOrderTrader(cfg)
    res = Simulator(SimConfig(seed=seed, horizon_s=horizon, seed_levels=50, seed_size=100000), [m]).run()
    return res, m


def test_children_of_a_parent_share_side_and_sum_to_its_size():
    cfg = MetaOrderConfig(rate=0.05, total=Const(40.0), child=Const(4.0), child_rate=20.0)
    res, m = run_meta(cfg, horizon=400.0)
    kids = [c for c in res.commands if isinstance(c, NewMarket)]
    assert m.parents_started > 5 and len(kids) == m.children_sent
    assert sum(c.qty for c in kids) == pytest.approx(40 * m.parents_started, abs=40 * 3)  # unfinished parents at the end
    # signs come in runs: with parents of 10 children each, sign changes are far rarer than for independent signs
    s = np.array([int(c.side) for c in kids])
    assert np.mean(s[1:] != s[:-1]) < 0.25


def test_final_child_carries_the_remainder_and_sizes_are_positive_ints():
    cfg = MetaOrderConfig(rate=0.2, total=Const(10.0), child=Const(3.0), child_rate=30.0)
    res, _ = run_meta(cfg, horizon=100.0)
    q = [c.qty for c in res.commands if isinstance(c, NewMarket)]
    assert set(q) <= {1, 3} and all(isinstance(x, int) and x >= 1 for x in q)  # 3,3,3,1 per parent


def test_sign_acf_of_metaorder_flow_is_persistent_with_theoretical_exponent():
    """alpha = 1.5 children tail => predicted sign-ACF exponent gamma = alpha - 1 = 0.5 (Lillo-Mike-Farmer)."""
    alpha = 1.5
    cfg = MetaOrderConfig(rate=0.6, total=Pareto(xm=3.0, alpha=alpha), child=Const(1.0), child_rate=50.0, max_total=20_000)
    res, _ = run_meta(cfg, seed=3, horizon=1500.0)
    s = np.array([int(c.side) for c in res.commands if isinstance(c, NewMarket)], dtype=float)
    a = acf(s, 100)
    assert a[0] > 0.3 and a[:20].min() > 0  # strongly persistent signs
    fit = power_law_exponent(a, lo=1, hi=60)
    assert fit["gamma"] == pytest.approx(alpha - 1, abs=0.2), fit
