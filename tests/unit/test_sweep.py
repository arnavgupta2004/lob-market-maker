"""Sweep framework, statistics additions and the new per-session metrics."""
import math

import numpy as np
import pytest

from backtest.runner import Scenario, run_session
from backtest.statistics import bootstrap_p, holm, paired_diff_ci
from backtest.sweep import Axis, cells, latency_axis, path_axis, pivot, replace_path, run_sweep
from simulator.latency.models import LatencyConfig
from simulator.simulator import SimConfig
from strategies.baseline_mm import ASConfig


def test_replace_path_is_nested_immutable_and_validates():
    sc = Scenario("t", SimConfig(horizon_s=10), mm=ASConfig(gamma=0.1))
    sc2 = replace_path(sc, "mm.gamma", 0.5)
    assert sc2.mm.gamma == 0.5 and sc.mm.gamma == 0.1  # original untouched
    sc3 = replace_path(sc, "sim.fundamental", None)
    assert sc3.sim.fundamental is None
    assert replace_path(sc, "noise.limit_rate", 7.0).noise.limit_rate == 7.0
    with pytest.raises(AttributeError):
        replace_path(sc, "mm.nonexistent", 1)
    with pytest.raises(AttributeError):
        replace_path(sc, "informed.rate", 1.0)  # informed is None: cannot descend


def test_axes_and_cells_cartesian_product_and_latency_axis():
    a, b = path_axis("mm.gamma", [0.1, 0.2]), latency_axis([0.0, 5.0])
    assert list(cells([a, b])) == [(0.1, 0.0), (0.1, 5.0), (0.2, 0.0), (0.2, 5.0)]
    sc = b.apply(Scenario("t", mm=ASConfig()), 5.0)
    assert sc.mm.latency == LatencyConfig.symmetric(5.0)


def test_run_sweep_is_deterministic_paired_and_pivots():
    base = Scenario("t", SimConfig(horizon_s=25), mm=ASConfig(gamma=0.01, k=0.9))
    axes = [path_axis("mm.gamma", [0.0, 0.05]), path_axis("mm.quote_size", [3, 6])]
    metrics = ["pnl_net", "n_fills", "adverse_cost_500ms"]
    r1, s1 = run_sweep(base, axes, [1, 2, 3], metrics)
    r2, s2 = run_sweep(base, axes, [1, 2, 3], metrics)
    assert r1 == r2 or all(math.isclose(a[k], b[k], rel_tol=0, abs_tol=0) or (a[k] != a[k] and b[k] != b[k]) for a, b in zip(r1, r2) for k in a)
    assert len(r1) == 4 and len(s1) == 12 and all(r["n"] == 3 for r in r1)
    xs, ys, Z = pivot(r1, "mm.gamma", "mm.quote_size", "n_fills")
    assert xs == [0.0, 0.05] and ys == [3, 6] and Z.shape == (2, 2) and np.isfinite(Z).all()
    # same seeds in every cell: the seed column repeats identically
    assert sorted({s["seed"] for s in s1}) == [1, 2, 3]


def test_paired_ci_pvalue_effect_size_and_holm():
    rng = np.random.default_rng(0)
    d = rng.normal(0.5, 1.0, 80)
    r = paired_diff_ci(d + 5.0, np.full(80, 5.0))
    assert r["significant"] and r["p_value"] < 0.01 and r["effect_size_dz"] == pytest.approx(0.5, abs=0.2)
    null = paired_diff_ci(rng.normal(0, 1, 80), np.zeros(80))
    assert null["p_value"] > 0.05
    assert bootstrap_p(np.zeros(10)) == 1.0 or bootstrap_p(np.zeros(10)) > 0.9
    adj = holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06]) and all(a >= p for a, p in zip(adj, [0.01, 0.04, 0.03]))
    assert holm([0.5]) == [0.5]


def test_session_reports_adverse_selection_and_quote_survival():
    s = run_session(Scenario("t", SimConfig(horizon_s=40), mm=ASConfig(gamma=0.01, k=0.9)), 2, keep_result=False).metrics
    for k in ("eff_half_spread_ticks", "postfill_10ms", "postfill_1s", "adverse_cost_500ms", "realized_half_spread_500ms",
              "quote_surv_250ms", "quote_surv_1s"):
        assert k in s and np.isfinite(s[k])
    assert s["adverse_cost_500ms"] == pytest.approx(-s["postfill_500ms"])
    assert s["realized_half_spread_500ms"] == pytest.approx(s["eff_half_spread_ticks"] + s["postfill_500ms"])
    assert 0 <= s["quote_surv_1s"] <= s["quote_surv_250ms"] <= 1
