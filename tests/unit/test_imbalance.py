import numpy as np
import pytest

from research.imbalance import OBI_EDGES, backward_changes, forward_changes, session_stats


def synthetic_samples(n=60_000, beta=0.4, noise=0.5, seed=0, dt=0.01, h_steps=10):
    """OBI is i.i.d. uniform; the mid's h-step-ahead change is beta*OBI + noise (plus a martingale walk)."""
    rng = np.random.default_rng(seed)
    obi = rng.uniform(-1, 1, n)
    steps = rng.normal(0, noise, n)
    # OBI at i adds beta/h to each of the next h steps => the h-step change after i contains beta*obi[i]
    ker = np.r_[0.0, np.full(h_steps, beta / h_steps)]
    steps += np.convolve(obi, ker)[:n]
    mid = 100 + np.cumsum(steps)
    # encode OBI via depths: (vb - va)/(vb + va) = obi with vb+va = 100
    vb, va = 50 * (1 + obi), 50 * (1 - obi)
    return {"t_ns": np.arange(n) * int(dt * 1e9), "mid": mid, "spread": np.ones(n), "bid_qty1": vb, "ask_qty1": va,
            "bid_depth5": vb, "ask_depth5": va}


def test_forward_and_backward_changes_alignment():
    m = np.array([1.0, 2.0, 4.0, 7.0])
    assert forward_changes(m, 1)[:3].tolist() == [1, 2, 3] and np.isnan(forward_changes(m, 1)[3])
    assert backward_changes(m, 2)[2:].tolist() == [3, 5] and np.isnan(backward_changes(m, 2)[0])


def test_recovers_known_predictive_slope_and_null_is_zero():
    s = synthetic_samples(beta=0.4, noise=0.05)
    r = session_stats(s, 0.01, [0.1, 0.5], start_s=0.0)
    assert r[0.1]["slope"] == pytest.approx(0.4, rel=0.1)
    # corr = beta*sd(obi) / sd(dm) ~ 0.37 here: earlier OBI values also drive the same window
    assert 0.3 < r[0.1]["corr"] < 0.45 and abs(r[0.1]["corr_shuffled"]) < 0.02
    assert abs(r[0.1]["corr_backward"]) < 0.05  # OBI here is exogenous: no reverse causality
    assert r[0.1]["dir_acc_excess"] > 0.1
    bm = np.array(r[0.1]["bin_mean"])
    assert np.all(np.diff(bm) > 0) and bm[0] < 0 < bm[-1]  # monotone in OBI


def test_no_relationship_gives_zero_everywhere():
    s = synthetic_samples(beta=0.0, noise=0.5, seed=3)
    r = session_stats(s, 0.01, [0.1], start_s=0.0)[0.1]
    assert abs(r["corr"]) < 0.02 and abs(r["slope"]) < 0.03


def test_reverse_causality_shows_up_in_backward_correlation():
    """If prices move first and OBI follows (OBI_t = recent move), backward correlation is large."""
    rng = np.random.default_rng(1)
    mid = 100 + np.cumsum(rng.normal(0, 0.5, 40_000))
    obi = np.clip(np.r_[np.zeros(10), mid[10:] - mid[:-10]] / 3, -1, 1)
    s = {"t_ns": np.arange(len(mid)) * 10_000_000, "mid": mid, "spread": np.ones(len(mid)),
         "bid_qty1": 50 * (1 + obi), "ask_qty1": 50 * (1 - obi), "bid_depth5": 50 * (1 + obi), "ask_depth5": 50 * (1 - obi)}
    r = session_stats(s, 0.01, [0.1], start_s=0.0)[0.1]
    assert r["corr_backward"] > 0.5 and abs(r["corr"]) < 0.05  # explained by past, no forward information


def test_spread_filter_and_warmup():
    s = synthetic_samples(n=20_000, beta=0.4, noise=0.05)
    s["spread"][::2] = 2
    a = session_stats(s, 0.01, [0.1], start_s=0.0, only_spread=1)[0.1]
    b = session_stats(s, 0.01, [0.1], start_s=0.0)[0.1]
    assert a["n"] < b["n"] * 0.6
    assert session_stats(s, 0.01, [0.1], start_s=1e9)[0.1]["n"] == 0
    assert len(OBI_EDGES) == 8
