import numpy as np
import pytest

from engine.common import Side
from research.queue_position import (
    FEATURES, Q_EDGES, QueueProbe, add_stats, feature_matrix, fit_logit, outcomes, predict_logit, q_bin_label,
    q_bin_stats,
)
from simulator.order_flow.noise import NoiseTrader
from simulator.simulator import NS, SimConfig, Simulator


def run_probe(seed=3, horizon=60.0, **kw):
    probe = QueueProbe(dwell_s=3.0, **kw)
    res = Simulator(SimConfig(seed=seed, horizon_s=horizon), [NoiseTrader(), probe]).run()
    return res, probe


def test_queue_ahead_never_increases_and_rows_are_consistent():
    res, probe = run_probe()
    d = probe.dataset()
    assert len(d["Q"]) > 500 and np.all(d["Q"] >= 0) and np.all(d["n_ahead"] >= 0)
    for pi in np.unique(d["probe"])[:300]:
        m = d["probe"] == pi
        q, age = d["Q"][m], d["age_s"][m]
        assert np.all(np.diff(q) <= 0), "orders behind us can never jump ahead"
        assert np.allclose(np.diff(age), 0.25) and age[-1] <= 3.0 + 1e-9
    assert set(np.unique(d["k"])) <= {0, 1, 2}
    assert np.all(d["d_touch"] >= 0)  # our price is never better than the same-side best
    assert np.all(d["d_touch"][d["k"] == 0] >= 0)


def test_probe_fills_match_the_exchange_tape():
    res, probe = run_probe(seed=4, horizon=120.0)
    tab = probe.probe_table()
    tape_fills = int(np.sum(res.trades["maker_owner"] == probe.owner_id))
    assert tape_fills == int(tab["filled"].sum()) > 20
    # a filled probe has a fill time within its dwell; unfilled ones are never marked
    for p in probe.probes:
        if p.fill_ns == p.fill_ns:
            assert 0 <= p.fill_ns - p.t0 <= 3.0 * NS + 1


def test_deeper_initial_queue_fills_less_often():
    tabs = []
    for seed in range(6):
        _, probe = run_probe(seed=seed, horizon=120.0)
        tabs.append(probe.probe_table())
    q0 = np.concatenate([t["q0"] for t in tabs]); f = np.concatenate([t["filled"] for t in tabs])
    assert f[q0 <= 5].mean() > f[q0 > 25].mean() + 0.1


def test_outcome_validity_excludes_incompletely_observed_windows():
    d = {"age_s": np.array([0.0, 1.0, 2.0, 2.75]), "t_check_ns": np.array([0, 1, 2, 2.75]) * NS,
         "fill_ns": np.full(4, 2.9 * NS)}
    y, valid = outcomes(d, 0.5, 3.0)
    assert valid.tolist() == [True, True, True, False]  # last window would extend to 3.25 > dwell
    # fill at 2.9 s: only the 2.75 s checkpoint has it inside (t, t + 0.5 s]; that row is flagged invalid anyway
    assert y.tolist() == [0, 0, 0, 1]


def test_q_bins_labels_and_additivity():
    Q = np.array([0, 0, 1, 2, 5, 10, 11, 100.0]); y = np.array([1, 0, 1, 0, 0, 1, 0, 0.0])
    st = q_bin_stats(Q, y)
    assert st[0].tolist() == [2, 1] and st[1].tolist() == [2, 1] and st[6].tolist() == [1, 0]
    assert [q_bin_label(i) for i in range(len(Q_EDGES) - 1)] == ["0", "1-2", "3-5", "6-10", "11-20", "21-40", "41+"]
    a = add_stats(q_bin_stats(Q[:4], y[:4]), q_bin_stats(Q[4:], y[4:]))
    assert all(np.allclose(a[k], st[k]) for k in st)


def test_logit_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60_000, 3)) * [1, 2, 0.5] + [0, 5, 1]
    true = np.array([-1.0, 0.5, 0.0])  # per-SD coefficients
    z = -0.3 + ((X - X.mean(0)) / X.std(0)) @ true
    y = (rng.random(len(X)) < 1 / (1 + np.exp(-z))).astype(float)
    m = fit_logit(X, y)
    assert m["intercept"] == pytest.approx(-0.3, abs=0.05)
    assert m["coef"] == pytest.approx(true, abs=0.05)
    p = predict_logit(m, X)
    assert p.mean() == pytest.approx(y.mean(), abs=0.01)


def test_feature_matrix_shape_matches_feature_names():
    _, probe = run_probe(horizon=30.0)
    X = feature_matrix(probe.dataset())
    assert X.shape[1] == len(FEATURES) and np.all(np.isfinite(X))
