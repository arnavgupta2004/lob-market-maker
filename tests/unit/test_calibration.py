"""The calibration estimators must recover known ground truth from synthetic data."""
import math

import numpy as np
import pytest

from research.as_calibration import (
    ProbeRecord, add_stats, fit_exponential_intensity, mid_diagnostics, rows_from_stats, session_bin_stats,
)


def synthetic_probes(A, k, deltas, n_per, dwell, seed):
    rng = np.random.default_rng(seed)
    recs = []
    for d in deltas:
        lam = A * math.exp(-k * d)
        for _ in range(n_per):
            t = rng.exponential(1 / lam)
            filled = t < dwell
            recs.append(ProbeRecord(1, d, 2.0, exposure_s=min(t, dwell), filled=filled,
                                    fill_after_s=t if filled else float("nan")))
    return recs


def test_censored_mle_recovers_intensity_and_k():
    A, k, dwell = 4.0, 0.8, 0.5
    recs = synthetic_probes(A, k, [0.5, 1, 1.5, 2, 3, 4, 5], 6000, dwell, seed=1)
    rows = rows_from_stats(session_bin_stats(recs, dwell))
    for r in rows:
        assert r["lam"] == pytest.approx(A * math.exp(-k * r["delta"]), rel=5 * 1 / math.sqrt(max(r["fills"], 1)))
    fit = fit_exponential_intensity(rows)
    assert fit["k"] == pytest.approx(k, abs=0.03) and fit["A"] == pytest.approx(A, rel=0.08)
    assert fit["r2"] > 0.99 and fit["chi2_per_dof"] < 3  # a truly exponential curve is not rejected


def test_constant_hazard_gives_equal_half_window_hazards():
    recs = synthetic_probes(3.0, 0.5, [1, 2], 20000, 0.5, seed=2)
    for r in rows_from_stats(session_bin_stats(recs, 0.5)):
        assert r["lam_second_half"] / r["lam_first_half"] == pytest.approx(1.0, abs=0.1)


def test_raw_fill_fraction_is_biased_but_censored_mle_is_not():
    recs = synthetic_probes(4.0, 0.0, [1.0], 20000, 0.5, seed=3)  # lam = 4/s
    (row,) = rows_from_stats(session_bin_stats(recs, 0.5))
    assert row["fill_frac"] == pytest.approx(1 - math.exp(-2.0), abs=0.01)  # 0.865 != 4
    assert row["lam"] == pytest.approx(4.0, rel=0.03)


def test_non_exponential_curve_is_flagged_by_chi2():
    rng = np.random.default_rng(4)
    recs = []
    for d in [0.5, 1, 1.5, 2, 3, 4, 5, 6]:
        lam = 4.0 / (1 + d) ** 2  # power law, not exponential
        for _ in range(8000):
            t = rng.exponential(1 / lam)
            recs.append(ProbeRecord(1, d, 2.0, min(t, 0.5), t < 0.5, t if t < 0.5 else float("nan")))
    fit = fit_exponential_intensity(rows_from_stats(session_bin_stats(recs, 0.5)))
    assert fit["chi2_per_dof"] > 10


def test_stats_are_additive_across_sessions():
    a = synthetic_probes(2.0, 0.5, [1, 2], 500, 0.5, 1)
    b = synthetic_probes(2.0, 0.5, [1, 2], 500, 0.5, 2)
    merged = add_stats(session_bin_stats(a, 0.5), session_bin_stats(b, 0.5))
    both = session_bin_stats(a + b, 0.5)
    for d in both:
        assert np.allclose(merged[d], both[d])


def test_mid_diagnostics_random_walk_vs_mean_reverting():
    rng = np.random.default_rng(5)
    rw = np.cumsum(rng.standard_normal(200_000))
    d = mid_diagnostics(rw, 0.1)
    assert d["vr_10"] == pytest.approx(1.0, abs=0.05) and d["excess_kurtosis"] == pytest.approx(0, abs=0.1)
    assert d["sigma_0.1s"] == pytest.approx(math.sqrt(1 / 0.1), rel=0.02)
    ou = np.zeros(200_000)
    e = rng.standard_normal(200_000)
    for i in range(1, len(ou)):
        ou[i] = 0.9 * ou[i - 1] + e[i]
    # AR(1): VR(q) = (1 - phi^q) / (q (1 - phi)) = 0.651 for phi = .9, q = 10
    assert mid_diagnostics(ou, 0.1)["vr_10"] == pytest.approx((1 - 0.9 ** 10) / (10 * 0.1), abs=0.02)
