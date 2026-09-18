"""Stylized-fact estimators against processes whose answer is known."""
import numpy as np
import pytest

from research.stylized_facts import acf, fano_factor, hill_tail_index, interarrival_stats, moments, power_law_exponent, spread_stats

RNG = np.random.default_rng(0)


def test_moments_against_distributions_with_known_kurtosis():
    g = moments(RNG.standard_normal(200_000))
    assert abs(g["skew"]) < 0.05 and abs(g["excess_kurtosis"]) < 0.1
    assert moments(RNG.laplace(size=400_000))["excess_kurtosis"] == pytest.approx(3.0, rel=0.1)  # Laplace: exactly 3
    assert moments(RNG.uniform(size=200_000))["excess_kurtosis"] == pytest.approx(-1.2, abs=0.05)  # uniform: exactly -1.2
    assert moments(RNG.exponential(size=400_000))["skew"] == pytest.approx(2.0, rel=0.1)  # exponential: skewness 2
    assert moments(np.ones(10))["std"] == 0.0


def test_hill_estimator_recovers_the_tail_exponent():
    for nu in (2.5, 3.0):
        h = hill_tail_index(RNG.standard_t(nu, 300_000), frac=0.02)
        assert h["alpha"] == pytest.approx(nu, rel=0.15), (nu, h)
        assert h["se"] == pytest.approx(h["alpha"] / np.sqrt(h["k"]))
    for a in (1.5, 3.0, 5.0):  # an exact Pareto is recovered at any moderate k
        assert hill_tail_index(RNG.pareto(a, 300_000) + 1, frac=0.05)["alpha"] == pytest.approx(a, rel=0.1)
    assert hill_tail_index(RNG.standard_normal(100_000), frac=0.02)["alpha"] > 5  # light tail => large index
    assert np.isnan(hill_tail_index(np.ones(10))["alpha"])


def test_hill_bias_for_student_t_shrinks_as_k_shrinks():
    """Student-t is only asymptotically Pareto: at the top 5% the Hill index of t5 is biased low; the bias falls with k."""
    x = np.random.default_rng(1).standard_t(5, 1_500_000)
    coarse, fine = hill_tail_index(x, frac=0.05)["alpha"], hill_tail_index(x, frac=0.001)["alpha"]
    assert coarse < 4.2 and abs(fine - 5) < abs(coarse - 5) and fine == pytest.approx(5.0, rel=0.15)


def test_acf_of_ar1_and_iid():
    n, phi = 200_000, 0.6
    e = RNG.standard_normal(n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    a = acf(x, 4)
    assert a[:3] == pytest.approx([phi, phi ** 2, phi ** 3], abs=0.02)
    assert np.abs(acf(RNG.standard_normal(100_000), 10)).max() < 0.02


def test_volatility_clustering_shows_in_abs_return_acf():
    n = 200_000
    eps = RNG.standard_normal(n)
    r, h = np.zeros(n), np.full(n, 1.0)
    for i in range(1, n):  # GARCH(1,1)
        h[i] = 0.05 + 0.15 * r[i - 1] ** 2 + 0.8 * h[i - 1]
        r[i] = np.sqrt(h[i]) * eps[i]
    assert np.abs(acf(r, 5)).max() < 0.03  # returns themselves uncorrelated
    assert acf(np.abs(r), 20)[[0, 9, 19]].min() > 0.05  # |r| positively correlated, slowly decaying
    assert np.abs(acf(np.abs(RNG.standard_normal(n)), 10)).max() < 0.02


def test_power_law_exponent_recovery():
    lags = np.arange(1, 101)
    c = lags ** -0.4 * np.exp(RNG.normal(0, 0.03, 100))
    fit = power_law_exponent(c)
    assert fit["gamma"] == pytest.approx(0.4, abs=0.03) and fit["r2"] > 0.95
    geo = power_law_exponent(0.5 ** lags[:30] + 1e-9)  # exponential decay is NOT a power law: poor fit, large exponent
    assert geo["r2"] < 0.9 or geo["gamma"] > 2
    assert np.isnan(power_law_exponent(np.array([-1.0, -0.5, -0.2, -0.1, -0.05]))["gamma"])


def test_sign_chain_acf_matches_theory():
    p, n = 0.8, 300_000  # persistence: P(same sign as previous)
    stay = RNG.random(n) < p
    s = np.ones(n)
    for i in range(1, n):
        s[i] = s[i - 1] if stay[i] else -s[i - 1]
    assert acf(s, 3) == pytest.approx([(2 * p - 1) ** k for k in (1, 2, 3)], abs=0.02)


def test_fano_factor_poisson_clustered_regular():
    T = 2000
    poisson = np.sort(RNG.uniform(0, T, 20_000)) * 1e9
    assert fano_factor(poisson, 1.0, 0, int(T * 1e9)) == pytest.approx(1.0, abs=0.12)
    # clustered: each parent spawns a burst of 10 children within 50 ms
    parents = np.sort(RNG.uniform(0, T, 2_000))
    clustered = np.sort(np.concatenate([p + RNG.uniform(0, 0.05, 10) for p in parents])) * 1e9
    assert fano_factor(clustered, 1.0, 0, int(T * 1e9)) > 5
    regular = np.arange(0, T, 0.05) * 1e9
    assert fano_factor(regular, 1.0, 0, int(T * 1e9)) < 0.1
    assert interarrival_stats(poisson)["cv"] == pytest.approx(1.0, abs=0.05)
    assert interarrival_stats(regular)["cv"] < 0.01


def test_spread_stats_fields():
    s = spread_stats(np.array([1, 1, 1, 2, 2, 3, 1, 1, np.nan]))
    assert s["median"] == 1 and s["frac_at_min"] == pytest.approx(5 / 8) and s["q99"] >= s["q90"]
