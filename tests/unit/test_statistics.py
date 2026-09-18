import numpy as np
import pytest

from backtest.statistics import bootstrap_ci, paired_diff_ci, summarize


def test_bootstrap_ci_is_deterministic_and_brackets_mean():
    x = np.random.default_rng(0).normal(5, 2, 50)
    a, b = bootstrap_ci(x, seed=1), bootstrap_ci(x, seed=1)
    assert a == b and a[0] < x.mean() < a[1]
    assert bootstrap_ci(x, seed=2) != a


def test_bootstrap_coverage_is_near_nominal():
    rng = np.random.default_rng(3)
    hits = 0
    for i in range(300):
        x = rng.normal(0.0, 1.0, 40)
        lo, hi = bootstrap_ci(x, n_boot=400, seed=i)
        hits += lo <= 0.0 <= hi
    assert 0.90 <= hits / 300 <= 0.99


def test_median_ci_via_custom_stat():
    x = np.random.default_rng(1).exponential(1.0, 200)
    lo, hi = bootstrap_ci(x, np.median, n_boot=800)
    assert lo < np.log(2) < hi  # median of Exp(1) = ln 2


def test_summarize_fields_and_nan_handling():
    s = summarize([1.0, 2.0, 3.0, float("nan")])
    assert s["n"] == 3 and s["mean"] == pytest.approx(2.0) and s["median"] == 2.0
    assert s["std"] == pytest.approx(1.0) and s["ci_lo"] <= 2.0 <= s["ci_hi"]
    assert np.isnan(bootstrap_ci([1.0])[0])


def test_paired_difference_detects_shift_and_respects_pairing():
    rng = np.random.default_rng(5)
    base = rng.normal(0, 10, 60)  # large between-seed variance shared by both arms
    a, b = base + 0.5 + rng.normal(0, 0.2, 60), base
    paired = paired_diff_ci(a, b)
    assert paired["significant"] and 0.3 < paired["mean_diff"] < 0.7
    assert paired["ci_hi"] - paired["ci_lo"] < 0.3  # pairing removes the shared variance
    unpaired_width = np.subtract(*bootstrap_ci(a)[::-1])
    assert unpaired_width > 10 * (paired["ci_hi"] - paired["ci_lo"])
    null = paired_diff_ci(base + rng.normal(0, 0.2, 60), base)
    assert not null["significant"]
    with pytest.raises(ValueError):
        paired_diff_ci([1, 2], [1])
