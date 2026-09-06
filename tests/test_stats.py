"""
tests/test_stats.py — Unit tests for atlas.stats.
"""

import numpy as np
import pytest
from atlas.stats import normalised_recovery, paired_bootstrap, success_rate_ci


def test_normalised_recovery_basic():
    nr = normalised_recovery(sr_fit=0.7, sr_oracle=0.9, sr_random=0.5)
    # (0.7 - 0.5) / (0.9 - 0.5) = 0.2 / 0.4 = 0.5
    assert abs(nr - 0.5) < 1e-6


def test_normalised_recovery_none_when_spread_small():
    nr = normalised_recovery(sr_fit=0.7, sr_oracle=0.75, sr_random=0.7)
    assert nr is None


def test_paired_bootstrap_zero_diff():
    a = np.ones(100)
    b = np.ones(100)
    mean_diff, (lo, hi) = paired_bootstrap(a, b)
    assert abs(mean_diff) < 1e-9
    assert abs(lo) < 1e-9
    assert abs(hi) < 1e-9


def test_paired_bootstrap_shape_mismatch():
    with pytest.raises(ValueError):
        paired_bootstrap(np.ones(10), np.ones(11))


def test_paired_bootstrap_positive_diff():
    a = np.ones(200)
    b = np.zeros(200)
    mean_diff, (lo, hi) = paired_bootstrap(a, b)
    assert abs(mean_diff - 1.0) < 1e-6
    assert lo > 0.9
    assert hi > 0.9


def test_success_rate_ci_all_ones():
    sr, (lo, hi) = success_rate_ci(np.ones(100))
    assert sr == 1.0
    assert lo > 0.95
    assert hi == 1.0


def test_success_rate_ci_all_zeros():
    sr, (lo, hi) = success_rate_ci(np.zeros(100))
    assert sr == 0.0
    assert lo == 0.0
    assert hi < 0.05
