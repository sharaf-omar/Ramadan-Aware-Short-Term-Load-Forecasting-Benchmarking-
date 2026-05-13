import numpy as np
import pytest
from src.evaluation.bootstrap import block_bootstrap_ci


def test_bootstrap_ci_returns_two_floats():
    rng = np.random.default_rng(42)
    errors = rng.normal(loc=10.0, scale=2.0, size=1000)
    lo, hi = block_bootstrap_ci(errors, block_size=24, n_resamples=200, seed=42)
    assert isinstance(lo, float)
    assert isinstance(hi, float)
    assert lo < hi


def test_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(0)
    errors = rng.normal(loc=5.0, scale=1.0, size=2000)
    lo, hi = block_bootstrap_ci(errors, block_size=24, n_resamples=500, seed=0)
    assert lo < 5.0 < hi


def test_bootstrap_ci_narrows_with_more_data():
    rng = np.random.default_rng(0)
    errors_small = rng.normal(size=200)
    errors_large = rng.normal(size=20000)
    lo_s, hi_s = block_bootstrap_ci(errors_small, 24, 200, seed=1)
    lo_l, hi_l = block_bootstrap_ci(errors_large, 24, 200, seed=1)
    assert (hi_l - lo_l) < (hi_s - lo_s)
