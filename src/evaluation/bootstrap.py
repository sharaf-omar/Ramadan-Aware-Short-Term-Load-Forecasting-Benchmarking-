"""Stationary block bootstrap (Politis & Romano 1994) for autocorrelated series."""
from __future__ import annotations

from typing import Callable

import numpy as np


def _stationary_block_resample(x: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """One stationary-bootstrap resample of length len(x).

    Block lengths are drawn from Geometric(1/block_size) so the mean block
    length is block_size; starts are uniform on [0, n).
    """
    n = len(x)
    p = 1.0 / block_size
    out = np.empty(n, dtype=x.dtype)
    pos = 0
    while pos < n:
        start = int(rng.integers(0, n))
        block_len = int(rng.geometric(p))
        for k in range(block_len):
            if pos >= n:
                break
            out[pos] = x[(start + k) % n]
            pos += 1
    return out


def block_bootstrap_ci(
    values: np.ndarray,
    block_size: int = 24,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
    statistic: Callable = np.mean,
) -> tuple[float, float]:
    """Bootstrap (1-alpha) CI for the statistic of an autocorrelated series.

    Parameters
    ----------
    values : 1-D array (e.g., absolute errors).
    block_size : mean stationary-bootstrap block length (default 24 = 1 day for hourly).
    n_resamples : number of bootstrap iterations.
    alpha : significance level (default 0.05 -> 95% CI).
    seed : RNG seed for reproducibility.
    statistic : function applied to each resample (default mean).

    Returns
    -------
    (low, high) tuple defining the (1-alpha) percentile CI.
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    stats_ = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        resample = _stationary_block_resample(values, block_size, rng)
        stats_[i] = statistic(resample)
    lo = float(np.quantile(stats_, alpha / 2))
    hi = float(np.quantile(stats_, 1 - alpha / 2))
    return lo, hi
