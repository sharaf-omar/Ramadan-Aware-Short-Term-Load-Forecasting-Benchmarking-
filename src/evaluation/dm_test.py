"""Diebold-Mariano test with Newey-West HAC standard errors.

Also exposes Holm-Bonferroni multiple-comparison correction.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import stats
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.regression.linear_model import OLS


def dm_test(y_true, y_pred_a, y_pred_b, h: int = 24, loss: str = "mae") -> tuple[float, float]:
    """Diebold-Mariano statistic and two-sided p-value.

    Parameters
    ----------
    y_true : array of true values.
    y_pred_a, y_pred_b : predictions from two models.
    h : forecast horizon, used to set HAC truncation lag = h - 1.
    loss : 'mae' or 'mse'.

    Returns
    -------
    (dm_stat, p_value).
    DM > 0 means model B has lower loss (loss_A > loss_B).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_a = np.asarray(y_pred_a, dtype=float)
    y_pred_b = np.asarray(y_pred_b, dtype=float)

    if loss == "mae":
        e_a = np.abs(y_true - y_pred_a)
        e_b = np.abs(y_true - y_pred_b)
    elif loss == "mse":
        e_a = (y_true - y_pred_a) ** 2
        e_b = (y_true - y_pred_b) ** 2
    else:
        raise ValueError("loss must be 'mae' or 'mse'")

    d = e_a - e_b
    n = len(d)

    if np.allclose(d, 0):
        return 0.0, 1.0

    # HAC variance via regression of d on a constant; sandwich estimator
    # gives Newey-West-style HAC.
    X = np.ones((n, 1))
    model = OLS(d, X).fit()
    nlags = max(h - 1, 1)
    cov = cov_hac(model, nlags=nlags)
    var_dbar = float(cov[0, 0])

    dbar = float(np.mean(d))
    if var_dbar <= 0:
        return 0.0, 1.0
    dm_stat = dbar / np.sqrt(var_dbar)
    p = float(2 * (1 - stats.norm.cdf(abs(dm_stat))))
    return float(dm_stat), p


def holm_bonferroni(p_values: Iterable[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment.

    Returns adjusted p-values in the *original* input order.
    """
    p = np.asarray(list(p_values), dtype=float)
    n = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    multipliers = np.arange(n, 0, -1)
    adjusted_sorted = np.maximum.accumulate(p_sorted * multipliers)
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()
