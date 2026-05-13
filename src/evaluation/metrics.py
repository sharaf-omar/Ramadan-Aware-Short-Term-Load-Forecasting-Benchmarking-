"""Forecast metrics: MAE, RMSE, MAPE, MASE (vs seasonal-naive lag-period)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error in %. Skips rows where y_true == 0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def mase(y_true, y_pred, y_train, period: int = 168) -> float:
    """Mean Absolute Scaled Error vs seasonal naive with lag = `period`."""
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= period:
        raise ValueError(f"y_train has {len(y_train)} obs, need > period={period}")
    scale = float(np.mean(np.abs(y_train[period:] - y_train[:-period])))
    if scale == 0:
        return float("inf")
    return float(mean_absolute_error(y_true, y_pred) / scale)
