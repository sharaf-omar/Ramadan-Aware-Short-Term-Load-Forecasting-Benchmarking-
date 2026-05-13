import numpy as np
import pytest
from src.evaluation.metrics import mae, rmse, mape, mase


def test_mae_simple():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.0, 4.0])
    # errors: 0.5, 0, 1.0 -> mean = 0.5
    assert mae(y_true, y_pred) == pytest.approx(0.5)


def test_rmse_simple():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([2.0, 4.0])
    # sq errors: 1, 4 -> mean 2.5 -> sqrt
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(2.5))


def test_mape_skips_zero_y():
    y_true = np.array([0.0, 100.0])
    y_pred = np.array([10.0, 110.0])
    # only the second observation counts: |100-110|/100 * 100 = 10
    assert mape(y_true, y_pred) == pytest.approx(10.0)


def test_mase_with_zero_scale_returns_inf():
    y_train = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    y_true = np.array([3.0])
    y_pred = np.array([2.0])
    # seasonal-naive period=2 errors: |y[2]-y[0]|=0, |y[3]-y[1]|=0, ... all 0.
    # Edge case: zero scale -> MASE returns inf.
    result = mase(y_true, y_pred, y_train, period=2)
    assert np.isinf(result)


def test_mase_typical():
    # y_train where seasonal-naive errors avg 1.0.
    y_train = np.arange(10, dtype=float)  # [0,1,2,..,9]
    # period=1: |y[1]-y[0]|=1, |y[2]-y[1]|=1, ... mean=1.
    y_true = np.array([10.0, 11.0])
    y_pred = np.array([10.5, 11.5])
    # MAE = 0.5, scale = 1.0, MASE = 0.5.
    assert mase(y_true, y_pred, y_train, period=1) == pytest.approx(0.5)
