import numpy as np
import pandas as pd
import pytest
from src.evaluation.regime_eval import evaluate_by_regime


def test_evaluate_by_regime_returns_one_row_per_regime():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 210.0, 310.0, 410.0])
    regimes = pd.Series(["Normal", "Normal", "Ramadan", "Heatwave"])
    y_train = np.arange(2000, dtype=float)
    out = evaluate_by_regime(y_true, y_pred, regimes, y_train, period=168)
    assert set(out["regime"]) == {"Normal", "Ramadan", "Heatwave", "Compound"}


def test_zero_count_regime_has_nan_metrics():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 210.0])
    regimes = pd.Series(["Normal", "Normal"])
    y_train = np.arange(2000, dtype=float)
    out = evaluate_by_regime(y_true, y_pred, regimes, y_train, period=168)
    compound_row = out[out["regime"] == "Compound"].iloc[0]
    assert compound_row["n"] == 0
    assert np.isnan(compound_row["mae"])


def test_normal_regime_mae_matches_expectation():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 220.0])
    regimes = pd.Series(["Normal", "Normal"])
    y_train = np.arange(2000, dtype=float)
    out = evaluate_by_regime(y_true, y_pred, regimes, y_train, period=168)
    normal_row = out[out["regime"] == "Normal"].iloc[0]
    # MAE = mean(|10|, |20|) = 15
    assert normal_row["mae"] == pytest.approx(15.0)
    assert normal_row["n"] == 2
