import numpy as np
import pandas as pd
import pytest
from src.models.tsfm._adapter import build_context_windows


def _make_load(n_hours: int = 1000) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n_hours, freq="h", tz="UTC")
    return pd.DataFrame({"actual_load": np.arange(n_hours, dtype=float)}, index=idx)


def test_context_window_correct_length_and_endpoint():
    df = _make_load(1000)
    forecast_times = df.index[500:510]
    L = 168
    contexts = build_context_windows(df["actual_load"], forecast_times, context_length=L)
    assert contexts.shape == (10, 168)
    # row 0 forecasts tau=500. issuance idx = 476. context = y[309..476]
    assert contexts[0, 0] == 309.0
    assert contexts[0, -1] == 476.0
    # row 9 forecasts tau=509. issuance idx = 485. context = y[318..485]
    assert contexts[9, 0] == 318.0
    assert contexts[9, -1] == 485.0


def test_context_window_drops_insufficient_history():
    df = _make_load(200)
    L = 168
    forecast_times = df.index[:200]
    contexts = build_context_windows(df["actual_load"], forecast_times, context_length=L)
    assert contexts.shape == (200, 168)
    assert np.isnan(contexts[0]).all()
    assert np.isnan(contexts[190]).all()  # tau=190: needs y[-1] - insufficient
    assert not np.isnan(contexts[191]).any()  # tau=191: y[0..167] available
