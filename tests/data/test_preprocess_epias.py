import pandas as pd
import numpy as np
import pytest
from src.data.preprocess_epias import build_lag_rolling_features


def _make_load_series(n_hours: int = 500) -> pd.Series:
    """Synthetic hourly load series with deterministic values for testing."""
    idx = pd.date_range("2024-01-01", periods=n_hours, freq="h", tz="UTC")
    return pd.Series(np.arange(n_hours, dtype=float), index=idx, name="actual_load")


def test_lag_24h_is_value_at_issuance():
    """For forecast time tau, y_lag_24h should be y at issuance t = tau-24."""
    y = _make_load_series(100)
    out = build_lag_rolling_features(y)
    # row tau=50 means forecast time idx 50, issuance idx 26 (50-24)
    # y_lag_24h should equal y[26] = 26
    assert out.loc[y.index[50], "y_lag_24h"] == 26.0


def test_lag_168h_is_value_one_week_before():
    y = _make_load_series(500)
    out = build_lag_rolling_features(y)
    # row tau=400 -> y_lag_168h = y[400-168] = y[232] = 232
    assert out.loc[y.index[400], "y_lag_168h"] == 232.0


def test_rolling_mean_24h_excludes_post_issuance_hours():
    """The 24h rolling mean at tau must use [tau-47, tau-24] inclusive (24 values,
    ending at issuance). NO peek past issuance."""
    y = _make_load_series(500)
    out = build_lag_rolling_features(y)
    # row tau=100 -> window covers y[53..76] inclusive (24 values).
    # Mean of arange(53,77) = (53+76)/2 = 64.5
    assert out.loc[y.index[100], "y_roll24_mean"] == pytest.approx(64.5)


def test_rolling_mean_168h_excludes_post_issuance_hours():
    """168h rolling mean at tau uses [tau-191, tau-24] inclusive (168 values)."""
    y = _make_load_series(500)
    out = build_lag_rolling_features(y)
    # row tau=300 -> window y[109..276] inclusive (168 values).
    # Mean = (109+276)/2 = 192.5
    assert out.loc[y.index[300], "y_roll168_mean"] == pytest.approx(192.5)


def test_early_rows_are_nan():
    """Rows where the longest lag window doesn't fit must be NaN.
    shift(336) -> iloc < 336 is NaN; iloc >= 336 is defined."""
    y = _make_load_series(500)
    out = build_lag_rolling_features(y)
    assert pd.isna(out["y_lag_336h"].iloc[335])
    assert not pd.isna(out["y_lag_336h"].iloc[336])
    # Rolling-mean-168 window ends at issuance: needs y at idx tau-191..tau-24.
    # First non-NaN row: tau = 191 (so tau-191 = 0 exists, tau-24 = 167 exists).
    assert pd.isna(out["y_roll168_mean"].iloc[190])
    assert not pd.isna(out["y_roll168_mean"].iloc[191])
