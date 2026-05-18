import pandas as pd
import numpy as np
from src.features.regimes import label_regimes


def _build_df(temps_per_day: list[float], starts: str = "2024-06-01"):
    """Build hourly DF where temps_per_day[i] is the daily max for day i.
    Returns a DataFrame with hourly temp_c (constant within day) and is_ramadan=0."""
    rows = []
    for i, t in enumerate(temps_per_day):
        day = pd.Timestamp(starts, tz="UTC") + pd.Timedelta(days=i)
        for h in range(24):
            rows.append({"timestamp": day + pd.Timedelta(hours=h), "temp_c": t, "is_ramadan": 0})
    df = pd.DataFrame(rows).set_index("timestamp")
    return df


def test_normal_when_cool_and_no_ramadan():
    df = _build_df([20.0, 20.0, 20.0])
    out = label_regimes(df)
    assert (out == "Normal").all()


def test_heatwave_requires_3_consecutive_days_at_35c():
    # Only 2 hot days: should NOT be heatwave.
    df = _build_df([20.0, 36.0, 36.0, 20.0])
    out = label_regimes(df)
    assert (out == "Normal").all()


def test_heatwave_3_consecutive_days_at_35c():
    df = _build_df([36.0, 36.0, 36.0])
    out = label_regimes(df)
    assert (out == "Heatwave").all()


def test_heatwave_5_day_block():
    df = _build_df([20.0, 36.0, 36.0, 36.0, 36.0, 36.0, 20.0])
    out = label_regimes(df)
    assert out.iloc[0:24].eq("Normal").all()
    assert out.iloc[24:24*6].eq("Heatwave").all()
    assert out.iloc[24*6:].eq("Normal").all()


def test_ramadan_only():
    df = _build_df([20.0, 20.0, 20.0])
    df["is_ramadan"] = 1
    out = label_regimes(df)
    assert (out == "Ramadan").all()


def test_compound_regime():
    df = _build_df([36.0, 36.0, 36.0])
    df["is_ramadan"] = 1
    out = label_regimes(df)
    assert (out == "Compound").all()


def test_returns_series_aligned_to_index():
    df = _build_df([20.0])
    out = label_regimes(df)
    assert isinstance(out, pd.Series)
    assert (out.index == df.index).all()
