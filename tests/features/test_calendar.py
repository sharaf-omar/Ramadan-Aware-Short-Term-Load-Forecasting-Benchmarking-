import numpy as np
import pandas as pd
from src.features.calendar import add_calendar_features


def test_calendar_columns_present():
    df = pd.DataFrame(index=pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC"))
    out = add_calendar_features(df)
    for col in [
        "hour", "day_of_week", "month",
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "month_sin", "month_cos",
        "is_weekend",
    ]:
        assert col in out.columns, f"missing column {col}"


def test_hour_sin_cos_periodic():
    df = pd.DataFrame(index=pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC"))
    out = add_calendar_features(df)
    next_day = pd.DataFrame(index=pd.date_range("2024-01-02 00:00", periods=1, freq="h", tz="UTC"))
    out_next = add_calendar_features(next_day)
    assert np.isclose(out["hour_sin"].iloc[0], out_next["hour_sin"].iloc[0])


def test_weekend_flag():
    # 2024-01-06 Saturday, 2024-01-07 Sunday, 2024-01-08 Monday.
    df = pd.DataFrame(index=pd.to_datetime(
        ["2024-01-06 12:00", "2024-01-07 12:00", "2024-01-08 12:00"], utc=True
    ))
    out = add_calendar_features(df)
    assert out["is_weekend"].tolist() == [1, 1, 0]
