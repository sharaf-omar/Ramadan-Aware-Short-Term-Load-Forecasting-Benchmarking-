import pandas as pd
import pytest
from src.features.hijri import add_hijri_features


def test_hijri_features_returns_required_columns():
    df = pd.DataFrame(index=pd.date_range("2024-03-10", "2024-03-12", freq="h", tz="UTC"))
    out = add_hijri_features(df)
    assert "is_ramadan" in out.columns
    assert "day_of_ramadan" in out.columns
    assert "is_eid" in out.columns


def test_hijri_ramadan_2024_window():
    # Ramadan 1445 (2024): roughly Mar 11 - Apr 9 (Turkey local).
    df = pd.DataFrame(index=pd.date_range("2024-03-15 12:00", periods=1, freq="h", tz="UTC"))
    out = add_hijri_features(df)
    assert out["is_ramadan"].iloc[0] == 1
    assert out["day_of_ramadan"].iloc[0] >= 1
    assert out["day_of_ramadan"].iloc[0] <= 30


def test_hijri_non_ramadan_zero():
    # July is never Ramadan in the 2020s.
    df = pd.DataFrame(index=pd.date_range("2024-07-15", periods=1, freq="h", tz="UTC"))
    out = add_hijri_features(df)
    assert out["is_ramadan"].iloc[0] == 0
    assert out["day_of_ramadan"].iloc[0] == 0
    assert out["is_eid"].iloc[0] == 0


def test_hijri_dtype_int():
    df = pd.DataFrame(index=pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC"))
    out = add_hijri_features(df)
    assert out["is_ramadan"].dtype.kind in ("i", "u")
    assert out["day_of_ramadan"].dtype.kind in ("i", "u")
    assert out["is_eid"].dtype.kind in ("i", "u")
