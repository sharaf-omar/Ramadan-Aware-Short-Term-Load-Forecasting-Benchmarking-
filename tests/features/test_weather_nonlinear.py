import pandas as pd
from src.features.weather_nonlinear import add_weather_nonlinear


def test_weather_columns_present():
    df = pd.DataFrame({
        "temp_c": [25.0, 36.0, 40.0],
    }, index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"))
    out = add_weather_nonlinear(df)
    assert "temp_sq" in out.columns
    assert "temp_above_35" in out.columns


def test_temp_squared_correct():
    df = pd.DataFrame({"temp_c": [10.0, 20.0, -5.0]},
                      index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"))
    out = add_weather_nonlinear(df)
    assert out["temp_sq"].tolist() == [100.0, 400.0, 25.0]


def test_temp_above_35_clipped():
    df = pd.DataFrame({"temp_c": [25.0, 35.0, 40.0]},
                      index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"))
    out = add_weather_nonlinear(df)
    assert out["temp_above_35"].tolist() == [0.0, 0.0, 5.0]
