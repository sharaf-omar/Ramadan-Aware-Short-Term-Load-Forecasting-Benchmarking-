import numpy as np
import pandas as pd
import pytest

from src.models.classical.sarimax import SARIMAXModel


def _synthetic_df(n: int = 1500) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    t = np.arange(n)
    load = (
        30000
        + 5000 * np.sin(2 * np.pi * t / 24)
        + 200 * (15 + 10 * np.sin(2 * np.pi * t / 24))
        + np.random.default_rng(0).normal(scale=300, size=n)
    )
    return pd.DataFrame({
        "actual_load": load,
        "temp_c": 15 + 10 * np.sin(2 * np.pi * t / 24),
        "dewpoint_c": 5.0,
        "wind_speed": 3.0,
        "solar_rad": 0.0,
        "temp_sq": (15 + 10 * np.sin(2 * np.pi * t / 24)) ** 2,
        "temp_above_35": 0.0,
        "is_ramadan": 0,
        "day_of_ramadan": 0,
        "is_eid": 0,
        "ramadan_x_heatwave": 0,
        "ramadan_x_temp_above_35": 0.0,
        "regime": "Normal",
    }, index=idx)


def test_sarimax_model_attributes():
    m = SARIMAXModel(variant="nohijri")
    assert m.name == "sarimax"
    assert m.needs_training is True
    assert m.supports_dynamic_covariates is True


def test_sarimax_predict_returns_unified_schema():
    df = _synthetic_df(1500)
    train = df.iloc[:1000]
    val = df.iloc[1000:1200]
    test = df.iloc[1200:]

    m = SARIMAXModel(variant="nohijri", order=(1, 0, 0), seasonal_order=(0, 0, 0, 0))
    m.fit(train, val, hijri=False, seed=0)
    out = m.predict(test)
    assert "y_true" in out.columns
    assert "y_pred" in out.columns
    assert "regime" in out.columns
    assert out["y_pred"].notna().all()


def test_sarimax_variant_rejects_unknown():
    with pytest.raises(ValueError):
        SARIMAXModel(variant="nonsense")


def test_sarimax_feature_set_per_variant():
    m_nh = SARIMAXModel(variant="nohijri")
    m_h  = SARIMAXModel(variant="hijri")
    m_pb = SARIMAXModel(variant="hijri_plusB")
    assert set(m_h.exog_features) - set(m_nh.exog_features) == {
        "is_ramadan", "day_of_ramadan", "is_eid",
    }
    assert set(m_pb.exog_features) - set(m_h.exog_features) == {
        "ramadan_x_heatwave", "ramadan_x_temp_above_35",
    }
