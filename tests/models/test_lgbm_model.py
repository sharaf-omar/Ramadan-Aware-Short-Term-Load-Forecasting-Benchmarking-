import numpy as np
import pandas as pd
from src.models.ml.lgbm import LightGBMModel


def _make_synthetic_df(n: int = 600, with_hijri: bool = True) -> pd.DataFrame:
    """Synthetic training-shaped data with all v2 columns we need."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    base_load = 30000 + 5000 * np.sin(2 * np.pi * np.arange(n) / 24)
    df = pd.DataFrame({
        "actual_load": base_load + rng.normal(scale=500, size=n),
        "temp_c": 15 + 10 * np.sin(2 * np.pi * np.arange(n) / 24),
        "dewpoint_c": 5.0,
        "wind_speed": 3.0,
        "solar_rad": np.maximum(0, 500 * np.sin(2 * np.pi * np.arange(n) / 24)),
        "temp_sq": (15 + 10 * np.sin(2 * np.pi * np.arange(n) / 24)) ** 2,
        "temp_above_35": 0.0,
        "hour_sin": np.sin(2 * np.pi * idx.hour / 24),
        "hour_cos": np.cos(2 * np.pi * idx.hour / 24),
        "dow_sin": np.sin(2 * np.pi * idx.dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * idx.dayofweek / 7),
        "month_sin": np.sin(2 * np.pi * idx.month / 12),
        "month_cos": np.cos(2 * np.pi * idx.month / 12),
        "hour": idx.hour, "day_of_week": idx.dayofweek, "month": idx.month,
        "is_weekend": (idx.dayofweek >= 5).astype(int),
        "y_lag_24h": base_load + rng.normal(scale=500, size=n),
        "y_lag_48h": base_load + rng.normal(scale=500, size=n),
        "y_lag_168h": base_load + rng.normal(scale=500, size=n),
        "y_lag_336h": base_load + rng.normal(scale=500, size=n),
        "y_roll24_mean": base_load,
        "y_roll24_std": 500.0,
        "y_roll168_mean": base_load,
        "y_roll168_std": 500.0,
        "heatwave_x_temp": 0.0,
        "is_ramadan": 0 if not with_hijri else (idx.month == 3).astype(int),
        "day_of_ramadan": 0,
        "is_eid": 0,
        "ramadan_x_hour_sin": 0.0,
        "ramadan_x_hour_cos": 0.0,
        "ramadan_x_weekend": 0,
        "ramadan_x_heatwave": 0,
        "ramadan_x_temp_above_35": 0.0,
        "regime": "Normal",
    }, index=idx)
    return df


def test_fit_and_predict_returns_unified_schema():
    df = _make_synthetic_df(n=1000)
    train_df = df.iloc[:600]
    val_df = df.iloc[600:800]
    test_df = df.iloc[800:]

    model = LightGBMModel(variant="hijri", n_estimators=50, learning_rate=0.1)
    model.fit(train_df, val_df, hijri=True, seed=42)
    out = model.predict(test_df)

    assert "y_true" in out.columns
    assert "y_pred" in out.columns
    assert "regime" in out.columns
    assert len(out) == len(test_df)
    assert out["y_pred"].notna().all()


def test_seed_reproducibility():
    df = _make_synthetic_df(n=1000)
    train_df = df.iloc[:600]
    val_df = df.iloc[600:800]
    test_df = df.iloc[800:]

    m1 = LightGBMModel(variant="hijri", n_estimators=20, learning_rate=0.1)
    m1.fit(train_df, val_df, hijri=True, seed=42)
    p1 = m1.predict(test_df)["y_pred"].values

    m2 = LightGBMModel(variant="hijri", n_estimators=20, learning_rate=0.1)
    m2.fit(train_df, val_df, hijri=True, seed=42)
    p2 = m2.predict(test_df)["y_pred"].values

    assert np.allclose(p1, p2)


def test_model_name_attribute():
    m = LightGBMModel(variant="hijri")
    assert m.name == "lgbm"
    assert m.needs_training is True
    assert m.supports_dynamic_covariates is True
