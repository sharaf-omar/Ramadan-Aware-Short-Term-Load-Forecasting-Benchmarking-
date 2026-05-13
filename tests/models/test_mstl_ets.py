import numpy as np
import pandas as pd
import pytest

from src.models.classical.mstl_ets import MSTLETSModel


def _synthetic_df(n: int = 2000) -> pd.DataFrame:
    """Hourly synthetic with daily + weekly seasonality + slow trend."""
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    t = np.arange(n)
    load = (
        30000
        + 5000 * np.sin(2 * np.pi * t / 24)
        + 2000 * np.sin(2 * np.pi * t / 168)
        + 0.5 * t
        + np.random.default_rng(0).normal(scale=200, size=n)
    )
    return pd.DataFrame({
        "actual_load": load,
        "is_ramadan": 0,
        "regime": "Normal",
    }, index=idx)


def test_mstl_ets_model_attributes():
    m = MSTLETSModel(variant="nohijri")
    assert m.name == "mstl_ets"
    assert m.needs_training is True
    assert m.supports_dynamic_covariates is False


def test_mstl_ets_predict_returns_unified_schema():
    df = _synthetic_df(2000)
    train = df.iloc[:1500]
    val = df.iloc[1500:1700]
    test = df.iloc[1700:]

    m = MSTLETSModel(variant="nohijri")
    m.fit(train, val, hijri=False, seed=0)
    out = m.predict(test)
    assert "y_true" in out.columns
    assert "y_pred" in out.columns
    assert "regime" in out.columns
    assert out["y_pred"].notna().all()


def test_mstl_ets_variant_rejects_unknown():
    with pytest.raises(ValueError):
        MSTLETSModel(variant="nonsense")


def test_mstl_ets_hijri_variant_loads():
    m = MSTLETSModel(variant="hijri")
    assert m.variant == "hijri"
