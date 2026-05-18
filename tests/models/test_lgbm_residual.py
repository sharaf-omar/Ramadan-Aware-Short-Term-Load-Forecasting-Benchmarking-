import numpy as np
import pandas as pd
import pytest

from src.models.residual.lgbm_residual import LGBMResidualModel


def _synthetic_tsfm_preds_and_features(n=3000, drift_per_hr=0.1):
    """Synthetic TSFM-style predictions + v2 features.

    Two Ramadan blocks so train can include one and test can include
    the other:
      - Ramadan #1: t in [500, 720)  -> in train
      - Ramadan #2: t in [2200, 2420) -> in test
    The TSFM under-predicts during Ramadan; the hijri residual head
    should learn to correct from block #1 and apply on block #2.
    """
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    t = np.arange(n)
    truth = 30000 + 5000 * np.sin(2 * np.pi * t / 24) + drift_per_hr * t
    ram_mask = ((t >= 500) & (t < 720)) | ((t >= 2200) & (t < 2420))
    truth = truth + np.where(ram_mask, -3000.0, 0.0)
    tsfm_pred = 30000 + 5000 * np.sin(2 * np.pi * t / 24) + drift_per_hr * t
    tsfm_df = pd.DataFrame({
        "y_true": truth,
        "y_pred": tsfm_pred,
        "regime": np.where(ram_mask, "Ramadan", "Normal"),
    }, index=idx)
    feat_df = pd.DataFrame({
        "actual_load": truth,
        "temp_c": 15.0 + 10.0 * np.sin(2 * np.pi * t / 24),
        "dewpoint_c": 5.0,
        "wind_speed": 3.0,
        "solar_rad": 0.0,
        "temp_sq": (15.0 + 10.0 * np.sin(2 * np.pi * t / 24)) ** 2,
        "temp_above_35": 0.0,
        "is_ramadan": ram_mask.astype(int),
        "day_of_ramadan": np.where(ram_mask, t % 30 + 1, 0),
        "is_eid": 0,
        "y_lag_24h": np.roll(truth, 24),
        "y_lag_168h": np.roll(truth, 168),
        "y_lag_336h": np.roll(truth, 336),
        "y_roll168_mean": pd.Series(truth, index=idx).rolling(168, min_periods=1).mean().values,
        "y_roll168_std": pd.Series(truth, index=idx).rolling(168, min_periods=1).std().fillna(0).values,
        "regime": np.where(ram_mask, "Ramadan", "Normal"),
    }, index=idx)
    return tsfm_df, feat_df


def test_lgbm_residual_model_attributes():
    m = LGBMResidualModel(variant="nohijri")
    assert m.name == "lgbm_residual"
    assert m.needs_training is True
    assert m.supports_dynamic_covariates is True


def test_lgbm_residual_variant_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown variant"):
        LGBMResidualModel(variant="nonsense")


def test_lgbm_residual_feature_set_per_variant():
    base = ["temp_c", "dewpoint_c", "wind_speed", "solar_rad",
            "temp_sq", "temp_above_35",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
            "y_lag_24h", "y_lag_168h", "y_lag_336h",
            "y_roll168_mean", "y_roll168_std"]
    hijri_extras = ["is_ramadan", "day_of_ramadan", "is_eid"]
    m_nh = LGBMResidualModel(variant="nohijri")
    m_h  = LGBMResidualModel(variant="hijri")
    assert set(m_nh.features) == set(base)
    assert set(m_h.features) - set(m_nh.features) == set(hijri_extras)


def test_lgbm_residual_correct_runs_and_reduces_ramadan_error():
    tsfm_df, feat_df = _synthetic_tsfm_preds_and_features(n=3000)
    # train sees Ramadan #1 (t=500..720); test sees Ramadan #2 (t=2200..2420).
    train_end = 2000
    val_end = 2100
    train_tsfm = tsfm_df.iloc[:train_end]
    val_tsfm   = tsfm_df.iloc[train_end:val_end]
    test_tsfm  = tsfm_df.iloc[val_end:]

    m = LGBMResidualModel(
        variant="hijri", n_estimators=300, learning_rate=0.05,
        # Loosen LGBM regularization so it can learn from a small synthetic
        # Ramadan signal (220 hours).
        min_data_in_leaf=20, num_leaves=31, early_stopping_rounds=20,
    )
    m.fit_residual(train_tsfm, feat_df, val_tsfm, feat_df, seed=0)
    corrected = m.correct(test_tsfm, feat_df)

    assert {"y_true", "y_pred", "regime"} <= set(corrected.columns)
    assert corrected.index.equals(test_tsfm.index)
    bare_ram_mae = (test_tsfm[test_tsfm.regime == "Ramadan"].y_true
                    - test_tsfm[test_tsfm.regime == "Ramadan"].y_pred).abs().mean()
    corr_ram_mae = (corrected[corrected.regime == "Ramadan"].y_true
                    - corrected[corrected.regime == "Ramadan"].y_pred).abs().mean()
    # Smoke check: residual must move predictions toward truth (any
    # improvement passes). Real evaluation lives in the run-script outputs.
    assert corr_ram_mae < bare_ram_mae, (
        f"Residual did not reduce Ramadan MAE: bare={bare_ram_mae:.1f} corr={corr_ram_mae:.1f}"
    )
