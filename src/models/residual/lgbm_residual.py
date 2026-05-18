"""LightGBM post-hoc residual head for a TSFM forecast.

Trained on (features) -> (y_true - y_pred_TSFM); the corrected forecast
is y_pred_TSFM + y_residual_hat. See
docs/superpowers/specs/2026-05-14-residual-correction-design.md.
"""
from __future__ import annotations

from typing import Literal

import lightgbm as lgb
import numpy as np
import pandas as pd


_WEATHER_FEATS = [
    "temp_c", "dewpoint_c", "wind_speed", "solar_rad",
    "temp_sq", "temp_above_35",
]
_CALENDAR_FEATS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
_LAG_FEATS = ["y_lag_24h", "y_lag_168h", "y_lag_336h",
              "y_roll168_mean", "y_roll168_std"]
_HIJRI_FEATS = ["is_ramadan", "day_of_ramadan", "is_eid"]


def _features_for_variant(variant: str) -> list[str]:
    base = _WEATHER_FEATS + _CALENDAR_FEATS + _LAG_FEATS
    if variant == "nohijri":
        return list(base)
    if variant == "hijri":
        return list(base) + list(_HIJRI_FEATS)
    raise ValueError(
        f"Unknown variant {variant!r}. Expected nohijri | hijri."
    )


def _ensure_calendar_features(feat_df: pd.DataFrame) -> pd.DataFrame:
    out = feat_df.copy()
    if "hour_sin" not in out.columns:
        h = out.index.hour.values
        out["hour_sin"] = np.sin(2 * np.pi * h / 24)
        out["hour_cos"] = np.cos(2 * np.pi * h / 24)
    if "dow_sin" not in out.columns:
        d = out.index.dayofweek.values
        out["dow_sin"] = np.sin(2 * np.pi * d / 7)
        out["dow_cos"] = np.cos(2 * np.pi * d / 7)
    if "is_weekend" not in out.columns:
        out["is_weekend"] = (out.index.dayofweek >= 5).astype(int)
    return out


class LGBMResidualModel:
    """LightGBM residual head for a single (TSFM, variant) combo."""
    name = "lgbm_residual"
    supports_dynamic_covariates = True
    needs_training = True

    def __init__(
        self,
        variant: Literal["nohijri", "hijri"] = "nohijri",
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        max_depth: int = -1,
        min_data_in_leaf: int = 50,
        early_stopping_rounds: int = 30,
    ):
        self.variant = variant
        self.features = _features_for_variant(variant)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_data_in_leaf = min_data_in_leaf
        self.early_stopping_rounds = early_stopping_rounds
        self._booster: lgb.Booster | None = None

    def _to_xy(
        self,
        tsfm_df: pd.DataFrame,
        feat_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        feat = _ensure_calendar_features(feat_df)
        common = tsfm_df.index.intersection(feat.index)
        if len(common) == 0:
            raise ValueError("tsfm_df and feat_df share no timestamps")
        X = feat.loc[common, self.features]
        residual = tsfm_df.loc[common, "y_true"] - tsfm_df.loc[common, "y_pred"]
        return X, residual

    def fit_residual(
        self,
        train_tsfm: pd.DataFrame,
        train_feat: pd.DataFrame,
        val_tsfm: pd.DataFrame,
        val_feat: pd.DataFrame,
        seed: int = 0,
    ) -> None:
        X_tr, y_tr = self._to_xy(train_tsfm, train_feat)
        X_va, y_va = self._to_xy(val_tsfm,   val_feat)
        dtrain = lgb.Dataset(X_tr, label=y_tr.values)
        dval = lgb.Dataset(X_va, label=y_va.values, reference=dtrain)
        params = {
            "objective": "regression_l1",
            "metric": "mae",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 5,
            "seed": seed,
            "verbose": -1,
        }
        self._booster = lgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

    def correct(
        self,
        test_tsfm: pd.DataFrame,
        test_feat: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply the trained residual head. Returns a DataFrame with the
        same index as test_tsfm (intersected with test_feat) and
        y_pred = y_pred_TSFM + y_residual_hat."""
        if self._booster is None:
            raise RuntimeError("Call fit_residual() before correct().")
        X, _ = self._to_xy(test_tsfm, test_feat)
        residual_hat = self._booster.predict(X)
        common = X.index
        corrected = test_tsfm.loc[common].copy()
        corrected["y_pred"] = corrected["y_pred"].values + residual_hat
        return corrected
