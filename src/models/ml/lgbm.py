"""LightGBM forecasting model wrapper.

Feature sets per ablation variant:
    nohijri      = BASE_FEATURES
    hijri        = BASE_FEATURES + HIJRI_FEATURES
    hijri_plusB  = BASE_FEATURES + HIJRI_FEATURES + ABLATION_B_FEATURES
"""
from __future__ import annotations

from typing import Any

import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error


BASE_FEATURES: list[str] = [
    # Weather
    "temp_c", "dewpoint_c", "wind_speed", "solar_rad",
    "temp_sq", "temp_above_35",
    # Calendar (cyclical)
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    # Raw calendar (kept for tree splits)
    "hour", "day_of_week", "month",
    "is_weekend",
    # Leak-free lags (computed in build_v2_dataset)
    "y_lag_24h", "y_lag_48h", "y_lag_168h", "y_lag_336h",
    # Leak-free rolling stats
    "y_roll24_mean", "y_roll24_std",
    "y_roll168_mean", "y_roll168_std",
    # Heatwave interaction (no Ramadan dependence)
    "heatwave_x_temp",
]


HIJRI_FEATURES: list[str] = [
    "is_ramadan", "day_of_ramadan", "is_eid",
    "ramadan_x_hour_sin", "ramadan_x_hour_cos", "ramadan_x_weekend",
]


ABLATION_B_FEATURES: list[str] = [
    "ramadan_x_heatwave",
    "ramadan_x_temp_above_35",
]


def feature_set_for_variant(variant: str) -> list[str]:
    """Return the feature column list for a given ablation variant."""
    if variant == "nohijri":
        return list(BASE_FEATURES)
    if variant == "hijri":
        return list(BASE_FEATURES) + list(HIJRI_FEATURES)
    if variant == "hijri_plusB":
        return list(BASE_FEATURES) + list(HIJRI_FEATURES) + list(ABLATION_B_FEATURES)
    raise ValueError(
        f"Unknown variant {variant!r}. Expected one of: nohijri, hijri, hijri_plusB."
    )


_DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 5.0,
    "verbose": -1,
    "n_jobs": -1,
}


class LightGBMModel:
    """LightGBM forecaster implementing the Model protocol."""
    name = "lgbm"
    supports_dynamic_covariates = True
    needs_training = True

    def __init__(self, variant: str, **param_overrides):
        self.variant = variant
        self.features = feature_set_for_variant(variant)
        self.params: dict[str, Any] = {**_DEFAULT_PARAMS, **param_overrides}
        self._model: lgb.LGBMRegressor | None = None

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        hijri: bool,
        seed: int,
    ) -> None:
        params = {**self.params, "random_state": seed}
        self._model = lgb.LGBMRegressor(**params)
        x_train = train_df[self.features]
        y_train = train_df["actual_load"]
        x_val = val_df[self.features]
        y_val = val_df["actual_load"]
        self._model.fit(
            x_train, y_train,
            eval_set=[(x_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

    def predict(
        self,
        test_df: pd.DataFrame,
        context_length: int | None = None,
    ) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        x = test_df[self.features]
        y_pred = self._model.predict(x)
        return pd.DataFrame({
            "y_true": test_df["actual_load"].values,
            "y_pred": y_pred,
            "regime": test_df["regime"].values,
        }, index=test_df.index)


def tune_with_optuna(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    variant: str,
    n_trials: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """50-trial TPE search on validation MAE. Returns best param dict."""
    features = feature_set_for_variant(variant)
    x_train = train_df[features]
    y_train = train_df["actual_load"]
    x_val = val_df[features]
    y_val = val_df["actual_load"]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression", "metric": "mae", "boosting_type": "gbdt",
            "verbose": -1, "random_state": seed, "n_jobs": -1,
            "n_estimators": 2000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 1,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(
            x_train, y_train,
            eval_set=[(x_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        pred = model.predict(x_val)
        return mean_absolute_error(y_val, pred)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return dict(study.best_params)
