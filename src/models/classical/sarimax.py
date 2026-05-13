"""SARIMAX classical baseline (proposal section 4.1).

Order fixed at (1,0,1)(0,1,1,24) — the Hyndman & Athanasopoulos default
for hourly electricity load (FPP3 §9.10). One seasonal difference at
lag 24 cleanly absorbs the dominant daily cycle while preserving level
anchoring (no non-seasonal differencing → predicted long-horizon paths
do not random-walk away from observed level). We initially tried
auto_arima on daily-resampled data; the selected (0,1,3)(1,0,1,24)
gave 21% worse aggregate MAE because d=1 differencing causes errors
to accumulate over the 47-step day-ahead horizon — particularly
during the morning peak ramp.

Per test day d:
  - issuance t = first_tau(d) - 24h
  - forecast 47 steps from issuance
  - slice [23:47] to align with day d hours 0..23
This makes day d hour 0 a strict t+24 forecast; subsequent hours are
progressively longer-horizon extrapolations from the same issuance —
the day-ahead protocol used throughout the benchmark.

Variants differ only in the exogenous feature set.
"""
from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


BASE_EXOG = [
    "temp_c", "dewpoint_c", "wind_speed", "solar_rad",
    "temp_sq", "temp_above_35",
]
HIJRI_EXOG = ["is_ramadan", "day_of_ramadan", "is_eid"]
ABLATION_B_EXOG = ["ramadan_x_heatwave", "ramadan_x_temp_above_35"]


HORIZON = 24
ISSUANCE_OFFSET = 24


def _exog_for_variant(variant: str) -> list[str]:
    if variant == "nohijri":
        return list(BASE_EXOG)
    if variant == "hijri":
        return list(BASE_EXOG) + list(HIJRI_EXOG)
    if variant == "hijri_plusB":
        return list(BASE_EXOG) + list(HIJRI_EXOG) + list(ABLATION_B_EXOG)
    raise ValueError(
        f"Unknown variant {variant!r}. Expected nohijri | hijri | hijri_plusB."
    )


class SARIMAXModel:
    """SARIMAX with weather + (optionally) Hijri exogenous regressors."""
    name = "sarimax"
    supports_dynamic_covariates = True
    needs_training = True

    def __init__(
        self,
        variant: Literal["nohijri", "hijri", "hijri_plusB"] = "nohijri",
        order: tuple[int, int, int] = (1, 0, 1),
        seasonal_order: tuple[int, int, int, int] = (0, 1, 1, 24),
    ):
        self.variant = variant
        self.exog_features = _exog_for_variant(variant)
        self.order = order
        self.seasonal_order = seasonal_order
        self._fitted = None
        self._full_endog: pd.Series | None = None
        self._full_exog: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, hijri: bool, seed: int) -> None:
        history = pd.concat([train_df, val_df]).sort_index()
        history = history[~history.index.duplicated(keep="last")]
        endog = history["actual_load"]
        exog = history[self.exog_features]
        # Drop tz for statsmodels
        if endog.index.tz is not None:
            endog = endog.copy(); endog.index = endog.index.tz_convert(None)
            exog = exog.copy(); exog.index = exog.index.tz_convert(None)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sarimax = SARIMAX(
                endog, exog=exog,
                order=self.order, seasonal_order=self.seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            )
            self._fitted = sarimax.fit(
                disp=False, maxiter=200, method="lbfgs",
            )
        self._full_endog = endog
        self._full_exog = exog

    def predict(self, test_df: pd.DataFrame, context_length: int | None = None) -> pd.DataFrame:
        """Day-ahead batched forecasts.

        Per day d:
          1. issuance = (first τ of day d) - 24h
          2. extend Kalman state to issuance using observed data
          3. forecast 47 steps from issuance (covers issuance+1 .. issuance+47)
          4. slice [23:47] to align with day d hours 0..23

        Forecast at day d hour 0 is a strict t+24 forecast.
        Forecast at day d hour h (h>0) is a t+(24+h) forecast — longer-
        horizon extrapolation from the same issuance. This is the
        day-ahead protocol; documented in module docstring.
        """
        if self._fitted is None:
            raise RuntimeError("Call fit() before predict().")

        test_endog = test_df["actual_load"].copy()
        test_exog = test_df[self.exog_features].copy()
        if test_endog.index.tz is not None:
            test_endog.index = test_endog.index.tz_convert(None)
            test_exog.index = test_exog.index.tz_convert(None)

        all_endog = pd.concat([self._full_endog, test_endog]).sort_index()
        all_endog = all_endog[~all_endog.index.duplicated(keep="last")]
        all_exog = pd.concat([self._full_exog, test_exog]).sort_index()
        all_exog = all_exog[~all_exog.index.duplicated(keep="last")]

        FORECAST_STEPS = HORIZON + HORIZON - 1  # 47

        test_dates = sorted({ts.date() for ts in test_df.index})
        results: list[pd.DataFrame] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for d in test_dates:
                day_rows = test_df[test_df.index.date == d]
                if len(day_rows) == 0:
                    continue
                earliest_tau_tz = day_rows.index.min()
                earliest_tau = (earliest_tau_tz.tz_convert(None)
                                if earliest_tau_tz.tz is not None else earliest_tau_tz)
                issuance = earliest_tau - pd.Timedelta(hours=ISSUANCE_OFFSET)
                endog_to_issuance = all_endog.loc[:issuance]
                exog_to_issuance = all_exog.loc[:issuance]
                if len(endog_to_issuance) < 10:
                    continue

                try:
                    extended = self._fitted.apply(
                        endog=endog_to_issuance,
                        exog=exog_to_issuance,
                        refit=False,
                    )
                except Exception as exc:
                    print(f"[WARN] SARIMAX apply failed for {d}: {exc}")
                    continue

                horizon_start = issuance + pd.Timedelta(hours=1)
                horizon_end = issuance + pd.Timedelta(hours=FORECAST_STEPS)
                horizon_exog = all_exog.loc[horizon_start:horizon_end]
                if len(horizon_exog) < FORECAST_STEPS:
                    # Not enough exog data for full forecast horizon.
                    continue
                try:
                    fc = extended.forecast(
                        steps=FORECAST_STEPS,
                        exog=horizon_exog.iloc[:FORECAST_STEPS],
                    )
                    block = np.asarray(fc, dtype=float)
                except Exception as exc:
                    print(f"[WARN] SARIMAX forecast failed for {d}: {exc}")
                    continue

                # Slice [23:47] to align with day d hours 0..23
                day_block = block[HORIZON - 1 : HORIZON - 1 + HORIZON]
                day_preds = pd.DataFrame({
                    "y_true": day_rows["actual_load"].values,
                    "y_pred": day_block[: len(day_rows)],
                    "regime": day_rows["regime"].values,
                }, index=day_rows.index)
                results.append(day_preds)
        return pd.concat(results) if results else pd.DataFrame(
            columns=["y_true", "y_pred", "regime"]
        )
