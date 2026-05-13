"""MSTL + ETS classical baseline (proposal section 4.1).

Strategy: fit MSTL ONCE on train+val with periods (24, 168). For each test
day, project the trend (last value), repeat the seasonal cycle, and refit
ETS(A,N,N) on the residual extended to the current issuance time.

The 'hijri' variant swaps the daily seasonal component for a Ramadan-only
estimate when the forecast issuance falls in Ramadan.

Why drop the yearly period (8766) from the proposal spec: at 50k+ training
hours, MSTL with periods (24, 168, 8766) takes ~3 min per fit. Even with
a single fit, we'd still pay that once; daily fits would be untenable.
The yearly seasonal contribution is mostly explained by temperature
(handled in other models via temp features) and the load data shows
weak yearly periodicity in the EDA.
"""
from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import MSTL


PERIODS = (24, 168)  # daily + weekly only (yearly dropped — see module docstring)
HORIZON = 24
ISSUANCE_OFFSET = 24


class MSTLETSModel:
    """MSTL decomposition + ETS(A,N,N) residual forecast."""
    name = "mstl_ets"
    supports_dynamic_covariates = False
    needs_training = True

    def __init__(self, variant: Literal["nohijri", "hijri"] = "nohijri"):
        if variant not in ("nohijri", "hijri"):
            raise ValueError(
                f"Unknown variant {variant!r}. Expected one of: nohijri, hijri."
            )
        self.variant = variant
        self._full_history: pd.DataFrame | None = None
        # Cached MSTL artifacts after fit()
        self._mstl_trend: pd.Series | None = None
        self._mstl_seasonal: pd.DataFrame | None = None
        self._mstl_residual: pd.Series | None = None
        self._ramadan_hourly_pattern: pd.Series | None = None  # for hijri variant

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, hijri: bool, seed: int) -> None:
        """Fit MSTL ONCE on train+val. Per-day predict() reuses this."""
        history = pd.concat([train_df, val_df]).sort_index()
        history = history[~history.index.duplicated(keep="last")]
        self._full_history = history

        y = history["actual_load"]
        if y.index.tz is not None:
            y = y.copy()
            y.index = y.index.tz_convert(None)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mstl = MSTL(y, periods=PERIODS).fit()
        self._mstl_trend = mstl.trend
        seasonal = mstl.seasonal
        if isinstance(seasonal, pd.Series):
            seasonal = seasonal.to_frame(name=f"seasonal_{PERIODS[0]}")
        self._mstl_seasonal = seasonal
        self._mstl_residual = mstl.resid

        # Pre-compute per-(hour) daily seasonal lookup (mean of daily seasonal
        # column grouped by hour-of-day from the most recent full year).
        recent_seasonal = seasonal.iloc[-365 * 24:] if len(seasonal) > 365 * 24 else seasonal
        daily_col = seasonal.columns[0]
        self._daily_by_hour = recent_seasonal[daily_col].groupby(
            recent_seasonal.index.hour
        ).mean()

        # Weekly seasonal lookup: mean by (day-of-week, hour-of-day)
        if len(seasonal.columns) > 1:
            weekly_col = seasonal.columns[1]
            recent_weekly = recent_seasonal[weekly_col]
            self._weekly_by_dwh = recent_weekly.groupby(
                [recent_weekly.index.dayofweek, recent_weekly.index.hour]
            ).mean()
        else:
            self._weekly_by_dwh = None

        # Pre-compute Ramadan-only hourly pattern for hijri variant
        if self.variant == "hijri" and "is_ramadan" in history.columns:
            ramadan_mask = history["is_ramadan"] == 1
            ram_y = y[ramadan_mask.values]
            if len(ram_y) >= 24 * 14:
                ram_smooth = ram_y.rolling(window=168, min_periods=24).mean()
                ram_detrended = ram_y - ram_smooth.fillna(ram_y.mean())
                self._ramadan_hourly_pattern = ram_detrended.groupby(
                    ram_detrended.index.hour
                ).mean()

    def predict(self, test_df: pd.DataFrame, context_length: int | None = None) -> pd.DataFrame:
        if self._mstl_seasonal is None:
            raise RuntimeError("Call fit() before predict().")

        # Build the full observed series (train+val+test actuals) so we can
        # extend the residual signal up to each test day's issuance time.
        all_data = pd.concat([self._full_history, test_df]).sort_index()
        all_data = all_data[~all_data.index.duplicated(keep="last")]
        y_full = all_data["actual_load"].copy()
        if y_full.index.tz is not None:
            y_full.index = y_full.index.tz_convert(None)

        # Pre-compute seasonal forecasts per (hour, weekday) once
        results: list[pd.DataFrame] = []
        test_dates = sorted({ts.date() for ts in test_df.index})

        for d in test_dates:
            day_rows = test_df[test_df.index.date == d]
            if len(day_rows) == 0:
                continue
            earliest_tau = day_rows.index.min()
            # tz-naive issuance for indexing into y_full
            issuance_tz = earliest_tau - pd.Timedelta(hours=ISSUANCE_OFFSET)
            issuance = issuance_tz.tz_convert(None) if issuance_tz.tz is not None else issuance_tz

            if issuance not in y_full.index:
                continue

            in_ramadan = bool(day_rows["is_ramadan"].iloc[0])

            block = self._forecast_one_day(
                y_full=y_full,
                issuance=issuance,
                horizon_start=(earliest_tau if earliest_tau.tz is None
                               else earliest_tau.tz_convert(None)),
                in_ramadan=in_ramadan,
            )
            day_preds = pd.DataFrame({
                "y_true": day_rows["actual_load"].values,
                "y_pred": block[: len(day_rows)],
                "regime": day_rows["regime"].values,
            }, index=day_rows.index)
            results.append(day_preds)

        return pd.concat(results) if results else pd.DataFrame(
            columns=["y_true", "y_pred", "regime"]
        )

    def _forecast_one_day(
        self,
        y_full: pd.Series,
        issuance: pd.Timestamp,
        horizon_start: pd.Timestamp,
        in_ramadan: bool,
    ) -> np.ndarray:
        """Produce a 24-hour forecast for the day starting at horizon_start.

        Trend: ETS(A,N,N) on observed series (recent slice) gives drift-aware
          near-future projection.
        Seasonal: per-(hour, weekday) lookup from cached MSTL decomposition.
        Residual: assumed zero-mean (already captured by the lookup).
        """
        # Compute the seasonal contribution for each of the 24 horizon hours.
        seasonal_fc = np.zeros(HORIZON)
        for h in range(HORIZON):
            ts = horizon_start + pd.Timedelta(hours=h)
            hod = ts.hour
            dow = ts.dayofweek
            daily_contrib = float(self._daily_by_hour.get(hod, 0.0))
            weekly_contrib = (
                float(self._weekly_by_dwh.get((dow, hod), 0.0))
                if self._weekly_by_dwh is not None else 0.0
            )
            seasonal_fc[h] = daily_contrib + weekly_contrib

        # Ramadan-aware daily seasonal override (replace the daily component).
        if self.variant == "hijri" and in_ramadan and self._ramadan_hourly_pattern is not None:
            for h in range(HORIZON):
                ts = horizon_start + pd.Timedelta(hours=h)
                hod = ts.hour
                # Subtract the standard daily lookup and add the Ramadan one.
                seasonal_fc[h] -= float(self._daily_by_hour.get(hod, 0.0))
                seasonal_fc[h] += float(self._ramadan_hourly_pattern.get(hod, 0.0))

        # Trend forecast: ETS(A,N,N) on the deseasonalized recent observed series.
        # Deseasonalize: y_observed - seasonal_lookup_at_each_timestamp.
        recent_window = y_full.loc[:issuance].iloc[-min(len(y_full), 24 * 60):]
        if len(recent_window) >= 48:
            recent_deseasoned = np.array([
                v - float(self._daily_by_hour.get(ts.hour, 0.0))
                  - (float(self._weekly_by_dwh.get((ts.dayofweek, ts.hour), 0.0))
                     if self._weekly_by_dwh is not None else 0.0)
                for ts, v in zip(recent_window.index, recent_window.values)
            ])
            try:
                ets = ExponentialSmoothing(
                    recent_deseasoned, trend=None, seasonal=None,
                ).fit(disp=False)
                trend_fc = np.full(HORIZON, float(ets.forecast(1)[0]))
            except Exception:
                trend_fc = np.full(HORIZON, float(np.mean(recent_deseasoned)))
        else:
            trend_fc = np.full(HORIZON, float(self._mstl_trend.iloc[-1]))

        return np.asarray(trend_fc + seasonal_fc, dtype=float)
