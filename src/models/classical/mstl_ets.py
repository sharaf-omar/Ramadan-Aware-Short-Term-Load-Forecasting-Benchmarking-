"""MSTL + ETS classical baseline (proposal section 4.1).

Strategy: fit MSTL ONCE on train+val with periods (24, 168). For each test
day, use a recent-window-weighted seasonal lookup, then fit Holt's linear
ETS on the deseasonalized recent residual to produce a *trajectory*
across the 24h horizon (not a flat trend).

Optimizations vs. the naive version:
  - Seasonal lookups use only the most recent 12 weeks (not the full
    last year) so they track current dynamics rather than being diluted
    by older, possibly out-of-distribution patterns.
  - ETS uses trend='add' (Holt's linear method) so the level evolves
    across the 24h horizon — peak/off-peak transitions are sharper
    than with a constant trend.
  - The 'hijri' Ramadan pattern uses the MOST RECENT contiguous Ramadan
    block (not a multi-year mean) so it captures current consumer
    behaviour instead of averaging over multiple shifting Ramadan years.

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
SEASONAL_LOOKBACK_WEEKS = 12  # recent window for seasonal lookups
ETS_WINDOW_HOURS = 24 * 30    # 30-day deseasonalized window for Holt's ETS


class MSTLETSModel:
    """MSTL decomposition + Holt's linear ETS trend forecast."""
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
        self._mstl_trend: pd.Series | None = None
        self._mstl_seasonal: pd.DataFrame | None = None
        self._mstl_residual: pd.Series | None = None
        self._daily_by_hour: pd.Series | None = None
        self._weekly_by_dwh: pd.Series | None = None
        self._ramadan_hourly_pattern: pd.Series | None = None

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

        # Most-recent-N-weeks slice gives lookups that track current dynamics
        # instead of being diluted by older patterns from the full history.
        lookback_hours = SEASONAL_LOOKBACK_WEEKS * 168
        recent_seasonal = (
            seasonal.iloc[-lookback_hours:]
            if len(seasonal) > lookback_hours else seasonal
        )

        daily_col = seasonal.columns[0]
        self._daily_by_hour = recent_seasonal[daily_col].groupby(
            recent_seasonal.index.hour
        ).mean()

        if len(seasonal.columns) > 1:
            weekly_col = seasonal.columns[1]
            recent_weekly = recent_seasonal[weekly_col]
            self._weekly_by_dwh = recent_weekly.groupby(
                [recent_weekly.index.dayofweek, recent_weekly.index.hour]
            ).mean()
        else:
            self._weekly_by_dwh = None

        # Ramadan pattern from the MOST RECENT contiguous Ramadan block so the
        # signal reflects current consumer behaviour, not a multi-year average.
        if self.variant == "hijri" and "is_ramadan" in history.columns:
            ram_mask = history["is_ramadan"] == 1
            if ram_mask.any():
                ram_y = y[ram_mask.values]
                # Identify contiguous blocks: any gap > 2h is a new block.
                breaks = ram_y.index.to_series().diff().gt(pd.Timedelta("2h"))
                block_id = breaks.cumsum()
                last_block_id = block_id.iloc[-1]
                last_ramadan = ram_y[block_id.values == last_block_id]
                if len(last_ramadan) >= 24 * 7:
                    ram_smooth = last_ramadan.rolling(window=24, min_periods=12).mean()
                    ram_detrended = last_ramadan - ram_smooth.fillna(last_ramadan.mean())
                    self._ramadan_hourly_pattern = ram_detrended.groupby(
                        ram_detrended.index.hour
                    ).mean()

    def _seasonal_at(self, ts: pd.Timestamp, in_ramadan: bool) -> float:
        """Combined (daily + weekly) seasonal contribution at timestamp ts.

        In Ramadan (hijri variant), the daily component is replaced by the
        Ramadan-only hourly pattern; the weekly day-of-week effect is kept.
        """
        daily = float(self._daily_by_hour.get(ts.hour, 0.0))
        weekly = (
            float(self._weekly_by_dwh.get((ts.dayofweek, ts.hour), 0.0))
            if self._weekly_by_dwh is not None else 0.0
        )
        if self.variant == "hijri" and in_ramadan and self._ramadan_hourly_pattern is not None:
            daily = float(self._ramadan_hourly_pattern.get(ts.hour, daily))
        return daily + weekly

    def predict(self, test_df: pd.DataFrame, context_length: int | None = None) -> pd.DataFrame:
        if self._mstl_seasonal is None:
            raise RuntimeError("Call fit() before predict().")

        all_data = pd.concat([self._full_history, test_df]).sort_index()
        all_data = all_data[~all_data.index.duplicated(keep="last")]
        y_full = all_data["actual_load"].copy()
        if y_full.index.tz is not None:
            y_full.index = y_full.index.tz_convert(None)

        results: list[pd.DataFrame] = []
        test_dates = sorted({ts.date() for ts in test_df.index})

        for d in test_dates:
            day_rows = test_df[test_df.index.date == d]
            if len(day_rows) == 0:
                continue
            earliest_tau = day_rows.index.min()
            issuance_tz = earliest_tau - pd.Timedelta(hours=ISSUANCE_OFFSET)
            issuance = issuance_tz.tz_convert(None) if issuance_tz.tz is not None else issuance_tz

            if issuance not in y_full.index:
                continue

            in_ramadan = (
                bool(day_rows["is_ramadan"].iloc[0])
                if "is_ramadan" in day_rows.columns else False
            )

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

        Trend: Holt's linear ETS on the deseasonalized recent observed
          series. Produces a *trajectory* — level + slope evolution across
          the 24h horizon, not a flat projection.
        Seasonal: per-(hour, weekday) lookup over the most recent
          SEASONAL_LOOKBACK_WEEKS of the MSTL decomposition.
        """
        seasonal_fc = np.array([
            self._seasonal_at(horizon_start + pd.Timedelta(hours=h), in_ramadan)
            for h in range(HORIZON)
        ])

        recent_window = y_full.loc[:issuance].iloc[-ETS_WINDOW_HOURS:]
        if len(recent_window) >= 48:
            recent_deseasoned = np.array([
                v - self._seasonal_at(ts, in_ramadan=False)
                for ts, v in zip(recent_window.index, recent_window.values)
            ])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ets = ExponentialSmoothing(
                        recent_deseasoned, trend="add", seasonal=None,
                        initialization_method="estimated",
                    ).fit(disp=False, optimized=True)
                    trend_fc = np.asarray(ets.forecast(HORIZON), dtype=float)
            except Exception:
                trend_fc = np.full(HORIZON, float(np.mean(recent_deseasoned[-24:])))
        else:
            trend_fc = np.full(HORIZON, float(self._mstl_trend.iloc[-1]))

        return np.asarray(trend_fc + seasonal_fc, dtype=float)
