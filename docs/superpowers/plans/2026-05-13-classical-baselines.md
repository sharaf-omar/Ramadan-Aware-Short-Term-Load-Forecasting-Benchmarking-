# Classical Baselines (MSTL+ETS + SARIMAX) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the two classical baselines from proposal §4.1 — MSTL+ETS with Ramadan-aware seasonal re-estimation, and SARIMAX with weather + Hijri exogenous regressors — running on the same v2 test set and emitting predictions to `data/predictions/` in the canonical schema. Adds 5 prediction parquets (2 MSTL+ETS variants + 3 SARIMAX variants) and updates the cross-model headline doc.

**Architecture:** Each model gets a thin wrapper in `src/models/classical/<name>.py` implementing the `Model` protocol. Both are CPU-bound; no GPU needed. To keep runtime tractable on a single laptop CPU: MSTL+ETS refits its decomposition **once per test day** (24 hourly forecasts share the same daily decomposition); SARIMAX fits its parameters **once** on the train window, then uses statsmodels' state-space `extend()` API to evolve the Kalman state with each test day's data and emit 24h forecasts. Both wrappers store per-forecast results to parquet via the existing `write_predictions()` helper.

**Scope decisions (vs the full spec):**
- **MSTL+ETS variants**: 2 — `nohijri` (standard MSTL on full history) and `hijri` (Ramadan-window seasonal re-estimation per proposal §4.1).
- **SARIMAX variants**: 3 — `nohijri` / `hijri` / `hijri_plusB`. Order selection runs ONCE on weekly-downsampled training data; the chosen `(p,d,q)(P,D,Q,24)` is reused across all variants. Parameters fit on train+val, then state extended per test day.
- **No Optuna**: classical baselines have hyperparameters set by proposal §4.1 (MSTL periods, SARIMAX order via auto_arima). No tuning loop.
- **Single seed**: deterministic models; seed parameter is accepted but unused.

**Tech Stack:** statsmodels (MSTL, ExponentialSmoothing, SARIMAX), pmdarima (auto_arima for order selection), pandas, numpy. CPU-only; no torch/CUDA.

**Reference docs:**
- Spec §4.1, §5.3, §5.4: `docs/superpowers/specs/2026-05-13-tsfm-ablations-completion-design.md`
- Predecessor plans: `docs/superpowers/plans/2026-05-13-foundation-and-lgbm-refactor.md`, `docs/superpowers/plans/2026-05-13-tsfm-zero-shot-baseline.md`, `docs/superpowers/plans/2026-05-13-tsfm-ablations-sweep.md`
- LGBM finding (sets the Ramadan-feature bar): `docs/v1_v2_lgbm_delta.md` — LGBM-hijri Ramadan MAE = 800.

---

## Phase 0 — Environment Bootstrap

### Task 0.1: Install pmdarima

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Install pmdarima for auto_arima**

Run:
```bash
.venv/Scripts/python.exe -m pip install "pmdarima>=2.0"
```
Expected: installs without conflict. If pmdarima requires older numpy or scipy, accept the warning — both packages keep their wheels regardless of pmdarima's deps.

- [ ] **Step 2: Smoke-test imports**

Run:
```bash
.venv/Scripts/python.exe -c "
import statsmodels; print(f'statsmodels={statsmodels.__version__}')
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima; print(f'pmdarima={pmdarima.__version__}')
from pmdarima import auto_arima
print('all classical imports ok')
"
```
Expected: all 5 import without errors. statsmodels already present from Plan 1.

If pmdarima install fails: report BLOCKED with the specific error. pmdarima sometimes has wheel issues on Python 3.12; fallback is `statsforecast`'s `AutoARIMA` which is a separate plan.

- [ ] **Step 3: Add pmdarima to requirements.txt**

Append:
```
pmdarima>=2.0
```

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore(env): add pmdarima for SARIMAX order selection"
```

---

## Phase 1 — MSTL + ETS

### Task 1.1: MSTL+ETS wrapper

**Files:**
- Create: `src/models/classical/mstl_ets.py`
- Create: `tests/models/test_mstl_ets.py`

- [ ] **Step 1: Write the failing test**

`tests/models/test_mstl_ets.py`:
```python
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
        + 5000 * np.sin(2 * np.pi * t / 24)            # daily
        + 2000 * np.sin(2 * np.pi * t / 168)           # weekly
        + 0.5 * t                                        # slow trend
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
    assert len(out) == len(test)
    assert out["y_pred"].notna().all()


def test_mstl_ets_variant_rejects_unknown():
    with pytest.raises(ValueError):
        MSTLETSModel(variant="nonsense")


def test_mstl_ets_hijri_variant_loads():
    m = MSTLETSModel(variant="hijri")
    assert m.variant == "hijri"
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/models/test_mstl_ets.py -v`
Expected: 4 failures with `ModuleNotFoundError: No module named 'src.models.classical.mstl_ets'`.

- [ ] **Step 3: Implement `src/models/classical/mstl_ets.py`**

```python
"""MSTL + ETS classical baseline (proposal §4.1).

MSTL decomposition with periods (24, 168, 8766) for daily/weekly/yearly
seasonality, then ETS(A,N,N) on the residual. The 'hijri' variant
re-estimates the daily seasonal component on Ramadan-only history when
the forecast issuance falls within a Ramadan window.

Refits the daily decomposition ONCE PER TEST DAY (not per hour) to keep
runtime tractable. Each daily fit produces a 24-hour forecast block.
"""
from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import MSTL


PERIODS = (24, 168, 8766)  # daily, weekly, yearly (per proposal §4.1)
HORIZON = 24
ISSUANCE_OFFSET = 24


class MSTLETSModel:
    """MSTL decomposition + ETS(A,N,N) residual forecast.

    variant="nohijri": standard MSTL on full history.
    variant="hijri":   when issuance is in Ramadan, re-estimate the daily
                       seasonal component from Ramadan-only history.
    """
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

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, hijri: bool, seed: int) -> None:
        """Cache the train+val history; MSTL fits are per-forecast."""
        self._full_history = pd.concat([train_df, val_df])

    def predict(self, test_df: pd.DataFrame, context_length: int | None = None) -> pd.DataFrame:
        if self._full_history is None:
            raise RuntimeError("Call fit() before predict().")
        # Combine all available history for context.
        all_data = pd.concat([self._full_history, test_df]).sort_index()
        all_data = all_data[~all_data.index.duplicated(keep="last")]
        y_full = all_data["actual_load"]

        # Group test forecast times by date to share decompositions.
        test_dates = sorted({ts.date() for ts in test_df.index})
        results: list[pd.DataFrame] = []
        for d in test_dates:
            day_rows = test_df[test_df.index.date == d]
            if len(day_rows) == 0:
                continue
            # Use the earliest forecast time of the day to define the
            # issuance window: earliest_tau - 24 = earliest issuance.
            earliest_tau = day_rows.index.min()
            issuance = earliest_tau - pd.Timedelta(hours=ISSUANCE_OFFSET)
            context = y_full.loc[:issuance]
            # Need at least 2 weekly cycles for MSTL to work reliably.
            if len(context) < 2 * max(PERIODS[:2]):  # 2*168 = 336
                continue

            in_ramadan = bool(day_rows["is_ramadan"].iloc[0])
            block = self._forecast_one_day(context, in_ramadan=in_ramadan)
            # Map the 24-step block to the day's forecast times (00-23 UTC).
            day_preds = pd.DataFrame({
                "y_true": day_rows["actual_load"].values,
                "y_pred": block[: len(day_rows)],
                "regime": day_rows["regime"].values,
            }, index=day_rows.index)
            results.append(day_preds)
        return pd.concat(results) if results else pd.DataFrame(
            columns=["y_true", "y_pred", "regime"]
        )

    def _forecast_one_day(self, context: pd.Series, in_ramadan: bool) -> np.ndarray:
        """Produce a 24-hour forecast from the given context."""
        # MSTL needs the series indexed without timezone for some statsmodels
        # versions; strip tz to be safe.
        ctx = context.copy()
        if ctx.index.tz is not None:
            ctx.index = ctx.index.tz_convert(None)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                mstl = MSTL(ctx, periods=PERIODS).fit()
            except Exception:
                # Fallback: drop yearly period if series too short.
                mstl = MSTL(ctx, periods=PERIODS[:2]).fit()

        trend = mstl.trend
        # MSTL.seasonal is a DataFrame with one column per period.
        seasonal = mstl.seasonal
        residual = mstl.resid

        # Ramadan-aware daily seasonal: replace the daily column with one
        # estimated from Ramadan-only hours when applicable.
        if self.variant == "hijri" and in_ramadan and "is_ramadan" in self._full_history.columns:
            daily_col_name = seasonal.columns[0]  # period 24
            # Compute Ramadan-only daily seasonal: detrended series, group by hour.
            hist = self._full_history.copy()
            if hist.index.tz is not None:
                hist.index = hist.index.tz_convert(None)
            ramadan_mask = hist["is_ramadan"] == 1
            ram_y = hist.loc[ramadan_mask, "actual_load"]
            if len(ram_y) >= 24 * 14:  # at least 2 weeks of Ramadan history
                # Subtract a slow trend (rolling 168h mean) before extracting hourly pattern.
                ram_smooth = ram_y.rolling(window=168, min_periods=24).mean()
                ram_detrended = ram_y - ram_smooth.fillna(ram_y.mean())
                hourly_pattern = ram_detrended.groupby(ram_detrended.index.hour).mean()
                # Replace the daily column at every position with the Ramadan pattern.
                hour_idx = seasonal.index.hour
                seasonal[daily_col_name] = hourly_pattern.reindex(hour_idx).values

        # Forecast each component over HORIZON steps:
        #   trend: last value, no drift (ETS(A,N,N) assumption)
        #   seasonal columns: periodic repeat
        #   residual: ETS(A,N,N)
        trend_fc = np.full(HORIZON, trend.iloc[-1])
        seasonal_fc = np.zeros(HORIZON)
        last_hour = ctx.index[-1].hour
        for period_idx, col in enumerate(seasonal.columns):
            period = PERIODS[period_idx]
            # Wrap around the seasonal cycle.
            seasonal_vals = seasonal[col].values
            for h in range(HORIZON):
                seasonal_fc[h] += seasonal_vals[-(period) + (h % period)]

        try:
            ets = ExponentialSmoothing(residual.dropna(), trend=None, seasonal=None).fit(disp=False)
            residual_fc = ets.forecast(HORIZON).values
        except Exception:
            residual_fc = np.zeros(HORIZON)

        block = trend_fc + seasonal_fc + residual_fc
        return np.asarray(block, dtype=float)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/models/test_mstl_ets.py -v`
Expected: 4 passed. (The MSTL+ETS test fit on synthetic data takes ~5 sec.)

If `test_mstl_ets_predict_returns_unified_schema` fails with "MSTL requires period <= n/2": the synthetic n=2000 is < 2×8766, so MSTL falls back to periods=(24, 168) — that's expected. The test passes when block forecast still produces 24 values.

- [ ] **Step 5: Commit**

```bash
git add src/models/classical/mstl_ets.py tests/models/test_mstl_ets.py
git commit -m "feat(models/classical): MSTL+ETS wrapper with Ramadan-aware seasonal"
```

---

### Task 1.2: Smoke-run MSTL+ETS on real test slice

- [ ] **Step 1: Sanity-test on a 7-day slice**

Run:
```bash
.venv/Scripts/python.exe -c "
import pandas as pd, time
from src.models.classical.mstl_ets import MSTLETSModel

df = pd.read_csv('data/processed/final_training_set_v2.csv', parse_dates=['timestamp']).set_index('timestamp')
df.index = df.index.tz_convert('UTC') if df.index.tz is not None else df.index.tz_localize('UTC')

train = df.loc['2018':'2022']
val = df.loc['2023']
test_slice = df.loc['2024-01-08':'2024-01-14']  # one week
print(f'train={len(train)} val={len(val)} test={len(test_slice)}')

m = MSTLETSModel(variant='nohijri')
m.fit(train, val, hijri=False, seed=0)
t0 = time.time()
out = m.predict(test_slice)
print(f'7-day predict: {len(out)} rows in {time.time()-t0:.1f}s')
print(f'y_pred range: [{out.y_pred.min():.1f}, {out.y_pred.max():.1f}]')
print(f'MAE: {(out.y_true - out.y_pred).abs().mean():.1f}')
"
```
Expected: 168 predictions in 30–120 sec (7 daily MSTL fits). MAE in 1000–3000 MW range (single-step MSTL+ETS isn't world-class but should be reasonable).

If MAE is wildly off (>10,000), inspect: y_pred range should be 15k–55k MW. Likely cause: trend forecast or seasonal indexing is off. Print the block forecast for the first day vs the actual.

- [ ] **Step 2: No commit yet — smoke is a check, not a deliverable.**

---

### Task 1.3: Add MSTL+ETS to `scripts/run_classical.py`

**Files:**
- Create: `scripts/run_classical.py`

- [ ] **Step 1: Implement the runner**

```python
"""Run a classical model on the v2 test set and save predictions to parquet.

Usage:
    .venv/Scripts/python.exe scripts/run_classical.py --model mstl_ets --variant nohijri
    .venv/Scripts/python.exe scripts/run_classical.py --model mstl_ets --variant hijri
    .venv/Scripts/python.exe scripts/run_classical.py --model sarimax --variant nohijri
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.evaluation.predictions_io import write_predictions


ROOT = Path(__file__).resolve().parents[1]
V2_CSV = ROOT / "data" / "processed" / "final_training_set_v2.csv"


MODEL_REGISTRY = {
    "mstl_ets": ("src.models.classical.mstl_ets", "MSTLETSModel"),
    "sarimax":  ("src.models.classical.sarimax",  "SARIMAXModel"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--variant", required=True,
                        choices=["nohijri", "hijri", "hijri_plusB"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"[1/5] Loading v2 dataset ...")
    df = pd.read_csv(V2_CSV, parse_dates=["timestamp"]).set_index("timestamp")
    df.index = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")

    # Drop early rows missing leak-free features.
    df = df.dropna(subset=["y_lag_336h", "y_roll168_mean"])

    train = df.loc["2018":"2022"]
    val = df.loc["2023"]
    test = df.loc["2024-01-01":"2025-03-31"]
    print(f"      train={len(train):,}  val={len(val):,}  test={len(test):,}")

    print(f"[2/5] Instantiating {args.model} variant={args.variant} ...")
    import importlib
    mod_path, cls_name = MODEL_REGISTRY[args.model]
    cls = getattr(importlib.import_module(mod_path), cls_name)
    model = cls(variant=args.variant)
    print(f"      name={model.name}")

    print(f"[3/5] Fitting ...")
    t0 = time.time()
    model.fit(train, val, hijri=(args.variant != "nohijri"), seed=args.seed)
    print(f"      fit done in {time.time()-t0:.1f}s")

    print(f"[4/5] Forecasting ...")
    t0 = time.time()
    preds = model.predict(test)
    print(f"      forecast done in {time.time()-t0:.1f}s  ({len(preds):,} rows)")

    print(f"[5/5] Writing parquet ...")
    path = write_predictions(
        preds, model=model.name, variant=args.variant,
        context_length=None, seed=args.seed,
    )
    print(f"      -> {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke the script with nohijri**

Run:
```bash
.venv/Scripts/python.exe scripts/run_classical.py --model mstl_ets --variant nohijri 2>&1 | tail -15
```
Expected: writes `data/predictions/mstl_ets__nohijri__seed0.parquet`. Wall-clock ~30-60 min (456 daily MSTL fits at ~5 sec each).

- [ ] **Step 3: Commit script + nohijri predictions**

```bash
git add scripts/run_classical.py data/predictions/mstl_ets__nohijri__seed0.parquet
git commit -m "feat(classical): MSTL+ETS nohijri run on v2 test set"
```

---

### Task 1.4: Run MSTL+ETS hijri variant

- [ ] **Step 1: Run**

Run:
```bash
.venv/Scripts/python.exe scripts/run_classical.py --model mstl_ets --variant hijri 2>&1 | tail -15
```
Expected: writes `data/predictions/mstl_ets__hijri__seed0.parquet`. Same wall-clock as nohijri.

- [ ] **Step 2: Quick comparison**

Run:
```bash
.venv/Scripts/python.exe -c "
import pandas as pd
from src.evaluation.predictions_io import read_predictions
from src.evaluation.regime_eval import evaluate_by_regime
TRAIN = pd.read_csv('data/processed/final_training_set_v2.csv', parse_dates=['timestamp']).set_index('timestamp').loc['2018':'2022']['actual_load'].values
for variant in ['nohijri', 'hijri']:
    p = read_predictions(model='mstl_ets', variant=variant, context_length=None, seed=0)
    print(f'\\nMSTL+ETS {variant}:')
    print(evaluate_by_regime(p['y_true'].values, p['y_pred'].values, regimes=p['regime'], y_train=TRAIN, period=168).to_string(index=False))
"
```
Expected: nohijri Ramadan MAE likely 1000-2500 MW; hijri Ramadan MAE should be smaller if the Ramadan-window re-estimation helps.

- [ ] **Step 3: Commit**

```bash
git add data/predictions/mstl_ets__hijri__seed0.parquet
git commit -m "feat(classical): MSTL+ETS hijri (Ramadan-window seasonal) run"
```

---

## Phase 2 — SARIMAX

### Task 2.1: SARIMAX wrapper (order selection + state-extension)

**Files:**
- Create: `src/models/classical/sarimax.py`
- Create: `tests/models/test_sarimax.py`

- [ ] **Step 1: Write the failing test**

`tests/models/test_sarimax.py`:
```python
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
        + 200 * (15 + 10 * np.sin(2 * np.pi * t / 24))   # weather coupling
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

    # Use a tiny fixed order to skip auto_arima in tests.
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
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/models/test_sarimax.py -v`
Expected: 4 failures with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/models/classical/sarimax.py`**

```python
"""SARIMAX classical baseline (proposal §4.1).

Order (p,d,q)(P,D,Q,24) selected once by pmdarima.auto_arima on
weekly-downsampled training data (full hourly is too slow). Parameters fit
on train+val with selected exogenous regressors, then the Kalman state is
extended day-by-day across the test window to produce 24h forecasts.

variant=nohijri:    weather-only exogenous
variant=hijri:      weather + Hijri (is_ramadan, day_of_ramadan, is_eid)
variant=hijri_plusB: weather + Hijri + interaction features (ramadan_x_heatwave,
                                       ramadan_x_temp_above_35)
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
        order: tuple[int, int, int] | None = None,
        seasonal_order: tuple[int, int, int, int] | None = None,
    ):
        self.variant = variant
        self.exog_features = _exog_for_variant(variant)
        # If order is None, auto_arima picks it during fit().
        self.order = order
        self.seasonal_order = seasonal_order
        self._fitted = None
        self._full_endog: pd.Series | None = None
        self._full_exog: pd.DataFrame | None = None

    def _select_order(self, train_df: pd.DataFrame) -> None:
        """Pick (p,d,q)(P,D,Q,24) once via auto_arima on weekly-downsampled data."""
        if self.order is not None and self.seasonal_order is not None:
            return
        from pmdarima import auto_arima

        weekly = train_df["actual_load"].resample("W").mean().dropna()
        if len(weekly) < 30:
            # Fallback: trivial order.
            self.order = (1, 1, 0)
            self.seasonal_order = (0, 0, 0, 0)
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = auto_arima(
                    weekly, seasonal=False, stepwise=True,
                    max_p=3, max_q=3, max_d=2,
                    suppress_warnings=True, error_action="ignore",
                )
                p, d, q = model.order
                # Manually set a daily seasonal component for the hourly refit
                # (auto_arima on weekly data has no daily info).
                self.order = (p, d, q)
                self.seasonal_order = (1, 0, 1, 24)
            except Exception:
                self.order = (1, 1, 0)
                self.seasonal_order = (1, 0, 1, 24)

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, hijri: bool, seed: int) -> None:
        self._select_order(train_df)
        history = pd.concat([train_df, val_df]).sort_index()
        history = history[~history.index.duplicated(keep="last")]
        endog = history["actual_load"]
        exog = history[self.exog_features]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sarimax = SARIMAX(
                endog, exog=exog,
                order=self.order, seasonal_order=self.seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            )
            self._fitted = sarimax.fit(disp=False, maxiter=50)
        self._full_endog = endog
        self._full_exog = exog

    def predict(self, test_df: pd.DataFrame, context_length: int | None = None) -> pd.DataFrame:
        if self._fitted is None:
            raise RuntimeError("Call fit() before predict().")

        all_endog = pd.concat([self._full_endog, test_df["actual_load"]]).sort_index()
        all_endog = all_endog[~all_endog.index.duplicated(keep="last")]
        all_exog = pd.concat([self._full_exog, test_df[self.exog_features]]).sort_index()
        all_exog = all_exog[~all_exog.index.duplicated(keep="last")]

        # Process by test day; for each issuance day, use Kalman-filter extension.
        test_dates = sorted({ts.date() for ts in test_df.index})
        results: list[pd.DataFrame] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for d in test_dates:
                day_rows = test_df[test_df.index.date == d]
                if len(day_rows) == 0:
                    continue
                earliest_tau = day_rows.index.min()
                issuance = earliest_tau - pd.Timedelta(hours=ISSUANCE_OFFSET)
                endog_to_issuance = all_endog.loc[:issuance]
                exog_to_issuance = all_exog.loc[:issuance]

                # extend the fitted model's state to issuance time with observed data.
                # `apply` uses fitted params but new endog/exog.
                extended = self._fitted.apply(
                    endog=endog_to_issuance, exog=exog_to_issuance,
                    refit=False,
                )

                horizon_exog = all_exog.loc[
                    issuance + pd.Timedelta(hours=1) : issuance + pd.Timedelta(hours=HORIZON)
                ]
                if len(horizon_exog) < HORIZON:
                    continue
                fc = extended.forecast(steps=HORIZON, exog=horizon_exog.iloc[:HORIZON])
                block = np.asarray(fc, dtype=float)

                day_preds = pd.DataFrame({
                    "y_true": day_rows["actual_load"].values,
                    "y_pred": block[: len(day_rows)],
                    "regime": day_rows["regime"].values,
                }, index=day_rows.index)
                results.append(day_preds)
        return pd.concat(results) if results else pd.DataFrame(
            columns=["y_true", "y_pred", "regime"]
        )
```

- [ ] **Step 4: Run tests, verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/models/test_sarimax.py -v`
Expected: 4 passed (synthetic fit takes ~10 sec with the trivial order).

If `test_sarimax_predict_returns_unified_schema` fails with convergence warnings — that's fine, the assertion only checks output schema.

- [ ] **Step 5: Commit**

```bash
git add src/models/classical/sarimax.py tests/models/test_sarimax.py
git commit -m "feat(models/classical): SARIMAX wrapper with auto_arima order + state-extend predict"
```

---

### Task 2.2: Smoke-run SARIMAX on real test slice

- [ ] **Step 1: Tiny smoke test**

Run:
```bash
.venv/Scripts/python.exe -c "
import pandas as pd, time
from src.models.classical.sarimax import SARIMAXModel

df = pd.read_csv('data/processed/final_training_set_v2.csv', parse_dates=['timestamp']).set_index('timestamp')
df.index = df.index.tz_convert('UTC') if df.index.tz is not None else df.index.tz_localize('UTC')
df = df.dropna(subset=['y_lag_336h', 'y_roll168_mean'])

train = df.loc['2018':'2022']
val = df.loc['2023']
test_slice = df.loc['2024-01-08':'2024-01-09']  # 2 days

m = SARIMAXModel(variant='nohijri', order=(1,1,0), seasonal_order=(1,0,1,24))
print(f'fitting on {len(train)+len(val):,} rows...')
t0 = time.time()
m.fit(train, val, hijri=False, seed=0)
print(f'fit done in {time.time()-t0:.1f}s')

print('forecasting 2 days...')
t0 = time.time()
out = m.predict(test_slice)
print(f'2-day predict: {len(out)} rows in {time.time()-t0:.1f}s')
print(f'y_pred range: [{out.y_pred.min():.1f}, {out.y_pred.max():.1f}]')
print(f'MAE: {(out.y_true - out.y_pred).abs().mean():.1f}')
"
```
Expected: SARIMAX fit takes ~3-10 min on 50k+ hourly rows with seasonal order. 2-day predict ~30 sec. MAE in 800-2500 MW range.

If fit hangs >30 min: order (1,1,0)(1,0,1,24) is too heavy. Drop seasonal order to (0,0,0,0) and accept lower forecast quality.

If predict produces NaN: check that `_full_exog` columns match `exog_features`. Likely cause: column ordering or missing column from test_df.

- [ ] **Step 2: No commit; smoke only.**

---

### Task 2.3: Full SARIMAX nohijri run

- [ ] **Step 1: Run**

Run:
```bash
.venv/Scripts/python.exe scripts/run_classical.py --model sarimax --variant nohijri 2>&1 | tail -15
```
Expected: fit takes 5–30 min depending on the auto_arima selection. Forecast takes 30–120 min (450 daily Kalman extensions × ~10-30 sec each). Total wall-clock 1-3 hours.

- [ ] **Step 2: Commit**

```bash
git add data/predictions/sarimax__nohijri__seed0.parquet
git commit -m "feat(classical): SARIMAX nohijri run on v2 test set"
```

---

### Task 2.4: SARIMAX hijri run

- [ ] **Step 1: Run**

Run:
```bash
.venv/Scripts/python.exe scripts/run_classical.py --model sarimax --variant hijri 2>&1 | tail -15
```
Expected: same wall-clock as nohijri.

- [ ] **Step 2: Commit**

```bash
git add data/predictions/sarimax__hijri__seed0.parquet
git commit -m "feat(classical): SARIMAX hijri run on v2 test set"
```

---

### Task 2.5: SARIMAX hijri_plusB run

- [ ] **Step 1: Run**

Run:
```bash
.venv/Scripts/python.exe scripts/run_classical.py --model sarimax --variant hijri_plusB 2>&1 | tail -15
```
Expected: same wall-clock. ABLATION_B columns are zero almost everywhere (Compound regime empty) so this should produce near-identical results to hijri.

- [ ] **Step 2: Commit**

```bash
git add data/predictions/sarimax__hijri_plusB__seed0.parquet
git commit -m "feat(classical): SARIMAX hijri_plusB run on v2 test set"
```

---

## Phase 3 — Classical Baseline Headline Doc

### Task 3.1: Build classical baseline comparison + DM tests

**Files:**
- Create: `docs/classical_baselines.md`

- [ ] **Step 1: Compute the table**

Run:
```bash
.venv/Scripts/python.exe -c "
import pandas as pd
from src.evaluation.predictions_io import read_predictions
from src.evaluation.regime_eval import evaluate_by_regime
from src.evaluation.dm_test import dm_test, holm_bonferroni
from src.evaluation.metrics import mae

TRAIN = pd.read_csv('data/processed/final_training_set_v2.csv', parse_dates=['timestamp']).set_index('timestamp').loc['2018':'2022']['actual_load'].values

CLASSICAL = [
    ('mstl_ets', 'nohijri'),
    ('mstl_ets', 'hijri'),
    ('sarimax',  'nohijri'),
    ('sarimax',  'hijri'),
    ('sarimax',  'hijri_plusB'),
]

print('=== Per-regime metrics ===')
for m, v in CLASSICAL:
    p = read_predictions(model=m, variant=v, context_length=None, seed=0)
    tab = evaluate_by_regime(p['y_true'].values, p['y_pred'].values, regimes=p['regime'], y_train=TRAIN, period=168)
    print(f'\\n[{m} {v}]  n={len(p)}')
    print(tab.to_string(index=False))

print('\\n=== Hijri ablation DM tests ===')
labels, pvals = [], []
for m in ['mstl_ets', 'sarimax']:
    nh = read_predictions(model=m, variant='nohijri', context_length=None, seed=0)
    hh = read_predictions(model=m, variant='hijri',   context_length=None, seed=0)
    shared = nh.index.intersection(hh.index)
    for regime in ['Normal', 'Ramadan', 'Heatwave']:
        rmask = (nh.loc[shared, 'regime'] == regime).values
        if rmask.sum() < 30: continue
        stat, p = dm_test(nh.loc[shared, 'y_true'].values[rmask],
                          nh.loc[shared, 'y_pred'].values[rmask],
                          hh.loc[shared, 'y_pred'].values[rmask], h=24)
        delta = mae(nh.loc[shared, 'y_true'].values[rmask], nh.loc[shared, 'y_pred'].values[rmask]) \
              - mae(hh.loc[shared, 'y_true'].values[rmask], hh.loc[shared, 'y_pred'].values[rmask])
        labels.append(f'{m}-{regime}'); pvals.append(p)
        print(f'  {m:<12} {regime:<10} stat={stat:+7.3f} p_raw={p:.4f} Delta(nh-h)={delta:+8.1f}')

p_adj = holm_bonferroni(pvals)
print('\\n=== Holm-Bonferroni adjusted p ===')
for lab, pr, ph in zip(labels, pvals, p_adj):
    sig = '***' if ph < 0.001 else '**' if ph < 0.01 else '*' if ph < 0.05 else ''
    print(f'  {lab:<25} p_raw={pr:.4f} p_holm={ph:.4f} {sig}')
"
```

Record the printed output for the next step.

- [ ] **Step 2: Write the doc**

Create `docs/classical_baselines.md`:
```markdown
# Classical Baselines (MSTL+ETS + SARIMAX)

Two classical baselines from proposal §4.1 on the v2 test set
(2024-01-01..2025-03-31, 10,944 hours).

## Setup

- **MSTL+ETS**: periods (24, 168, 8766); ETS(A,N,N) on residual; one
  decomposition per test day (24 hourly forecasts share the same fit).
  `hijri` variant re-estimates the daily seasonal component from
  Ramadan-only history when issuance falls in Ramadan.
- **SARIMAX**: order selected via `pmdarima.auto_arima` on weekly-downsampled
  training data; manual seasonal_order=(1,0,1,24). Parameters fit once on
  train+val; Kalman state extended per test day via `_fitted.apply()`.
  Exogenous regressors: weather (`temp_c`, `dewpoint_c`, `wind_speed`,
  `solar_rad`, `temp_sq`, `temp_above_35`) + (hijri) + (hijri_plusB).

## Per-regime MAE

(paste step-1 per-regime tables here)

## Hijri ablation A — DM tests

(paste step-1 DM table here, Holm-corrected)

## Findings

(write 3-5 sentences after seeing numbers; e.g., "MSTL+ETS hijri reduces
Ramadan MAE by X% (DM p_holm=Y, significant). SARIMAX hijri behavior is Z.")

## Comparison with the full benchmark

Cross-reference `docs/tsfm_zero_shot_baseline.md` for the headline cross-model
table. The classical baselines should fall between LightGBM (best on Normal
and Ramadan) and the small TSFMs (best on Heatwave with long context).

## Files

- `data/predictions/mstl_ets__{nohijri,hijri}__seed0.parquet`
- `data/predictions/sarimax__{nohijri,hijri,hijri_plusB}__seed0.parquet`
```

- [ ] **Step 3: Commit**

```bash
git add docs/classical_baselines.md
git commit -m "docs: classical baseline results (MSTL+ETS, SARIMAX) with DM tests"
```

---

### Task 3.2: Update cross-model headline doc

**Files:**
- Modify: `docs/tsfm_zero_shot_baseline.md`

- [ ] **Step 1: Rebuild best-per-model table with classical baselines**

Run:
```bash
.venv/Scripts/python.exe -c "
import pandas as pd
from src.evaluation.predictions_io import read_predictions
from src.evaluation.regime_eval import evaluate_by_regime
from src.evaluation.metrics import mae

TRAIN = pd.read_csv('data/processed/final_training_set_v2.csv', parse_dates=['timestamp']).set_index('timestamp').loc['2018':'2022']['actual_load'].values

# Best-per-model. Determine each classical's best variant by aggregate test MAE.
def best_variant(m, variants):
    best = None
    for v in variants:
        p = read_predictions(model=m, variant=v, context_length=None, seed=0)
        cur = mae(p.y_true, p.y_pred)
        if best is None or cur < best[1]:
            best = (v, cur)
    return best[0]

best_mstl = best_variant('mstl_ets', ['nohijri', 'hijri'])
best_sar  = best_variant('sarimax', ['nohijri', 'hijri', 'hijri_plusB'])

BEST = [
    ('LightGBM hijri seed=44',     'lgbm',              'hijri',   None, 44),
    ('Chronos-Bolt-Base L=720',    'chronos_bolt_base', 'nohijri', 720,  0),
    ('TimesFM 2.5-200M L=168',     'timesfm_2_5',       'nohijri', 168,  0),
    ('Moirai-1.1-R-Small L=336',   'moirai_1_1_small',  'nohijri', 336,  0),
    ('Time-MoE-200M L=720',        'time_moe_200m',     'nohijri', 720,  0),
    (f'MSTL+ETS {best_mstl}',      'mstl_ets',          best_mstl, None, 0),
    (f'SARIMAX {best_sar}',        'sarimax',           best_sar,  None, 0),
]

preds = [(lab, read_predictions(model=m, variant=v, context_length=L, seed=s)) for lab,m,v,L,s in BEST]
shared = preds[0][1].index
for _, p in preds[1:]:
    shared = shared.intersection(p.index)
print(f'shared rows: {len(shared)}')
print(f'\\nAggregate MAE:')
for lab, p in preds:
    p2 = p.loc[shared]
    print(f'  {lab:<32}  MAE={mae(p2.y_true,p2.y_pred):8.1f}')
print(f'\\nPer-regime MAE:')
print(f'{\"model\":<32}{\"Normal\":>10}{\"Ramadan\":>10}{\"Heatwave\":>10}')
for lab, p in preds:
    p2 = p.loc[shared]
    cells = [lab.ljust(32)]
    for regime in ['Normal', 'Ramadan', 'Heatwave']:
        mask = (p2.regime == regime).values
        cells.append(f'{mae(p2.y_true.values[mask], p2.y_pred.values[mask]):>10.1f}')
    print(''.join(cells))
"
```

- [ ] **Step 2: Update the relevant tables in `docs/tsfm_zero_shot_baseline.md`**

Open the existing doc and:
- Replace the aggregate metrics table with the 7-row version from step 1.
- Replace the per-regime table with the 7-row version.
- Update the "Where each model wins" table.
- Add a one-paragraph "Classical baselines verdict" section pointing to `docs/classical_baselines.md`.

- [ ] **Step 3: Commit**

```bash
git add docs/tsfm_zero_shot_baseline.md
git commit -m "docs: extend headline cross-model table with MSTL+ETS and SARIMAX"
```

---

## Phase 4 — Smoke Tests and Wrap-Up

### Task 4.1: Extend `tests/test_smoke_pipeline.py`

**Files:**
- Modify: `tests/test_smoke_pipeline.py`

- [ ] **Step 1: Append classical smoke checks**

Add to `tests/test_smoke_pipeline.py`:
```python
# Plan 4: Classical baselines (CPU-only, deterministic single seed).
CLASSICAL_RUNS = [
    ("mstl_ets", "nohijri"),
    ("mstl_ets", "hijri"),
    ("sarimax",  "nohijri"),
    ("sarimax",  "hijri"),
    ("sarimax",  "hijri_plusB"),
]


@pytest.mark.parametrize("model_name,variant", CLASSICAL_RUNS)
def test_classical_prediction_exists(model_name, variant):
    p = PRED_DIR / f"{model_name}__{variant}__seed0.parquet"
    assert p.exists(), f"Missing {p}. Run scripts/run_classical.py."
    df = pd.read_parquet(p)
    assert "y_true" in df.columns
    assert "y_pred" in df.columns
    assert df["y_pred"].notna().all()
    assert len(df) > 5000
```

- [ ] **Step 2: Run smoke**

Run: `.venv/Scripts/python.exe -m pytest tests/test_smoke_pipeline.py -v`
Expected: prior 35 + 5 new = 40 smoke tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_smoke_pipeline.py
git commit -m "test: classical baseline smoke checks (5 prediction parquets)"
```

---

### Task 4.2: Update README milestone tracker

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Flip Plan 4 to [x]**

Change:
```
- [ ] Plan 4: Classical baselines (MSTL+ETS, SARIMAX)
```
to:
```
- [x] Plan 4: Classical baselines (MSTL+ETS, SARIMAX)
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: mark Plan 4 milestone complete in README"
```

---

### Task 4.3: Full pytest green

- [ ] **Step 1: Run full suite**

Run: `.venv/Scripts/python.exe -m pytest -v`
Expected: ~106 prior + 4 MSTL + 4 SARIMAX + 5 smoke = ~119 passing.

- [ ] **Step 2: Commit fixups if any**

If any test failed, fix and commit. Otherwise skip.

---

## Self-Review Checklist

Before claiming Plan 4 complete:

1. **5 classical prediction parquets exist** (mstl_ets ×2, sarimax ×3).
2. **`docs/classical_baselines.md` has** all per-regime tables and DM significance results filled in.
3. **`docs/tsfm_zero_shot_baseline.md` headline tables** updated to include classical baselines.
4. **README Plan 4 = [x]**.
5. **Full pytest green**.

## Risks and Escalation

- **`pmdarima` Python 3.12 wheel**: If install fails, fallback is to manually set `(p,d,q)(P,D,Q,24) = (1,1,0)(1,0,1,24)` and skip auto_arima. The proposal calls for AICc selection but a reasonable default is acceptable with a footnote.
- **SARIMAX convergence**: with hourly data and a daily seasonal order, MLE optimization can fail to converge in 50 iterations. The wrapper has `enforce_stationarity=False, enforce_invertibility=False` to allow non-stationary fits; convergence warnings are suppressed. If predictions are all NaN, raise `maxiter` to 200.
- **SARIMAX runtime**: 450 daily Kalman extensions × ~10-30 sec/extension = 1-4 hours per variant. Run overnight if needed.
- **MSTL with 2018-2025 data**: periods (24, 168, 8766) require ≥17532 observations. Training set has 43,000+ — fine. If MSTL raises "series too short", the fallback drops the yearly period.
- **MSTL+ETS predictions far from LGBM scale**: if y_pred values are off by >5000 MW from the true range, suspect the seasonal forecast indexing. The wrapper uses `seasonal_vals[-(period) + (h % period)]` which assumes the last `period` values represent one full cycle; verify with a single-day inspection.
