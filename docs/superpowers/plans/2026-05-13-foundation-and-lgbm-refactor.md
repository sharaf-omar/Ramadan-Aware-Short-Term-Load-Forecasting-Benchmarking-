# Foundation, Evaluation Harness, and LightGBM Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the leakage-free t+24 data pipeline, the unified evaluation harness, and refactor the LightGBM baseline into the new `src/` package — producing the first end-to-end working slice on `final_training_set_v2.csv` with 5 seeds × 3 feature variants, per-regime metrics with bootstrap CIs, and predictions persisted to parquet.

**Architecture:** Three layered packages under `src/`: `features/` (feature builders), `models/` (Model protocol + LGBM), `evaluation/` (metrics, regime stratification, DM tests, bootstrap, parquet I/O). Test the package contract with pytest; run an end-to-end smoke via a thin notebook. This plan is **Plan 1 of ~6**; follow-on plans cover TSFMs, classical baselines, PatchTST, residual correction, statistical analysis, and reporting.

**Tech Stack:** pandas, NumPy, scikit-learn, LightGBM, Optuna, statsmodels (HAC), `arch` (block bootstrap), pyarrow (parquet), hijridate, pytest.

**Reference docs:**
- Spec: `docs/superpowers/specs/2026-05-13-tsfm-ablations-completion-design.md`
- Execution guide: `docs/tsfm_execution_guide.md`

---

## Phase 1 — Foundation: Leakage-Free Preprocessing

### Task 1.1: Project scaffolding

**Files:**
- Create: `src/__init__.py`
- Create: `src/features/__init__.py`
- Create: `src/models/__init__.py`
- Create: `src/models/classical/__init__.py`
- Create: `src/models/ml/__init__.py`
- Create: `src/models/dl/__init__.py`
- Create: `src/models/tsfm/__init__.py`
- Create: `src/evaluation/__init__.py`
- Create: `src/reporting/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/features/__init__.py`
- Create: `tests/evaluation/__init__.py`
- Create: `tests/models/__init__.py`
- Create: `pyproject.toml`
- Create: `requirements.txt`

- [ ] **Step 1: Create empty package files**

Create each `__init__.py` listed above with content `""` (empty string is fine, single newline).

- [ ] **Step 2: Write pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "ramadan-stlf"
version = "0.1.0"
description = "Ramadan-Aware Short-Term Load Forecasting Benchmark"
requires-python = ">=3.10"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = "-v --strict-markers"

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]
```

- [ ] **Step 3: Write requirements.txt**

```
pandas>=2.0,<3.0
numpy>=1.24,<2.0
scikit-learn>=1.3
lightgbm>=4.0
optuna>=3.5
statsmodels>=0.14
arch>=6.0
pyarrow>=14.0
hijridate>=2.4
matplotlib>=3.7
seaborn>=0.13
scipy>=1.11
xarray>=2023.10
netCDF4>=1.6
python-dotenv>=1.0
eptr2>=0.4
pytest>=7.4
pytest-cov>=4.1
```

- [ ] **Step 4: Verify install**

Run: `pip install -e .`
Run: `pip install -r requirements.txt`
Expected: both succeed without errors.

Run: `pytest --collect-only`
Expected: "no tests ran in 0.0s" (no tests yet, but pytest finds the configuration).

- [ ] **Step 5: Commit**

```bash
git add src tests pyproject.toml requirements.txt
git commit -m "chore: project scaffolding for src/ package + pytest config"
```

---

### Task 1.2: Hijri features module

**Files:**
- Create: `src/features/hijri.py`
- Create: `tests/features/test_hijri.py`

- [ ] **Step 1: Write the failing test**

`tests/features/test_hijri.py`:
```python
import pandas as pd
import pytest
from src.features.hijri import add_hijri_features


def test_hijri_features_returns_required_columns():
    df = pd.DataFrame(index=pd.date_range("2024-03-10", "2024-03-12", freq="H", tz="UTC"))
    out = add_hijri_features(df)
    assert "is_ramadan" in out.columns
    assert "day_of_ramadan" in out.columns
    assert "is_eid" in out.columns


def test_hijri_ramadan_2024_window():
    # Ramadan 1445 (2024): roughly Mar 11 - Apr 9 (Turkey local).
    df = pd.DataFrame(index=pd.date_range("2024-03-15 12:00", periods=1, freq="H", tz="UTC"))
    out = add_hijri_features(df)
    assert out["is_ramadan"].iloc[0] == 1
    assert out["day_of_ramadan"].iloc[0] >= 1
    assert out["day_of_ramadan"].iloc[0] <= 30


def test_hijri_non_ramadan_zero():
    # July is never Ramadan in the 2020s.
    df = pd.DataFrame(index=pd.date_range("2024-07-15", periods=1, freq="H", tz="UTC"))
    out = add_hijri_features(df)
    assert out["is_ramadan"].iloc[0] == 0
    assert out["day_of_ramadan"].iloc[0] == 0
    assert out["is_eid"].iloc[0] == 0


def test_hijri_dtype_int():
    df = pd.DataFrame(index=pd.date_range("2024-01-01", periods=10, freq="H", tz="UTC"))
    out = add_hijri_features(df)
    assert out["is_ramadan"].dtype.kind in ("i", "u")
    assert out["day_of_ramadan"].dtype.kind in ("i", "u")
    assert out["is_eid"].dtype.kind in ("i", "u")
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/features/test_hijri.py -v`
Expected: 4 failures with `ModuleNotFoundError: No module named 'src.features.hijri'`.

- [ ] **Step 3: Implement `src/features/hijri.py`**

```python
"""Hijri calendar features (is_ramadan, day_of_ramadan, is_eid).

Implementation note: hijridate's Gregorian→Hijri conversion uses calendar-day
granularity. Hijri days actually start at maghrib (sunset). Boundary hours may
be miscategorized by up to ~6h. This is acceptable per the proposal but should
be cited in the LGBM technical report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hijridate import Gregorian


def _hijri_tuple(ts: pd.Timestamp) -> tuple[int, int, int]:
    """Return (is_ramadan, is_eid, day_of_ramadan) for a single UTC timestamp.

    Computed against Turkey local calendar day (Europe/Istanbul) since the
    proposal grids the load data on Turkish national consumption.
    """
    local_ts = ts.tz_convert("Europe/Istanbul")
    h = Gregorian(local_ts.year, local_ts.month, local_ts.day).to_hijri()
    is_ramadan = int(h.month == 9)
    is_eid = int(h.month == 10 and h.day <= 3)
    day_of_ramadan = int(h.day) if h.month == 9 else 0
    return is_ramadan, is_eid, day_of_ramadan


def add_hijri_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_ramadan, is_eid, day_of_ramadan columns.

    Parameters
    ----------
    df : DataFrame with a UTC-aware DatetimeIndex.

    Returns
    -------
    A copy of df with three new int columns appended.
    """
    if df.index.tz is None:
        raise ValueError("DataFrame index must be timezone-aware (UTC).")

    out = df.copy()
    tuples = [_hijri_tuple(ts) for ts in out.index]
    arr = np.asarray(tuples, dtype=np.int8)
    out["is_ramadan"] = arr[:, 0]
    out["is_eid"] = arr[:, 1]
    out["day_of_ramadan"] = arr[:, 2]
    return out
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/features/test_hijri.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/features/hijri.py tests/features/test_hijri.py
git commit -m "feat(features): add hijri feature module with is_ramadan/is_eid/day_of_ramadan"
```

---

### Task 1.3: Calendar features module

**Files:**
- Create: `src/features/calendar.py`
- Create: `tests/features/test_calendar.py`

- [ ] **Step 1: Write failing test**

`tests/features/test_calendar.py`:
```python
import numpy as np
import pandas as pd
from src.features.calendar import add_calendar_features


def test_calendar_columns_present():
    df = pd.DataFrame(index=pd.date_range("2024-01-01", periods=48, freq="H", tz="UTC"))
    out = add_calendar_features(df)
    for col in [
        "hour", "day_of_week", "month",
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "month_sin", "month_cos",
        "is_weekend",
    ]:
        assert col in out.columns, f"missing column {col}"


def test_hour_sin_cos_periodic():
    df = pd.DataFrame(index=pd.date_range("2024-01-01", periods=24, freq="H", tz="UTC"))
    out = add_calendar_features(df)
    # sin(2π * 0/24) == sin(2π * 24/24)
    next_day = pd.DataFrame(index=pd.date_range("2024-01-02 00:00", periods=1, freq="H", tz="UTC"))
    out_next = add_calendar_features(next_day)
    assert np.isclose(out["hour_sin"].iloc[0], out_next["hour_sin"].iloc[0])


def test_weekend_flag():
    # 2024-01-06 is a Saturday, 2024-01-07 Sunday, 2024-01-08 Monday.
    df = pd.DataFrame(index=pd.to_datetime(
        ["2024-01-06 12:00", "2024-01-07 12:00", "2024-01-08 12:00"], utc=True
    ))
    out = add_calendar_features(df)
    assert out["is_weekend"].tolist() == [1, 1, 0]
```

- [ ] **Step 2: Run test, verify fail**

Run: `pytest tests/features/test_calendar.py -v`
Expected: 3 failures with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/features/calendar.py`**

```python
"""Calendar features: cyclical hour/dow/month + weekend flag."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day_of_week, month, sin/cos encodings, is_weekend."""
    if df.index.tz is None:
        raise ValueError("DataFrame index must be timezone-aware (UTC).")
    out = df.copy()
    out["hour"] = out.index.hour
    out["day_of_week"] = out.index.dayofweek
    out["month"] = out.index.month
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(np.int8)
    return out
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/features/test_calendar.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/features/calendar.py tests/features/test_calendar.py
git commit -m "feat(features): add calendar features (cyclical encodings + weekend)"
```

---

### Task 1.4: Weather nonlinear features

**Files:**
- Create: `src/features/weather_nonlinear.py`
- Create: `tests/features/test_weather_nonlinear.py`

- [ ] **Step 1: Write failing test**

`tests/features/test_weather_nonlinear.py`:
```python
import pandas as pd
from src.features.weather_nonlinear import add_weather_nonlinear


def test_weather_columns_present():
    df = pd.DataFrame({
        "temp_c": [25.0, 36.0, 40.0],
    }, index=pd.date_range("2024-01-01", periods=3, freq="H", tz="UTC"))
    out = add_weather_nonlinear(df)
    assert "temp_sq" in out.columns
    assert "temp_above_35" in out.columns


def test_temp_squared_correct():
    df = pd.DataFrame({"temp_c": [10.0, 20.0, -5.0]},
                      index=pd.date_range("2024-01-01", periods=3, freq="H", tz="UTC"))
    out = add_weather_nonlinear(df)
    assert out["temp_sq"].tolist() == [100.0, 400.0, 25.0]


def test_temp_above_35_clipped():
    df = pd.DataFrame({"temp_c": [25.0, 35.0, 40.0]},
                      index=pd.date_range("2024-01-01", periods=3, freq="H", tz="UTC"))
    out = add_weather_nonlinear(df)
    assert out["temp_above_35"].tolist() == [0.0, 0.0, 5.0]
```

- [ ] **Step 2: Run test, verify fail**

Run: `pytest tests/features/test_weather_nonlinear.py -v`
Expected: 3 failures with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/features/weather_nonlinear.py`**

```python
"""Nonlinear weather feature transforms."""
from __future__ import annotations

import pandas as pd


def add_weather_nonlinear(df: pd.DataFrame, temp_col: str = "temp_c") -> pd.DataFrame:
    """Add temp_sq and temp_above_35.

    Parameters
    ----------
    df : DataFrame with a temperature column (default name 'temp_c').
    temp_col : name of the temperature column.
    """
    if temp_col not in df.columns:
        raise KeyError(f"DataFrame missing required column {temp_col!r}")
    out = df.copy()
    out["temp_sq"] = out[temp_col] ** 2
    out["temp_above_35"] = (out[temp_col] - 35.0).clip(lower=0.0)
    return out
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/features/test_weather_nonlinear.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/features/weather_nonlinear.py tests/features/test_weather_nonlinear.py
git commit -m "feat(features): add nonlinear weather features (temp_sq, temp_above_35)"
```

---

### Task 1.5: Regimes module (4-regime labeler)

**Files:**
- Create: `src/features/regimes.py`
- Create: `tests/features/test_regimes.py`

- [ ] **Step 1: Write failing tests**

`tests/features/test_regimes.py`:
```python
import pandas as pd
import numpy as np
from src.features.regimes import label_regimes


def _build_df(temps_per_day: list[float], starts: str = "2024-06-01"):
    """Build hourly DF where temps_per_day[i] is the daily max for day i.
    Returns a DataFrame with hourly temp_c (constant within day) and is_ramadan=0."""
    rows = []
    for i, t in enumerate(temps_per_day):
        day = pd.Timestamp(starts, tz="UTC") + pd.Timedelta(days=i)
        for h in range(24):
            rows.append({"timestamp": day + pd.Timedelta(hours=h), "temp_c": t, "is_ramadan": 0})
    df = pd.DataFrame(rows).set_index("timestamp")
    return df


def test_normal_when_cool_and_no_ramadan():
    df = _build_df([20.0, 20.0, 20.0])
    out = label_regimes(df)
    assert (out == "Normal").all()


def test_heatwave_requires_3_consecutive_days_at_35c():
    # Only 2 hot days: should NOT be heatwave.
    df = _build_df([20.0, 36.0, 36.0, 20.0])
    out = label_regimes(df)
    assert (out == "Normal").all()


def test_heatwave_3_consecutive_days_at_35c():
    df = _build_df([36.0, 36.0, 36.0])
    out = label_regimes(df)
    assert (out == "Heatwave").all()


def test_heatwave_5_day_block():
    df = _build_df([20.0, 36.0, 36.0, 36.0, 36.0, 36.0, 20.0])
    out = label_regimes(df)
    assert out.iloc[0:24].eq("Normal").all()
    assert out.iloc[24:24*6].eq("Heatwave").all()
    assert out.iloc[24*6:].eq("Normal").all()


def test_ramadan_only():
    df = _build_df([20.0, 20.0, 20.0])
    df["is_ramadan"] = 1
    out = label_regimes(df)
    assert (out == "Ramadan").all()


def test_compound_regime():
    df = _build_df([36.0, 36.0, 36.0])
    df["is_ramadan"] = 1
    out = label_regimes(df)
    assert (out == "Compound").all()


def test_returns_series_aligned_to_index():
    df = _build_df([20.0])
    out = label_regimes(df)
    assert isinstance(out, pd.Series)
    assert (out.index == df.index).all()
```

- [ ] **Step 2: Run test, verify fail**

Run: `pytest tests/features/test_regimes.py -v`
Expected: 7 failures with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/features/regimes.py`**

```python
"""4-regime labeling per proposal Table 3.

Heatwave := daily T_max >= 35°C for >= 3 consecutive days.
Compound := Ramadan AND Heatwave.
Ramadan  := Ramadan AND NOT Heatwave.
Normal   := otherwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


HEATWAVE_TEMP_THRESHOLD = 35.0  # °C
HEATWAVE_MIN_RUN_DAYS = 3


def _compute_heatwave_days(daily_max: pd.Series) -> pd.Series:
    """Return a boolean series indexed by date: True iff day is part of a
    consecutive run of >= HEATWAVE_MIN_RUN_DAYS days at or above the temp
    threshold."""
    hot = daily_max >= HEATWAVE_TEMP_THRESHOLD
    # Run-length encoding via cumsum on transition points.
    run_id = (hot != hot.shift(fill_value=False)).cumsum()
    run_len = hot.groupby(run_id).transform("size")
    return hot & (run_len >= HEATWAVE_MIN_RUN_DAYS)


def label_regimes(
    df: pd.DataFrame,
    temp_col: str = "temp_c",
    ramadan_col: str = "is_ramadan",
) -> pd.Series:
    """Label each row with one of Normal/Ramadan/Heatwave/Compound.

    Parameters
    ----------
    df : DataFrame with UTC DatetimeIndex, a temperature column, and an
        is_ramadan column.
    temp_col : name of the hourly temperature column.
    ramadan_col : name of the binary Ramadan indicator column.

    Returns
    -------
    pd.Series of dtype 'object' aligned to df.index.
    """
    for col in (temp_col, ramadan_col):
        if col not in df.columns:
            raise KeyError(f"DataFrame missing required column {col!r}")

    daily_max = df.groupby(df.index.date)[temp_col].max()
    heatwave_day = _compute_heatwave_days(daily_max)
    hw_map = heatwave_day.to_dict()

    is_hw_row = pd.Series(
        [bool(hw_map.get(ts.date(), False)) for ts in df.index],
        index=df.index,
    )
    is_ram_row = df[ramadan_col].astype(bool)

    labels = np.where(
        is_ram_row & is_hw_row, "Compound",
        np.where(
            is_ram_row & ~is_hw_row, "Ramadan",
            np.where(
                ~is_ram_row & is_hw_row, "Heatwave",
                "Normal",
            ),
        ),
    )
    return pd.Series(labels, index=df.index, name="regime")
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/features/test_regimes.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/features/regimes.py tests/features/test_regimes.py
git commit -m "feat(features): 4-regime labeler (Normal/Ramadan/Heatwave/Compound) per proposal Table 3"
```

---

### Task 1.6: Rewrite `preprocess_epias.py` with leak-free t+24 features

**Files:**
- Modify: `src/data/preprocess_epias.py` (full rewrite)
- Create: `tests/data/__init__.py`
- Create: `tests/data/test_preprocess_epias.py`

- [ ] **Step 1: Write failing test**

`tests/data/__init__.py`: empty.

`tests/data/test_preprocess_epias.py`:
```python
import pandas as pd
import numpy as np
import pytest
from src.data.preprocess_epias import build_lag_rolling_features


def _make_load_series(n_hours: int = 500) -> pd.Series:
    """Synthetic hourly load series with deterministic values for testing."""
    idx = pd.date_range("2024-01-01", periods=n_hours, freq="H", tz="UTC")
    # Deterministic values: load = hour index
    return pd.Series(np.arange(n_hours, dtype=float), index=idx, name="actual_load")


def test_lag_24h_is_value_at_issuance():
    """For forecast time τ, y_lag_24h should be y at issuance t = τ-24."""
    y = _make_load_series(100)
    out = build_lag_rolling_features(y)
    # row τ=50 means forecast time idx 50, issuance idx 26 (50-24)
    # y_lag_24h should equal y[26] = 26
    assert out.loc[y.index[50], "y_lag_24h"] == 26.0


def test_lag_168h_is_value_one_week_before():
    y = _make_load_series(500)
    out = build_lag_rolling_features(y)
    # row τ=400 → y_lag_168h = y[400-168] = y[232] = 232
    assert out.loc[y.index[400], "y_lag_168h"] == 232.0


def test_rolling_mean_24h_excludes_post_issuance_hours():
    """The 24h rolling mean at τ must use [τ-47, τ-24] inclusive (24 values,
    ending at issuance). NO peek past issuance."""
    y = _make_load_series(500)
    out = build_lag_rolling_features(y)
    # row τ=100 → window covers y[53..76] inclusive (24 values).
    # Mean of arange(53,77) = (53+76)/2 = 64.5
    assert out.loc[y.index[100], "y_roll24_mean"] == pytest.approx(64.5)


def test_rolling_mean_168h_excludes_post_issuance_hours():
    """168h rolling mean at τ uses [τ-191, τ-24] inclusive (168 values)."""
    y = _make_load_series(500)
    out = build_lag_rolling_features(y)
    # row τ=300 → window y[109..276] inclusive (168 values).
    # Mean = (109+276)/2 = 192.5
    assert out.loc[y.index[300], "y_roll168_mean"] == pytest.approx(192.5)


def test_early_rows_are_nan():
    """Rows where the longest lag/rolling window doesn't fit must be NaN
    (i.e., τ < 336 + 24 - 1 = 359 for lag_336h)."""
    y = _make_load_series(500)
    out = build_lag_rolling_features(y)
    assert out["y_lag_336h"].iloc[358].is_integer() is False  # NaN check via isna below
    assert pd.isna(out["y_lag_336h"].iloc[358])
    assert not pd.isna(out["y_lag_336h"].iloc[360])
```

- [ ] **Step 2: Run test, verify fail**

Run: `pytest tests/data/test_preprocess_epias.py -v`
Expected: 5 failures (module exists but `build_lag_rolling_features` is missing).

- [ ] **Step 3: Rewrite `src/data/preprocess_epias.py`**

```python
"""EPIAS preprocessing → final_training_set_v2.csv with leak-free t+24 features.

Row convention: each output row is indexed by forecast time τ. Issuance time
is t = τ - 24. ALL lag and rolling features are built from y at indices ≤ t.

Run as: python -m src.data.preprocess_epias
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from eptr2 import EPTR2

from src.features.hijri import add_hijri_features


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DOTENV_PATH = ROOT_DIR / ".env"
OUTPUT_CSV = PROCESSED_DIR / "epias_processed_final.csv"


def _fetch_buffer_2017() -> pd.DataFrame:
    """Fetch Dec 2017 buffer needed for early-2018 lag features."""
    load_dotenv(DOTENV_PATH)
    eptr = EPTR2(use_dotenv=True, dotenv_path=str(DOTENV_PATH), recycle_tgt=True)
    try:
        buf = eptr.call(
            "rt-cons", start_date="2017-12-01", end_date="2017-12-31"
        ).rename(columns={"consumption": "actual_load"})
    except Exception as exc:
        print(f"[WARN] Could not fetch 2017 buffer: {exc}. Lags in early 2018 will be NaN.")
        buf = pd.DataFrame()
    return buf


def build_lag_rolling_features(y: pd.Series) -> pd.DataFrame:
    """Build leak-free lag and rolling features for a y_{t+24} forecast.

    Each row τ holds features computed from y at indices ≤ τ-24 only.
    Concretely:
        y_lag_24h     = y[τ-24]              # value at issuance
        y_lag_48h     = y[τ-48]
        y_lag_168h    = y[τ-168]
        y_lag_336h    = y[τ-336]
        y_roll24_mean = mean(y[τ-47..τ-24])  # 24-value window ending at issuance
        y_roll24_std  = std (y[τ-47..τ-24])
        y_roll168_mean= mean(y[τ-191..τ-24])
        y_roll168_std = std (y[τ-191..τ-24])

    Parameters
    ----------
    y : hourly load Series with a UTC-aware DatetimeIndex.

    Returns
    -------
    DataFrame indexed identically to y with the columns above.
    Early rows where the longest window doesn't fit are NaN.
    """
    out = pd.DataFrame(index=y.index)
    out["y_lag_24h"] = y.shift(24)
    out["y_lag_48h"] = y.shift(48)
    out["y_lag_168h"] = y.shift(168)
    out["y_lag_336h"] = y.shift(336)

    # Anchor the rolling base at the issuance time (τ-24), then roll backward.
    issuance = y.shift(24)
    out["y_roll24_mean"] = issuance.rolling(window=24).mean()
    out["y_roll24_std"] = issuance.rolling(window=24).std()
    out["y_roll168_mean"] = issuance.rolling(window=168).mean()
    out["y_roll168_std"] = issuance.rolling(window=168).std()
    return out


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    print("[1/4] Loading EPIAS load CSV ...")
    df_main = pd.read_csv(
        RAW_DIR / "electricity_consumption_2018_2025.csv"
    ).rename(columns={"consumption": "actual_load"})

    print("[2/4] Fetching 2017 buffer ...")
    df_buf = _fetch_buffer_2017()

    print("[3/4] Concatenating, aligning timestamps ...")
    df = pd.concat([df_buf, df_main], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["date"], utc=True)
    df = (
        df.drop(columns=[c for c in ("date", "time") if c in df.columns])
          .sort_values("timestamp")
          .set_index("timestamp")
    )
    df["actual_load"] = df["actual_load"].interpolate(method="linear")

    print("[4/4] Building leak-free lag/rolling features ...")
    feat = build_lag_rolling_features(df["actual_load"])
    df = df.join(feat)

    # Add Hijri features (computed at row timestamp = forecast time τ).
    df = add_hijri_features(df)

    # Slice off the 2017 buffer rows.
    df = df.loc[df.index >= "2018-01-01 00:00:00+00:00"].copy()

    df.to_csv(OUTPUT_CSV)
    print(f"[OK] wrote {OUTPUT_CSV} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/data/test_preprocess_epias.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/data/preprocess_epias.py tests/data/test_preprocess_epias.py
git commit -m "feat(data): leak-free t+24 lag/rolling features in preprocess_epias.py"
```

---

### Task 1.7: Rename typo download script

**Files:**
- Rename: `src/data/dounload_epias.py` → `src/data/download_epias.py`

- [ ] **Step 1: Rename**

Run: `git mv src/data/dounload_epias.py src/data/download_epias.py`

- [ ] **Step 2: Verify**

Run: `ls src/data/`
Expected: see `download_epias.py`, no `dounload_epias.py`.

- [ ] **Step 3: Commit**

```bash
git commit -m "chore: fix download_epias.py typo"
```

---

### Task 1.8: Build v2 dataset and meta sidecar

**Files:**
- Modify: `src/data/spatial_weights.py` (only the output filename + add meta sidecar)
- Create: `src/data/build_v2_dataset.py`
- Create: `tests/data/test_build_v2_dataset.py`

- [ ] **Step 1: Write failing test**

`tests/data/test_build_v2_dataset.py`:
```python
from pathlib import Path
import pandas as pd
import json
from src.data.build_v2_dataset import build_v2, V2_OUTPUT_CSV, V2_META_JSON


def test_v2_outputs_exist_after_build(tmp_path, monkeypatch):
    # Smoke: just verify the function is callable and outputs the expected paths.
    # Full integration test happens manually after running preprocessing.
    assert V2_OUTPUT_CSV.name == "final_training_set_v2.csv"
    assert V2_META_JSON.name == "final_training_set_v2.meta.json"


def test_v2_columns_after_build_exists():
    """Run only if v2 already built — verifies column schema."""
    if not V2_OUTPUT_CSV.exists():
        import pytest
        pytest.skip("v2 dataset not built yet; run build_v2() first")

    df = pd.read_csv(V2_OUTPUT_CSV, nrows=10)
    required = [
        "timestamp", "actual_load",
        "y_lag_24h", "y_lag_48h", "y_lag_168h", "y_lag_336h",
        "y_roll24_mean", "y_roll24_std",
        "y_roll168_mean", "y_roll168_std",
        "temp_c", "dewpoint_c", "wind_speed", "solar_rad",
        "temp_sq", "temp_above_35",
        "hour", "day_of_week", "month",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        "is_weekend",
        "is_ramadan", "day_of_ramadan", "is_eid",
        "ramadan_x_hour_sin", "ramadan_x_hour_cos", "ramadan_x_weekend",
        "ramadan_x_heatwave", "ramadan_x_temp_above_35", "heatwave_x_temp",
        "regime",
    ]
    for col in required:
        assert col in df.columns, f"v2 dataset missing required column: {col}"
```

- [ ] **Step 2: Run test, verify fail**

Run: `pytest tests/data/test_build_v2_dataset.py -v`
Expected: 2 failures with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/data/build_v2_dataset.py`**

```python
"""Build final_training_set_v2.csv: leak-free t+24 + all engineered features.

Inputs:
    data/processed/epias_processed_final.csv  (from preprocess_epias.py)
    data/processed/weather_proxy.csv          (from spatial_weights.py)

Outputs:
    data/processed/final_training_set_v2.csv
    data/processed/final_training_set_v2.meta.json
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.features.calendar import add_calendar_features
from src.features.weather_nonlinear import add_weather_nonlinear
from src.features.regimes import label_regimes


ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

EPIAS_CSV = PROCESSED_DIR / "epias_processed_final.csv"
WEATHER_CSV = PROCESSED_DIR / "weather_proxy.csv"
V2_OUTPUT_CSV = PROCESSED_DIR / "final_training_set_v2.csv"
V2_META_JSON = PROCESSED_DIR / "final_training_set_v2.meta.json"

TEST_START = "2024-01-01 00:00:00+00:00"
TEST_END = "2025-03-31 23:00:00+00:00"


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def build_v2() -> pd.DataFrame:
    """Build v2 dataset and write CSV + meta sidecar. Returns the DataFrame."""
    if not EPIAS_CSV.exists():
        raise FileNotFoundError(
            f"{EPIAS_CSV} missing — run `python -m src.data.preprocess_epias` first."
        )
    if not WEATHER_CSV.exists():
        raise FileNotFoundError(
            f"{WEATHER_CSV} missing — run `python -m src.data.spatial_weights` first."
        )

    epias = pd.read_csv(EPIAS_CSV, parse_dates=["timestamp"]).set_index("timestamp")
    epias.index = epias.index.tz_convert("UTC") if epias.index.tz is not None \
        else epias.index.tz_localize("UTC")

    weather = pd.read_csv(WEATHER_CSV, parse_dates=["timestamp"]).set_index("timestamp")
    weather.index = weather.index.tz_convert("UTC") if weather.index.tz is not None \
        else weather.index.tz_localize("UTC")

    df = epias.join(weather, how="left")
    df[["temp_c", "dewpoint_c", "wind_speed", "solar_rad"]] = \
        df[["temp_c", "dewpoint_c", "wind_speed", "solar_rad"]].interpolate(
            method="time", limit=3
        )

    df = add_calendar_features(df)
    df = add_weather_nonlinear(df)

    # Ramadan × calendar interactions (covariate features used by tree models).
    df["ramadan_x_hour_sin"] = df["is_ramadan"] * df["hour_sin"]
    df["ramadan_x_hour_cos"] = df["is_ramadan"] * df["hour_cos"]
    df["ramadan_x_weekend"] = df["is_ramadan"] * df["is_weekend"]

    # Ablation B interactions.
    df["heatwave_x_temp"] = 0  # placeholder filled after regime labeling
    df["ramadan_x_temp_above_35"] = df["is_ramadan"] * df["temp_above_35"]

    df["regime"] = label_regimes(df)
    is_hw_row = df["regime"].isin(["Heatwave", "Compound"]).astype(int)
    df["heatwave_x_temp"] = is_hw_row * df["temp_c"]
    df["ramadan_x_heatwave"] = df["is_ramadan"] * is_hw_row

    # Test-window NaN assertion.
    test_slice = df.loc[TEST_START:TEST_END]
    n_nan = test_slice["actual_load"].isna().sum()
    if n_nan > 0:
        last_valid = df["actual_load"].last_valid_index()
        print(
            f"[WARN] {n_nan} NaN actual_load rows in test window "
            f"({TEST_START} .. {TEST_END}). Last valid: {last_valid}"
        )

    df.to_csv(V2_OUTPUT_CSV)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "source_files": {
            EPIAS_CSV.name: _hash_file(EPIAS_CSV),
            WEATHER_CSV.name: _hash_file(WEATHER_CSV),
        },
        "row_count": int(len(df)),
        "date_range": [str(df.index[0]), str(df.index[-1])],
        "heatwave_threshold_c": 35.0,
        "heatwave_min_run_days": 3,
        "test_window_nan_actual_load": int(n_nan),
        "feature_columns": sorted(df.columns.tolist()),
    }
    V2_META_JSON.write_text(json.dumps(meta, indent=2))
    print(f"[OK] wrote {V2_OUTPUT_CSV} ({len(df):,} rows)")
    print(f"[OK] wrote {V2_META_JSON}")
    return df


if __name__ == "__main__":
    build_v2()
```

- [ ] **Step 4: Run tests, verify they pass (skip for missing file)**

Run: `pytest tests/data/test_build_v2_dataset.py -v`
Expected: `test_v2_outputs_exist_after_build` PASS, `test_v2_columns_after_build_exists` SKIP (until step 5 generates the file).

- [ ] **Step 5: Generate v2 dataset end-to-end**

Run: `python -m src.data.preprocess_epias`
Expected: writes `data/processed/epias_processed_final.csv`. Reports row count and date range.

Run: `python -m src.data.build_v2_dataset`
Expected: writes `data/processed/final_training_set_v2.csv` and `final_training_set_v2.meta.json`. Prints OK lines.

Run: `pytest tests/data/test_build_v2_dataset.py -v`
Expected: both tests PASS now that v2 exists.

- [ ] **Step 6: Spot-check v2 row count**

Run:
```bash
python -c "import pandas as pd; df = pd.read_csv('data/processed/final_training_set_v2.csv', parse_dates=['timestamp']); print(f'rows={len(df)}, range=[{df.timestamp.min()}, {df.timestamp.max()}]'); print('test window NaN actual_load:', df.set_index('timestamp').loc['2024-01-01':'2025-03-31', 'actual_load'].isna().sum())"
```
Expected: ~63,000 rows, date range covers 2018-01-01 to ~2025-03-31, NaN count in test window = 0 (or low; investigate if >100).

- [ ] **Step 7: Commit**

```bash
git add src/data/build_v2_dataset.py tests/data/test_build_v2_dataset.py data/processed/final_training_set_v2.csv data/processed/final_training_set_v2.meta.json
git commit -m "feat(data): build_v2_dataset.py with leak-free features and meta sidecar"
```

---

## Phase 2 — Evaluation Harness

### Task 2.1: Metrics module

**Files:**
- Create: `src/evaluation/metrics.py`
- Create: `tests/evaluation/test_metrics.py`

- [ ] **Step 1: Write failing test**

`tests/evaluation/test_metrics.py`:
```python
import numpy as np
import pytest
from src.evaluation.metrics import mae, rmse, mape, mase


def test_mae_simple():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.0, 4.0])
    # errors: 0.5, 0, 1.0 → mean = 0.5
    assert mae(y_true, y_pred) == pytest.approx(0.5)


def test_rmse_simple():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([2.0, 4.0])
    # sq errors: 1, 4 → mean 2.5 → sqrt ≈ 1.5811
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(2.5))


def test_mape_skips_zero_y():
    y_true = np.array([0.0, 100.0])
    y_pred = np.array([10.0, 110.0])
    # only the second observation counts: |100-110|/100 * 100 = 10
    assert mape(y_true, y_pred) == pytest.approx(10.0)


def test_mase_with_known_scale():
    y_train = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    y_true = np.array([3.0])
    y_pred = np.array([2.0])
    # seasonal-naive period=2 errors: |y[2]-y[0]|=0, |y[3]-y[1]|=0, ... all 0.
    # Edge case: zero scale → MASE returns inf. We assert that fallback.
    result = mase(y_true, y_pred, y_train, period=2)
    assert np.isinf(result)


def test_mase_typical():
    # Construct y_train where seasonal-naive errors avg 1.0.
    y_train = np.arange(10, dtype=float)  # [0,1,2,..,9]
    # period=1: |y[1]-y[0]|=1, |y[2]-y[1]|=1, ... mean=1.
    y_true = np.array([10.0, 11.0])
    y_pred = np.array([10.5, 11.5])
    # MAE = 0.5, scale = 1.0, MASE = 0.5.
    assert mase(y_true, y_pred, y_train, period=1) == pytest.approx(0.5)
```

- [ ] **Step 2: Run test, verify fail**

Run: `pytest tests/evaluation/test_metrics.py -v`
Expected: 5 failures with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/evaluation/metrics.py`**

```python
"""Forecast metrics: MAE, RMSE, MAPE, MASE (vs seasonal-naive lag-period)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error in %. Skips rows where y_true == 0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def mase(y_true, y_pred, y_train, period: int = 168) -> float:
    """Mean Absolute Scaled Error vs seasonal naive with lag = `period`."""
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= period:
        raise ValueError(f"y_train has {len(y_train)} obs, need > period={period}")
    scale = float(np.mean(np.abs(y_train[period:] - y_train[:-period])))
    if scale == 0:
        return float("inf")
    return float(mean_absolute_error(y_true, y_pred) / scale)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/evaluation/test_metrics.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/metrics.py tests/evaluation/test_metrics.py
git commit -m "feat(eval): MAE/RMSE/MAPE/MASE metric helpers"
```

---

### Task 2.2: Regime evaluation

**Files:**
- Create: `src/evaluation/regime_eval.py`
- Create: `tests/evaluation/test_regime_eval.py`

- [ ] **Step 1: Write failing test**

`tests/evaluation/test_regime_eval.py`:
```python
import numpy as np
import pandas as pd
import pytest
from src.evaluation.regime_eval import evaluate_by_regime


def test_evaluate_by_regime_returns_one_row_per_regime():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 210.0, 310.0, 410.0])
    regimes = pd.Series(["Normal", "Normal", "Ramadan", "Heatwave"])
    y_train = np.arange(2000, dtype=float)  # any series long enough for MASE
    out = evaluate_by_regime(y_true, y_pred, regimes, y_train, period=168)
    # All 4 proposal regimes appear, even when count is 0 (Compound).
    assert set(out["regime"]) == {"Normal", "Ramadan", "Heatwave", "Compound"}


def test_zero_count_regime_has_nan_metrics():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 210.0])
    regimes = pd.Series(["Normal", "Normal"])
    y_train = np.arange(2000, dtype=float)
    out = evaluate_by_regime(y_true, y_pred, regimes, y_train, period=168)
    compound_row = out[out["regime"] == "Compound"].iloc[0]
    assert compound_row["n"] == 0
    assert np.isnan(compound_row["mae"])


def test_normal_regime_mae_matches_expectation():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 220.0])
    regimes = pd.Series(["Normal", "Normal"])
    y_train = np.arange(2000, dtype=float)
    out = evaluate_by_regime(y_true, y_pred, regimes, y_train, period=168)
    normal_row = out[out["regime"] == "Normal"].iloc[0]
    # MAE = mean(|10|, |20|) = 15
    assert normal_row["mae"] == pytest.approx(15.0)
    assert normal_row["n"] == 2
```

- [ ] **Step 2: Run test, verify fail**

Run: `pytest tests/evaluation/test_regime_eval.py -v`
Expected: 3 failures with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/evaluation/regime_eval.py`**

```python
"""Per-regime metric stratification."""
from __future__ import annotations

import numpy as np
import pandas as pd
from .metrics import mae as mae_fn, rmse as rmse_fn, mape as mape_fn, mase as mase_fn


REGIMES = ("Normal", "Ramadan", "Heatwave", "Compound")


def evaluate_by_regime(
    y_true,
    y_pred,
    regimes: pd.Series,
    y_train,
    period: int = 168,
) -> pd.DataFrame:
    """Compute MAE, RMSE, MAPE, MASE per regime.

    Parameters
    ----------
    y_true, y_pred : array-like of equal length.
    regimes : pd.Series of regime labels aligned to y_true/y_pred (length-matched).
    y_train : training-split y values used to compute MASE seasonal-naive scale.
    period : seasonal-naive lag (default 168 = weekly hourly).

    Returns
    -------
    DataFrame with columns: regime, n, mae, rmse, mape, mase.
    One row per regime in REGIMES (zero-count regimes get NaN metrics).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    regimes_arr = np.asarray(regimes)

    rows = []
    for r in REGIMES:
        mask = regimes_arr == r
        n = int(mask.sum())
        if n == 0:
            rows.append(
                dict(regime=r, n=0, mae=np.nan, rmse=np.nan, mape=np.nan, mase=np.nan)
            )
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        rows.append(dict(
            regime=r,
            n=n,
            mae=mae_fn(yt, yp),
            rmse=rmse_fn(yt, yp),
            mape=mape_fn(yt, yp),
            mase=mase_fn(yt, yp, y_train, period=period),
        ))
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/evaluation/test_regime_eval.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/regime_eval.py tests/evaluation/test_regime_eval.py
git commit -m "feat(eval): per-regime metric stratification (4 regimes always present)"
```

---

### Task 2.3: DM test with Newey-West HAC + Holm-Bonferroni

**Files:**
- Create: `src/evaluation/dm_test.py`
- Create: `tests/evaluation/test_dm_test.py`

- [ ] **Step 1: Write failing test**

`tests/evaluation/test_dm_test.py`:
```python
import numpy as np
import pytest
from src.evaluation.dm_test import dm_test, holm_bonferroni


def test_dm_identical_predictions_pvalue_one():
    """If two models predict identically, DM should be ~0 and p ~ 1."""
    rng = np.random.default_rng(42)
    y_true = rng.normal(size=500)
    y_pred = y_true + rng.normal(scale=0.1, size=500)
    stat, p = dm_test(y_true, y_pred, y_pred, h=24)
    assert abs(stat) < 1e-9
    assert p == pytest.approx(1.0)


def test_dm_model_b_clearly_better_pvalue_low():
    """Model B is much closer to y_true than model A → DM stat strongly negative
    (loss_A > loss_B), p-value should be very small."""
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=2000)
    y_pred_a = y_true + rng.normal(scale=2.0, size=2000)   # noisy
    y_pred_b = y_true + rng.normal(scale=0.5, size=2000)   # accurate
    stat, p = dm_test(y_true, y_pred_a, y_pred_b, h=24)
    assert stat > 0   # loss A > loss B → d_t > 0 → positive DM statistic
    assert p < 0.01


def test_holm_bonferroni_basic():
    p_values = [0.001, 0.04, 0.03, 0.5]
    adjusted = holm_bonferroni(p_values)
    # Sorted ascending: 0.001, 0.03, 0.04, 0.5; multipliers 4, 3, 2, 1.
    # Adjusted: 0.004, 0.09, 0.08 → cumulative max → 0.09, 0.5 → 0.5
    # Mapped back to original order: 0.004, 0.09, 0.09, 0.5
    assert adjusted[0] == pytest.approx(0.004)
    assert adjusted[3] == pytest.approx(0.5)


def test_holm_bonferroni_monotone():
    """Holm correction preserves order of raw p-values."""
    p = [0.01, 0.02, 0.03, 0.04, 0.05]
    adj = holm_bonferroni(p)
    for i in range(len(adj) - 1):
        assert adj[i] <= adj[i + 1]
```

- [ ] **Step 2: Run test, verify fail**

Run: `pytest tests/evaluation/test_dm_test.py -v`
Expected: 4 failures with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/evaluation/dm_test.py`**

```python
"""Diebold-Mariano test with Newey-West HAC standard errors.

Also exposes Holm-Bonferroni multiple-comparison correction.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import stats
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.regression.linear_model import OLS


def dm_test(y_true, y_pred_a, y_pred_b, h: int = 24, loss: str = "mae") -> tuple[float, float]:
    """Diebold-Mariano statistic and two-sided p-value.

    Parameters
    ----------
    y_true : array of true values.
    y_pred_a, y_pred_b : predictions from two models.
    h : forecast horizon, used to set HAC truncation lag = h - 1.
    loss : 'mae' or 'mse'.

    Returns
    -------
    (dm_stat, p_value).
    DM > 0 means model B has lower loss (loss_A > loss_B).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_a = np.asarray(y_pred_a, dtype=float)
    y_pred_b = np.asarray(y_pred_b, dtype=float)

    if loss == "mae":
        e_a = np.abs(y_true - y_pred_a)
        e_b = np.abs(y_true - y_pred_b)
    elif loss == "mse":
        e_a = (y_true - y_pred_a) ** 2
        e_b = (y_true - y_pred_b) ** 2
    else:
        raise ValueError("loss must be 'mae' or 'mse'")

    d = e_a - e_b
    n = len(d)

    if np.allclose(d, 0):
        return 0.0, 1.0

    # HAC variance via regression of d on a constant; sandwich estimator
    # gives Newey-West-style HAC.
    X = np.ones((n, 1))
    model = OLS(d, X).fit()
    nlags = max(h - 1, 1)
    cov = cov_hac(model, nlags=nlags)
    var_dbar = float(cov[0, 0])

    dbar = float(np.mean(d))
    if var_dbar <= 0:
        return 0.0, 1.0
    dm_stat = dbar / np.sqrt(var_dbar)
    p = float(2 * (1 - stats.norm.cdf(abs(dm_stat))))
    return float(dm_stat), p


def holm_bonferroni(p_values: Iterable[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment.

    Returns adjusted p-values in the *original* input order.
    """
    p = np.asarray(list(p_values), dtype=float)
    n = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    # Multipliers: n, n-1, ..., 1
    multipliers = np.arange(n, 0, -1)
    adjusted_sorted = np.minimum.accumulate(
        # Take cumulative max so values are monotone non-decreasing.
        np.maximum.accumulate(p_sorted * multipliers)
    )
    # Clip to <= 1.
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    # Unsort back to original order.
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/evaluation/test_dm_test.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/dm_test.py tests/evaluation/test_dm_test.py
git commit -m "feat(eval): DM test (Newey-West HAC) and Holm-Bonferroni correction"
```

---

### Task 2.4: Block bootstrap CIs

**Files:**
- Create: `src/evaluation/bootstrap.py`
- Create: `tests/evaluation/test_bootstrap.py`

- [ ] **Step 1: Write failing test**

`tests/evaluation/test_bootstrap.py`:
```python
import numpy as np
import pytest
from src.evaluation.bootstrap import block_bootstrap_ci


def test_bootstrap_ci_returns_two_floats():
    rng = np.random.default_rng(42)
    errors = rng.normal(loc=10.0, scale=2.0, size=1000)
    lo, hi = block_bootstrap_ci(errors, block_size=24, n_resamples=200, seed=42)
    assert isinstance(lo, float)
    assert isinstance(hi, float)
    assert lo < hi


def test_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(0)
    errors = rng.normal(loc=5.0, scale=1.0, size=2000)
    lo, hi = block_bootstrap_ci(errors, block_size=24, n_resamples=500, seed=0)
    # The 95% CI of the bootstrap distribution of the mean should bracket 5.0.
    assert lo < 5.0 < hi


def test_bootstrap_ci_narrows_with_more_data():
    rng = np.random.default_rng(0)
    errors_small = rng.normal(size=200)
    errors_large = rng.normal(size=20000)
    lo_s, hi_s = block_bootstrap_ci(errors_small, 24, 200, seed=1)
    lo_l, hi_l = block_bootstrap_ci(errors_large, 24, 200, seed=1)
    assert (hi_l - lo_l) < (hi_s - lo_s)
```

- [ ] **Step 2: Run test, verify fail**

Run: `pytest tests/evaluation/test_bootstrap.py -v`
Expected: 3 failures with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/evaluation/bootstrap.py`**

```python
"""Stationary block bootstrap (Politis & Romano 1994) for autocorrelated series."""
from __future__ import annotations

import numpy as np


def _stationary_block_resample(x: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """One stationary-bootstrap resample of length len(x).

    Block lengths are drawn from Geometric(1/block_size) so the mean block
    length is block_size; starts are uniform on [0, n).
    """
    n = len(x)
    p = 1.0 / block_size
    out = np.empty(n, dtype=x.dtype)
    pos = 0
    while pos < n:
        start = int(rng.integers(0, n))
        # Geometric draw: number of trials until first success, with prob p.
        block_len = int(rng.geometric(p))
        for k in range(block_len):
            if pos >= n:
                break
            out[pos] = x[(start + k) % n]
            pos += 1
    return out


def block_bootstrap_ci(
    values: np.ndarray,
    block_size: int = 24,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
    statistic: callable = np.mean,
) -> tuple[float, float]:
    """Bootstrap (1-alpha) CI for the statistic of an autocorrelated series.

    Parameters
    ----------
    values : 1-D array (e.g., absolute errors).
    block_size : mean stationary-bootstrap block length (default 24 = 1 day for hourly data).
    n_resamples : number of bootstrap iterations.
    alpha : significance level (default 0.05 → 95% CI).
    seed : RNG seed for reproducibility.
    statistic : function applied to each resample (default mean).

    Returns
    -------
    (low, high) tuple defining the (1-alpha) percentile CI.
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    stats_ = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        resample = _stationary_block_resample(values, block_size, rng)
        stats_[i] = statistic(resample)
    lo = float(np.quantile(stats_, alpha / 2))
    hi = float(np.quantile(stats_, 1 - alpha / 2))
    return lo, hi
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/evaluation/test_bootstrap.py -v`
Expected: 3 passed (this test runs ~500 resamples; may take ~5 s).

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/bootstrap.py tests/evaluation/test_bootstrap.py
git commit -m "feat(eval): stationary block bootstrap CIs (Politis & Romano)"
```

---

### Task 2.5: Predictions parquet I/O

**Files:**
- Create: `src/evaluation/predictions_io.py`
- Create: `tests/evaluation/test_predictions_io.py`

- [ ] **Step 1: Write failing test**

`tests/evaluation/test_predictions_io.py`:
```python
import pandas as pd
import pytest
from src.evaluation.predictions_io import (
    write_predictions, read_predictions, predictions_path,
)


def _sample_predictions(n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="H", tz="UTC")
    return pd.DataFrame({
        "y_true": list(range(n)),
        "y_pred": [v + 0.5 for v in range(n)],
        "regime": ["Normal"] * n,
    }, index=idx).rename_axis("timestamp")


def test_write_and_read_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("RAMADAN_PRED_DIR", str(tmp_path))
    df = _sample_predictions()
    p = write_predictions(
        df, model="lgbm", variant="hijri", context_length=None, seed=42,
    )
    assert p.exists()
    out = read_predictions(model="lgbm", variant="hijri", context_length=None, seed=42)
    assert (out["y_pred"] == df["y_pred"]).all()


def test_predictions_path_format(tmp_path, monkeypatch):
    monkeypatch.setenv("RAMADAN_PRED_DIR", str(tmp_path))
    p = predictions_path(model="chronos_bolt_base", variant="nohijri", context_length=168, seed=42)
    assert p.name == "chronos_bolt_base__nohijri__L168__seed42.parquet"


def test_predictions_path_no_context_length(tmp_path, monkeypatch):
    monkeypatch.setenv("RAMADAN_PRED_DIR", str(tmp_path))
    p = predictions_path(model="lgbm", variant="hijri", context_length=None, seed=42)
    assert p.name == "lgbm__hijri__seed42.parquet"
```

- [ ] **Step 2: Run test, verify fail**

Run: `pytest tests/evaluation/test_predictions_io.py -v`
Expected: 3 failures with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/evaluation/predictions_io.py`**

```python
"""Predictions parquet I/O with a stable filename convention.

Schema:
    timestamp (UTC, idx) | y_true | y_pred | regime | y_block (optional, list[float, 24])

Convention:
    data/predictions/<model>__<variant>__L<ctx>__seed<seed>.parquet
    (L<ctx> omitted when context_length is None — i.e., tabular models)
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


_DEFAULT_PRED_DIR = Path(__file__).resolve().parents[2] / "data" / "predictions"


def _pred_dir() -> Path:
    """Allow override via env var (used by tests)."""
    p = Path(os.environ.get("RAMADAN_PRED_DIR", str(_DEFAULT_PRED_DIR)))
    p.mkdir(parents=True, exist_ok=True)
    return p


def predictions_path(
    *,
    model: str,
    variant: str,
    context_length: int | None,
    seed: int,
) -> Path:
    """Canonical filename for a (model, variant, ctx, seed) tuple."""
    parts = [model, variant]
    if context_length is not None:
        parts.append(f"L{context_length}")
    parts.append(f"seed{seed}")
    fname = "__".join(parts) + ".parquet"
    return _pred_dir() / fname


def write_predictions(
    df: pd.DataFrame,
    *,
    model: str,
    variant: str,
    context_length: int | None,
    seed: int,
) -> Path:
    """Persist predictions to parquet under the canonical path."""
    required = {"y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"predictions DataFrame missing columns: {missing}")
    path = predictions_path(
        model=model, variant=variant, context_length=context_length, seed=seed,
    )
    df.to_parquet(path)
    return path


def read_predictions(
    *,
    model: str,
    variant: str,
    context_length: int | None,
    seed: int,
) -> pd.DataFrame:
    """Load predictions from the canonical path."""
    path = predictions_path(
        model=model, variant=variant, context_length=context_length, seed=seed,
    )
    return pd.read_parquet(path)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/evaluation/test_predictions_io.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/predictions_io.py tests/evaluation/test_predictions_io.py
git commit -m "feat(eval): predictions parquet I/O with canonical naming"
```

---

## Phase 3 — Model Protocol and LightGBM Refactor

### Task 3.1: Model protocol

**Files:**
- Create: `src/models/base.py`
- Create: `tests/models/test_base.py`

- [ ] **Step 1: Write failing test**

`tests/models/test_base.py`:
```python
import pandas as pd
from src.models.base import Model


def test_model_protocol_has_expected_attrs():
    # Protocol classes can't be instantiated, but we can check attribute names.
    assert hasattr(Model, "fit")
    assert hasattr(Model, "predict")


def test_concrete_model_satisfies_protocol():
    """A simple class with the right shape should pass isinstance check."""
    class Toy:
        name = "toy"
        supports_dynamic_covariates = False
        needs_training = True

        def fit(self, train_df, val_df, hijri, seed):
            return None

        def predict(self, test_df, context_length=None):
            return pd.DataFrame({"y_true": [], "y_pred": []})

    instance = Toy()
    assert isinstance(instance, Model)
```

- [ ] **Step 2: Run test, verify fail**

Run: `pytest tests/models/test_base.py -v`
Expected: 2 failures with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/models/base.py`**

```python
"""Model protocol — all model wrappers in src/models/* implement this."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Model(Protocol):
    """Common interface for every forecasting model in this benchmark.

    Concrete attributes
    -------------------
    name : str
        Stable identifier used in predictions filenames (e.g., "lgbm",
        "chronos_bolt_base", "patchtst").
    supports_dynamic_covariates : bool
        True iff the model accepts hour-aligned covariates over context+horizon.
        Used by ablation orchestration to decide between in-band covariates
        and post-hoc residual correction.
    needs_training : bool
        True for fit-then-predict models (LGBM, classical, PatchTST);
        False for zero-shot TSFMs.
    """
    name: str
    supports_dynamic_covariates: bool
    needs_training: bool

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        hijri: bool,
        seed: int,
    ) -> None:
        ...

    def predict(
        self,
        test_df: pd.DataFrame,
        context_length: int | None = None,
    ) -> pd.DataFrame:
        """Returns DataFrame with index=τ (UTC) and at least columns
        {y_true, y_pred}. Optional column y_block holds full 24-step block
        for block-forecasters."""
        ...
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/models/test_base.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models/base.py tests/models/test_base.py
git commit -m "feat(models): Model protocol for unified fit/predict interface"
```

---

### Task 3.2: LightGBM feature-set definitions

**Files:**
- Create: `src/models/ml/__init__.py` (already created in Task 1.1 — verify)
- Create: `src/models/ml/lgbm.py` (skeleton + feature sets only this task)
- Create: `tests/models/test_lgbm_features.py`

- [ ] **Step 1: Write failing test**

`tests/models/test_lgbm_features.py`:
```python
from src.models.ml.lgbm import (
    BASE_FEATURES, HIJRI_FEATURES, ABLATION_B_FEATURES,
    feature_set_for_variant,
)


def test_base_features_no_hijri():
    for f in BASE_FEATURES:
        assert "ramadan" not in f.lower()
        assert "eid" not in f.lower()


def test_hijri_features_block_isolated():
    expected = {
        "is_ramadan", "day_of_ramadan", "is_eid",
        "ramadan_x_hour_sin", "ramadan_x_hour_cos", "ramadan_x_weekend",
    }
    assert set(HIJRI_FEATURES) == expected


def test_ablation_b_features_block_isolated():
    assert set(ABLATION_B_FEATURES) == {"ramadan_x_heatwave", "ramadan_x_temp_above_35"}


def test_feature_set_nohijri():
    fs = feature_set_for_variant("nohijri")
    assert set(fs) == set(BASE_FEATURES)


def test_feature_set_hijri():
    fs = feature_set_for_variant("hijri")
    assert set(fs) == set(BASE_FEATURES) | set(HIJRI_FEATURES)


def test_feature_set_hijri_plus_b():
    fs = feature_set_for_variant("hijri_plusB")
    assert set(fs) == set(BASE_FEATURES) | set(HIJRI_FEATURES) | set(ABLATION_B_FEATURES)


def test_feature_set_unknown_variant_raises():
    import pytest
    with pytest.raises(ValueError):
        feature_set_for_variant("nonsense")
```

- [ ] **Step 2: Run test, verify fail**

Run: `pytest tests/models/test_lgbm_features.py -v`
Expected: 7 failures with `ImportError`.

- [ ] **Step 3: Implement skeleton `src/models/ml/lgbm.py`**

```python
"""LightGBM forecasting model wrapper.

Feature sets per ablation variant:
    nohijri      = BASE_FEATURES
    hijri        = BASE_FEATURES + HIJRI_FEATURES
    hijri_plusB  = BASE_FEATURES + HIJRI_FEATURES + ABLATION_B_FEATURES
"""
from __future__ import annotations


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
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/models/test_lgbm_features.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models/ml/lgbm.py tests/models/test_lgbm_features.py
git commit -m "feat(models): LightGBM feature-set definitions per ablation variant"
```

---

### Task 3.3: LightGBM class — fit and predict

**Files:**
- Modify: `src/models/ml/lgbm.py` (append `LightGBMModel` class)
- Create: `tests/models/test_lgbm_model.py`

- [ ] **Step 1: Write failing test**

`tests/models/test_lgbm_model.py`:
```python
import numpy as np
import pandas as pd
import pytest
from src.models.ml.lgbm import LightGBMModel


def _make_synthetic_df(n: int = 600, with_hijri: bool = True) -> pd.DataFrame:
    """Synthetic training-shaped data with all v2 columns we need."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=n, freq="H", tz="UTC")
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
```

- [ ] **Step 2: Run test, verify fail**

Run: `pytest tests/models/test_lgbm_model.py -v`
Expected: 3 failures (`LightGBMModel` not defined).

- [ ] **Step 3: Append `LightGBMModel` class to `src/models/ml/lgbm.py`**

Add to the end of the file:
```python
from typing import Any

import lightgbm as lgb
import pandas as pd


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
    """LightGBM forecaster implementing the Model protocol.

    Attributes
    ----------
    name : "lgbm"
    supports_dynamic_covariates : True (covariates passed as columns)
    needs_training : True
    """
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
        # `hijri` flag is informational; variant is the source of truth for features.
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
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/models/test_lgbm_model.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models/ml/lgbm.py tests/models/test_lgbm_model.py
git commit -m "feat(models): LightGBMModel fit/predict implementing Model protocol"
```

---

### Task 3.4: LightGBM Optuna hyperparameter search

**Files:**
- Modify: `src/models/ml/lgbm.py` (append `tune_with_optuna`)
- Create: `tests/models/test_lgbm_optuna.py`

- [ ] **Step 1: Write failing test**

`tests/models/test_lgbm_optuna.py`:
```python
import numpy as np
import pandas as pd
from src.models.ml.lgbm import LightGBMModel, tune_with_optuna


def _make_synthetic_df(n: int = 600) -> pd.DataFrame:
    """Synthetic training-shaped data with all v2 columns we need.

    Duplicated from test_lgbm_model intentionally: cross-test imports are
    fragile because tests/ is not an installed package.
    """
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=n, freq="H", tz="UTC")
    base_load = 30000 + 5000 * np.sin(2 * np.pi * np.arange(n) / 24)
    df = pd.DataFrame({
        "actual_load": base_load + rng.normal(scale=500, size=n),
        "temp_c": 15 + 10 * np.sin(2 * np.pi * np.arange(n) / 24),
        "dewpoint_c": 5.0, "wind_speed": 3.0,
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
        "y_roll24_mean": base_load, "y_roll24_std": 500.0,
        "y_roll168_mean": base_load, "y_roll168_std": 500.0,
        "heatwave_x_temp": 0.0,
        "is_ramadan": (idx.month == 3).astype(int),
        "day_of_ramadan": 0, "is_eid": 0,
        "ramadan_x_hour_sin": 0.0, "ramadan_x_hour_cos": 0.0,
        "ramadan_x_weekend": 0,
        "ramadan_x_heatwave": 0, "ramadan_x_temp_above_35": 0.0,
        "regime": "Normal",
    }, index=idx)
    return df


def test_tune_with_optuna_runs_few_trials_and_returns_params():
    df = _make_synthetic_df(n=1000)
    train_df = df.iloc[:600]
    val_df = df.iloc[600:800]

    best_params = tune_with_optuna(
        train_df, val_df, variant="hijri", n_trials=3, seed=42,
    )
    expected_keys = {
        "learning_rate", "num_leaves", "max_depth", "min_child_samples",
        "feature_fraction", "bagging_fraction", "lambda_l1", "lambda_l2",
        "min_split_gain",
    }
    assert expected_keys.issubset(best_params.keys())


def test_tune_reproducible_same_seed():
    df = _make_synthetic_df(n=1000)
    train_df = df.iloc[:600]
    val_df = df.iloc[600:800]

    p1 = tune_with_optuna(train_df, val_df, variant="hijri", n_trials=3, seed=42)
    p2 = tune_with_optuna(train_df, val_df, variant="hijri", n_trials=3, seed=42)
    assert p1 == p2
```

- [ ] **Step 2: Run test, verify fail**

Run: `pytest tests/models/test_lgbm_optuna.py -v`
Expected: 2 failures (`tune_with_optuna` not defined).

- [ ] **Step 3: Append `tune_with_optuna` to `src/models/ml/lgbm.py`**

```python
import optuna
from sklearn.metrics import mean_absolute_error


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
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/models/test_lgbm_optuna.py -v`
Expected: 2 passed (each trial is small; ~10–20s total).

- [ ] **Step 5: Commit**

```bash
git add src/models/ml/lgbm.py tests/models/test_lgbm_optuna.py
git commit -m "feat(models): Optuna TPE hyperparameter search for LightGBM"
```

---

### Task 3.5: Thin LGBM runner notebook

**Files:**
- Create: `notebooks/02_lgbm.ipynb`

This task creates a new notebook that drives the LGBM training end-to-end on v2 data, all 3 variants × 5 seeds, writes predictions to parquet, generates the regime metric table.

- [ ] **Step 1: Create notebook scaffold**

Run this Python snippet (or use Jupyter UI) to build a minimal notebook:
```python
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell("# LightGBM Runner — v2 Pipeline\n\n"
                              "3 variants × 5 seeds. Saves predictions to `data/predictions/`."),
    nbf.v4.new_code_cell("""\
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from pathlib import Path

from src.models.ml.lgbm import LightGBMModel, tune_with_optuna
from src.evaluation.predictions_io import write_predictions
from src.evaluation.regime_eval import evaluate_by_regime
from src.evaluation.bootstrap import block_bootstrap_ci

ROOT = Path('.').resolve()
DATA = pd.read_csv(ROOT / 'data' / 'processed' / 'final_training_set_v2.csv',
                   parse_dates=['timestamp']).set_index('timestamp')
# Drop early rows where leak-free features are NaN
DATA = DATA.dropna(subset=['y_lag_336h', 'y_roll168_mean'])

TRAIN = DATA.loc['2018':'2022']
VAL   = DATA.loc['2023']
TEST  = DATA.loc['2024':'2025-03']
print(f'Train {len(TRAIN):,}  Val {len(VAL):,}  Test {len(TEST):,}')
"""),
    nbf.v4.new_code_cell("""\
# 1) Optuna tune ONCE on the hijri variant, seed 42.
best_params = tune_with_optuna(TRAIN, VAL, variant='hijri', n_trials=50, seed=42)
print('Best Optuna params:'); print(best_params)
"""),
    nbf.v4.new_code_cell("""\
# 2) Run 3 variants × 5 seeds with the tuned params.
VARIANTS = ['nohijri', 'hijri', 'hijri_plusB']
SEEDS = [42, 43, 44, 45, 46]
for variant in VARIANTS:
    for seed in SEEDS:
        model = LightGBMModel(variant=variant, n_estimators=3000, **best_params)
        model.fit(TRAIN, VAL, hijri=(variant != 'nohijri'), seed=seed)
        preds = model.predict(TEST)
        path = write_predictions(preds, model='lgbm', variant=variant, context_length=None, seed=seed)
        print(f'wrote {path.name} | rows={len(preds)}')
"""),
    nbf.v4.new_code_cell("""\
# 3) Regime metrics on median-seed predictions (seed 44 is the middle of 42..46).
from src.evaluation.predictions_io import read_predictions
results = []
for variant in VARIANTS:
    p = read_predictions(model='lgbm', variant=variant, context_length=None, seed=44)
    tab = evaluate_by_regime(
        p['y_true'].values, p['y_pred'].values,
        regimes=p['regime'], y_train=TRAIN['actual_load'].values, period=168,
    )
    tab.insert(0, 'variant', variant)
    results.append(tab)
import pandas as pd
print(pd.concat(results).to_string(index=False))
"""),
    nbf.v4.new_code_cell("""\
# 4) Bootstrap CIs on aggregate MAE per variant (across-regime).
for variant in VARIANTS:
    p = read_predictions(model='lgbm', variant=variant, context_length=None, seed=44)
    abs_err = (p['y_true'] - p['y_pred']).abs().values
    lo, hi = block_bootstrap_ci(abs_err, block_size=24, n_resamples=1000, seed=42)
    print(f'{variant:<14} MAE={abs_err.mean():.2f}  CI95=[{lo:.2f}, {hi:.2f}]')
"""),
]
nbf.write(nb, 'notebooks/02_lgbm.ipynb')
```

Save the snippet as `_build_nb_02.py` at the repo root, run it once:
```bash
python _build_nb_02.py
rm _build_nb_02.py
```
Expected: `notebooks/02_lgbm.ipynb` created.

- [ ] **Step 2: Execute notebook end-to-end**

Open `notebooks/02_lgbm.ipynb` in Jupyter (or run via `jupyter nbconvert --to notebook --execute notebooks/02_lgbm.ipynb`).

Expected output:
- Optuna prints 50 trial values, best Val MAE in 1000–1500 MW range.
- 15 parquet files in `data/predictions/` (3 variants × 5 seeds).
- Regime table printed showing 4 regimes × 3 variants, no NaN MAE/RMSE/MAPE/MASE in populated regimes.
- Bootstrap CIs printed for each variant with low/high straddling the point estimate.

- [ ] **Step 3: Spot-check predictions**

```bash
python -c "from pathlib import Path; print(sorted(p.name for p in Path('data/predictions').glob('lgbm__*.parquet')))"
```
Expected: 15 filenames listed in alphabetical order.

```bash
python -c "import pandas as pd; df = pd.read_parquet('data/predictions/lgbm__hijri__seed42.parquet'); print(df.head()); print('rows:', len(df))"
```
Expected: rows ≈ 10800, columns include y_true/y_pred/regime.

- [ ] **Step 4: Commit**

```bash
git add notebooks/02_lgbm.ipynb data/predictions/lgbm__*.parquet
git commit -m "feat(notebooks): 02_lgbm.ipynb runs 3 variants × 5 seeds on v2 data"
```

---

### Task 3.6: V1 vs V2 LGBM delta documentation

**Files:**
- Create: `docs/v1_v2_lgbm_delta.md`

This task records the metric shift caused by fixing the rolling-feature leakage. Reviewer-facing transparency.

- [ ] **Step 1: Extract V1 metrics from existing notebook**

Open `notebooks/lgbm_training.ipynb`. Find the "Test Set Model Comparison" table cell (around cell 20). Record the LGBM-Hijri-tuned row: MAE, RMSE, MAPE, MASE.

- [ ] **Step 2: Compute V2 metrics from `notebooks/02_lgbm.ipynb`**

Run:
```bash
python -c "
import pandas as pd
from src.evaluation.metrics import mae, rmse, mape, mase
p = pd.read_parquet('data/predictions/lgbm__hijri__seed44.parquet')
TRAIN = pd.read_csv('data/processed/final_training_set_v2.csv', parse_dates=['timestamp']).set_index('timestamp').loc['2018':'2022']['actual_load']
print(f'MAE  : {mae(p[\"y_true\"], p[\"y_pred\"]):.2f}')
print(f'RMSE : {rmse(p[\"y_true\"], p[\"y_pred\"]):.2f}')
print(f'MAPE : {mape(p[\"y_true\"], p[\"y_pred\"]):.4f}')
print(f'MASE : {mase(p[\"y_true\"], p[\"y_pred\"], TRAIN.values, period=168):.4f}')
"
```

- [ ] **Step 3: Write `docs/v1_v2_lgbm_delta.md`**

```markdown
# LightGBM v1 → v2 Data Quality Delta

Comparison of LightGBM-Hijri-Tuned test metrics before (v1) and after (v2)
fixing the rolling-feature leakage. v1 rolling features used
`y.shift(1).rolling(24)` which peeked 23 hours past issuance time;
v2 uses `y.shift(24).rolling(24)` which ends exactly at issuance.

Test set: 2024-01-01 .. 2025-03-31. Median seed (44) reported.

| Metric | v1 (leaky)    | v2 (clean)    | Δ (v2 − v1) |
|--------|---------------|---------------|-------------|
| MAE    | <FILL FROM STEP 1> | <FILL FROM STEP 2> | <CALC>      |
| RMSE   | <FILL FROM STEP 1> | <FILL FROM STEP 2> | <CALC>      |
| MAPE   | <FILL FROM STEP 1> | <FILL FROM STEP 2> | <CALC>      |
| MASE   | <FILL FROM STEP 1> | <FILL FROM STEP 2> | <CALC>      |

**Expected sign:** v2 metrics should be *worse* than v1 by a modest margin
(leakage inflated v1 accuracy). Magnitude expected to be small (~1–5%) because
the leakage only affected the rolling-mean window which is one feature among
many.

**If v2 is significantly better than v1:** investigate for an opposite bug;
the v2 features may not be properly aligned.

**If v2 is much worse (>15% drop):** the leakage was load-bearing; document
prominently in the final report and re-tune hyperparameters with `tune_with_optuna`
on the clean features in case the optimum has shifted.
```

Manually fill in the v1 and v2 numbers and compute the delta column.

- [ ] **Step 4: Commit**

```bash
git add docs/v1_v2_lgbm_delta.md
git commit -m "docs: record v1→v2 LGBM metric delta from leakage fix"
```

---

## Phase 4 — Smoke Test and Plan 1 Wrap-Up

### Task 4.1: End-to-end smoke pytest

**Files:**
- Create: `tests/test_smoke_pipeline.py`

- [ ] **Step 1: Write the smoke test**

`tests/test_smoke_pipeline.py`:
```python
"""End-to-end smoke: v2 data exists, LGBM ran on all variants × seeds,
predictions parquets exist, and basic sanity checks pass."""
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "data" / "predictions"
V2_CSV = ROOT / "data" / "processed" / "final_training_set_v2.csv"


def test_v2_dataset_exists():
    assert V2_CSV.exists(), "Run `python -m src.data.build_v2_dataset` first."


def test_v2_dataset_test_window_no_nan_actual_load():
    df = pd.read_csv(V2_CSV, parse_dates=["timestamp"]).set_index("timestamp")
    df.index = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
    test_slice = df.loc["2024-01-01":"2025-03-31"]
    assert test_slice["actual_load"].isna().sum() == 0


@pytest.mark.parametrize("variant", ["nohijri", "hijri", "hijri_plusB"])
@pytest.mark.parametrize("seed", [42, 43, 44, 45, 46])
def test_lgbm_prediction_exists(variant, seed):
    p = PRED_DIR / f"lgbm__{variant}__seed{seed}.parquet"
    assert p.exists(), f"Missing {p}. Re-run notebooks/02_lgbm.ipynb."
    df = pd.read_parquet(p)
    assert "y_true" in df.columns
    assert "y_pred" in df.columns
    assert df["y_pred"].notna().all()
    assert len(df) > 5000  # sanity: should be ~10,800 test hours
```

- [ ] **Step 2: Run smoke test**

Run: `pytest tests/test_smoke_pipeline.py -v`
Expected: 17 passed (2 dataset checks + 15 prediction-existence checks).

If any fail, fix the underlying issue (re-run preprocessing or notebook) — do not modify the test to make it pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_smoke_pipeline.py
git commit -m "test: end-to-end smoke checks for v2 dataset and LGBM predictions"
```

---

### Task 4.2: Full test suite green

- [ ] **Step 1: Run full pytest**

Run: `pytest -v`
Expected: all tests pass (count should be ~50+ across all modules).

If anything fails, fix it. Do not move to follow-on plans with a broken pipeline.

- [ ] **Step 2: Commit any fix-up changes**

If you needed to fix bugs found by full suite: commit them with a descriptive message.

If the suite was already green: no commit needed; proceed.

---

### Task 4.3: README and follow-on plan placeholder

**Files:**
- Create: `README.md`
- Modify: `docs/superpowers/plans/2026-05-13-foundation-and-lgbm-refactor.md` (this file — append "Status: Complete")

- [ ] **Step 1: Write `README.md`**

```markdown
# Ramadan-Aware Short-Term Load Forecasting Benchmark

Capstone project benchmarking time-series foundation models (TSFMs) against
classical and ML baselines on Turkish national electricity load, with
controlled ablations on Hijri-calendar regime shifts (Ramadan, Eid) and
extreme-heat regimes.

## Status (current milestone)

- [x] Plan 1: Foundation + evaluation harness + LightGBM refactor
- [ ] Plan 2: TSFM zero-shot evaluation (Chronos, TimesFM, Moirai, Time-MoE)
- [ ] Plan 3: Classical baselines (MSTL+ETS, SARIMAX)
- [ ] Plan 4: PatchTST
- [ ] Plan 5: Post-hoc residual correction + ablation orchestration
- [ ] Plan 6: Statistical analysis + deeper analysis + report artifacts

See `docs/superpowers/specs/2026-05-13-tsfm-ablations-completion-design.md` for
the full design and `docs/superpowers/plans/` for active plans.

## Reproduction

1. `pip install -e .` and `pip install -r requirements.txt`.
2. Copy `.env-example` to `.env` and fill EPIAS credentials.
3. `python -m src.data.preprocess_epias` → builds `data/processed/epias_processed_final.csv`.
4. `python -m src.data.spatial_weights` → builds `data/processed/weather_proxy.csv`.
5. `python -m src.data.build_v2_dataset` → builds `data/processed/final_training_set_v2.csv`.
6. Open `notebooks/02_lgbm.ipynb` and run all cells.
7. `pytest -v` to verify the harness.

For TSFM and PatchTST GPU work, see `docs/tsfm_execution_guide.md`.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with milestone tracking and reproduction steps"
```

- [ ] **Step 3: Announce completion**

Plan 1 is complete. Foundation, evaluation harness, and LightGBM refactor are
in place. Next plan: TSFM zero-shot evaluation (Plan 2). Request the
`writing-plans` skill to generate that plan when ready.

---

## Self-Review Checklist (engineer should not skip)

Before claiming Plan 1 complete:

1. **Every test passes:** `pytest -v` shows all green.
2. **V2 dataset has the right columns:** the meta JSON lists 30+ columns including all engineered features.
3. **Predictions parquets are 15 files** (3 variants × 5 seeds) for LGBM only.
4. **v1→v2 delta documented:** `docs/v1_v2_lgbm_delta.md` filled in.
5. **No code references `final_training_set_v1.csv`** anywhere in `src/` (verify with `grep -rn 'v1.csv' src/`).
6. **`download_epias.py` renamed** (no `dounload_epias.py` in `src/data/`).

If any item fails, address it before invoking writing-plans for Plan 2.
