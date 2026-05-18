# Plan 6 — Post-Hoc Residual Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a LightGBM residual head on each of 4 TSFMs (Chronos-L720, Time-MoE-L720, TimesFM-L168, Moirai-L336) and 2 residual feature sets (nohijri, hijri), producing 8 corrected-forecast parquets to test whether post-hoc residual correction beats the in-band covariate path that Plan 3 showed hurts TSFMs.

**Architecture:** Stage 1 adds `--window` flag to `scripts/run_tsfm.py` so it saves predictions for the full train+val+test window (4 extra parquets, no extra GPU since inference already runs on full df, just filtered out). Stage 2 introduces `src/models/residual/lgbm_residual.py` — a `LGBMResidualModel` class that takes a TSFM-prediction parquet plus the v2 feature dataframe, fits LightGBM on `(features) → (y_true - y_pred_TSFM)` over train+val, and emits a corrected prediction parquet for the test window. Stage 3 regenerates the statistical appendix with 20 models and writes a new results doc.

**Tech Stack:** Python 3.12, pandas 2.1.4, lightgbm (already pinned), existing TSFM wrappers, existing `src.evaluation.predictions_io`. No new dependencies.

**Spec:** [`docs/superpowers/specs/2026-05-14-residual-correction-design.md`](../specs/2026-05-14-residual-correction-design.md)

---

## File structure (locked in)

| File | Responsibility |
|------|----------------|
| `scripts/run_tsfm.py` | **Modify**: add `--window {test,all}` flag; default `test` keeps current behaviour, `all` saves the full preds_all parquet to `<model>__<variant>__L<ctx>__seed<s>__window_all.parquet` |
| `src/models/residual/__init__.py` | New: re-export `LGBMResidualModel` |
| `src/models/residual/lgbm_residual.py` | New: `LGBMResidualModel` class — fits LGBM on residual target, corrects test predictions, conforms to existing `Model` protocol shape |
| `scripts/run_residual.py` | New: CLI per `(tsfm_model, L, residual_variant)` triple — loads TSFM-all parquet, fits residual head, writes corrected test parquet |
| `tests/models/test_lgbm_residual.py` | New: 4 unit tests (model attributes, feature set per variant, fit smoke on synthetic, corrected output schema) |
| `tests/test_smoke_pipeline.py` | **Modify**: add 8 parquet existence checks for the corrected test outputs |
| `docs/residual_correction.md` | New: per-TSFM bare vs +residual-nohijri vs +residual-hijri MAE table (agg + 3 regimes), DM tests, runtime |
| `docs/statistical_appendix.md` | **Regenerate** with 20 models (12 existing + 8 corrected); existing builder works unchanged |
| `docs/tsfm_zero_shot_baseline.md` | **Modify**: add the best-of-residual row(s) to the headline tables |

`scripts/build_statistical_appendix.py` MODELS list needs updating with the 8 new entries — included in Task 5 below.

---

## Task 0: Branch setup

**Files:** none (git only)

- [ ] **Step 1: Create branch off the latest plan-5/7 work**

```bash
# Assumes Plan 5 grid has completed and Plan 5 Tasks 9-11 are merged.
# If not, branch off the current head which already has Plans 5+7 design + Plan 5/7 partial commits.
git checkout -b plan-6-residual
```

Expected: `Switched to a new branch 'plan-6-residual'`

- [ ] **Step 2: Verify TSFM test parquets present (the 4 we'll correct)**

```bash
ls data/predictions/{chronos_bolt_base__nohijri__L720,timesfm_2_5__nohijri__L168,moirai_1_1_small__nohijri__L336,time_moe_200m__nohijri__L720}__seed0.parquet
```

Expected: 4 files listed. (These exist from Plans 2+3.)

- [ ] **Step 3: Verify lightgbm importable**

```bash
.venv/Scripts/python.exe -c "import lightgbm; print(lightgbm.__version__)"
```

Expected: prints a version (4.x).

---

## Task 1: `--window` flag in `run_tsfm.py`

**Files:**
- Modify: `scripts/run_tsfm.py`

- [ ] **Step 1: Read the relevant section of run_tsfm.py**

The current per-L block is:

```python
preds_all = model.predict(df, context_length=L)
test_preds = preds_all.loc[test_window.index.intersection(preds_all.index)]
elapsed = time.time() - t0
print(f"      L={L} done in {elapsed:.1f}s  ({len(test_preds):,} test predictions; {len(preds_all):,} total)")
# ... later writes test_preds
```

We need to optionally save `preds_all` to a different parquet filename suffixed with `__window_all`.

- [ ] **Step 2: Add the flag to argparse**

In `scripts/run_tsfm.py`, find the argparse block and add:

```python
parser.add_argument(
    "--window", default="test", choices=["test", "all"],
    help="`test` (default) saves only the test window (2024-01-01..2025-03-31). "
         "`all` saves the full preds_all parquet (train+val+test) for use as "
         "Plan 6 residual-correction input.",
)
```

- [ ] **Step 3: Branch on the flag at write time**

Find the line that writes the parquet (uses `write_predictions(test_preds, ...)`). Just below the existing print, add:

```python
        if args.window == "all":
            # Save full preds_all (train+val+test) for Plan 6 residual training.
            from src.evaluation.predictions_io import predictions_path
            # Construct the all-window filename by appending __window_all suffix.
            from pathlib import Path as _P
            base = predictions_path(
                model=model.name, variant=args.variant,
                context_length=L, seed=args.seed,
            )
            all_path = _P(str(base).replace(".parquet", "__window_all.parquet"))
            preds_all.to_parquet(all_path)
            print(f"      -> {all_path}  ({len(preds_all):,} rows)")
```

Place this BEFORE the existing `write_predictions(test_preds, ...)` call (which always runs and writes the test parquet).

- [ ] **Step 4: Smoke-test the new flag with the smallest/fastest TSFM**

Chronos-Bolt-Base at L=336 takes ~28s on the test set:

```bash
.venv/Scripts/python.exe scripts/run_tsfm.py --model chronos --context-length 336 --window all
```

Expected output includes both:
- `-> .../chronos_bolt_base__nohijri__L336__seed0__window_all.parquet  (~59,000 rows)`
- `-> .../chronos_bolt_base__nohijri__L336__seed0.parquet  (10,944 rows)` (the existing test parquet, overwritten with identical content)

- [ ] **Step 5: Commit**

```bash
git add scripts/run_tsfm.py
git commit -m "feat(tsfm): --window flag to save full train+val+test predictions"
```

---

## Task 2: TSFM train+val+test parquets for all 4 TSFMs

**Files:** none (runtime executions)

GPU wall-clock estimate per L: same as the original test-only run since inference already covers the full df.

- [ ] **Step 1: Chronos-Bolt-Base at L=720**

```bash
.venv/Scripts/python.exe scripts/run_tsfm.py --model chronos --context-length 720 --window all
```

Expected: ~85 s; produces `chronos_bolt_base__nohijri__L720__seed0__window_all.parquet`.

- [ ] **Step 2: Moirai-1.1-small at L=336**

```bash
.venv/Scripts/python.exe scripts/run_tsfm.py --model moirai --context-length 336 --window all
```

Expected: ~50 s; produces `moirai_1_1_small__nohijri__L336__seed0__window_all.parquet`.

- [ ] **Step 3: TimesFM 2.5-200M at L=168**

```bash
.venv/Scripts/python.exe scripts/run_tsfm.py --model timesfm --context-length 168 --window all
```

Expected: ~3-4 min; produces `timesfm_2_5__nohijri__L168__seed0__window_all.parquet`.

- [ ] **Step 4: Time-MoE-200M at L=720**

```bash
.venv/Scripts/python.exe scripts/run_tsfm.py --model timemoe --context-length 720 --window all
```

Expected: ~30-40 min; produces `time_moe_200m__nohijri__L720__seed0__window_all.parquet`. This is the long-tail step; run unattended or in background.

- [ ] **Step 5: Verify all 4 train+val parquets exist with the expected row count**

```bash
.venv/Scripts/python.exe -c "
import pandas as pd
for f in [
    'chronos_bolt_base__nohijri__L720__seed0__window_all.parquet',
    'moirai_1_1_small__nohijri__L336__seed0__window_all.parquet',
    'timesfm_2_5__nohijri__L168__seed0__window_all.parquet',
    'time_moe_200m__nohijri__L720__seed0__window_all.parquet',
]:
    df = pd.read_parquet(f'data/predictions/{f}')
    print(f'{f}: {len(df):,} rows, span {df.index.min()} .. {df.index.max()}')
"
```

Expected: each parquet has 35-50k rows (depending on L burn-in) spanning from ~2018-02 to 2025-03-31.

- [ ] **Step 6: Commit**

```bash
git add data/predictions/*__window_all.parquet
git commit -m "feat(tsfm): full-window predictions for 4 TSFMs (Plan 6 residual training inputs)"
```

---

## Task 3: `LGBMResidualModel` class (unit-tested)

**Files:**
- Create: `src/models/residual/__init__.py`
- Create: `src/models/residual/lgbm_residual.py`
- Create: `tests/models/test_lgbm_residual.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/models/test_lgbm_residual.py`:

```python
import numpy as np
import pandas as pd
import pytest

from src.models.residual.lgbm_residual import LGBMResidualModel


def _synthetic_tsfm_preds_and_features(n=2000, drift_per_hr=0.1):
    """Synthetic TSFM-style predictions DataFrame + v2 feature DataFrame.

    TSFM is under-predicting during Ramadan windows; residual model
    should learn to correct upward when is_ramadan==1.
    """
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    t = np.arange(n)
    truth = 30000 + 5000 * np.sin(2 * np.pi * t / 24) + drift_per_hr * t
    # Ramadan window = middle 30 days
    ram_mask = (idx >= idx[n // 2]) & (idx < idx[n // 2 + 24 * 30])
    truth = truth + np.where(ram_mask, -3000.0, 0.0)  # Ramadan dip
    # TSFM predicts the baseline, missing the Ramadan effect.
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
        "day_of_ramadan": np.where(ram_mask, np.arange(n) % 30 + 1, 0),
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
    """Smoke test: residual head should reduce Ramadan-period error
    when given Hijri features."""
    tsfm_df, feat_df = _synthetic_tsfm_preds_and_features(n=2000)
    # Synthetic train/val/test split
    train_end = 1500
    val_end = 1800
    train_tsfm = tsfm_df.iloc[:train_end]
    train_feat = feat_df.iloc[:train_end]
    val_tsfm = tsfm_df.iloc[train_end:val_end]
    val_feat = feat_df.iloc[train_end:val_end]
    test_tsfm = tsfm_df.iloc[val_end:]
    test_feat = feat_df.iloc[val_end:]

    m = LGBMResidualModel(variant="hijri", n_estimators=200, learning_rate=0.05)
    m.fit_residual(train_tsfm, train_feat, val_tsfm, val_feat, seed=0)
    corrected = m.correct(test_tsfm, test_feat)
    assert {"y_true", "y_pred", "regime"} <= set(corrected.columns)
    assert corrected.index.equals(test_tsfm.index)
    bare_ramadan_mae = (test_tsfm[test_tsfm.regime == "Ramadan"].y_true
                        - test_tsfm[test_tsfm.regime == "Ramadan"].y_pred).abs().mean()
    corrected_ramadan_mae = (corrected[corrected.regime == "Ramadan"].y_true
                             - corrected[corrected.regime == "Ramadan"].y_pred).abs().mean()
    # Residual should reduce Ramadan MAE materially (we crafted a 3000 MW dip).
    assert corrected_ramadan_mae < bare_ramadan_mae * 0.7, (
        f"Residual did not reduce Ramadan MAE: bare={bare_ramadan_mae:.1f}, "
        f"corrected={corrected_ramadan_mae:.1f}"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/models/test_lgbm_residual.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.models.residual'`.

- [ ] **Step 3: Implement the package skeleton**

Create `src/models/residual/__init__.py`:

```python
"""Post-hoc residual correction wrappers for zero-shot TSFMs."""
from src.models.residual.lgbm_residual import LGBMResidualModel

__all__ = ["LGBMResidualModel"]
```

Create `src/models/residual/lgbm_residual.py`:

```python
"""LightGBM post-hoc residual head for a TSFM forecast.

Trained on (features) -> (y_true - y_pred_TSFM) over train+val; the
corrected forecast on test is y_pred_TSFM + y_residual_hat. See
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
    """Compute hour/day-of-week sin/cos + is_weekend if not already present."""
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
    """LightGBM residual head for a single (TSFM, residual-variant) combo."""
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
        """Align tsfm_df.index with feat_df.index, compute residual target."""
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
        X_va, y_va = self._to_xy(val_tsfm, val_feat)
        dtrain = lgb.Dataset(X_tr, label=y_tr.values)
        dval = lgb.Dataset(X_va, label=y_va.values, reference=dtrain)
        params = {
            "objective": "regression_l1",   # MAE-style loss matches our eval
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
        """Apply the trained residual head; return corrected DataFrame
        with y_pred = y_pred_TSFM + y_residual_hat."""
        if self._booster is None:
            raise RuntimeError("Call fit_residual() before correct().")
        X, _ = self._to_xy(test_tsfm, test_feat)
        residual_hat = self._booster.predict(X)
        common = X.index
        corrected = test_tsfm.loc[common].copy()
        corrected["y_pred"] = corrected["y_pred"].values + residual_hat
        return corrected
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/models/test_lgbm_residual.py -v
```

Expected: 4 tests pass (~5-10 s wall clock; the smoke test trains a small LGBM).

- [ ] **Step 5: Commit**

```bash
git add src/models/residual/ tests/models/test_lgbm_residual.py
git commit -m "feat(residual): LGBMResidualModel with fit_residual()/correct() API"
```

---

## Task 4: `scripts/run_residual.py` CLI

**Files:**
- Create: `scripts/run_residual.py`

- [ ] **Step 1: Implement the CLI**

Create `scripts/run_residual.py`:

```python
"""Fit a LightGBM residual head on a TSFM's full-window forecast and
write the corrected test parquet.

Usage:
    .venv/Scripts/python.exe scripts/run_residual.py \\
        --tsfm-parquet chronos_bolt_base__nohijri__L720__seed0__window_all.parquet \\
        --tsfm-name chronos_bolt_base --context-length 720 \\
        --variant hijri --seed 0
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.evaluation.predictions_io import write_predictions, predictions_path
from src.models.residual import LGBMResidualModel


ROOT = Path(__file__).resolve().parents[1]
V2_CSV = ROOT / "data" / "processed" / "final_training_set_v2.csv"
PRED_DIR = ROOT / "data" / "predictions"


def _load_v2() -> pd.DataFrame:
    df = pd.read_csv(V2_CSV, parse_dates=["timestamp"]).set_index("timestamp")
    df.index = (df.index.tz_convert("UTC")
                if df.index.tz is not None else df.index.tz_localize("UTC"))
    df = df.dropna(subset=["y_lag_336h", "y_roll168_mean"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsfm-parquet", required=True,
                        help="Filename in data/predictions/ ending in __window_all.parquet")
    parser.add_argument("--tsfm-name", required=True,
                        help="Used to construct the output filename, e.g. chronos_bolt_base")
    parser.add_argument("--context-length", type=int, required=True,
                        help="Context length used by the source TSFM; for output filename.")
    parser.add_argument("--variant", required=True, choices=["nohijri", "hijri"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("[1/5] Loading TSFM full-window predictions ...")
    tsfm_path = PRED_DIR / args.tsfm_parquet
    if not tsfm_path.exists():
        raise FileNotFoundError(
            f"{tsfm_path} not found. Run scripts/run_tsfm.py --window all first."
        )
    tsfm_all = pd.read_parquet(tsfm_path)
    print(f"      {len(tsfm_all):,} predicted rows, span {tsfm_all.index.min()} .. {tsfm_all.index.max()}")

    print("[2/5] Loading v2 dataset features ...")
    v2 = _load_v2()
    print(f"      {len(v2):,} v2 rows")

    print("[3/5] Splitting into train/val/test ...")
    train_end = pd.Timestamp("2022-12-31 23:00", tz="UTC")
    val_end = pd.Timestamp("2023-12-31 23:00", tz="UTC")
    train_tsfm = tsfm_all.loc[:train_end]
    val_tsfm = tsfm_all.loc[train_end + pd.Timedelta(hours=1):val_end]
    test_tsfm = tsfm_all.loc[val_end + pd.Timedelta(hours=1):]
    print(f"      train={len(train_tsfm):,}  val={len(val_tsfm):,}  test={len(test_tsfm):,}")

    print(f"[4/5] Fitting LGBMResidualModel (variant={args.variant}) ...")
    m = LGBMResidualModel(variant=args.variant)
    t0 = time.time()
    m.fit_residual(train_tsfm, v2, val_tsfm, v2, seed=args.seed)
    print(f"      fit done in {time.time()-t0:.1f}s")

    print("[5/5] Applying residual to test, writing corrected parquet ...")
    corrected = m.correct(test_tsfm, v2)
    print(f"      {len(corrected):,} corrected rows")

    # Output naming: <tsfm>__L<ctx>__residual_<variant>__seed<s>.parquet
    out_path = predictions_path(
        model=f"{args.tsfm_name}__residual",
        variant=args.variant,
        context_length=args.context_length,
        seed=args.seed,
    )
    corrected.to_parquet(out_path)
    print(f"      -> {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test on the cheap TSFM (chronos L=336 from Task 1 Step 4)**

```bash
.venv/Scripts/python.exe scripts/run_residual.py \
    --tsfm-parquet chronos_bolt_base__nohijri__L336__seed0__window_all.parquet \
    --tsfm-name chronos_bolt_base --context-length 336 \
    --variant hijri --seed 0
```

Expected: prints 5 stages, completes in ~30 s, writes
`chronos_bolt_base__residual__hijri__L336__seed0.parquet` (~11k rows).

- [ ] **Step 3: Quick numeric sanity check**

```bash
.venv/Scripts/python.exe -c "
import pandas as pd
bare = pd.read_parquet('data/predictions/chronos_bolt_base__nohijri__L336__seed0.parquet')
corr = pd.read_parquet('data/predictions/chronos_bolt_base__residual__hijri__L336__seed0.parquet')
print(f'bare  Chronos-L336 agg MAE: {(bare.y_true-bare.y_pred).abs().mean():.1f}')
print(f'+residual-hijri agg MAE:    {(corr.y_true-corr.y_pred).abs().mean():.1f}')
for r in [\"Normal\",\"Ramadan\",\"Heatwave\"]:
    bb = bare[bare.regime==r]
    cc = corr[corr.regime==r]
    print(f'  {r:9s}  bare={(bb.y_true-bb.y_pred).abs().mean():7.1f}  +res={(cc.y_true-cc.y_pred).abs().mean():7.1f}')
"
```

Expected: corrected agg MAE is similar to or better than bare (this is the smoke; the real test is in Task 5 with Chronos-L720). Ramadan should improve materially if Hijri features help.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_residual.py
git commit -m "feat(residual): scripts/run_residual.py CLI per (TSFM, variant) combo"
```

---

## Task 5: Run all 8 residual combos

**Files:** none (runtime executions)

8 combos total. CPU only, ~30 s per fit + 5 s per correct = ~5 min total.

- [ ] **Step 1: Chronos-Bolt-Base L=720 × 2 variants**

```bash
.venv/Scripts/python.exe scripts/run_residual.py \
    --tsfm-parquet chronos_bolt_base__nohijri__L720__seed0__window_all.parquet \
    --tsfm-name chronos_bolt_base --context-length 720 \
    --variant nohijri --seed 0

.venv/Scripts/python.exe scripts/run_residual.py \
    --tsfm-parquet chronos_bolt_base__nohijri__L720__seed0__window_all.parquet \
    --tsfm-name chronos_bolt_base --context-length 720 \
    --variant hijri --seed 0
```

Expected: 2 parquets produced.

- [ ] **Step 2: Moirai-1.1-small L=336 × 2 variants**

```bash
.venv/Scripts/python.exe scripts/run_residual.py \
    --tsfm-parquet moirai_1_1_small__nohijri__L336__seed0__window_all.parquet \
    --tsfm-name moirai_1_1_small --context-length 336 \
    --variant nohijri --seed 0

.venv/Scripts/python.exe scripts/run_residual.py \
    --tsfm-parquet moirai_1_1_small__nohijri__L336__seed0__window_all.parquet \
    --tsfm-name moirai_1_1_small --context-length 336 \
    --variant hijri --seed 0
```

- [ ] **Step 3: TimesFM 2.5-200M L=168 × 2 variants**

```bash
.venv/Scripts/python.exe scripts/run_residual.py \
    --tsfm-parquet timesfm_2_5__nohijri__L168__seed0__window_all.parquet \
    --tsfm-name timesfm_2_5 --context-length 168 \
    --variant nohijri --seed 0

.venv/Scripts/python.exe scripts/run_residual.py \
    --tsfm-parquet timesfm_2_5__nohijri__L168__seed0__window_all.parquet \
    --tsfm-name timesfm_2_5 --context-length 168 \
    --variant hijri --seed 0
```

- [ ] **Step 4: Time-MoE-200M L=720 × 2 variants**

```bash
.venv/Scripts/python.exe scripts/run_residual.py \
    --tsfm-parquet time_moe_200m__nohijri__L720__seed0__window_all.parquet \
    --tsfm-name time_moe_200m --context-length 720 \
    --variant nohijri --seed 0

.venv/Scripts/python.exe scripts/run_residual.py \
    --tsfm-parquet time_moe_200m__nohijri__L720__seed0__window_all.parquet \
    --tsfm-name time_moe_200m --context-length 720 \
    --variant hijri --seed 0
```

- [ ] **Step 5: Verify 8 corrected parquets exist**

```bash
ls -la data/predictions/*__residual__*.parquet
```

Expected: 8 files listed (4 TSFMs × 2 variants).

- [ ] **Step 6: Quick aggregate-MAE table**

```bash
.venv/Scripts/python.exe -c "
import pandas as pd
rows = []
for tsfm, L in [('chronos_bolt_base', 720), ('moirai_1_1_small', 336),
                ('timesfm_2_5', 168), ('time_moe_200m', 720)]:
    bare = pd.read_parquet(f'data/predictions/{tsfm}__nohijri__L{L}__seed0.parquet')
    bare_mae = (bare.y_true - bare.y_pred).abs().mean()
    rows.append((f'{tsfm}-L{L}-bare', bare_mae))
    for v in ['nohijri', 'hijri']:
        d = pd.read_parquet(f'data/predictions/{tsfm}__residual__{v}__L{L}__seed0.parquet')
        rows.append((f'{tsfm}-L{L}-residual-{v}', (d.y_true-d.y_pred).abs().mean()))
for n, m in rows:
    print(f'  {n:50s} agg MAE = {m:.1f}')
"
```

Expected: a 12-row table showing bare and +residual MAEs per TSFM.

- [ ] **Step 7: Commit**

```bash
git add data/predictions/*__residual__*.parquet
git commit -m "feat(residual): 8 corrected TSFM parquets (4 TSFMs x 2 residual variants)"
```

---

## Task 6: Regenerate the statistical appendix with 20 models

**Files:**
- Modify: `scripts/build_statistical_appendix.py`

- [ ] **Step 1: Extend the MODELS constant**

Open `scripts/build_statistical_appendix.py` and find the `MODELS:` list. Append these 8 entries at the bottom (preserving the existing 12):

```python
    # Plan 6: post-hoc LGBM residual heads on the 4 TSFMs, 2 variants each.
    ("chronos-bolt-L720+res-nh",  "chronos_bolt_base__residual__nohijri__L720__seed0.parquet"),
    ("chronos-bolt-L720+res-h",   "chronos_bolt_base__residual__hijri__L720__seed0.parquet"),
    ("moirai-L336+res-nh",        "moirai_1_1_small__residual__nohijri__L336__seed0.parquet"),
    ("moirai-L336+res-h",         "moirai_1_1_small__residual__hijri__L336__seed0.parquet"),
    ("timesfm-L168+res-nh",       "timesfm_2_5__residual__nohijri__L168__seed0.parquet"),
    ("timesfm-L168+res-h",        "timesfm_2_5__residual__hijri__L168__seed0.parquet"),
    ("time-moe-L720+res-nh",      "time_moe_200m__residual__nohijri__L720__seed0.parquet"),
    ("time-moe-L720+res-h",       "time_moe_200m__residual__hijri__L720__seed0.parquet"),
```

- [ ] **Step 2: Re-run the builder**

```bash
.venv/Scripts/python.exe scripts/build_statistical_appendix.py
```

Expected: runs in ~5-10 min (now 20 models × 4 regimes = 80 CI cells + 20·19/2 = 190 DM pairs per regime).

Outputs overwritten:
- `docs/statistical_appendix.md`
- `data/statistical_appendix/{ci_table,dm_aggregate,dm_Normal,dm_Ramadan,dm_Heatwave}.csv`

- [ ] **Step 3: Sanity-check the appendix has 20-model rows**

```bash
.venv/Scripts/python.exe -c "
import pandas as pd
ci = pd.read_csv('data/statistical_appendix/ci_table.csv')
agg = ci[ci.regime == 'aggregate'].sort_values('mae')
print(agg[['model', 'mae', 'ci_lo', 'ci_hi']].to_string(index=False))
"
```

Expected: 20 rows, sorted by aggregate MAE. The four +residual-hijri rows should appear competitive with the bare TSFMs (this is the headline test).

- [ ] **Step 4: Commit**

```bash
git add scripts/build_statistical_appendix.py docs/statistical_appendix.md data/statistical_appendix/
git commit -m "stats: regenerate statistical appendix with 20 models (incl. 8 residual)"
```

---

## Task 7: Write `docs/residual_correction.md`

**Files:**
- Create: `docs/residual_correction.md`

- [ ] **Step 1: Compute the headline numbers**

```bash
.venv/Scripts/python.exe -c "
import pandas as pd, numpy as np
from src.evaluation.dm_test import dm_test

rows = []
for tsfm, L in [('chronos_bolt_base', 720), ('moirai_1_1_small', 336),
                ('timesfm_2_5', 168), ('time_moe_200m', 720)]:
    bare = pd.read_parquet(f'data/predictions/{tsfm}__nohijri__L{L}__seed0.parquet')
    res_nh = pd.read_parquet(f'data/predictions/{tsfm}__residual__nohijri__L{L}__seed0.parquet')
    res_h  = pd.read_parquet(f'data/predictions/{tsfm}__residual__hijri__L{L}__seed0.parquet')

    def m(df, regime=None):
        sub = df if regime is None else df[df.regime == regime]
        if len(sub) == 0: return float('nan')
        return (sub.y_true - sub.y_pred).abs().mean()

    def dm(a, b, regime=None):
        a2 = a if regime is None else a[a.regime == regime]
        b2 = b if regime is None else b[b.regime == regime]
        common = a2.index.intersection(b2.index)
        if len(common) < 5: return float('nan'), float('nan')
        return dm_test(a2.loc[common].y_true.values,
                       a2.loc[common].y_pred.values,
                       b2.loc[common].y_pred.values,
                       h=24, loss='mae')

    print(f'\\n=== {tsfm}-L{L} ===')
    for regime in [None, 'Normal', 'Ramadan', 'Heatwave']:
        rlbl = regime or 'aggregate'
        print(f'  {rlbl:9s}  bare={m(bare,regime):7.1f}  +res-nh={m(res_nh,regime):7.1f}  +res-h={m(res_h,regime):7.1f}')
    s, p = dm(bare, res_h, 'Ramadan')
    print(f'  DM bare vs +res-h on Ramadan: stat={s:.2f} p={p:.2e}')
    s, p = dm(bare, res_h)
    print(f'  DM bare vs +res-h on aggregate: stat={s:.2f} p={p:.2e}')
"
```

Save the printed output — paste relevant numbers into the doc template below.

- [ ] **Step 2: Write the doc**

Create `docs/residual_correction.md` (replace `_FROM_STEP_1_` placeholders with the actual numbers printed above):

````markdown
# Plan 6 — Post-Hoc LGBM Residual Correction Results

For each of the 4 TSFMs, a small LightGBM head was fit on
`(features) → (y_true − y_pred_TSFM)` over train (2018-2022) + val
(2023, early-stop), then applied to the test window (2024-01-01 to
2025-03-31). The corrected forecast is `y_pred_TSFM + y_residual_hat`.

See [`docs/superpowers/specs/2026-05-14-residual-correction-design.md`](superpowers/specs/2026-05-14-residual-correction-design.md).

## Setup

- 4 TSFMs at their best L from Plan 3: Chronos-L720, Time-MoE-L720,
  TimesFM-L168, Moirai-L336.
- 2 residual variants: `nohijri` (weather + calendar + lag features
  only) and `hijri` (above + `is_ramadan`, `day_of_ramadan`, `is_eid`).
- LightGBM: `objective="regression_l1"`, lr=0.05, 1000 estimators with
  early stop patience 30 on val MAE.
- Test n = 10,944 (intersection with the rest of the headline cohort).

## Per-TSFM result table

| TSFM | Variant | Agg MAE | Normal | Ramadan | Heatwave |
|---|---|---|---|---|---|
| chronos-bolt-L720 | bare        | _FROM_STEP_1_ | _ | _ | _ |
| chronos-bolt-L720 | +residual-nh | _FROM_STEP_1_ | _ | _ | _ |
| chronos-bolt-L720 | +residual-h | _FROM_STEP_1_ | _ | _ | _ |
| moirai-L336       | bare        | _FROM_STEP_1_ | _ | _ | _ |
| moirai-L336       | +residual-nh | _FROM_STEP_1_ | _ | _ | _ |
| moirai-L336       | +residual-h | _FROM_STEP_1_ | _ | _ | _ |
| timesfm-L168      | bare        | _FROM_STEP_1_ | _ | _ | _ |
| timesfm-L168      | +residual-nh | _FROM_STEP_1_ | _ | _ | _ |
| timesfm-L168      | +residual-h | _FROM_STEP_1_ | _ | _ | _ |
| time-moe-L720     | bare        | _FROM_STEP_1_ | _ | _ | _ |
| time-moe-L720     | +residual-nh | _FROM_STEP_1_ | _ | _ | _ |
| time-moe-L720     | +residual-h | _FROM_STEP_1_ | _ | _ | _ |

## DM tests: bare vs +residual-hijri (Ramadan only, HAC h=24, MAE loss)

| TSFM | DM stat | Raw p | Verdict |
|---|---|---|---|
| chronos-bolt-L720 | _FROM_STEP_1_ | _ | _ |
| moirai-L336       | _FROM_STEP_1_ | _ | _ |
| timesfm-L168      | _FROM_STEP_1_ | _ | _ |
| time-moe-L720     | _FROM_STEP_1_ | _ | _ |

## Headline finding

_Fill in based on the table: "X of 4 TSFMs gained significantly on
Ramadan from post-hoc residual correction. The best post-hoc combo is
... at MAE Y, vs the bare TSFM Z." Compare to LightGBM-hijri's 800.0_

## Comparison to Plan 3 in-band covariate path

Plan 3 found that injecting Hijri features through the HuggingFace
covariate API *hurt* both TimesFM and Moirai (DM p<0.001). For the
two covariate-capable TSFMs:

| Model | bare MAE (Ramadan) | +HF covariate (Ramadan) | +LGBM residual (Ramadan) |
|---|---|---|---|
| timesfm-2.5-L168  | _from `tsfm_hijri_covariates.md`_ | _ditto_ | _from Step 1_ |
| moirai-1.1-L336   | _from `tsfm_hijri_covariates.md`_ | _ditto_ | _from Step 1_ |

_If post-hoc residual beats in-band covariate: that's the principled
finding the proposal predicted. Document._

## Runtime

| Stage | Wall-clock |
|---|---|
| Stage 1 (4 TSFM full-window inferences) | ~40 min GPU |
| Stage 2 (8 LGBM residual fits) | ~5 min CPU |
| Stage 3 (rebuild statistical appendix) | ~10 min CPU |

## Files

- `data/predictions/<tsfm>__residual__{nohijri,hijri}__L<L>__seed0.parquet` (8 files)
- `data/predictions/<tsfm>__nohijri__L<L>__seed0__window_all.parquet` (4 source parquets)

## Reproduction

```bash
# Stage 1 — generate TSFM full-window predictions (~40 min GPU)
.venv/Scripts/python.exe scripts/run_tsfm.py --model chronos --context-length 720 --window all
.venv/Scripts/python.exe scripts/run_tsfm.py --model moirai  --context-length 336 --window all
.venv/Scripts/python.exe scripts/run_tsfm.py --model timesfm --context-length 168 --window all
.venv/Scripts/python.exe scripts/run_tsfm.py --model timemoe --context-length 720 --window all

# Stage 2 — fit + apply residual heads (~5 min CPU)
for TSFM_NAME in chronos_bolt_base moirai_1_1_small timesfm_2_5 time_moe_200m; do
  for V in nohijri hijri; do
    .venv/Scripts/python.exe scripts/run_residual.py \\
      --tsfm-parquet ${TSFM_NAME}__nohijri__L<L>__seed0__window_all.parquet \\
      --tsfm-name ${TSFM_NAME} --context-length <L> \\
      --variant $V --seed 0
  done
done

# Stage 3 — regenerate stats
.venv/Scripts/python.exe scripts/build_statistical_appendix.py
```
````

- [ ] **Step 3: Commit**

```bash
git add docs/residual_correction.md
git commit -m "docs(plan-6): residual_correction.md results doc"
```

---

## Task 8: Smoke tests + headline updates + final pytest

**Files:**
- Modify: `tests/test_smoke_pipeline.py`
- Modify: `docs/tsfm_zero_shot_baseline.md`

- [ ] **Step 1: Add residual-parquet smoke tests**

Append to `tests/test_smoke_pipeline.py`:

```python
# Plan 6: post-hoc LGBM residual heads on 4 TSFMs.
RESIDUAL_RUNS = [
    ("chronos_bolt_base", 720),
    ("moirai_1_1_small", 336),
    ("timesfm_2_5",       168),
    ("time_moe_200m",     720),
]


@pytest.mark.parametrize("tsfm_name,L", RESIDUAL_RUNS)
@pytest.mark.parametrize("variant", ["nohijri", "hijri"])
def test_residual_prediction_exists(tsfm_name, L, variant):
    p = PRED_DIR / f"{tsfm_name}__residual__{variant}__L{L}__seed0.parquet"
    assert p.exists(), (
        f"Missing {p}. Re-run "
        f"scripts/run_residual.py --tsfm-parquet "
        f"{tsfm_name}__nohijri__L{L}__seed0__window_all.parquet "
        f"--tsfm-name {tsfm_name} --context-length {L} "
        f"--variant {variant} --seed 0."
    )
    df = pd.read_parquet(p)
    assert {"y_true", "y_pred", "regime"} <= set(df.columns)
    assert df["y_pred"].notna().all()
    assert len(df) > 5000
```

- [ ] **Step 2: Add best-residual rows to the headline TSFM doc**

Open `docs/tsfm_zero_shot_baseline.md`. Do all three:

(a) In the **Aggregate test metrics** table, add one row per TSFM
showing the best-of-(residual-nohijri, residual-hijri) corrected MAE
in MAE-sorted order. Row label format: `<TSFM> + residual-<variant>`
e.g., `Chronos-Bolt-Base L=720 + residual-hijri`.

(b) In the **Per-regime MAE** table, add the same rows with the
per-regime MAE values from Task 7 Step 1.

(c) Add a bullet under the **Headline** sub-section pointing to the new
doc, e.g.: `- **Plan 6 finding:** post-hoc LGBM residual correction
[helps/hurts/has-mixed-effect] on Ramadan for the TSFMs that can't
take native covariates. See [residual_correction.md](residual_correction.md).`

If the residual correction did NOT materially improve any TSFM,
document the negative result the same way — that's still a publishable
finding.

- [ ] **Step 3: Run all smoke tests**

```bash
.venv/Scripts/python.exe -m pytest tests/test_smoke_pipeline.py -v -k "residual"
```

Expected: 8 tests pass.

- [ ] **Step 4: Full pytest green**

```bash
.venv/Scripts/python.exe -m pytest -q
```

Expected: 140 prior + 4 (test_lgbm_residual.py) + 8 (residual smoke) = 152 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_smoke_pipeline.py docs/tsfm_zero_shot_baseline.md
git commit -m "test+docs(plan-6): smoke checks + headline updates for residual rows"
```

---

## Self-check before merging

- 4 TSFM `__window_all.parquet` files exist
- 8 `__residual__*` test parquets exist
- `docs/residual_correction.md` exists with non-placeholder numbers
- `docs/statistical_appendix.md` regenerated with 20 models
- `docs/tsfm_zero_shot_baseline.md` updated headline tables
- `pytest -q` reports 152 passed
- The headline finding (post-hoc residual rescues TSFMs on Ramadan) is
  either confirmed (write up) or refuted (document why)
