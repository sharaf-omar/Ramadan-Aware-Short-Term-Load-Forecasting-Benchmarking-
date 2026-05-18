# TSFM Zero-Shot Baseline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire up the 4 proposal TSFMs (Chronos-Bolt, TimesFM 2.0, Moirai-1.1, Time-MoE) as zero-shot block-forecasters that consume the v2 dataset's context window, emit 24-step forecasts, and persist predictions in the same parquet schema used by LightGBM — producing the first cross-architecture comparison on the Turkish STLF test set.

**Architecture:** Each TSFM gets a thin wrapper in `src/models/tsfm/<model>.py` implementing the `Model` protocol. Shared infra (context-window slicing, batched bf16 inference, output unpacking) lives in `src/models/tsfm/_adapter.py`. Predictions land in `data/predictions/<model>__nohijri__L168__seed0.parquet`. Local-only constraint: 8GB VRAM on RTX 4070 mobile means **Chronos-Bolt-Base, TimesFM 2.0 (500M), Moirai-1.1-R-Small, Time-MoE-200M** — substitutions from the proposal's "Large" variants documented in the v2 metadata.

**Scope decisions (vs the full spec):**
- **Single context length L=336** (≈2 weeks) for the headline run. Context-length sweep (L ∈ 96, 168, 336, 720) is Plan 3.
- **Univariate framing for ALL 4 models** in this plan — no Hijri dynamic covariates yet. TimesFM/Moirai covariate variants are Plan 3.
- **Single seed (0)** since TSFMs are deterministic zero-shot.
- **Block-forecast retained** — every prediction parquet stores both `y_pred` (the t+24 point) and `y_block` (the 24-step list) for downstream per-horizon analysis.

**Tech Stack:** PyTorch (with CUDA), Hugging Face `transformers`, Amazon `chronos-forecasting`, Google `timesfm`, Salesforce `uni2ts` (Moirai), `Maple728/TimeMoE`, pandas, pyarrow.

**Reference docs:**
- Spec: `docs/superpowers/specs/2026-05-13-tsfm-ablations-completion-design.md`
- Execution guide: `docs/tsfm_execution_guide.md`
- Predecessor plan: `docs/superpowers/plans/2026-05-13-foundation-and-lgbm-refactor.md`

---

## Phase 0 — Environment Bootstrap

### Task 0.1: Install PyTorch with CUDA support

**Files:**
- Modify: `requirements.txt` (no — torch is a separate install with CUDA wheel index)
- Run: pip install commands

- [ ] **Step 1: Confirm GPU + driver state**

Run: `nvidia-smi | head -10`
Expected: RTX 4070 8GB visible, driver ≥ 525, CUDA-runtime field shown.

- [ ] **Step 2: Install torch + torchvision wheels (CUDA 12.4 build)**

Run:
```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
```
Expected: completes without error; reports torch version 2.4+ with CUDA wheel.

- [ ] **Step 3: Verify CUDA works**

Run:
```bash
python -c "import torch; print(f'torch={torch.__version__}'); print(f'cuda={torch.cuda.is_available()}'); print(f'device={torch.cuda.get_device_name(0)}'); print(f'vram_GB={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}')"
```
Expected:
```
torch=2.4.x+cu124
cuda=True
device=NVIDIA GeForce RTX 4070 Laptop GPU
vram_GB=8.0
```

If `cuda=False`: STOP and ask the user. Driver mismatch or wheel build mismatch.

- [ ] **Step 4: Quick CUDA tensor smoke**

Run:
```bash
python -c "import torch; x = torch.randn(1024, 1024, device='cuda'); y = (x @ x).sum().item(); print(f'cuda matmul ok, sum={y:.2f}')"
```
Expected: prints a finite float.

- [ ] **Step 5: Add torch to requirements.txt as a comment-only entry**

Append to `requirements.txt`:
```
# torch and torchvision: install separately with CUDA wheel index:
#   pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
```

- [ ] **Step 6: Commit**

```bash
git add requirements.txt
git commit -m "chore(env): document torch+CUDA install in requirements.txt"
```

---

### Task 0.2: Install TSFM packages

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Install transformers + accelerate + chronos-forecasting**

Run:
```bash
pip install "transformers>=4.40" "accelerate>=0.30" "chronos-forecasting>=1.4"
```
Expected: all install without conflict. Chronos-forecasting brings in `gluonts` as a dep.

- [ ] **Step 2: Install timesfm**

Run:
```bash
pip install "timesfm[torch]>=1.2"
```
Expected: installs Google TimesFM 2.0 wrapper.

- [ ] **Step 3: Install uni2ts (Moirai)**

Run:
```bash
pip install "uni2ts>=1.2"
```
Expected: installs Salesforce uni2ts and its hydra/lightning deps.

- [ ] **Step 4: Smoke-test imports**

Run:
```bash
python -c "
import chronos
import timesfm
import uni2ts
print(f'chronos={chronos.__version__ if hasattr(chronos, \"__version__\") else \"unknown\"}')
print(f'timesfm={timesfm.__version__ if hasattr(timesfm, \"__version__\") else \"unknown\"}')
print(f'uni2ts={uni2ts.__version__ if hasattr(uni2ts, \"__version__\") else \"unknown\"}')
"
```
Expected: all 3 import without errors. (Time-MoE uses transformers directly — no separate package needed.)

If any import fails: report `NEEDS_CONTEXT` with the exact ImportError, do not improvise. Some packages have aggressive version constraints on Python.

- [ ] **Step 5: Add TSFM deps to requirements.txt**

Append to `requirements.txt`:
```
transformers>=4.40
accelerate>=0.30
chronos-forecasting>=1.4
timesfm[torch]>=1.2
uni2ts>=1.2
```

- [ ] **Step 6: Commit**

```bash
git add requirements.txt
git commit -m "chore(env): add TSFM library dependencies"
```

---

## Phase 1 — TSFM Adapter Skeleton

### Task 1.1: Context-window builder utility

**Files:**
- Create: `src/models/tsfm/_adapter.py`
- Create: `tests/models/test_tsfm_adapter.py`

- [ ] **Step 1: Write the failing test**

`tests/models/test_tsfm_adapter.py`:
```python
import numpy as np
import pandas as pd
import pytest
from src.models.tsfm._adapter import build_context_windows


def _make_load(n_hours: int = 1000) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n_hours, freq="h", tz="UTC")
    return pd.DataFrame({"actual_load": np.arange(n_hours, dtype=float)}, index=idx)


def test_context_window_correct_length_and_endpoint():
    df = _make_load(1000)
    forecast_times = df.index[500:510]  # 10 forecast targets
    L = 168
    contexts = build_context_windows(df["actual_load"], forecast_times, context_length=L)
    assert contexts.shape == (10, 168)
    # row 0 forecasts tau=500. issuance = tau-24 = 476. context = y[476-168+1..476]
    # = y[309..476] inclusive, 168 values
    assert contexts[0, 0] == 309.0
    assert contexts[0, -1] == 476.0
    # row 9 forecasts tau=509. issuance = 485. context = y[318..485]
    assert contexts[9, 0] == 318.0
    assert contexts[9, -1] == 485.0


def test_context_window_drops_insufficient_history():
    df = _make_load(200)
    L = 168
    forecast_times = df.index[:200]  # all rows
    contexts = build_context_windows(df["actual_load"], forecast_times, context_length=L)
    # Need history y[tau-24-L+1 .. tau-24]. Smallest valid tau = 24+L-1 = 191.
    # So only forecast_times at indices >= 191 yield a context.
    # The function returns an array with NaN rows for insufficient history,
    # callers can mask them.
    assert contexts.shape == (200, 168)
    assert np.isnan(contexts[0]).all()    # tau=0 has no history
    assert np.isnan(contexts[190]).all()  # tau=190: would need y[-1] - insufficient
    assert not np.isnan(contexts[191]).any()  # tau=191: y[0..167] available
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/models/test_tsfm_adapter.py -v`
Expected: 2 failures, `ModuleNotFoundError: No module named 'src.models.tsfm._adapter'`.

- [ ] **Step 3: Implement `src/models/tsfm/_adapter.py`**

```python
"""Shared infrastructure for TSFM wrappers.

Provides:
- build_context_windows: slice y into (N, L) tensor for batched TSFM inference
- HORIZON: the proposal-fixed forecast horizon (24 hours)
- ISSUANCE_OFFSET: the gap between issuance time t and forecast time tau
"""
from __future__ import annotations

import numpy as np
import pandas as pd


HORIZON = 24
ISSUANCE_OFFSET = 24  # tau = t + 24


def build_context_windows(
    y: pd.Series,
    forecast_times: pd.DatetimeIndex,
    context_length: int,
) -> np.ndarray:
    """Build (N, L) array of context windows for batched TSFM inference.

    For each forecast time tau in forecast_times, the row contains
    y[tau-24-L+1 .. tau-24] inclusive (L values ending at issuance time t=tau-24).

    Parameters
    ----------
    y : pd.Series of hourly load with UTC DatetimeIndex.
    forecast_times : pd.DatetimeIndex of forecast targets.
    context_length : L.

    Returns
    -------
    np.ndarray shape (len(forecast_times), context_length), dtype float64.
    Rows with insufficient history (less than L values before issuance)
    are filled with NaN. Caller is responsible for masking those rows.
    """
    y_values = y.values
    y_index = y.index
    # Map forecast_times to integer positions in y_index.
    positions = y_index.get_indexer(forecast_times)
    if (positions == -1).any():
        missing = forecast_times[positions == -1]
        raise KeyError(
            f"{len(missing)} forecast times not present in y index. "
            f"First missing: {missing[0]}"
        )

    n = len(forecast_times)
    L = context_length
    out = np.full((n, L), np.nan, dtype=np.float64)

    for i, tau_pos in enumerate(positions):
        end = tau_pos - ISSUANCE_OFFSET  # inclusive end index in y
        start = end - L + 1
        if start < 0:
            continue  # insufficient history
        out[i, :] = y_values[start : end + 1]
    return out
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/models/test_tsfm_adapter.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models/tsfm/_adapter.py tests/models/test_tsfm_adapter.py
git commit -m "feat(tsfm): adapter — context window builder for batched TSFM inference"
```

---

### Task 1.2: TSFM common base class

**Files:**
- Modify: `src/models/tsfm/_adapter.py` (append `TSFMBase`)
- Create: `tests/models/test_tsfm_base.py`

- [ ] **Step 1: Write the failing test**

`tests/models/test_tsfm_base.py`:
```python
import numpy as np
import pandas as pd
from src.models.tsfm._adapter import TSFMBase, HORIZON


class _FakeTSFM(TSFMBase):
    """Test double: returns context[-1] repeated 24 times as the forecast."""
    name = "fake_tsfm"
    supports_dynamic_covariates = False
    needs_training = False

    def _forecast_batch(self, contexts: np.ndarray) -> np.ndarray:
        # contexts: (B, L). Return (B, 24) where each row is last context value.
        return np.tile(contexts[:, -1:], (1, HORIZON))


def _make_test_df(n: int = 500) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "actual_load": np.arange(n, dtype=float),
        "regime": ["Normal"] * n,
    }, index=idx)


def test_tsfm_predict_returns_unified_schema():
    df = _make_test_df(500)
    model = _FakeTSFM()
    out = model.predict(df, context_length=168)
    assert "y_true" in out.columns
    assert "y_pred" in out.columns
    assert "regime" in out.columns
    assert "y_block" in out.columns
    # Rows with insufficient history are dropped.
    # Smallest valid tau index = 24 + 168 - 1 = 191. So out has 500-191 = 309 rows.
    assert len(out) == 309


def test_tsfm_predict_block_has_24_entries():
    df = _make_test_df(500)
    model = _FakeTSFM()
    out = model.predict(df, context_length=168)
    first_block = out["y_block"].iloc[0]
    assert len(first_block) == HORIZON


def test_tsfm_y_pred_is_24th_block_entry():
    df = _make_test_df(500)
    model = _FakeTSFM()
    out = model.predict(df, context_length=168)
    # _FakeTSFM returns last context value repeated 24 times.
    # row tau=191 -> issuance idx 167 -> last context value = y[167] = 167
    # y_pred = block[23] = 167.0
    first_row = out.iloc[0]
    assert first_row["y_pred"] == 167.0
    assert first_row["y_block"][-1] == 167.0


def test_tsfm_fit_is_noop():
    df = _make_test_df(500)
    model = _FakeTSFM()
    # Should not raise; zero-shot models have a no-op fit
    model.fit(df, df, hijri=False, seed=0)
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/models/test_tsfm_base.py -v`
Expected: 4 failures, `ImportError: cannot import name 'TSFMBase'`.

- [ ] **Step 3: Append `TSFMBase` to `src/models/tsfm/_adapter.py`**

```python
import abc

import pandas as pd


class TSFMBase(abc.ABC):
    """Common base for zero-shot TSFM wrappers implementing the Model protocol.

    Subclasses provide:
        name : str
        supports_dynamic_covariates : bool
        _forecast_batch(contexts: np.ndarray) -> np.ndarray

    where contexts is shape (B, L) and the returned array is shape (B, HORIZON).
    """
    needs_training = False  # zero-shot

    @abc.abstractmethod
    def _forecast_batch(self, contexts: np.ndarray) -> np.ndarray:
        """Forecast (B, HORIZON) given context windows (B, L)."""
        raise NotImplementedError

    def fit(self, train_df, val_df, hijri, seed) -> None:
        """Zero-shot models do not train. No-op."""
        return None

    def predict(
        self,
        test_df: pd.DataFrame,
        context_length: int | None = None,
    ) -> pd.DataFrame:
        if context_length is None:
            raise ValueError("TSFM models require context_length.")

        contexts = build_context_windows(
            test_df["actual_load"], test_df.index, context_length=context_length,
        )
        valid_mask = ~np.isnan(contexts).any(axis=1)
        valid_contexts = contexts[valid_mask]

        if len(valid_contexts) == 0:
            raise RuntimeError(
                "No rows in test_df have sufficient history for context_length="
                f"{context_length}. Got {len(test_df)} test rows total."
            )

        blocks = self._forecast_batch(valid_contexts)  # (N_valid, HORIZON)
        assert blocks.shape == (len(valid_contexts), HORIZON), (
            f"_forecast_batch returned {blocks.shape}, expected "
            f"({len(valid_contexts)}, {HORIZON})"
        )

        out = pd.DataFrame({
            "y_true": test_df["actual_load"].values[valid_mask],
            "y_pred": blocks[:, -1],  # the t+24 point
            "y_block": [b.tolist() for b in blocks],
            "regime": test_df["regime"].values[valid_mask],
        }, index=test_df.index[valid_mask])
        return out
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/models/test_tsfm_base.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models/tsfm/_adapter.py tests/models/test_tsfm_base.py
git commit -m "feat(tsfm): TSFMBase shared base class with unified predict() output"
```

---

## Phase 2 — Chronos-Bolt-Base (smallest model first)

### Task 2.1: Chronos wrapper

**Files:**
- Create: `src/models/tsfm/chronos_bolt.py`
- Create: `tests/models/test_chronos_bolt.py`

- [ ] **Step 1: Write the failing test (mocks the underlying pipeline)**

`tests/models/test_chronos_bolt.py`:
```python
import numpy as np
import pytest

# This module imports lazily, so an import-only test is cheap.
from src.models.tsfm.chronos_bolt import ChronosBoltModel


def test_chronos_model_attributes():
    m = ChronosBoltModel()
    assert m.name == "chronos_bolt_base"
    assert m.needs_training is False
    assert m.supports_dynamic_covariates is False


def test_chronos_forecast_batch_shape_smoke():
    """Smoke test against the real model on tiny data. Skipped on CPU/no-GPU.

    Validates: model loads, forward pass returns (B, 24) float array."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    m = ChronosBoltModel()
    rng = np.random.default_rng(0)
    contexts = rng.normal(size=(4, 168)).astype(np.float32)
    out = m._forecast_batch(contexts)
    assert out.shape == (4, 24)
    assert np.isfinite(out).all()
```

- [ ] **Step 2: Run test, verify ImportError**

Run: `pytest tests/models/test_chronos_bolt.py -v`
Expected: 2 failures with ImportError.

- [ ] **Step 3: Implement `src/models/tsfm/chronos_bolt.py`**

```python
"""Chronos-Bolt zero-shot wrapper.

Model: amazon/chronos-bolt-base (T5-encoder-decoder over tokenized time series).
Univariate; no dynamic covariates.
"""
from __future__ import annotations

import numpy as np
import torch

from ._adapter import TSFMBase, HORIZON


class ChronosBoltModel(TSFMBase):
    name = "chronos_bolt_base"
    supports_dynamic_covariates = False

    def __init__(self, checkpoint: str = "amazon/chronos-bolt-base", batch_size: int = 64):
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self._pipeline = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        # Lazy import so test collection does not download a model.
        from chronos import ChronosBoltPipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self._pipeline = ChronosBoltPipeline.from_pretrained(
            self.checkpoint,
            device_map=device,
            torch_dtype=dtype,
        )

    def _forecast_batch(self, contexts: np.ndarray) -> np.ndarray:
        """contexts: (B, L) float. Returns (B, HORIZON) median quantile."""
        self._load()

        # Chronos expects a list of 1-D torch tensors.
        ctx_tensors = [torch.tensor(row, dtype=torch.float32) for row in contexts]

        all_blocks: list[np.ndarray] = []
        bs = self.batch_size
        for i in range(0, len(ctx_tensors), bs):
            batch = ctx_tensors[i : i + bs]
            # predict_quantiles returns (B, H, Q) and (B, H) for the mean.
            quantiles, mean = self._pipeline.predict_quantiles(
                context=batch,
                prediction_length=HORIZON,
                quantile_levels=[0.5],
            )
            # quantiles shape: (B, H, 1) since we asked for the median only.
            block = quantiles[:, :, 0].float().cpu().numpy()
            all_blocks.append(block)
        return np.concatenate(all_blocks, axis=0)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/models/test_chronos_bolt.py -v`
Expected: `test_chronos_model_attributes` PASSES. `test_chronos_forecast_batch_shape_smoke` either passes or downloads/loads the model (~~140MB) and then passes.

If `test_chronos_forecast_batch_shape_smoke` fails with OOM: drop `batch_size` from 64 to 16.

- [ ] **Step 5: Commit**

```bash
git add src/models/tsfm/chronos_bolt.py tests/models/test_chronos_bolt.py
git commit -m "feat(tsfm): Chronos-Bolt-Base zero-shot wrapper"
```

---

### Task 2.2: Run Chronos-Bolt on the test set

**Files:**
- Create: `scripts/run_tsfm.py`

- [ ] **Step 1: Implement the script**

```python
"""Run a TSFM on the v2 test set and save predictions to parquet.

Usage:
    python scripts/run_tsfm.py --model chronos --context-length 336
    python scripts/run_tsfm.py --model timesfm --context-length 336
    python scripts/run_tsfm.py --model moirai  --context-length 336
    python scripts/run_tsfm.py --model timemoe --context-length 336
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
    "chronos": ("src.models.tsfm.chronos_bolt", "ChronosBoltModel"),
    "timesfm": ("src.models.tsfm.timesfm", "TimesFMModel"),
    "moirai":  ("src.models.tsfm.moirai",  "MoiraiModel"),
    "timemoe": ("src.models.tsfm.time_moe", "TimeMoEModel"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--variant", default="nohijri", choices=["nohijri", "hijri"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"[1/4] Loading v2 dataset ...")
    df = pd.read_csv(V2_CSV, parse_dates=["timestamp"]).set_index("timestamp")
    df.index = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")

    # Use the full historical y for context (TSFM looks back), but only emit
    # predictions for forecast times in the test window.
    test_window = df.loc["2024-01-01":"2025-03-31"]
    print(f"      test forecast hours: {len(test_window):,}")

    print(f"[2/4] Instantiating {args.model} ...")
    mod_path, cls_name = MODEL_REGISTRY[args.model]
    import importlib
    cls = getattr(importlib.import_module(mod_path), cls_name)
    model = cls()
    print(f"      name={model.name} supports_dynamic_covariates={model.supports_dynamic_covariates}")

    print(f"[3/4] Forecasting (L={args.context_length}) ...")
    t0 = time.time()
    # IMPORTANT: pass the FULL df as test_df so build_context_windows can look
    # back before 2024-01-01. Filter to test_window AFTER prediction.
    preds_all = model.predict(df, context_length=args.context_length)
    test_preds = preds_all.loc[test_window.index.intersection(preds_all.index)]
    elapsed = time.time() - t0
    print(f"      done in {elapsed:.1f}s  ({len(test_preds):,} predictions)")

    print(f"[4/4] Writing parquet ...")
    path = write_predictions(
        test_preds,
        model=model.name,
        variant=args.variant,
        context_length=args.context_length,
        seed=args.seed,
    )
    print(f"      -> {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run on Chronos (small test set slice first)**

Run:
```bash
python -c "
import pandas as pd
from src.models.tsfm.chronos_bolt import ChronosBoltModel
df = pd.read_csv('data/processed/final_training_set_v2.csv', parse_dates=['timestamp']).set_index('timestamp')
df.index = df.index.tz_convert('UTC') if df.index.tz is not None else df.index.tz_localize('UTC')
# Take a tiny slice for smoke test (2024-01-15 to 2024-01-20: 5 days = 120 forecast hours)
slice_df = df.loc['2024-01-15':'2024-01-20']
print(f'slice rows = {len(slice_df)}')
m = ChronosBoltModel()
out = m.predict(df.loc[:'2024-01-20'], context_length=336)
print(f'predictions = {len(out)}')
print(out.head().to_string())
print(f'y_pred range: [{out[\"y_pred\"].min():.1f}, {out[\"y_pred\"].max():.1f}]')
"
```
Expected: ~120 predictions, y_pred values in plausible MW range (~25000-45000). Runtime < 2 min.

- [ ] **Step 3: Full Chronos run on test set**

Run:
```bash
python scripts/run_tsfm.py --model chronos --context-length 336
```
Expected:
- Prints "test forecast hours: 10,944"
- Forecasts complete in 5–15 min
- Writes `data/predictions/chronos_bolt_base__nohijri__L336__seed0.parquet`

- [ ] **Step 4: Sanity-check vs LGBM on Normal regime**

Run:
```bash
python -c "
from src.evaluation.predictions_io import read_predictions
from src.evaluation.regime_eval import evaluate_by_regime
import pandas as pd

TRAIN = pd.read_csv('data/processed/final_training_set_v2.csv', parse_dates=['timestamp']).set_index('timestamp').loc['2018':'2022']['actual_load'].values

chronos = read_predictions(model='chronos_bolt_base', variant='nohijri', context_length=336, seed=0)
lgbm    = read_predictions(model='lgbm', variant='nohijri', context_length=None, seed=44)

# Intersect on shared timestamps (LGBM has no L-dropoff, chronos has L=336 dropoff)
shared = chronos.index.intersection(lgbm.index)
c2 = chronos.loc[shared]
l2 = lgbm.loc[shared]

print(f'shared rows: {len(shared):,}')
print()
print('Chronos-Bolt-Base (L=336) per-regime:')
print(evaluate_by_regime(c2['y_true'].values, c2['y_pred'].values, regimes=c2['regime'], y_train=TRAIN, period=168).to_string(index=False))
print()
print('LGBM hijri seed=44 on SAME rows (subset):')
print(evaluate_by_regime(l2['y_true'].values, l2['y_pred'].values, regimes=l2['regime'], y_train=TRAIN, period=168).to_string(index=False))
"
```
Expected: Chronos Normal MAE likely 1200–2500 MW (worse than LGBM ~890), Heatwave MAE higher than Normal (as with LGBM). If Chronos Normal MAE is wildly off (>5000 MW), investigate.

- [ ] **Step 5: Commit predictions + script**

```bash
git add scripts/run_tsfm.py data/predictions/chronos_bolt_base__nohijri__L336__seed0.parquet
git commit -m "feat(tsfm): run Chronos-Bolt-Base on v2 test set at L=336"
```

---

## Phase 3 — TimesFM 2.0

### Task 3.1: TimesFM wrapper

**Files:**
- Create: `src/models/tsfm/timesfm.py`
- Create: `tests/models/test_timesfm.py`

- [ ] **Step 1: Write failing test**

`tests/models/test_timesfm.py`:
```python
import numpy as np
import pytest

from src.models.tsfm.timesfm import TimesFMModel


def test_timesfm_model_attributes():
    m = TimesFMModel()
    assert m.name == "timesfm_2_0"
    assert m.needs_training is False
    # Univariate framing in this plan (covariates added in Plan 3).
    assert m.supports_dynamic_covariates is True


def test_timesfm_forecast_batch_shape_smoke():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    m = TimesFMModel()
    rng = np.random.default_rng(0)
    contexts = rng.normal(size=(4, 336)).astype(np.float32)
    out = m._forecast_batch(contexts)
    assert out.shape == (4, 24)
    assert np.isfinite(out).all()
```

- [ ] **Step 2: Run test, verify ImportError**

Run: `pytest tests/models/test_timesfm.py -v`
Expected: 2 failures with ImportError.

- [ ] **Step 3: Implement wrapper**

`src/models/tsfm/timesfm.py`:
```python
"""TimesFM 2.0 zero-shot wrapper.

Model: google/timesfm-2.0-500m-pytorch (decoder-only patched Transformer).
Supports dynamic covariates (deferred to Plan 3).
"""
from __future__ import annotations

import numpy as np
import torch

from ._adapter import TSFMBase, HORIZON


class TimesFMModel(TSFMBase):
    name = "timesfm_2_0"
    supports_dynamic_covariates = True

    def __init__(
        self,
        checkpoint: str = "google/timesfm-2.0-500m-pytorch",
        batch_size: int = 32,
    ):
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import timesfm
        backend = "gpu" if torch.cuda.is_available() else "cpu"
        self._model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=backend,
                per_core_batch_size=self.batch_size,
                horizon_len=HORIZON,
                context_len=512,  # max supported; we'll pad smaller contexts
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=self.checkpoint,
            ),
        )

    def _forecast_batch(self, contexts: np.ndarray) -> np.ndarray:
        """contexts: (B, L). Returns (B, HORIZON) point forecast."""
        self._load()
        forecast_input = [row.astype(np.float32) for row in contexts]
        # frequency 0 = high-frequency (hourly fits here).
        freq = [0] * len(forecast_input)
        point_forecast, _quantile = self._model.forecast(
            forecast_input,
            freq=freq,
        )
        # point_forecast shape: (B, HORIZON)
        return np.asarray(point_forecast)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/models/test_timesfm.py -v`
Expected: 2 passed (or skipped without CUDA). First run downloads ~1GB.

- [ ] **Step 5: Commit**

```bash
git add src/models/tsfm/timesfm.py tests/models/test_timesfm.py
git commit -m "feat(tsfm): TimesFM 2.0 zero-shot wrapper"
```

---

### Task 3.2: Run TimesFM on test set

- [ ] **Step 1: Run via the script**

Run:
```bash
python scripts/run_tsfm.py --model timesfm --context-length 336
```
Expected: ~10,944 predictions, runtime 15–30 min, writes `timesfm_2_0__nohijri__L336__seed0.parquet`.

- [ ] **Step 2: Sanity-check**

Run:
```bash
python -c "
import pandas as pd
from src.evaluation.predictions_io import read_predictions
from src.evaluation.regime_eval import evaluate_by_regime

TRAIN = pd.read_csv('data/processed/final_training_set_v2.csv', parse_dates=['timestamp']).set_index('timestamp').loc['2018':'2022']['actual_load'].values
p = read_predictions(model='timesfm_2_0', variant='nohijri', context_length=336, seed=0)
print(evaluate_by_regime(p['y_true'].values, p['y_pred'].values, regimes=p['regime'], y_train=TRAIN, period=168).to_string(index=False))
"
```
Expected: per-regime metrics printed; Heatwave MAE higher than Normal as before.

- [ ] **Step 3: Commit**

```bash
git add data/predictions/timesfm_2_0__nohijri__L336__seed0.parquet
git commit -m "feat(tsfm): run TimesFM 2.0 on v2 test set at L=336"
```

---

## Phase 4 — Moirai-1.1

### Task 4.1: Moirai wrapper

**Files:**
- Create: `src/models/tsfm/moirai.py`
- Create: `tests/models/test_moirai.py`

- [ ] **Step 1: Write failing test**

`tests/models/test_moirai.py`:
```python
import numpy as np
import pytest

from src.models.tsfm.moirai import MoiraiModel


def test_moirai_model_attributes():
    m = MoiraiModel()
    assert m.name == "moirai_1_1_small"
    assert m.needs_training is False
    assert m.supports_dynamic_covariates is True


def test_moirai_forecast_batch_shape_smoke():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    m = MoiraiModel()
    rng = np.random.default_rng(0)
    contexts = rng.normal(size=(4, 336)).astype(np.float32)
    out = m._forecast_batch(contexts)
    assert out.shape == (4, 24)
    assert np.isfinite(out).all()
```

- [ ] **Step 2: Run test, verify ImportError**

Run: `pytest tests/models/test_moirai.py -v`
Expected: 2 failures.

- [ ] **Step 3: Implement wrapper**

`src/models/tsfm/moirai.py`:
```python
"""Moirai-1.1 zero-shot wrapper.

Model: Salesforce/moirai-1.1-R-small (masked encoder, any-variate).
8GB VRAM forces 'small' instead of 'large' from the proposal; documented in spec.
Supports dynamic covariates (deferred to Plan 3).
"""
from __future__ import annotations

import numpy as np
import torch

from ._adapter import TSFMBase, HORIZON


class MoiraiModel(TSFMBase):
    name = "moirai_1_1_small"
    supports_dynamic_covariates = True

    def __init__(
        self,
        checkpoint: str = "Salesforce/moirai-1.1-R-small",
        batch_size: int = 32,
        patch_size: int = 32,
    ):
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.patch_size = patch_size
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        device = "cuda" if torch.cuda.is_available() else "cpu"
        module = MoiraiModule.from_pretrained(self.checkpoint).to(device).eval()
        self._module = module
        self._device = device

    def _forecast_batch(self, contexts: np.ndarray) -> np.ndarray:
        """contexts: (B, L). Returns (B, HORIZON) median point forecast."""
        self._load()
        from uni2ts.model.moirai import MoiraiForecast

        L = contexts.shape[1]
        forecaster = MoiraiForecast(
            module=self._module,
            prediction_length=HORIZON,
            context_length=L,
            patch_size=self.patch_size,
            num_samples=20,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        ).to(self._device).eval()

        # MoiraiForecast.forward expects:
        #   past_target: (B, L, target_dim)
        #   past_observed_target: (B, L, target_dim)
        #   past_is_pad: (B, L)
        past_target = torch.tensor(
            contexts[..., None], dtype=torch.float32, device=self._device
        )
        past_observed = torch.ones_like(past_target, dtype=torch.bool)
        past_is_pad = torch.zeros(past_target.shape[:2], dtype=torch.bool, device=self._device)

        all_blocks: list[np.ndarray] = []
        bs = self.batch_size
        with torch.no_grad():
            for i in range(0, past_target.shape[0], bs):
                pt = past_target[i : i + bs]
                po = past_observed[i : i + bs]
                pp = past_is_pad[i : i + bs]
                # forward returns (B, num_samples, HORIZON, target_dim)
                samples = forecaster(past_target=pt, past_observed_target=po, past_is_pad=pp)
                # Take median across samples; drop singleton target_dim.
                med = samples.float().median(dim=1).values[:, :, 0]
                all_blocks.append(med.cpu().numpy())
        return np.concatenate(all_blocks, axis=0)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/models/test_moirai.py -v`
Expected: 2 passed (or skipped without CUDA). First run downloads ~150MB.

If you hit `OSError` or `KeyError` on Moirai loading: report `NEEDS_CONTEXT`. The Salesforce uni2ts API has changed between versions; the engineer may need to pin a specific version.

- [ ] **Step 5: Commit**

```bash
git add src/models/tsfm/moirai.py tests/models/test_moirai.py
git commit -m "feat(tsfm): Moirai-1.1-R-Small zero-shot wrapper"
```

---

### Task 4.2: Run Moirai on test set

- [ ] **Step 1: Run via script**

Run:
```bash
python scripts/run_tsfm.py --model moirai --context-length 336
```
Expected: ~10,944 predictions, runtime 15–30 min, writes `moirai_1_1_small__nohijri__L336__seed0.parquet`.

- [ ] **Step 2: Sanity-check**

Run:
```bash
python -c "
import pandas as pd
from src.evaluation.predictions_io import read_predictions
from src.evaluation.regime_eval import evaluate_by_regime

TRAIN = pd.read_csv('data/processed/final_training_set_v2.csv', parse_dates=['timestamp']).set_index('timestamp').loc['2018':'2022']['actual_load'].values
p = read_predictions(model='moirai_1_1_small', variant='nohijri', context_length=336, seed=0)
print(evaluate_by_regime(p['y_true'].values, p['y_pred'].values, regimes=p['regime'], y_train=TRAIN, period=168).to_string(index=False))
"
```

- [ ] **Step 3: Commit**

```bash
git add data/predictions/moirai_1_1_small__nohijri__L336__seed0.parquet
git commit -m "feat(tsfm): run Moirai-1.1-R-Small on v2 test set at L=336"
```

---

## Phase 5 — Time-MoE

### Task 5.1: Time-MoE wrapper

**Files:**
- Create: `src/models/tsfm/time_moe.py`
- Create: `tests/models/test_time_moe.py`

- [ ] **Step 1: Write failing test**

`tests/models/test_time_moe.py`:
```python
import numpy as np
import pytest

from src.models.tsfm.time_moe import TimeMoEModel


def test_timemoe_model_attributes():
    m = TimeMoEModel()
    assert m.name == "time_moe_200m"
    assert m.needs_training is False
    assert m.supports_dynamic_covariates is False  # channel-independent


def test_timemoe_forecast_batch_shape_smoke():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    m = TimeMoEModel()
    rng = np.random.default_rng(0)
    contexts = rng.normal(size=(4, 336)).astype(np.float32)
    out = m._forecast_batch(contexts)
    assert out.shape == (4, 24)
    assert np.isfinite(out).all()
```

- [ ] **Step 2: Run test, verify ImportError**

Run: `pytest tests/models/test_time_moe.py -v`
Expected: 2 failures.

- [ ] **Step 3: Implement wrapper**

`src/models/tsfm/time_moe.py`:
```python
"""Time-MoE 200M zero-shot wrapper.

Model: Maple728/TimeMoE-200M (Mixture-of-Experts decoder).
8GB VRAM forces 200M instead of "Large" from the proposal; documented in spec.
Channel-independent; no dynamic covariates inline.
"""
from __future__ import annotations

import numpy as np
import torch

from ._adapter import TSFMBase, HORIZON


class TimeMoEModel(TSFMBase):
    name = "time_moe_200m"
    supports_dynamic_covariates = False

    def __init__(
        self,
        checkpoint: str = "Maple728/TimeMoE-200M",
        batch_size: int = 32,
    ):
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device).eval()
        self._device = device

    def _forecast_batch(self, contexts: np.ndarray) -> np.ndarray:
        """contexts: (B, L). Returns (B, HORIZON) point forecast.

        Time-MoE expects context to be normalized per series internally; we just
        pass raw inputs. The model's `generate` produces the next HORIZON points.
        """
        self._load()
        all_blocks: list[np.ndarray] = []
        bs = self.batch_size
        with torch.no_grad():
            for i in range(0, len(contexts), bs):
                batch = torch.tensor(
                    contexts[i : i + bs], dtype=torch.float32, device=self._device
                )
                # Time-MoE per-series normalization: subtract mean, divide by std.
                mean = batch.mean(dim=1, keepdim=True)
                std = batch.std(dim=1, keepdim=True).clamp(min=1e-8)
                batch_norm = (batch - mean) / std
                # generate produces (B, L + HORIZON); take the last HORIZON tokens
                # and denormalize.
                preds = self._model.generate(
                    inputs=batch_norm,
                    max_new_tokens=HORIZON,
                )
                tail = preds[:, -HORIZON:].float()
                denorm = tail * std + mean
                all_blocks.append(denorm.cpu().numpy())
        return np.concatenate(all_blocks, axis=0)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/models/test_time_moe.py -v`
Expected: 2 passed (or skipped without CUDA). First run downloads ~450MB.

If the call to `generate(inputs=...)` raises (signature mismatch), check Time-MoE's example code on the HuggingFace model card — the `generate` interface for Time-MoE is non-standard and changes occasionally. Report NEEDS_CONTEXT with the exact error if so.

- [ ] **Step 5: Commit**

```bash
git add src/models/tsfm/time_moe.py tests/models/test_time_moe.py
git commit -m "feat(tsfm): Time-MoE-200M zero-shot wrapper"
```

---

### Task 5.2: Run Time-MoE on test set

- [ ] **Step 1: Run via script**

Run:
```bash
python scripts/run_tsfm.py --model timemoe --context-length 336
```
Expected: ~10,944 predictions, runtime 10–25 min, writes `time_moe_200m__nohijri__L336__seed0.parquet`.

- [ ] **Step 2: Sanity-check**

Run:
```bash
python -c "
import pandas as pd
from src.evaluation.predictions_io import read_predictions
from src.evaluation.regime_eval import evaluate_by_regime

TRAIN = pd.read_csv('data/processed/final_training_set_v2.csv', parse_dates=['timestamp']).set_index('timestamp').loc['2018':'2022']['actual_load'].values
p = read_predictions(model='time_moe_200m', variant='nohijri', context_length=336, seed=0)
print(evaluate_by_regime(p['y_true'].values, p['y_pred'].values, regimes=p['regime'], y_train=TRAIN, period=168).to_string(index=False))
"
```

- [ ] **Step 3: Commit**

```bash
git add data/predictions/time_moe_200m__nohijri__L336__seed0.parquet
git commit -m "feat(tsfm): run Time-MoE-200M on v2 test set at L=336"
```

---

## Phase 6 — Cross-Model Comparison Table

### Task 6.1: Comparison summary doc

**Files:**
- Create: `docs/tsfm_zero_shot_baseline.md`

- [ ] **Step 1: Compute the cross-model table**

Run:
```bash
python -c "
import pandas as pd
from src.evaluation.predictions_io import read_predictions
from src.evaluation.regime_eval import evaluate_by_regime
from src.evaluation.metrics import mae, rmse, mape, mase

TRAIN = pd.read_csv('data/processed/final_training_set_v2.csv', parse_dates=['timestamp']).set_index('timestamp').loc['2018':'2022']['actual_load'].values

MODELS = [
    ('lgbm', 'hijri', None, 44),
    ('chronos_bolt_base', 'nohijri', 336, 0),
    ('timesfm_2_0',       'nohijri', 336, 0),
    ('moirai_1_1_small',  'nohijri', 336, 0),
    ('time_moe_200m',     'nohijri', 336, 0),
]

# Use intersection of timestamps across all models for fair comparison.
preds = []
for m, v, L, s in MODELS:
    p = read_predictions(model=m, variant=v, context_length=L, seed=s)
    preds.append((m, p))
shared = preds[0][1].index
for _, p in preds[1:]:
    shared = shared.intersection(p.index)
print(f'shared rows across all models: {len(shared):,}')
print()

print('=== Aggregate MAE on shared rows ===')
print(f'{\"model\":<22} {\"MAE\":>10} {\"RMSE\":>10} {\"MAPE\":>8} {\"MASE\":>8}')
for m, p in preds:
    p2 = p.loc[shared]
    print(f'{m:<22} {mae(p2.y_true,p2.y_pred):>10.2f} {rmse(p2.y_true,p2.y_pred):>10.2f} {mape(p2.y_true,p2.y_pred):>8.4f} {mase(p2.y_true,p2.y_pred,TRAIN,168):>8.4f}')

print()
print('=== Per-regime MAE on shared rows ===')
for m, p in preds:
    p2 = p.loc[shared]
    tab = evaluate_by_regime(p2['y_true'].values, p2['y_pred'].values, regimes=p2['regime'], y_train=TRAIN, period=168)
    print(f'\n{m}:')
    print(tab.to_string(index=False))
" > /tmp/tsfm_comparison.txt 2>&1
cat /tmp/tsfm_comparison.txt
```

- [ ] **Step 2: Write `docs/tsfm_zero_shot_baseline.md`**

Capture the output of step 1 inside markdown. Use this template (fill in the actual numbers from step 1's output):

```markdown
# TSFM Zero-Shot Baseline Results (L=336, seed=0)

First cross-architecture comparison of the four proposal TSFMs against the
LightGBM baseline on the Turkish STLF test set (2024-01-01 to 2025-03-31).

## Setup

- Context length: 336 hours (~2 weeks)
- Forecast horizon: 24 hours; report on the t+24 point
- Single seed (0): TSFMs are deterministic zero-shot
- All 4 models in **univariate framing** (no Hijri covariates) for this plan;
  TimesFM and Moirai will be re-run with Hijri dynamic covariates in Plan 3.
- Hardware substitutions due to local 8GB VRAM:
  - Chronos-Bolt-Base (proposal said Large)
  - Moirai-1.1-R-Small (proposal said Large)
  - Time-MoE-200M (proposal said the sparse "Large" variant)

## Aggregate metrics (on shared timestamps across all 5 models)

(paste step-1 aggregate table here)

## Per-regime breakdown

(paste step-1 per-regime tables here)

## Interpretation

(write 2-3 sentences after seeing the numbers; e.g., which model is closest
to LGBM on Normal, which is most affected by the Heatwave regime, etc.)

## Files

- Plan: `docs/superpowers/plans/2026-05-13-tsfm-zero-shot-baseline.md`
- Predictions: `data/predictions/chronos_bolt_base__nohijri__L336__seed0.parquet`,
  `timesfm_2_0__nohijri__L336__seed0.parquet`,
  `moirai_1_1_small__nohijri__L336__seed0.parquet`,
  `time_moe_200m__nohijri__L336__seed0.parquet`.
```

- [ ] **Step 3: Commit**

```bash
git add docs/tsfm_zero_shot_baseline.md
git commit -m "docs: TSFM zero-shot baseline results (4 models at L=336)"
```

---

## Phase 7 — Smoke Tests and Plan Wrap-Up

### Task 7.1: Extend smoke pytest

**Files:**
- Modify: `tests/test_smoke_pipeline.py`

- [ ] **Step 1: Append TSFM smoke checks**

Add to `tests/test_smoke_pipeline.py`:
```python
TSFM_MODELS_L336 = [
    "chronos_bolt_base",
    "timesfm_2_0",
    "moirai_1_1_small",
    "time_moe_200m",
]


@pytest.mark.parametrize("model_name", TSFM_MODELS_L336)
def test_tsfm_prediction_exists_L336(model_name):
    p = PRED_DIR / f"{model_name}__nohijri__L336__seed0.parquet"
    assert p.exists(), f"Missing {p}. Run scripts/run_tsfm.py --model <short_name> --context-length 336."
    df = pd.read_parquet(p)
    assert "y_true" in df.columns
    assert "y_pred" in df.columns
    assert "y_block" in df.columns
    assert df["y_pred"].notna().all()
    # Block length must be 24
    assert all(len(b) == 24 for b in df["y_block"].head(20))
    assert len(df) > 5000  # most of 10,944 test hours
```

- [ ] **Step 2: Run smoke**

Run: `pytest tests/test_smoke_pipeline.py -v`
Expected: all existing tests + 4 new TSFM tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_smoke_pipeline.py
git commit -m "test: smoke checks for 4 TSFM prediction parquets at L=336"
```

---

### Task 7.2: Update README milestone tracker

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Flip Plan 2 from `[ ]` to `[x]` in README.md**

Edit the `## Status (current milestone)` section: change
```
- [ ] Plan 2: TSFM zero-shot evaluation (Chronos, TimesFM, Moirai, Time-MoE)
```
to
```
- [x] Plan 2: TSFM zero-shot evaluation (Chronos, TimesFM, Moirai, Time-MoE) — single L=336, univariate
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: mark Plan 2 milestone complete in README"
```

---

### Task 7.3: Full pytest green

- [ ] **Step 1: Run full suite**

Run: `pytest -v`
Expected: all tests pass (≈ 71 from Plan 1 + 4 adapter/base + 4 per-model attribute + 4 smoke = ~83 passing; the per-model smoke tests requiring CUDA are skipped or pass depending on CUDA availability at test time).

- [ ] **Step 2: Commit any fix-ups**

If something broke, fix it and commit. Otherwise no commit needed.

---

## Self-Review Checklist (engineer should not skip)

Before claiming Plan 2 complete:

1. **Every test passes:** `pytest -v` all green.
2. **4 TSFM prediction parquets exist** at L=336, seed=0, nohijri variant.
3. **Each parquet has** y_true / y_pred / y_block (24 floats) / regime columns.
4. **Cross-model comparison table** computed and saved in `docs/tsfm_zero_shot_baseline.md`.
5. **No model substitution undocumented** — small variants used due to VRAM are cited in the doc.
6. **All commits on `plan-2-tsfm` branch** (not on `plan-1-foundation` or `main`).

## Risks and Escalation

- **uni2ts API drift**: Moirai loader changed between 1.1 and 1.2. If `MoiraiForecast` constructor rejects the kwargs in Task 4.1, pin to `uni2ts==1.2.1` and retry.
- **Time-MoE `generate` signature**: The HF model card uses `model.generate(inputs=context_tensor, max_new_tokens=H)`. If transformers complains, check Time-MoE's example on HuggingFace and adapt.
- **VRAM OOM mid-run**: drop batch_size to 16 or 8 in the offending wrapper's constructor.
- **HuggingFace 401**: some checkpoints require accepting a license; run `huggingface-cli login` and accept on the model page.
- **Disk space**: model caches total ~3 GB. Set `$env:HF_HOME` to a non-C drive if C: is tight.
