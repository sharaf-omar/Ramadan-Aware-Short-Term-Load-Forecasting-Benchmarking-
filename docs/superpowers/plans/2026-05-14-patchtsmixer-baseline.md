# PatchTSMixer Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deep-learning baseline (HuggingFace PatchTSMixer with cross-channel mixing) trained from scratch on Turkish electricity load, to slot between LightGBM/TSFMs and the classical baselines in the headline table.

**Architecture:** A single Python wrapper (`PatchTSMixerModel`) wraps HuggingFace's `PatchTSMixerForPrediction`. A `WindowedDataset(torch.utils.data.Dataset)` produces sliding (context, target) pairs from the v2 dataset. Training uses HuggingFace `Trainer` with bf16, cosine LR schedule, and val-MAE early stopping. Inference is per-τ on the test set with day-ahead alignment (`y_pred[τ] = forecast_block[23]`).

**Tech Stack:** Python 3.12, `torch 2.4.1+cu124`, `transformers 4.48.3` (already pinned; ships `PatchTSMixerForPrediction`), `pandas 2.1.4`, `numpy 1.26.4`. No new dependencies.

**Spec:** [`docs/superpowers/specs/2026-05-14-patchtsmixer-baseline-design.md`](../specs/2026-05-14-patchtsmixer-baseline-design.md)

---

## File structure (locked in)

| File | Responsibility |
|------|----------------|
| `src/models/dl/patchtsmixer.py` | `WindowedDataset` + `PatchTSMixerModel` wrapper (fit/predict conforming to `src/models/base.py::Model`) |
| `src/models/dl/__init__.py` | Re-export `PatchTSMixerModel` |
| `scripts/run_patchtsmixer.py` | CLI runner that mirrors `scripts/run_classical.py` but passes `--context-length` and writes parquet with `context_length=L` |
| `tests/models/test_patchtsmixer.py` | Unit tests: attributes, variant feature sets, fit-1-epoch smoke, predict schema |
| `tests/test_smoke_pipeline.py` | Extend with `patchtsmixer__*` parquet existence checks |
| `docs/patchtsmixer_baseline.md` | Results doc (L-probe, headline grid, regime metrics, DM tests) |
| `docs/tsfm_zero_shot_baseline.md` | Add PatchTSMixer rows to aggregate and per-regime tables |
| `README.md` | Mark Plan 5 complete |

---

## Task 0: Branch setup

**Files:** none (git only)

- [ ] **Step 1: Create branch off main**

Run:
```bash
git checkout main && git pull origin main
git checkout -b plan-5-patchtst
```

Expected: `Switched to a new branch 'plan-5-patchtst'`

- [ ] **Step 2: Verify dataset present**

Run:
```bash
ls -la data/processed/final_training_set_v2.csv
```

Expected: File exists, ~50 MB. If missing, the engineer must rebuild via `python -m src.data.build_v2_dataset` (see README §Reproduction).

- [ ] **Step 3: Verify transformers ships PatchTSMixer**

Run:
```bash
.venv/Scripts/python.exe -c "from transformers import PatchTSMixerForPrediction, PatchTSMixerConfig; print(PatchTSMixerConfig().mode)"
```

Expected: prints `common_channel` (the default — confirms the symbols import cleanly).

- [ ] **Step 4: Verify GPU + bf16 available**

Run:
```bash
.venv/Scripts/python.exe -c "import torch; print('cuda', torch.cuda.is_available(), 'bf16', torch.cuda.is_bf16_supported())"
```

Expected: `cuda True bf16 True`

---

## Task 1: `WindowedDataset` (sliding-window pairs)

**Files:**
- Create: `src/models/dl/patchtsmixer.py`
- Test: `tests/models/test_patchtsmixer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/models/test_patchtsmixer.py`:

```python
import numpy as np
import pandas as pd
import pytest
import torch

from src.models.dl.patchtsmixer import WindowedDataset


def _synthetic_arr(n: int = 100, c: int = 3) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, c)).astype(np.float32)


def test_windowed_dataset_shapes():
    arr = _synthetic_arr(n=100, c=3)
    ds = WindowedDataset(arr, context_length=24, prediction_length=24)
    # n=100, L=24, h=24 -> 100 - 24 - 24 + 1 = 53 samples
    assert len(ds) == 53
    sample = ds[0]
    assert sample["past_values"].shape == (24, 3)
    assert sample["past_values"].dtype == torch.float32
    # future_values: shape (h, 1) — y-channel only (col 0)
    assert sample["future_values"].shape == (24, 1)
    assert sample["future_values"].dtype == torch.float32


def test_windowed_dataset_alignment():
    arr = _synthetic_arr(n=60, c=2)
    ds = WindowedDataset(arr, context_length=12, prediction_length=12)
    sample = ds[5]
    # sample 5: past = arr[5:17], future_y = arr[17:29, 0]
    assert np.allclose(sample["past_values"].numpy(), arr[5:17])
    assert np.allclose(sample["future_values"].squeeze(-1).numpy(), arr[17:29, 0])
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/models/test_patchtsmixer.py::test_windowed_dataset_shapes -v
```

Expected: `ModuleNotFoundError: No module named 'src.models.dl.patchtsmixer'`

- [ ] **Step 3: Write minimal `WindowedDataset` implementation**

Create `src/models/dl/patchtsmixer.py`:

```python
"""PatchTSMixer deep-learning baseline (proposal §4.2 substitute).

Substitutes HuggingFace's PatchTSMixer (cross-channel mixing variant) for
vanilla PatchTST so that Hijri/weather channels actually condition the
y forecast. See docs/superpowers/specs/2026-05-14-patchtsmixer-baseline-design.md.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowedDataset(Dataset):
    """Sliding-window dataset over a (T, C) float32 array.

    Yields per-sample dicts compatible with HuggingFace PatchTSMixerForPrediction:
        - past_values:   (context_length, num_channels) float32
        - future_values: (prediction_length, 1)         float32  — y-channel only (col 0)

    The y channel is assumed to be column 0 of `arr`.
    """

    def __init__(self, arr: np.ndarray, context_length: int, prediction_length: int):
        assert arr.dtype == np.float32, f"arr must be float32, got {arr.dtype}"
        assert arr.ndim == 2, f"arr must be 2D (T, C), got shape {arr.shape}"
        if len(arr) < context_length + prediction_length:
            raise ValueError(
                f"arr length {len(arr)} < L+h ({context_length}+{prediction_length})"
            )
        self.arr = arr
        self.L = context_length
        self.H = prediction_length
        self._len = len(arr) - self.L - self.H + 1

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        past = self.arr[i : i + self.L]                       # (L, C)
        future_y = self.arr[i + self.L : i + self.L + self.H, 0:1]  # (H, 1)
        return {
            "past_values":   torch.from_numpy(past),
            "future_values": torch.from_numpy(future_y),
        }
```

- [ ] **Step 4: Run tests to verify pass**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/models/test_patchtsmixer.py::test_windowed_dataset_shapes tests/models/test_patchtsmixer.py::test_windowed_dataset_alignment -v
```

Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add src/models/dl/patchtsmixer.py tests/models/test_patchtsmixer.py
git commit -m "feat(patchtsmixer): WindowedDataset for sliding context/target pairs"
```

---

## Task 2: `PatchTSMixerModel` — attributes and variant feature sets

**Files:**
- Modify: `src/models/dl/patchtsmixer.py`
- Modify: `tests/models/test_patchtsmixer.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/models/test_patchtsmixer.py`:

```python
from src.models.dl.patchtsmixer import PatchTSMixerModel


def test_patchtsmixer_model_attributes():
    m = PatchTSMixerModel(variant="nohijri", context_length=96)
    assert m.name == "patchtsmixer"
    assert m.needs_training is True
    assert m.supports_dynamic_covariates is True


def test_patchtsmixer_variant_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown variant"):
        PatchTSMixerModel(variant="nonsense", context_length=96)


def test_patchtsmixer_feature_set_per_variant():
    base = ["actual_load", "temp_c", "dewpoint_c", "wind_speed",
            "solar_rad", "temp_sq", "temp_above_35"]
    hijri = ["is_ramadan", "day_of_ramadan", "is_eid"]
    ablB  = ["ramadan_x_heatwave", "ramadan_x_temp_above_35"]

    m_nh = PatchTSMixerModel(variant="nohijri", context_length=96)
    m_h  = PatchTSMixerModel(variant="hijri", context_length=96)
    m_pb = PatchTSMixerModel(variant="hijri_plusB", context_length=96)

    assert m_nh.channels == base
    assert m_h.channels  == base + hijri
    assert m_pb.channels == base + hijri + ablB
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/models/test_patchtsmixer.py -v -k "model_attributes or variant_rejects or feature_set"
```

Expected: 3 failures with `ImportError: cannot import name 'PatchTSMixerModel'` or similar.

- [ ] **Step 3: Implement the class skeleton**

Append to `src/models/dl/patchtsmixer.py`:

```python
from typing import Literal

import pandas as pd


_BASE_CHANNELS = [
    "actual_load",          # y, MUST be column 0
    "temp_c", "dewpoint_c", "wind_speed", "solar_rad",
    "temp_sq", "temp_above_35",
]
_HIJRI_CHANNELS = ["is_ramadan", "day_of_ramadan", "is_eid"]
_ABLATION_B_CHANNELS = ["ramadan_x_heatwave", "ramadan_x_temp_above_35"]


def _channels_for_variant(variant: str) -> list[str]:
    if variant == "nohijri":
        return list(_BASE_CHANNELS)
    if variant == "hijri":
        return list(_BASE_CHANNELS) + list(_HIJRI_CHANNELS)
    if variant == "hijri_plusB":
        return list(_BASE_CHANNELS) + list(_HIJRI_CHANNELS) + list(_ABLATION_B_CHANNELS)
    raise ValueError(
        f"Unknown variant {variant!r}. Expected nohijri | hijri | hijri_plusB."
    )


class PatchTSMixerModel:
    """PatchTSMixer (HF transformers) trained from scratch on Turkish load.

    Conforms to src/models/base.py::Model protocol.
    """
    name = "patchtsmixer"
    supports_dynamic_covariates = True
    needs_training = True

    def __init__(
        self,
        variant: Literal["nohijri", "hijri", "hijri_plusB"] = "nohijri",
        context_length: int = 336,
        prediction_length: int = 24,
        patch_length: int = 16,
        patch_stride: int = 8,
        d_model: int = 128,
        num_layers: int = 3,
        expansion_factor: int = 2,
        dropout: float = 0.1,
        head_dropout: float = 0.1,
        max_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_steps: int = 500,
        early_stopping_patience: int = 10,
    ):
        self.variant = variant
        self.channels = _channels_for_variant(variant)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.d_model = d_model
        self.num_layers = num_layers
        self.expansion_factor = expansion_factor
        self.dropout = dropout
        self.head_dropout = head_dropout
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.early_stopping_patience = early_stopping_patience
        self._fitted_model = None
        self._fit_history_arr: np.ndarray | None = None  # train+val concat for predict()
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/models/test_patchtsmixer.py -v -k "model_attributes or variant_rejects or feature_set"
```

Expected: 3 passes.

- [ ] **Step 5: Commit**

```bash
git add src/models/dl/patchtsmixer.py tests/models/test_patchtsmixer.py
git commit -m "feat(patchtsmixer): PatchTSMixerModel class skeleton with variant feature sets"
```

---

## Task 3: `PatchTSMixerModel.fit()` — trains on synthetic data

**Files:**
- Modify: `src/models/dl/patchtsmixer.py`
- Modify: `tests/models/test_patchtsmixer.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/models/test_patchtsmixer.py`:

```python
def _synthetic_df(n: int = 600) -> pd.DataFrame:
    """Hourly synthetic with daily seasonality + temperature signal."""
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    t = np.arange(n)
    load = (
        30000.0
        + 5000.0 * np.sin(2 * np.pi * t / 24)
        + np.random.default_rng(0).normal(scale=300.0, size=n)
    ).astype(np.float32)
    temp = (15.0 + 10.0 * np.sin(2 * np.pi * t / 24)).astype(np.float32)
    return pd.DataFrame({
        "actual_load": load,
        "temp_c": temp,
        "dewpoint_c": 5.0,
        "wind_speed": 3.0,
        "solar_rad": 0.0,
        "temp_sq": (temp ** 2),
        "temp_above_35": 0.0,
        "is_ramadan": 0,
        "day_of_ramadan": 0,
        "is_eid": 0,
        "ramadan_x_heatwave": 0,
        "ramadan_x_temp_above_35": 0.0,
        "regime": "Normal",
    }, index=idx)


def test_patchtsmixer_fit_runs_one_epoch():
    df = _synthetic_df(n=600)
    train = df.iloc[:400]
    val   = df.iloc[400:550]
    m = PatchTSMixerModel(
        variant="nohijri",
        context_length=48, prediction_length=24,
        patch_length=8, patch_stride=4,
        d_model=32, num_layers=1, expansion_factor=2,
        max_epochs=1, batch_size=16, warmup_steps=5,
        early_stopping_patience=10,
    )
    m.fit(train, val, hijri=False, seed=0)
    assert m._fitted_model is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/models/test_patchtsmixer.py::test_patchtsmixer_fit_runs_one_epoch -v
```

Expected: `AttributeError: 'PatchTSMixerModel' object has no attribute 'fit'`

- [ ] **Step 3: Add module-level helpers + extend `PatchTSMixerModel` with `fit()`**

Add these imports at the top of `src/models/dl/patchtsmixer.py` (alongside the existing `numpy/torch/Dataset` imports):

```python
import random
import tempfile
from pathlib import Path

from transformers import (
    EarlyStoppingCallback,
    PatchTSMixerConfig,
    PatchTSMixerForPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)
```

Add these module-level helpers (after `_channels_for_variant`, before the class):

```python
def _df_to_array(df: pd.DataFrame, channels: list[str]) -> np.ndarray:
    """Return (T, C) float32 array of channels from df, in the given order."""
    missing = [c for c in channels if c not in df.columns]
    if missing:
        raise KeyError(f"DataFrame missing channels: {missing}")
    return df[channels].to_numpy(dtype=np.float32, copy=True)


def _compute_mae(eval_pred) -> dict[str, float]:
    """HF Trainer compute_metrics: returns MAE on the y-channel.

    eval_pred.predictions: (N, H, 1) from PatchTSMixerForPredictionOutput
    eval_pred.label_ids:   (N, H, 1) — future_values
    """
    preds = np.asarray(eval_pred.predictions)
    labels = np.asarray(eval_pred.label_ids)
    if preds.ndim > labels.ndim:
        # Some HF heads wrap in extra tuple; take first element
        preds = preds[0]
    mae = float(np.mean(np.abs(preds.reshape(-1) - labels.reshape(-1))))
    return {"mae": mae}
```

Add a `_build_model()` private method and a `fit()` method to the existing `PatchTSMixerModel` class (do NOT redefine the class — just add these two methods to the body):

```python
    def _build_model(self) -> "PatchTSMixerForPrediction":
        config = PatchTSMixerConfig(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            patch_length=self.patch_length,
            patch_stride=self.patch_stride,
            num_input_channels=len(self.channels),
            d_model=self.d_model,
            num_layers=self.num_layers,
            expansion_factor=self.expansion_factor,
            dropout=self.dropout,
            head_dropout=self.head_dropout,
            mode="mix_channel",
            scaling="std",
            prediction_channel_indices=[0],  # y is channel 0
            loss="mse",
        )
        return PatchTSMixerForPrediction(config)

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        hijri: bool,
        seed: int,
    ) -> None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        train_arr = _df_to_array(train_df, self.channels)
        val_arr   = _df_to_array(val_df,   self.channels)
        train_ds = WindowedDataset(train_arr, self.context_length, self.prediction_length)
        val_ds   = WindowedDataset(val_arr,   self.context_length, self.prediction_length)

        model = self._build_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = TrainingArguments(
                output_dir=str(Path(tmpdir) / "out"),
                num_train_epochs=self.max_epochs,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size * 4,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                warmup_steps=self.warmup_steps,
                lr_scheduler_type="cosine",
                max_grad_norm=1.0,
                bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                eval_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=1,
                load_best_model_at_end=True,
                metric_for_best_model="mae",
                greater_is_better=False,
                logging_steps=50,
                report_to="none",
                seed=seed,
                data_seed=seed,
                dataloader_num_workers=0,
            )
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                compute_metrics=_compute_mae,
                callbacks=[EarlyStoppingCallback(
                    early_stopping_patience=self.early_stopping_patience,
                )],
            )
            trainer.train()
            # load_best_model_at_end=True ensures trainer.model is the
            # best-val-MAE checkpoint after train() returns.
            self._fitted_model = trainer.model.eval()

        # Concat train + val so predict() can grab context windows that
        # straddle the train/val/test boundaries.
        self._fit_history_arr = np.concatenate([train_arr, val_arr], axis=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/models/test_patchtsmixer.py::test_patchtsmixer_fit_runs_one_epoch -v -s
```

Expected: passes (~30s). The `-s` flag prints training logs so you can see one epoch ran.

- [ ] **Step 5: Verify earlier tests still pass**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/models/test_patchtsmixer.py -v
```

Expected: all 5 pass.

- [ ] **Step 6: Commit**

```bash
git add src/models/dl/patchtsmixer.py tests/models/test_patchtsmixer.py
git commit -m "feat(patchtsmixer): fit() trains via HF Trainer with val-MAE early stop"
```

---

## Task 4: `PatchTSMixerModel.predict()` — per-τ batched inference

**Files:**
- Modify: `src/models/dl/patchtsmixer.py`
- Modify: `tests/models/test_patchtsmixer.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/models/test_patchtsmixer.py`:

```python
def test_patchtsmixer_predict_returns_unified_schema():
    df = _synthetic_df(n=800)
    train = df.iloc[:500]
    val   = df.iloc[500:650]
    test  = df.iloc[650:]
    m = PatchTSMixerModel(
        variant="nohijri",
        context_length=48, prediction_length=24,
        patch_length=8, patch_stride=4,
        d_model=32, num_layers=1, expansion_factor=2,
        max_epochs=1, batch_size=16, warmup_steps=5,
    )
    m.fit(train, val, hijri=False, seed=0)
    out = m.predict(test)
    assert {"y_true", "y_pred", "regime"} <= set(out.columns)
    assert out["y_pred"].notna().all()
    # Index is the subset of test τ values where a full L-hour past window
    # is available (i.e., τ - 24 - L lies in train+val+test history).
    assert len(out) > 0
    assert out.index.is_monotonic_increasing
    # y_block must be a 24-vector for each row
    if "y_block" in out.columns:
        assert all(len(b) == 24 for b in out["y_block"].head(5))
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/models/test_patchtsmixer.py::test_patchtsmixer_predict_returns_unified_schema -v
```

Expected: `AttributeError: 'PatchTSMixerModel' object has no attribute 'predict'`

- [ ] **Step 3: Implement `predict()`**

Append to the `PatchTSMixerModel` class body in `src/models/dl/patchtsmixer.py`:

```python
    def predict(
        self,
        test_df: pd.DataFrame,
        context_length: int | None = None,
    ) -> pd.DataFrame:
        """Per-τ day-ahead inference.

        For each test τ, build the past-L window ending at τ-24, forward through
        the model, take the 24th forecast step as y_pred[τ]. The full 24-step
        block is also saved as y_block[τ] for the per-horizon analysis.
        """
        if self._fitted_model is None:
            raise RuntimeError("Call fit() before predict().")
        L = context_length or self.context_length
        H = self.prediction_length
        assert L == self.context_length, (
            f"predict() called with L={L} but model was fit at L={self.context_length}"
        )

        test_arr = _df_to_array(test_df, self.channels)
        # Concat fit-time history with test so τ < first-test + L + 24 can still
        # form a valid past window.
        full_arr = np.concatenate([self._fit_history_arr, test_arr], axis=0)
        history_len = len(self._fit_history_arr)  # offset of test row 0 in full_arr

        # For each test τ (index i_test in test_df), past window is full_arr
        # [history_len + i_test - 24 - L : history_len + i_test - 24]. Drop τ
        # values where this window would go negative.
        n_test = len(test_df)
        valid_mask = np.zeros(n_test, dtype=bool)
        past_starts = np.zeros(n_test, dtype=np.int64)
        for i in range(n_test):
            start = history_len + i - 24 - L
            if start >= 0:
                valid_mask[i] = True
                past_starts[i] = start
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) == 0:
            return pd.DataFrame(columns=["y_true", "y_pred", "regime"])

        device = next(self._fitted_model.parameters()).device
        batch_size = 256
        y_pred = np.empty(len(valid_idx), dtype=np.float32)
        y_block = np.empty((len(valid_idx), H), dtype=np.float32)

        self._fitted_model.eval()
        with torch.no_grad():
            for b_start in range(0, len(valid_idx), batch_size):
                b_idx = valid_idx[b_start : b_start + batch_size]
                past = np.stack(
                    [full_arr[past_starts[i] : past_starts[i] + L] for i in b_idx],
                    axis=0,
                )  # (B, L, C)
                past_t = torch.from_numpy(past).to(device)
                outputs = self._fitted_model(past_values=past_t)
                # PatchTSMixerForPredictionOutput.prediction_outputs: (B, H, 1)
                pred = outputs.prediction_outputs.detach().float().cpu().numpy()
                pred = pred.reshape(pred.shape[0], H)  # (B, H)
                y_block[b_start : b_start + len(b_idx)] = pred
                y_pred[b_start : b_start + len(b_idx)] = pred[:, -1]  # horizon-24

        out_idx = test_df.index[valid_idx]
        if "regime" in test_df.columns:
            regimes = test_df["regime"].values[valid_idx]
        else:
            regimes = np.full(len(valid_idx), "Normal", dtype=object)
        out = pd.DataFrame({
            "y_true": test_df["actual_load"].values[valid_idx],
            "y_pred": y_pred,
            "regime": regimes,
            "y_block": [list(map(float, row)) for row in y_block],
        }, index=out_idx)
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/models/test_patchtsmixer.py::test_patchtsmixer_predict_returns_unified_schema -v -s
```

Expected: passes (~30s including 1-epoch fit).

- [ ] **Step 5: Run full test file**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/models/test_patchtsmixer.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/models/dl/patchtsmixer.py tests/models/test_patchtsmixer.py
git commit -m "feat(patchtsmixer): predict() per-tau batched inference with y_block"
```

---

## Task 5: `src/models/dl/__init__.py` re-export

**Files:**
- Modify: `src/models/dl/__init__.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/models/test_patchtsmixer.py`:

```python
def test_patchtsmixer_reexported_from_dl_package():
    from src.models.dl import PatchTSMixerModel as Re
    assert Re is PatchTSMixerModel
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/models/test_patchtsmixer.py::test_patchtsmixer_reexported_from_dl_package -v
```

Expected: `ImportError: cannot import name 'PatchTSMixerModel' from 'src.models.dl'`

- [ ] **Step 3: Update `__init__.py`**

Replace contents of `src/models/dl/__init__.py` with:

```python
"""Deep-learning baselines."""
from src.models.dl.patchtsmixer import PatchTSMixerModel

__all__ = ["PatchTSMixerModel"]
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/models/test_patchtsmixer.py::test_patchtsmixer_reexported_from_dl_package -v
```

Expected: passes.

- [ ] **Step 5: Commit**

```bash
git add src/models/dl/__init__.py tests/models/test_patchtsmixer.py
git commit -m "feat(patchtsmixer): re-export PatchTSMixerModel from src.models.dl"
```

---

## Task 6: CLI runner

**Files:**
- Create: `scripts/run_patchtsmixer.py`

- [ ] **Step 1: Implement the CLI**

Create `scripts/run_patchtsmixer.py`:

```python
"""Run PatchTSMixer on the v2 test set and save predictions to parquet.

Usage:
    .venv/Scripts/python.exe scripts/run_patchtsmixer.py --variant nohijri --context-length 336 --seed 42
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.evaluation.predictions_io import write_predictions
from src.models.dl import PatchTSMixerModel


ROOT = Path(__file__).resolve().parents[1]
V2_CSV = ROOT / "data" / "processed" / "final_training_set_v2.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True,
                        choices=["nohijri", "hijri", "hijri_plusB"])
    parser.add_argument("--context-length", type=int, required=True,
                        choices=[96, 168, 336, 720])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print(f"[1/5] Loading v2 dataset ...")
    df = pd.read_csv(V2_CSV, parse_dates=["timestamp"]).set_index("timestamp")
    df.index = (df.index.tz_convert("UTC")
                if df.index.tz is not None else df.index.tz_localize("UTC"))
    df = df.dropna(subset=["y_lag_336h", "y_roll168_mean"])

    train = df.loc["2018":"2022"]
    val   = df.loc["2023"]
    test  = df.loc["2024-01-01":"2025-03-31"]
    print(f"      train={len(train):,}  val={len(val):,}  test={len(test):,}")

    print(f"[2/5] Instantiating patchtsmixer variant={args.variant} L={args.context_length} ...")
    model = PatchTSMixerModel(
        variant=args.variant,
        context_length=args.context_length,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
    )

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
        context_length=args.context_length, seed=args.seed,
    )
    print(f"      -> {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the CLI on 1-day window with tiny config**

Run (only purpose is to verify the script wires up correctly — not a real run):
```bash
.venv/Scripts/python.exe -c "
import pandas as pd
from src.models.dl import PatchTSMixerModel
df = pd.read_csv('data/processed/final_training_set_v2.csv', parse_dates=['timestamp']).set_index('timestamp')
df.index = df.index.tz_convert('UTC') if df.index.tz is not None else df.index.tz_localize('UTC')
df = df.dropna(subset=['y_lag_336h', 'y_roll168_mean'])
m = PatchTSMixerModel(variant='nohijri', context_length=96, max_epochs=1, batch_size=32)
m.fit(df.loc['2018':'2018-06'], df.loc['2018-07':'2018-08'], hijri=False, seed=0)
out = m.predict(df.loc['2018-09-01':'2018-09-02'])
print(f'rows={len(out)} MAE={(out.y_true-out.y_pred).abs().mean():.1f}')
"
```

Expected: prints rows count and an MAE number; no crash. (MAE will be terrible — 1 epoch on partial year — but that's fine; this only validates the CLI path.)

- [ ] **Step 3: Commit**

```bash
git add scripts/run_patchtsmixer.py
git commit -m "feat(patchtsmixer): CLI runner"
```

---

## Task 7: L-probe — pick the best context length

**Files:** none (runtime executions)

Run each command, watch wall-clock and aggregate MAE. After all 4, pick the L with the lowest aggregate test MAE as `BEST_L` for Task 8.

- [ ] **Step 1: L=336 first (proposal default — sanity baseline)**

```bash
.venv/Scripts/python.exe scripts/run_patchtsmixer.py --variant nohijri --context-length 336 --seed 0
```

Expected wall-clock: ~15–25 min. After completion, evaluate:
```bash
.venv/Scripts/python.exe -c "
import pandas as pd
df = pd.read_parquet('data/predictions/patchtsmixer__nohijri__L336__seed0.parquet')
print(f'L=336  rows={len(df)}  agg MAE={(df.y_true-df.y_pred).abs().mean():.1f}')
for r in [\"Normal\",\"Ramadan\",\"Heatwave\"]:
    sub = df[df.regime==r]
    if len(sub): print(f'  {r:9s}: MAE={(sub.y_true-sub.y_pred).abs().mean():.1f}')
"
```

- [ ] **Step 2: L=168**

```bash
.venv/Scripts/python.exe scripts/run_patchtsmixer.py --variant nohijri --context-length 168 --seed 0
```

Expected wall-clock: ~10–18 min. Evaluate as in Step 1 with `L168`.

- [ ] **Step 3: L=96**

```bash
.venv/Scripts/python.exe scripts/run_patchtsmixer.py --variant nohijri --context-length 96 --seed 0
```

Expected wall-clock: ~8–15 min. Evaluate as above.

- [ ] **Step 4: L=720**

```bash
.venv/Scripts/python.exe scripts/run_patchtsmixer.py --variant nohijri --context-length 720 --seed 0
```

Expected wall-clock: ~30–45 min. If it OOMs, retry with `--batch-size 16`.
Evaluate as above.

- [ ] **Step 5: Pick best L and note it for Task 8**

Inspect all 4 aggregate MAEs. `BEST_L = the L with lowest aggregate MAE`.
Tie-breaking: if two L values are within 5 MW, prefer the smaller L (faster
training in Task 8).

Record `BEST_L` in a note for yourself. Common-sense expectation: 168 or 336.

- [ ] **Step 6: Commit the 4 L-probe parquets**

```bash
git add data/predictions/patchtsmixer__nohijri__L*.parquet
git commit -m "feat(patchtsmixer): L-probe parquets at L in {96,168,336,720}, seed 0"
```

---

## Task 8: Headline grid — 3 variants × 5 seeds at BEST_L

**Files:** none (runtime executions). Replace `BEST_L` below with the value chosen in Task 7 Step 5.

- [ ] **Step 1: Run nohijri × 5 seeds**

```bash
for SEED in 42 43 44 45 46; do
  .venv/Scripts/python.exe scripts/run_patchtsmixer.py --variant nohijri --context-length BEST_L --seed $SEED
done
```

(On PowerShell: `foreach ($s in 42,43,44,45,46) { .venv\Scripts\python.exe scripts\run_patchtsmixer.py --variant nohijri --context-length BEST_L --seed $s }`)

Expected wall-clock: 5 × per-run time (e.g., 5×20min = 100min at L=336).
Run unattended overnight or in background.

- [ ] **Step 2: Run hijri × 5 seeds**

```bash
for SEED in 42 43 44 45 46; do
  .venv/Scripts/python.exe scripts/run_patchtsmixer.py --variant hijri --context-length BEST_L --seed $SEED
done
```

- [ ] **Step 3: Run hijri_plusB × 5 seeds**

```bash
for SEED in 42 43 44 45 46; do
  .venv/Scripts/python.exe scripts/run_patchtsmixer.py --variant hijri_plusB --context-length BEST_L --seed $SEED
done
```

- [ ] **Step 4: Verify all 15 parquets exist**

```bash
ls -la data/predictions/patchtsmixer__*__LBEST_L__seed4*.parquet
```

Expected: 15 files (3 variants × 5 seeds).

- [ ] **Step 5: Commit the 15 headline parquets**

```bash
git add data/predictions/patchtsmixer__*__LBEST_L__seed4*.parquet
git commit -m "feat(patchtsmixer): headline grid (3 variants × 5 seeds) at L=BEST_L"
```

---

## Task 9: Smoke tests + pytest green

**Files:**
- Modify: `tests/test_smoke_pipeline.py`

- [ ] **Step 1: Add classical-style smoke parametrized test for the headline grid**

Insert into `tests/test_smoke_pipeline.py` (after the classical block; replace `BEST_L` with the chosen L):

```python
# Plan 5: PatchTSMixer headline grid.
PATCHTSMIXER_BEST_L = BEST_L  # value chosen in Plan 5 Task 7 Step 5
PATCHTSMIXER_RUNS = [
    (variant, seed)
    for variant in ["nohijri", "hijri", "hijri_plusB"]
    for seed in [42, 43, 44, 45, 46]
]


@pytest.mark.parametrize("variant,seed", PATCHTSMIXER_RUNS)
def test_patchtsmixer_prediction_exists(variant, seed):
    p = PRED_DIR / f"patchtsmixer__{variant}__L{PATCHTSMIXER_BEST_L}__seed{seed}.parquet"
    assert p.exists(), (
        f"Missing {p}. Re-run "
        f"scripts/run_patchtsmixer.py --variant {variant} "
        f"--context-length {PATCHTSMIXER_BEST_L} --seed {seed}."
    )
    df = pd.read_parquet(p)
    assert {"y_true", "y_pred", "regime"} <= set(df.columns)
    assert df["y_pred"].notna().all()
    assert len(df) > 5000  # ~10k after dropping τ with insufficient context
```

Also add a separate parametrized test for the 4 L-probe runs:

```python
PATCHTSMIXER_L_PROBE = [96, 168, 336, 720]


@pytest.mark.parametrize("L", PATCHTSMIXER_L_PROBE)
def test_patchtsmixer_l_probe_exists(L):
    p = PRED_DIR / f"patchtsmixer__nohijri__L{L}__seed0.parquet"
    assert p.exists(), (
        f"Missing {p}. Re-run "
        f"scripts/run_patchtsmixer.py --variant nohijri "
        f"--context-length {L} --seed 0."
    )
    df = pd.read_parquet(p)
    assert "y_pred" in df.columns
    assert df["y_pred"].notna().all()
    assert len(df) > 5000
```

- [ ] **Step 2: Run smoke tests**

```bash
.venv/Scripts/python.exe -m pytest tests/test_smoke_pipeline.py -v -k "patchtsmixer"
```

Expected: 19 tests pass.

- [ ] **Step 3: Run full pytest**

```bash
.venv/Scripts/python.exe -m pytest -q
```

Expected: prior 119 + 6 new wrapper tests + 19 new smoke tests = 144 passed.

- [ ] **Step 4: Commit**

```bash
git add tests/test_smoke_pipeline.py
git commit -m "test(patchtsmixer): parquet existence smoke checks for 19 runs"
```

---

## Task 10: Results doc

**Files:**
- Create: `docs/patchtsmixer_baseline.md`

- [ ] **Step 1: Compute headline metrics and DM tests**

Run the following analyzer (it produces the table rows and DM stats you'll paste into the doc):

```bash
.venv/Scripts/python.exe -c "
import pandas as pd, numpy as np
from src.evaluation.dm_test import dm_test, holm_bonferroni

BEST_L = BEST_L  # paste numeric value
SEEDS = [42, 43, 44, 45, 46]

def load(variant, seed, L=BEST_L):
    return pd.read_parquet(f'data/predictions/patchtsmixer__{variant}__L{L}__seed{seed}.parquet')

def regime_mae(d, regime):
    sub = d[d.regime == regime]
    return (sub.y_true - sub.y_pred).abs().mean() if len(sub) else float('nan')

rows = []
for v in ['nohijri', 'hijri', 'hijri_plusB']:
    seed_dfs = [load(v, s) for s in SEEDS]
    aggs = [(d.y_true - d.y_pred).abs().mean() for d in seed_dfs]
    rmses = [float(np.sqrt(((d.y_true - d.y_pred) ** 2).mean())) for d in seed_dfs]
    medians_d = seed_dfs[2]  # seed 44 = median seed
    rows.append({
        'variant': v,
        'agg_mae_mean': np.mean(aggs), 'agg_mae_std': np.std(aggs),
        'agg_rmse_mean': np.mean(rmses), 'agg_rmse_std': np.std(rmses),
        'normal_seed44': regime_mae(medians_d, 'Normal'),
        'ramadan_seed44': regime_mae(medians_d, 'Ramadan'),
        'heatwave_seed44': regime_mae(medians_d, 'Heatwave'),
    })

print('Headline:')
for r in rows:
    print(r)

# DM tests vs LGBM-hijri (seed 44)
lgbm_h = pd.read_parquet('data/predictions/lgbm__hijri__seed44.parquet')
print()
print('DM vs LGBM-hijri (seed 44):')
candidates = [('hijri', 44), ('nohijri', 44)]
for v, s in candidates:
    d = load(v, s)
    merged = lgbm_h[['y_true','y_pred']].join(d[['y_pred']].rename(columns={'y_pred':'y_pred_b'}), how='inner')
    stat, p = dm_test(merged.y_true.values, merged.y_pred.values, merged.y_pred_b.values, h=24, loss='mae')
    print(f'  LGBM-hijri vs patchtsmixer-{v}: stat={stat:.2f} p={p:.2e} n={len(merged)}')

# Within-PatchTSMixer nohijri vs hijri (seed 44)
print()
da = load('nohijri', 44)
db = load('hijri', 44)
merged = da[['y_true','y_pred']].rename(columns={'y_pred':'y_pred_a'}).join(db[['y_pred']].rename(columns={'y_pred':'y_pred_b'}), how='inner')
stat, p = dm_test(merged.y_true.values, merged.y_pred_a.values, merged.y_pred_b.values, h=24, loss='mae')
print(f'Within patchtsmixer nohijri vs hijri: stat={stat:.2f} p={p:.2e}')
"
```

Save the printed output — you'll paste it into the doc next.

- [ ] **Step 2: Write the doc**

Create `docs/patchtsmixer_baseline.md` with this template — replace `BEST_L` and the metric values from Step 1:

````markdown
# PatchTSMixer Baseline — Plan 5

Deep-learning baseline. Substitutes `transformers.models.patchtsmixer.PatchTSMixerForPrediction` (cross-channel-mixing variant) for vanilla PatchTST, because HuggingFace `PatchTSTForPrediction` is channel-independent by design — Hijri/weather channels would not actually condition the y forecast in that architecture. See [`docs/superpowers/specs/2026-05-14-patchtsmixer-baseline-design.md`](superpowers/specs/2026-05-14-patchtsmixer-baseline-design.md).

## Setup

- Architecture: PatchTSMixer with patch=16/stride=8, d_model=128, num_layers=3, expansion_factor=2, dropout=0.1, mode="mix_channel", scaling="std", prediction_length=24, prediction_channel_indices=[0]
- Best context length (from L-probe): **L=BEST_L**
- Training: AdamW lr=1e-4, weight_decay=1e-2, cosine schedule with 500 warmup steps, batch 32, max 100 epochs, bf16, early stop on val MAE patience 10
- Test window: 2024-01-01 .. 2025-03-31 (10,944 rows; minus τ values dropped for insufficient context)
- Variants: nohijri / hijri / hijri_plusB. Seeds: 42–46 (5 seeds; median seed = 44 for headline)

## L-probe (nohijri × seed 0)

| L   | Aggregate MAE | Normal | Ramadan | Heatwave | Wall-clock |
|-----|---------------|--------|---------|----------|-----------:|
|  96 | _from probe_  | _x_    | _x_     | _x_      | _x_ min    |
| 168 | _from probe_  | _x_    | _x_     | _x_      | _x_ min    |
| 336 | _from probe_  | _x_    | _x_     | _x_      | _x_ min    |
| 720 | _from probe_  | _x_    | _x_     | _x_      | _x_ min    |

Best: **L=BEST_L**.

## Headline metrics (5-seed mean ± std; per-regime at median seed 44)

| Variant       | Agg MAE (mean ± std) | Agg RMSE (mean ± std) | Normal | Ramadan | Heatwave |
|---------------|----------------------|------------------------|--------|---------|----------|
| nohijri       | _from analyzer_      | _from analyzer_        | _x_    | _x_     | _x_      |
| hijri         | _from analyzer_      | _from analyzer_        | _x_    | _x_     | _x_      |
| hijri_plusB   | _from analyzer_      | _from analyzer_        | _x_    | _x_     | _x_      |

## Diebold-Mariano tests

Aggregate, HAC h=24, Holm-Bonferroni adjusted across the comparisons listed below. **DM > 0 means model B has lower loss.**

| Comparison (A vs B)                       | Stat | Raw p | Verdict |
|--------------------------------------------|------|-------|---------|
| LGBM-hijri vs PatchTSMixer-hijri (seed 44) | _x_ | _x_ | _x_ |
| LGBM-hijri vs PatchTSMixer-nohijri (seed 44) | _x_ | _x_ | _x_ |
| Within-PatchTSMixer: nohijri vs hijri (seed 44) | _x_ | _x_ | _x_ |

## hijri_plusB is structurally inactive (again)

As with SARIMAX in Plan 4, the Ablation B features (`ramadan_x_heatwave`, `ramadan_x_temp_above_35`) are constant zero in train/val/test. PatchTSMixer hijri_plusB predictions are statistically indistinguishable from hijri (DM stat ≈ 0, p ≈ 1). Documented for proposal alignment; the two parquets exist but should be read as one variant.

## Files

- `data/predictions/patchtsmixer__nohijri__L{96,168,336,720}__seed0.parquet` (L-probe)
- `data/predictions/patchtsmixer__{nohijri,hijri,hijri_plusB}__LBEST_L__seed{42,43,44,45,46}.parquet` (headline grid, 15 files)

## Reproduction

```bash
# L-probe (1 seed × 4 L)
for L in 96 168 336 720; do
  .venv/Scripts/python.exe scripts/run_patchtsmixer.py --variant nohijri --context-length $L --seed 0
done

# Headline grid (3 variants × 5 seeds at BEST_L)
for VARIANT in nohijri hijri hijri_plusB; do
  for SEED in 42 43 44 45 46; do
    .venv/Scripts/python.exe scripts/run_patchtsmixer.py --variant $VARIANT --context-length BEST_L --seed $SEED
  done
done
```
````

- [ ] **Step 3: Commit the doc**

```bash
git add docs/patchtsmixer_baseline.md
git commit -m "docs(patchtsmixer): results doc (L-probe, headline grid, DM tests)"
```

---

## Task 11: Update headline tables and README

**Files:**
- Modify: `docs/tsfm_zero_shot_baseline.md`
- Modify: `README.md`

- [ ] **Step 1: Add PatchTSMixer rows to the aggregate table in `docs/tsfm_zero_shot_baseline.md`**

Open `docs/tsfm_zero_shot_baseline.md`. Find the aggregate-metrics table (currently has rows for Chronos, LightGBM, Time-MoE, TimesFM, MSTL+ETS, Moirai, SARIMAX). Insert PatchTSMixer rows in MAE-sorted order using the values from Task 10 Step 1.

Example insertion (paste actual numbers):

```markdown
| PatchTSMixer hijri (seed 44, L=BEST_L)        |   _xxx.x_ | _xxxx.x_ |
| PatchTSMixer nohijri (seed 44, L=BEST_L)      |   _xxx.x_ | _xxxx.x_ |
```

Similarly add to the per-regime table.

- [ ] **Step 2: Update README milestone**

In `README.md`, change:

```markdown
- [ ] Plan 5: PatchTST
```

to:

```markdown
- [x] Plan 5: PatchTSMixer (deep-learning baseline) — see [`docs/patchtsmixer_baseline.md`](docs/patchtsmixer_baseline.md)
```

Also update the `src/models/dl/` comment in the README's project-structure block:

```markdown
    dl/          # PatchTSMixer (Plan 5)
```

- [ ] **Step 3: Final full pytest**

```bash
.venv/Scripts/python.exe -m pytest -q
```

Expected: 144 passed (or 144 + however many new tests added in execution).

- [ ] **Step 4: Commit**

```bash
git add docs/tsfm_zero_shot_baseline.md README.md
git commit -m "docs(plan-5): add PatchTSMixer to headline + mark Plan 5 complete"
```

---

## Task 12: Finalize branch

**Files:** none (git only)

- [ ] **Step 1: Push branch**

```bash
git push -u origin plan-5-patchtst
```

- [ ] **Step 2: Open PR (or merge per project convention)**

Use `gh pr create` (the user merged Plan 4 themselves; default to opening a PR and letting the user decide):

```bash
gh pr create --title "Plan 5: PatchTSMixer baseline (deep learning, cross-channel mixing)" --body "$(cat <<'EOF'
## Summary

- Adds PatchTSMixer baseline trained from scratch on Turkish load (5-seed × 3-variant grid at the L picked from a 4-L probe).
- Substitutes HuggingFace PatchTSMixer for vanilla PatchTST so that Hijri/weather channels actually condition the y forecast via cross-channel mixing.
- Slots between LightGBM/TSFMs and classical baselines in the headline table.

Detail: docs/patchtsmixer_baseline.md.

## Test plan

- [x] tests/models/test_patchtsmixer.py (6 unit tests pass)
- [x] tests/test_smoke_pipeline.py (19 new parquet-existence checks pass)
- [x] Full pytest: 144 passed
- [x] L-probe parquets at L∈{96,168,336,720} exist
- [x] Headline grid (3 variants × 5 seeds at best L) parquets exist

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-check before merging

- All 144+ tests green
- 19 parquets present in `data/predictions/`
- `docs/patchtsmixer_baseline.md` exists with non-placeholder numbers
- `docs/tsfm_zero_shot_baseline.md` has PatchTSMixer rows in headline
- `README.md` Plan 5 marked complete
