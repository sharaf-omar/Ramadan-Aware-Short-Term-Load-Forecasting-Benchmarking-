"""PatchTSMixer deep-learning baseline (proposal §4.2 substitute).

Substitutes HuggingFace's PatchTSMixer (cross-channel mixing variant) for
vanilla PatchTST so that Hijri/weather channels actually condition the
y forecast. See docs/superpowers/specs/2026-05-14-patchtsmixer-baseline-design.md.
"""
from __future__ import annotations

import random
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    EarlyStoppingCallback,
    PatchTSMixerConfig,
    PatchTSMixerForPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)


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


def _df_to_array(df: pd.DataFrame, channels: list[str]) -> np.ndarray:
    """Return (T, C) float32 array of channels from df, in the given order."""
    missing = [c for c in channels if c not in df.columns]
    if missing:
        raise KeyError(f"DataFrame missing channels: {missing}")
    return df[channels].to_numpy(dtype=np.float32, copy=True)


def _compute_mae(eval_pred) -> dict[str, float]:
    """HF Trainer compute_metrics: returns MAE on the y-channel.

    PatchTSMixerForPredictionOutput has multiple fields (prediction_outputs,
    last_hidden_state, ...). Trainer passes them as a tuple via
    eval_pred.predictions; we take the first element (prediction_outputs)
    which has shape (N, H, num_targets) = (N, H, 1).
    eval_pred.label_ids has shape (N, H, 1) from future_values.
    """
    preds = eval_pred.predictions
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    preds = np.asarray(preds)
    labels = np.asarray(eval_pred.label_ids)
    mae = float(np.mean(np.abs(preds.reshape(-1) - labels.reshape(-1))))
    return {"mae": mae}


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
        self._fitted_model: PatchTSMixerForPrediction | None = None
        self._fit_history_arr: np.ndarray | None = None  # train+val concat for predict()

    def _build_model(self) -> PatchTSMixerForPrediction:
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
        val_arr = _df_to_array(val_df, self.channels)
        train_ds = WindowedDataset(train_arr, self.context_length, self.prediction_length)
        val_ds = WindowedDataset(val_arr, self.context_length, self.prediction_length)

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
                # Tell Trainer which dict key holds the labels so it routes
                # future_values to EvalPrediction.label_ids instead of dropping
                # it.
                label_names=["future_values"],
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
        # Concat fit-time history with test so τ values near test start can
        # still form a valid past window (window crosses val/test boundary).
        full_arr = np.concatenate([self._fit_history_arr, test_arr], axis=0)
        history_len = len(self._fit_history_arr)

        # For each test τ (index i in test_df), past window is
        # full_arr[history_len + i - 24 - L : history_len + i - 24].
        # Drop τ values where this window would have a negative start.
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
