"""PatchTSMixer deep-learning baseline (proposal §4.2 substitute).

Substitutes HuggingFace's PatchTSMixer (cross-channel mixing variant) for
vanilla PatchTST so that Hijri/weather channels actually condition the
y forecast. See docs/superpowers/specs/2026-05-14-patchtsmixer-baseline-design.md.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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
        self._fitted_model = None
        self._fit_history_arr: np.ndarray | None = None  # train+val concat for predict()
