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
