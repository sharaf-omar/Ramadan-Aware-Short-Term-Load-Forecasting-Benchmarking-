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
        from chronos import ChronosBoltPipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self._pipeline = ChronosBoltPipeline.from_pretrained(
            self.checkpoint,
            device_map=device,
            torch_dtype=dtype,
        )

    def _forecast_batch(self, contexts: np.ndarray) -> np.ndarray:
        """contexts: (B, L) float. Returns (B, HORIZON) median forecast."""
        self._load()
        ctx_tensors = [torch.tensor(row, dtype=torch.float32) for row in contexts]

        all_blocks: list[np.ndarray] = []
        bs = self.batch_size
        for i in range(0, len(ctx_tensors), bs):
            batch = ctx_tensors[i : i + bs]
            quantiles, _mean = self._pipeline.predict_quantiles(
                context=batch,
                prediction_length=HORIZON,
                quantile_levels=[0.5],
            )
            # quantiles shape: (B, H, 1)
            block = quantiles[:, :, 0].float().cpu().numpy()
            all_blocks.append(block)
        return np.concatenate(all_blocks, axis=0)
