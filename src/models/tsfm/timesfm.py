"""TimesFM 2.5 zero-shot wrapper.

Model: google/timesfm-2.5-200m-pytorch (decoder-only patched Transformer).
Note: substituted TimesFM 2.5 for the proposal's 2.0 because the PyPI 1.0
package is broken (lingvo dep) and 2.5 is what installs from GitHub HEAD.
Architecturally the same family; documented in the report.
Supports dynamic covariates (deferred to Plan 3).
"""
from __future__ import annotations

import numpy as np
import torch

from ._adapter import TSFMBase, HORIZON


class TimesFMModel(TSFMBase):
    name = "timesfm_2_5"
    supports_dynamic_covariates = True

    def __init__(
        self,
        checkpoint: str = "google/timesfm-2.5-200m-pytorch",
        batch_size: int = 32,
    ):
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self._model = None
        self._compiled_max_context: int | None = None

    def _load(self, max_context: int) -> None:
        from timesfm import TimesFM_2p5_200M_torch, ForecastConfig

        if self._model is None:
            # Monkey-patch: huggingface_hub's ModelHubMixin.from_pretrained passes
            # `proxies` (and other download-related kwargs) through to __init__,
            # but TimesFM_2p5_200M_torch.__init__ doesn't accept them.
            original_init = TimesFM_2p5_200M_torch.__init__
            _hub_kwargs = {"proxies", "force_download", "resume_download", "token",
                           "cache_dir", "local_files_only", "revision",
                           "subfolder", "trust_remote_code"}

            def _patched_init(self, *args, **kwargs):
                for k in list(kwargs):
                    if k in _hub_kwargs:
                        kwargs.pop(k)
                return original_init(self, *args, **kwargs)

            TimesFM_2p5_200M_torch.__init__ = _patched_init
            try:
                self._model = TimesFM_2p5_200M_torch.from_pretrained(self.checkpoint)
            finally:
                TimesFM_2p5_200M_torch.__init__ = original_init

        # Recompile if context length changed (a single compiled model is
        # tied to a specific max_context; reusing it with smaller/larger
        # contexts silently gives wrong/stale outputs).
        if self._compiled_max_context != max_context:
            self._model.compile(
                ForecastConfig(
                    max_context=max_context,
                    max_horizon=HORIZON,
                    normalize_inputs=True,
                    use_continuous_quantile_head=False,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    per_core_batch_size=self.batch_size,
                )
            )
            self._compiled_max_context = max_context

    def _forecast_batch(self, contexts: np.ndarray) -> np.ndarray:
        """contexts: (B, L). Returns (B, HORIZON) point forecast."""
        L = contexts.shape[1]
        self._load(max_context=L)

        forecast_inputs = [row.astype(np.float32) for row in contexts]
        # forecast returns (point_forecast, quantile_forecast) where
        # point_forecast has shape (B, HORIZON).
        point_forecast, _quantiles = self._model.forecast(
            horizon=HORIZON,
            inputs=forecast_inputs,
        )
        return np.asarray(point_forecast)
