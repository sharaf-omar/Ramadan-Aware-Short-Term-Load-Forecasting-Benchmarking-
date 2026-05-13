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
        self._compiled_max_context = None
        self._needs_backcast = False  # set True before _load when using covariates

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
        compile_key = (max_context, self._needs_backcast)
        if self._compiled_max_context != compile_key:
            self._model.compile(
                ForecastConfig(
                    max_context=max_context,
                    max_horizon=HORIZON,
                    normalize_inputs=True,
                    use_continuous_quantile_head=False,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    per_core_batch_size=self.batch_size,
                    return_backcast=self._needs_backcast,
                )
            )
            self._compiled_max_context = compile_key

    def _forecast_batch(self, contexts: np.ndarray) -> np.ndarray:
        """contexts: (B, L). Returns (B, HORIZON) point forecast."""
        L = contexts.shape[1]
        self._needs_backcast = False
        self._load(max_context=L)

        forecast_inputs = [row.astype(np.float32) for row in contexts]
        point_forecast, _quantiles = self._model.forecast(
            horizon=HORIZON,
            inputs=forecast_inputs,
        )
        return np.asarray(point_forecast)

    def _forecast_batch_with_covariates(
        self,
        contexts: np.ndarray,
        past_cov: np.ndarray,
        future_cov: np.ndarray,
    ) -> np.ndarray:
        """contexts: (B, L); past_cov: (B, L, C); future_cov: (B, H, C).
        Returns (B, HORIZON)."""
        L = contexts.shape[1]
        self._needs_backcast = True
        self._load(max_context=L)

        forecast_inputs = [row.astype(np.float32).tolist() for row in contexts]
        B, _, C = past_cov.shape
        # TimesFM expects each covariate as a list of length-(L+H) sequences.
        dynamic_numerical_covariates: dict[str, list] = {}
        for c in range(C):
            per_series = []
            for b in range(B):
                past = past_cov[b, :, c].astype(np.float32)
                fut = future_cov[b, :, c].astype(np.float32)
                per_series.append(np.concatenate([past, fut]).tolist())
            dynamic_numerical_covariates[f"cov{c}"] = per_series

        _timesfm_pred, xreg_pred = self._model.forecast_with_covariates(
            inputs=forecast_inputs,
            dynamic_numerical_covariates=dynamic_numerical_covariates,
            xreg_mode="xreg + timesfm",
            normalize_xreg_target_per_input=True,
            ridge=0.0,
        )
        # Empirically both `xreg_pred` and `timesfm_pred` are total forecasts
        # in MW scale; they should not be summed. `xreg_pred` is the
        # covariate-aware prediction (TimesFM + linear xreg correction); use
        # its mean across the quantile axis as the point estimate.
        out = np.zeros((B, HORIZON), dtype=np.float32)
        for b in range(B):
            xreg_b = np.asarray(xreg_pred[b], dtype=np.float32)
            if xreg_b.ndim == 2:
                point = xreg_b.mean(axis=-1)[:HORIZON]
            else:
                point = xreg_b.reshape(-1)[:HORIZON]
            out[b] = point
        return out
