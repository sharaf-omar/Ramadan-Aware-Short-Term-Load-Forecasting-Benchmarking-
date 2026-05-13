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
        num_samples: int = 20,
        patch_size: int = 32,
    ):
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.patch_size = patch_size
        self._module = None
        self._device = None

    def _load(self) -> None:
        if self._module is not None:
            return
        from uni2ts.model.moirai import MoiraiModule
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._module = MoiraiModule.from_pretrained(self.checkpoint).to(device).eval()
        self._device = device

    def _forecast_batch(self, contexts: np.ndarray) -> np.ndarray:
        """contexts: (B, L). Returns (B, HORIZON) median forecast across samples."""
        self._load()
        from uni2ts.model.moirai import MoiraiForecast

        L = contexts.shape[1]
        forecaster = MoiraiForecast(
            module=self._module,
            prediction_length=HORIZON,
            context_length=L,
            patch_size=self.patch_size,
            num_samples=self.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        ).to(self._device).eval()

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
                samples = forecaster(
                    past_target=pt,
                    past_observed_target=po,
                    past_is_pad=pp,
                )
                # samples last dim is target_dim=1; squeeze it.
                if samples.dim() == 4:
                    samples = samples.squeeze(-1)
                # samples now (B, num_samples, HORIZON). Median across samples.
                med = samples.float().median(dim=1).values
                all_blocks.append(med.cpu().numpy())
        return np.concatenate(all_blocks, axis=0)
