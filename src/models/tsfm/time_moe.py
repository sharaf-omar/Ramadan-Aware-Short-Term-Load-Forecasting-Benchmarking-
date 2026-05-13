"""Time-MoE 200M zero-shot wrapper — DEFERRED to Plan 3.

Model: Maple728/TimeMoE-200M (Mixture-of-Experts decoder).
8GB VRAM forces 200M instead of "Large" from the proposal; documented in spec.
Channel-independent; no dynamic covariates inline.

Plan 2 status: NOT RUN. The model's bundled `ts_generation_mixin.py` and
`modeling_time_moe.py` (downloaded via trust_remote_code) use a pre-4.46
transformers Cache API that is incompatible with our pinned transformers
4.48.3 (which we need for chronos-forecasting 1.5.2). Patching one issue
exposes a deeper incompatibility in `prepare_inputs_for_generation`.

Plan 3 will revisit by either:
- Implementing manual autoregressive forward (skip transformers `generate`)
- Switching to an A100 stack where we can pin transformers ~4.40 + a Chronos
  version compatible with that range
- Using the Time-MoE-50M variant if it's been re-released with newer code

The wrapper class below is preserved as a placeholder; calling _load() will
raise NotImplementedError to make the deferral explicit.
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
        self._device = None
        self._dtype = None

    def _load(self) -> None:
        raise NotImplementedError(
            "Time-MoE-200M zero-shot inference is deferred to Plan 3. "
            "The model's bundled remote code is incompatible with the "
            "transformers 4.48.3 pin required by chronos-forecasting. "
            "See module docstring for details and follow-on plan."
        )

    def _forecast_batch(self, contexts: np.ndarray) -> np.ndarray:
        """contexts: (B, L). Returns (B, HORIZON) point forecast.

        Time-MoE per-series normalization (subtract mean, divide by std) before
        generation, then denormalize the forecast.
        """
        self._load()
        all_blocks: list[np.ndarray] = []
        bs = self.batch_size
        with torch.no_grad():
            for i in range(0, len(contexts), bs):
                batch = torch.tensor(
                    contexts[i : i + bs], dtype=torch.float32, device=self._device
                )
                mean = batch.mean(dim=1, keepdim=True)
                std = batch.std(dim=1, keepdim=True).clamp(min=1e-8)
                batch_norm = ((batch - mean) / std).to(self._dtype)
                # generate returns (B, L + HORIZON). Take the tail HORIZON.
                preds = self._model.generate(
                    inputs=batch_norm,
                    max_new_tokens=HORIZON,
                )
                tail = preds[:, -HORIZON:].float()
                denorm = tail * std + mean
                all_blocks.append(denorm.cpu().numpy())
        return np.concatenate(all_blocks, axis=0)
