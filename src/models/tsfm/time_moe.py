"""Time-MoE 200M zero-shot wrapper.

Model: Maple728/TimeMoE-200M (Mixture-of-Experts decoder).
Time-MoE has prediction heads for horizons [1, 8, 32, 64]. Plan 3 uses the
32-step head (smallest >= our HORIZON=24) in a single forward call, sliced to
24 steps. This avoids both:
  - the bundled generate() (broken on transformers 4.48.3 Cache API), and
  - a 24-step autoregressive loop (would take ~16h on RTX 4070 for the full
    test set).
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
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(device).eval()
        self._device = device
        self._dtype = dtype

    def _forecast_batch(self, contexts: np.ndarray) -> np.ndarray:
        """contexts: (B, L). Returns (B, HORIZON) point forecast.

        Uses the 32-step prediction head (lm_heads[2]) in a single forward call
        per batch, then slices to the first HORIZON=24 steps.
        """
        self._load()
        all_blocks: list[np.ndarray] = []
        bs = self.batch_size
        with torch.no_grad():
            for i in range(0, len(contexts), bs):
                batch = torch.tensor(
                    contexts[i : i + bs], dtype=torch.float32, device=self._device
                )
                # Per-series normalize over context.
                mean = batch.mean(dim=1, keepdim=True)
                std = batch.std(dim=1, keepdim=True).clamp(min=1e-8)
                seq = ((batch - mean) / std).to(self._dtype)  # (B, L)
                # max_horizon_length=32 -> picks the 32-step head and returns
                # logits shape (B, L, 32). We take the last position.
                out = self._model(input_ids=seq, max_horizon_length=32)
                logits = out.logits  # (B, L, 32)
                last_pos = logits[:, -1, :]  # (B, 32)
                block_norm = last_pos[:, :HORIZON].float()  # (B, HORIZON)
                denorm = block_norm * std + mean
                all_blocks.append(denorm.cpu().numpy())
        return np.concatenate(all_blocks, axis=0)
