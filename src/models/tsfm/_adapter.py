"""Shared infrastructure for TSFM wrappers.

Provides:
- build_context_windows: slice y into (N, L) tensor for batched TSFM inference
- TSFMBase: abstract base class implementing the Model protocol
- HORIZON: the proposal-fixed forecast horizon (24 hours)
- ISSUANCE_OFFSET: the gap between issuance time t and forecast time tau
"""
from __future__ import annotations

import abc

import numpy as np
import pandas as pd


HORIZON = 24
ISSUANCE_OFFSET = 24  # tau = t + 24


def build_context_windows(
    y: pd.Series,
    forecast_times: pd.DatetimeIndex,
    context_length: int,
) -> np.ndarray:
    """Build (N, L) array of context windows for batched TSFM inference.

    For each forecast time tau in forecast_times, the row contains
    y[tau-24-L+1 .. tau-24] inclusive (L values ending at issuance time t=tau-24).
    """
    y_values = y.values
    y_index = y.index
    positions = y_index.get_indexer(forecast_times)
    if (positions == -1).any():
        missing = forecast_times[positions == -1]
        raise KeyError(
            f"{len(missing)} forecast times not present in y index. "
            f"First missing: {missing[0]}"
        )

    n = len(forecast_times)
    L = context_length
    out = np.full((n, L), np.nan, dtype=np.float64)

    for i, tau_pos in enumerate(positions):
        end = tau_pos - ISSUANCE_OFFSET  # inclusive end index in y
        start = end - L + 1
        if start < 0:
            continue
        out[i, :] = y_values[start : end + 1]
    return out


class TSFMBase(abc.ABC):
    """Common base for zero-shot TSFM wrappers implementing the Model protocol.

    Subclasses provide:
        name : str
        supports_dynamic_covariates : bool
        _forecast_batch(contexts: np.ndarray) -> np.ndarray

    where contexts is shape (B, L) and the returned array is shape (B, HORIZON).
    """
    needs_training = False  # zero-shot

    @abc.abstractmethod
    def _forecast_batch(self, contexts: np.ndarray) -> np.ndarray:
        """Forecast (B, HORIZON) given context windows (B, L)."""
        raise NotImplementedError

    def fit(self, train_df, val_df, hijri, seed) -> None:
        """Zero-shot models do not train. No-op."""
        return None

    def predict(
        self,
        test_df: pd.DataFrame,
        context_length: int | None = None,
    ) -> pd.DataFrame:
        if context_length is None:
            raise ValueError("TSFM models require context_length.")

        contexts = build_context_windows(
            test_df["actual_load"], test_df.index, context_length=context_length,
        )
        valid_mask = ~np.isnan(contexts).any(axis=1)
        valid_contexts = contexts[valid_mask]

        if len(valid_contexts) == 0:
            raise RuntimeError(
                "No rows in test_df have sufficient history for context_length="
                f"{context_length}. Got {len(test_df)} test rows total."
            )

        blocks = self._forecast_batch(valid_contexts)
        if blocks.shape != (len(valid_contexts), HORIZON):
            raise AssertionError(
                f"_forecast_batch returned {blocks.shape}, expected "
                f"({len(valid_contexts)}, {HORIZON})"
            )

        out = pd.DataFrame({
            "y_true": test_df["actual_load"].values[valid_mask],
            "y_pred": blocks[:, -1],  # the t+24 point
            "y_block": [b.tolist() for b in blocks],
            "regime": test_df["regime"].values[valid_mask],
        }, index=test_df.index[valid_mask])
        return out
