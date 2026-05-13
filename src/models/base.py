"""Model protocol - all model wrappers in src/models/* implement this."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Model(Protocol):
    """Common interface for every forecasting model in this benchmark.

    Concrete attributes
    -------------------
    name : str
        Stable identifier used in predictions filenames (e.g., "lgbm",
        "chronos_bolt_base", "patchtst").
    supports_dynamic_covariates : bool
        True iff the model accepts hour-aligned covariates over context+horizon.
        Used by ablation orchestration to decide between in-band covariates
        and post-hoc residual correction.
    needs_training : bool
        True for fit-then-predict models (LGBM, classical, PatchTST);
        False for zero-shot TSFMs.
    """
    name: str
    supports_dynamic_covariates: bool
    needs_training: bool

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        hijri: bool,
        seed: int,
    ) -> None:
        ...

    def predict(
        self,
        test_df: pd.DataFrame,
        context_length: int | None = None,
    ) -> pd.DataFrame:
        """Returns DataFrame with index=tau (UTC) and at least columns
        {y_true, y_pred}. Optional column y_block holds full 24-step block
        for block-forecasters."""
        ...
