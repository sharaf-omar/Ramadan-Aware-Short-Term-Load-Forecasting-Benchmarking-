"""Predictions parquet I/O with a stable filename convention.

Schema:
    timestamp (UTC, idx) | y_true | y_pred | regime | y_block (optional, list[float, 24])

Convention:
    data/predictions/<model>__<variant>__L<ctx>__seed<seed>.parquet
    (L<ctx> omitted when context_length is None - i.e., tabular models)
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


_DEFAULT_PRED_DIR = Path(__file__).resolve().parents[2] / "data" / "predictions"


def _pred_dir() -> Path:
    """Allow override via env var (used by tests)."""
    p = Path(os.environ.get("RAMADAN_PRED_DIR", str(_DEFAULT_PRED_DIR)))
    p.mkdir(parents=True, exist_ok=True)
    return p


def predictions_path(
    *,
    model: str,
    variant: str,
    context_length: int | None,
    seed: int,
) -> Path:
    """Canonical filename for a (model, variant, ctx, seed) tuple."""
    parts = [model, variant]
    if context_length is not None:
        parts.append(f"L{context_length}")
    parts.append(f"seed{seed}")
    fname = "__".join(parts) + ".parquet"
    return _pred_dir() / fname


def write_predictions(
    df: pd.DataFrame,
    *,
    model: str,
    variant: str,
    context_length: int | None,
    seed: int,
) -> Path:
    """Persist predictions to parquet under the canonical path."""
    required = {"y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"predictions DataFrame missing columns: {missing}")
    path = predictions_path(
        model=model, variant=variant, context_length=context_length, seed=seed,
    )
    df.to_parquet(path)
    return path


def read_predictions(
    *,
    model: str,
    variant: str,
    context_length: int | None,
    seed: int,
) -> pd.DataFrame:
    """Load predictions from the canonical path."""
    path = predictions_path(
        model=model, variant=variant, context_length=context_length, seed=seed,
    )
    return pd.read_parquet(path)
