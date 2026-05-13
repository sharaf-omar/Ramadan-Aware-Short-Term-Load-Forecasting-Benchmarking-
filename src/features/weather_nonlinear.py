"""Nonlinear weather feature transforms."""
from __future__ import annotations

import pandas as pd


def add_weather_nonlinear(df: pd.DataFrame, temp_col: str = "temp_c") -> pd.DataFrame:
    """Add temp_sq and temp_above_35.

    Parameters
    ----------
    df : DataFrame with a temperature column (default name 'temp_c').
    temp_col : name of the temperature column.
    """
    if temp_col not in df.columns:
        raise KeyError(f"DataFrame missing required column {temp_col!r}")
    out = df.copy()
    out["temp_sq"] = out[temp_col] ** 2
    out["temp_above_35"] = (out[temp_col] - 35.0).clip(lower=0.0)
    return out
