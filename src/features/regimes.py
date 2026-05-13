"""4-regime labeling per proposal Table 3.

Heatwave := daily T_max >= 35 C for >= 3 consecutive days.
Compound := Ramadan AND Heatwave.
Ramadan  := Ramadan AND NOT Heatwave.
Normal   := otherwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


HEATWAVE_TEMP_THRESHOLD = 35.0  # degrees Celsius
HEATWAVE_MIN_RUN_DAYS = 3


def _compute_heatwave_days(daily_max: pd.Series) -> pd.Series:
    """Return a boolean series indexed by date: True iff day is part of a
    consecutive run of >= HEATWAVE_MIN_RUN_DAYS days at or above the temp
    threshold."""
    hot = daily_max >= HEATWAVE_TEMP_THRESHOLD
    run_id = (hot != hot.shift(fill_value=False)).cumsum()
    run_len = hot.groupby(run_id).transform("size")
    return hot & (run_len >= HEATWAVE_MIN_RUN_DAYS)


def label_regimes(
    df: pd.DataFrame,
    temp_col: str = "temp_c",
    ramadan_col: str = "is_ramadan",
) -> pd.Series:
    """Label each row with one of Normal/Ramadan/Heatwave/Compound.

    Parameters
    ----------
    df : DataFrame with UTC DatetimeIndex, a temperature column, and an
        is_ramadan column.
    temp_col : name of the hourly temperature column.
    ramadan_col : name of the binary Ramadan indicator column.

    Returns
    -------
    pd.Series of dtype 'object' aligned to df.index.
    """
    for col in (temp_col, ramadan_col):
        if col not in df.columns:
            raise KeyError(f"DataFrame missing required column {col!r}")

    daily_max = df.groupby(df.index.date)[temp_col].max()
    heatwave_day = _compute_heatwave_days(daily_max)
    hw_map = heatwave_day.to_dict()

    is_hw_row = pd.Series(
        [bool(hw_map.get(ts.date(), False)) for ts in df.index],
        index=df.index,
    )
    is_ram_row = df[ramadan_col].astype(bool)

    labels = np.where(
        is_ram_row & is_hw_row, "Compound",
        np.where(
            is_ram_row & ~is_hw_row, "Ramadan",
            np.where(
                ~is_ram_row & is_hw_row, "Heatwave",
                "Normal",
            ),
        ),
    )
    return pd.Series(labels, index=df.index, name="regime")
