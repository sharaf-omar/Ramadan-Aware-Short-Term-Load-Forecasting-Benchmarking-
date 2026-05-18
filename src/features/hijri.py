"""Hijri calendar features (is_ramadan, day_of_ramadan, is_eid).

Implementation note: hijridate's Gregorian->Hijri conversion uses calendar-day
granularity. Hijri days actually start at maghrib (sunset). Boundary hours may
be miscategorized by up to ~6h. This is acceptable per the proposal but should
be cited in the LGBM technical report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hijridate import Gregorian


def _hijri_tuple(ts: pd.Timestamp) -> tuple[int, int, int]:
    """Return (is_ramadan, is_eid, day_of_ramadan) for a single UTC timestamp.

    Computed against Turkey local calendar day (Europe/Istanbul) since the
    proposal grids the load data on Turkish national consumption.
    """
    local_ts = ts.tz_convert("Europe/Istanbul")
    h = Gregorian(local_ts.year, local_ts.month, local_ts.day).to_hijri()
    is_ramadan = int(h.month == 9)
    is_eid = int(h.month == 10 and h.day <= 3)
    day_of_ramadan = int(h.day) if h.month == 9 else 0
    return is_ramadan, is_eid, day_of_ramadan


def add_hijri_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_ramadan, is_eid, day_of_ramadan columns.

    Parameters
    ----------
    df : DataFrame with a UTC-aware DatetimeIndex.

    Returns
    -------
    A copy of df with three new int columns appended.
    """
    if df.index.tz is None:
        raise ValueError("DataFrame index must be timezone-aware (UTC).")

    out = df.copy()
    tuples = [_hijri_tuple(ts) for ts in out.index]
    arr = np.asarray(tuples, dtype=np.int8)
    out["is_ramadan"] = arr[:, 0]
    out["is_eid"] = arr[:, 1]
    out["day_of_ramadan"] = arr[:, 2]
    return out
