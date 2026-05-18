"""Per-regime metric stratification."""
from __future__ import annotations

import numpy as np
import pandas as pd
from .metrics import mae as mae_fn, rmse as rmse_fn, mape as mape_fn, mase as mase_fn


REGIMES = ("Normal", "Ramadan", "Heatwave", "Compound")


def evaluate_by_regime(
    y_true,
    y_pred,
    regimes: pd.Series,
    y_train,
    period: int = 168,
) -> pd.DataFrame:
    """Compute MAE, RMSE, MAPE, MASE per regime.

    Returns a DataFrame with columns: regime, n, mae, rmse, mape, mase.
    One row per regime in REGIMES (zero-count regimes get NaN metrics).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    regimes_arr = np.asarray(regimes)

    rows = []
    for r in REGIMES:
        mask = regimes_arr == r
        n = int(mask.sum())
        if n == 0:
            rows.append(
                dict(regime=r, n=0, mae=np.nan, rmse=np.nan, mape=np.nan, mase=np.nan)
            )
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        rows.append(dict(
            regime=r,
            n=n,
            mae=mae_fn(yt, yp),
            rmse=rmse_fn(yt, yp),
            mape=mape_fn(yt, yp),
            mase=mase_fn(yt, yp, y_train, period=period),
        ))
    return pd.DataFrame(rows)
