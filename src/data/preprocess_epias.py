"""EPIAS preprocessing -> epias_processed_final.csv with leak-free t+24 features.

Row convention: each output row is indexed by forecast time tau. Issuance time
is t = tau - 24. ALL lag and rolling features are built from y at indices <= t.

Run as: python -m src.data.preprocess_epias
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.hijri import add_hijri_features


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DOTENV_PATH = ROOT_DIR / ".env"
OUTPUT_CSV = PROCESSED_DIR / "epias_processed_final.csv"


def _fetch_buffer_2017() -> pd.DataFrame:
    """Fetch Dec 2017 buffer needed for early-2018 lag features.

    Imported lazily so test collection does not trigger EPTR2 authentication.
    """
    from eptr2 import EPTR2

    eptr = EPTR2(use_dotenv=True, dotenv_path=str(DOTENV_PATH), recycle_tgt=True)
    try:
        buf = eptr.call(
            "rt-cons", start_date="2017-12-01", end_date="2017-12-31"
        ).rename(columns={"consumption": "actual_load"})
    except Exception as exc:
        print(f"[WARN] Could not fetch 2017 buffer: {exc}. Lags in early 2018 will be NaN.")
        buf = pd.DataFrame()
    return buf


def build_lag_rolling_features(y: pd.Series) -> pd.DataFrame:
    """Build leak-free lag and rolling features for a y_{t+24} forecast.

    Each row tau holds features computed from y at indices <= tau-24 only.
    Concretely:
        y_lag_24h     = y[tau-24]                # value at issuance
        y_lag_48h     = y[tau-48]
        y_lag_168h    = y[tau-168]
        y_lag_336h    = y[tau-336]
        y_roll24_mean = mean(y[tau-47..tau-24])  # 24-value window ending at issuance
        y_roll24_std  = std (y[tau-47..tau-24])
        y_roll168_mean= mean(y[tau-191..tau-24])
        y_roll168_std = std (y[tau-191..tau-24])

    Parameters
    ----------
    y : hourly load Series with a UTC-aware DatetimeIndex.

    Returns
    -------
    DataFrame indexed identically to y with the columns above.
    Early rows where the longest window doesn't fit are NaN.
    """
    out = pd.DataFrame(index=y.index)
    out["y_lag_24h"] = y.shift(24)
    out["y_lag_48h"] = y.shift(48)
    out["y_lag_168h"] = y.shift(168)
    out["y_lag_336h"] = y.shift(336)

    # Anchor the rolling base at the issuance time (tau-24), then roll backward.
    issuance = y.shift(24)
    out["y_roll24_mean"] = issuance.rolling(window=24).mean()
    out["y_roll24_std"] = issuance.rolling(window=24).std()
    out["y_roll168_mean"] = issuance.rolling(window=168).mean()
    out["y_roll168_std"] = issuance.rolling(window=168).std()
    return out


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading EPIAS load CSV ...")
    df_main = pd.read_csv(
        RAW_DIR / "electricity_consumption_2018_2025.csv"
    ).rename(columns={"consumption": "actual_load"})

    print("[2/4] Fetching 2017 buffer ...")
    df_buf = _fetch_buffer_2017()

    print("[3/4] Concatenating, aligning timestamps ...")
    df = pd.concat([df_buf, df_main], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["date"], utc=True)
    df = (
        df.drop(columns=[c for c in ("date", "time") if c in df.columns])
          .sort_values("timestamp")
          .set_index("timestamp")
    )
    df["actual_load"] = df["actual_load"].interpolate(method="linear")

    print("[4/4] Building leak-free lag/rolling features ...")
    feat = build_lag_rolling_features(df["actual_load"])
    df = df.join(feat)

    df = add_hijri_features(df)

    df = df.loc[df.index >= "2018-01-01 00:00:00+00:00"].copy()

    df.to_csv(OUTPUT_CSV)
    print(f"[OK] wrote {OUTPUT_CSV} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
