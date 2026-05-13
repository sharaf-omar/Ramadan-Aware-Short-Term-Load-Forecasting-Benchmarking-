"""Build final_training_set_v2.csv: leak-free t+24 + all engineered features.

Inputs:
    data/processed/epias_processed_final.csv  (from preprocess_epias.py)
    data/processed/weather_proxy.csv          (from spatial_weights.py)

Outputs:
    data/processed/final_training_set_v2.csv
    data/processed/final_training_set_v2.meta.json
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.features.calendar import add_calendar_features
from src.features.weather_nonlinear import add_weather_nonlinear
from src.features.regimes import label_regimes


ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

EPIAS_CSV = PROCESSED_DIR / "epias_processed_final.csv"
WEATHER_CSV = PROCESSED_DIR / "weather_proxy.csv"
SOUTHERN_TEMP_CSV = PROCESSED_DIR / "southern_temp.csv"
V2_OUTPUT_CSV = PROCESSED_DIR / "final_training_set_v2.csv"
V2_META_JSON = PROCESSED_DIR / "final_training_set_v2.meta.json"

TEST_START = "2024-01-01 00:00:00+00:00"
TEST_END = "2025-03-31 23:00:00+00:00"


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def build_v2() -> pd.DataFrame:
    """Build v2 dataset and write CSV + meta sidecar. Returns the DataFrame."""
    if not EPIAS_CSV.exists():
        raise FileNotFoundError(
            f"{EPIAS_CSV} missing - run `python -m src.data.preprocess_epias` first."
        )
    if not WEATHER_CSV.exists():
        raise FileNotFoundError(
            f"{WEATHER_CSV} missing - run `python -m src.data.spatial_weights` first."
        )

    epias = pd.read_csv(EPIAS_CSV, parse_dates=["timestamp"]).set_index("timestamp")
    if epias.index.tz is None:
        epias.index = epias.index.tz_localize("UTC")
    else:
        epias.index = epias.index.tz_convert("UTC")

    weather = pd.read_csv(WEATHER_CSV, parse_dates=["timestamp"]).set_index("timestamp")
    if weather.index.tz is None:
        weather.index = weather.index.tz_localize("UTC")
    else:
        weather.index = weather.index.tz_convert("UTC")

    if not SOUTHERN_TEMP_CSV.exists():
        raise FileNotFoundError(
            f"{SOUTHERN_TEMP_CSV} missing - generate via "
            f"`python -c \"from src.data.spatial_weights import build_southern_temp_series, PROCESSED_DIR; "
            f"build_southern_temp_series().reset_index().rename(columns={{'index':'timestamp'}}).to_csv(PROCESSED_DIR/'southern_temp.csv', index=False)\"`"
        )
    south = pd.read_csv(SOUTHERN_TEMP_CSV, parse_dates=["timestamp"]).set_index("timestamp")
    if south.index.tz is None:
        south.index = south.index.tz_localize("UTC")
    else:
        south.index = south.index.tz_convert("UTC")

    df = epias.join(weather, how="left").join(south, how="left")
    df[["temp_c", "dewpoint_c", "wind_speed", "solar_rad", "temp_c_south"]] = (
        df[["temp_c", "dewpoint_c", "wind_speed", "solar_rad", "temp_c_south"]].interpolate(
            method="time", limit=3
        )
    )

    df = add_calendar_features(df)
    df = add_weather_nonlinear(df)

    # Ramadan x calendar interactions (covariate features used by tree models).
    df["ramadan_x_hour_sin"] = df["is_ramadan"] * df["hour_sin"]
    df["ramadan_x_hour_cos"] = df["is_ramadan"] * df["hour_cos"]
    df["ramadan_x_weekend"] = df["is_ramadan"] * df["is_weekend"]

    # Ablation B interactions.
    df["ramadan_x_temp_above_35"] = df["is_ramadan"] * df["temp_above_35"]

    # Regime detection uses the southern-region temperature (35 C threshold,
    # >= 3 consecutive days). The pop-weighted national temp_c rarely crosses
    # 35 C so heatwave label would be empty on that signal.
    df["regime"] = label_regimes(df, temp_col="temp_c_south")
    is_hw_row = df["regime"].isin(["Heatwave", "Compound"]).astype(int)
    df["heatwave_x_temp"] = is_hw_row * df["temp_c"]
    df["ramadan_x_heatwave"] = df["is_ramadan"] * is_hw_row

    # Test-window NaN assertion.
    test_slice = df.loc[TEST_START:TEST_END]
    n_nan = test_slice["actual_load"].isna().sum()
    if n_nan > 0:
        last_valid = df["actual_load"].last_valid_index()
        print(
            f"[WARN] {n_nan} NaN actual_load rows in test window "
            f"({TEST_START} .. {TEST_END}). Last valid: {last_valid}"
        )

    df.to_csv(V2_OUTPUT_CSV)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "source_files": {
            EPIAS_CSV.name: _hash_file(EPIAS_CSV),
            WEATHER_CSV.name: _hash_file(WEATHER_CSV),
            SOUTHERN_TEMP_CSV.name: _hash_file(SOUTHERN_TEMP_CSV),
        },
        "regime_temp_col": "temp_c_south",
        "regime_temp_cities": [
            "Adana", "Şanlıurfa", "Gaziantep", "Diyarbakır",
            "Mersin", "Konya", "Antalya",
        ],
        "row_count": int(len(df)),
        "date_range": [str(df.index[0]), str(df.index[-1])],
        "heatwave_threshold_c": 35.0,
        "heatwave_min_run_days": 3,
        "test_window_nan_actual_load": int(n_nan),
        "feature_columns": sorted(df.columns.tolist()),
    }
    V2_META_JSON.write_text(json.dumps(meta, indent=2))
    print(f"[OK] wrote {V2_OUTPUT_CSV} ({len(df):,} rows)")
    print(f"[OK] wrote {V2_META_JSON}")
    return df


if __name__ == "__main__":
    build_v2()
