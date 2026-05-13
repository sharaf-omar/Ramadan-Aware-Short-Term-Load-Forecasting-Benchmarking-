from pathlib import Path
import pandas as pd
import pytest
from src.data.build_v2_dataset import V2_OUTPUT_CSV, V2_META_JSON


def test_v2_output_path_constants():
    assert V2_OUTPUT_CSV.name == "final_training_set_v2.csv"
    assert V2_META_JSON.name == "final_training_set_v2.meta.json"


def test_v2_columns_after_build_exists():
    """Run only if v2 already built - verifies column schema."""
    if not V2_OUTPUT_CSV.exists():
        pytest.skip("v2 dataset not built yet; run build_v2() first")

    df = pd.read_csv(V2_OUTPUT_CSV, nrows=10)
    required = [
        "timestamp", "actual_load",
        "y_lag_24h", "y_lag_48h", "y_lag_168h", "y_lag_336h",
        "y_roll24_mean", "y_roll24_std",
        "y_roll168_mean", "y_roll168_std",
        "temp_c", "dewpoint_c", "wind_speed", "solar_rad",
        "temp_sq", "temp_above_35",
        "hour", "day_of_week", "month",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        "is_weekend",
        "is_ramadan", "day_of_ramadan", "is_eid",
        "ramadan_x_hour_sin", "ramadan_x_hour_cos", "ramadan_x_weekend",
        "ramadan_x_heatwave", "ramadan_x_temp_above_35", "heatwave_x_temp",
        "regime",
    ]
    for col in required:
        assert col in df.columns, f"v2 dataset missing required column: {col}"
