"""End-to-end smoke: v2 data exists, LGBM ran on all variants x seeds,
predictions parquets exist, and basic sanity checks pass."""
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "data" / "predictions"
V2_CSV = ROOT / "data" / "processed" / "final_training_set_v2.csv"


def test_v2_dataset_exists():
    assert V2_CSV.exists(), "Run `python -m src.data.build_v2_dataset` first."


def test_v2_dataset_test_window_no_nan_actual_load():
    df = pd.read_csv(V2_CSV, parse_dates=["timestamp"]).set_index("timestamp")
    df.index = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
    test_slice = df.loc["2024-01-01":"2025-03-31"]
    assert test_slice["actual_load"].isna().sum() == 0


@pytest.mark.parametrize("variant", ["nohijri", "hijri", "hijri_plusB"])
@pytest.mark.parametrize("seed", [42, 43, 44, 45, 46])
def test_lgbm_prediction_exists(variant, seed):
    p = PRED_DIR / f"lgbm__{variant}__seed{seed}.parquet"
    assert p.exists(), f"Missing {p}. Re-run notebooks/02_lgbm.ipynb."
    df = pd.read_parquet(p)
    assert "y_true" in df.columns
    assert "y_pred" in df.columns
    assert df["y_pred"].notna().all()
    assert len(df) > 5000  # sanity: should be ~10,800 test hours


# Plan 2: TSFM zero-shot baseline at L=336 (Time-MoE deferred to Plan 3).
TSFM_MODELS_L336 = [
    "chronos_bolt_base",
    "timesfm_2_5",
    "moirai_1_1_small",
]


@pytest.mark.parametrize("model_name", TSFM_MODELS_L336)
def test_tsfm_prediction_exists_L336(model_name):
    p = PRED_DIR / f"{model_name}__nohijri__L336__seed0.parquet"
    assert p.exists(), f"Missing {p}. Run scripts/run_tsfm.py --model <short> --context-length 336."
    df = pd.read_parquet(p)
    assert "y_true" in df.columns
    assert "y_pred" in df.columns
    assert "y_block" in df.columns
    assert df["y_pred"].notna().all()
    # Block length must be 24
    assert all(len(b) == 24 for b in df["y_block"].head(20))
    assert len(df) > 5000
