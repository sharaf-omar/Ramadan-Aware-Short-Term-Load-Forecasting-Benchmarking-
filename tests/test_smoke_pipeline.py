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


# Plan 3 full grid: 4 TSFMs × 4 L values, all nohijri.
TSFM_MODELS_FULL = [
    "chronos_bolt_base",
    "timesfm_2_5",
    "moirai_1_1_small",
    "time_moe_200m",
]
TSFM_LS = [96, 168, 336, 720]


@pytest.mark.parametrize("model_name", TSFM_MODELS_FULL)
@pytest.mark.parametrize("L", TSFM_LS)
def test_tsfm_prediction_exists_sweep(model_name, L):
    p = PRED_DIR / f"{model_name}__nohijri__L{L}__seed0.parquet"
    assert p.exists(), f"Missing {p}. Re-run Plan 3 Phase 2."
    df = pd.read_parquet(p)
    assert "y_true" in df.columns
    assert "y_pred" in df.columns
    assert "y_block" in df.columns
    assert df["y_pred"].notna().all()
    assert all(len(b) == 24 for b in df["y_block"].head(20))
    assert len(df) > 5000


# Plan 3 Ablation A: Hijri-covariate variants for the 2 covariate-capable
# TSFMs at L=336.
COVARIATE_TSFMS = ["timesfm_2_5", "moirai_1_1_small"]


@pytest.mark.parametrize("model_name", COVARIATE_TSFMS)
def test_tsfm_hijri_prediction_exists(model_name):
    p = PRED_DIR / f"{model_name}__hijri__L336__seed0.parquet"
    assert p.exists(), f"Missing {p}. Re-run Plan 3 Phase 3."
    df = pd.read_parquet(p)
    assert "y_true" in df.columns
    assert "y_pred" in df.columns
    assert df["y_pred"].notna().all()
    assert len(df) > 5000


# Plan 4: classical baselines.
CLASSICAL_RUNS = [
    ("mstl_ets", "nohijri"),
    ("mstl_ets", "hijri"),
    ("sarimax",  "nohijri"),
    ("sarimax",  "hijri"),
    ("sarimax",  "hijri_plusB"),
]


@pytest.mark.parametrize("model_name,variant", CLASSICAL_RUNS)
def test_classical_prediction_exists(model_name, variant):
    p = PRED_DIR / f"{model_name}__{variant}__seed0.parquet"
    assert p.exists(), (
        f"Missing {p}. Re-run "
        f"scripts/run_classical.py --model {model_name} --variant {variant}."
    )
    df = pd.read_parquet(p)
    assert "y_true" in df.columns
    assert "y_pred" in df.columns
    assert "regime" in df.columns
    assert df["y_pred"].notna().all()
    assert len(df) > 5000


# Plan 7 sub-task: statistical appendix.
STATS_DIR = ROOT / "data" / "statistical_appendix"
STATS_DOC = ROOT / "docs" / "statistical_appendix.md"
STATS_CSVS = [
    "ci_table.csv",
    "dm_aggregate.csv",
    "dm_Normal.csv",
    "dm_Ramadan.csv",
    "dm_Heatwave.csv",
]


def test_statistical_appendix_doc_exists():
    assert STATS_DOC.exists(), (
        f"Missing {STATS_DOC}. Re-run scripts/build_statistical_appendix.py."
    )
    txt = STATS_DOC.read_text(encoding="utf-8")
    assert "# Statistical Appendix" in txt
    assert "## Bootstrap MAE confidence intervals" in txt
    assert "## Pairwise Diebold-Mariano tests" in txt


@pytest.mark.parametrize("name", STATS_CSVS)
def test_statistical_appendix_csv_exists(name):
    p = STATS_DIR / name
    assert p.exists(), (
        f"Missing {p}. Re-run scripts/build_statistical_appendix.py."
    )
    df = pd.read_csv(p)
    assert len(df) > 0
