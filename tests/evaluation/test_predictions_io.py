import pandas as pd
from src.evaluation.predictions_io import (
    write_predictions, read_predictions, predictions_path,
)


def _sample_predictions(n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "y_true": list(range(n)),
        "y_pred": [v + 0.5 for v in range(n)],
        "regime": ["Normal"] * n,
    }, index=idx).rename_axis("timestamp")


def test_write_and_read_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("RAMADAN_PRED_DIR", str(tmp_path))
    df = _sample_predictions()
    p = write_predictions(
        df, model="lgbm", variant="hijri", context_length=None, seed=42,
    )
    assert p.exists()
    out = read_predictions(model="lgbm", variant="hijri", context_length=None, seed=42)
    assert (out["y_pred"] == df["y_pred"]).all()


def test_predictions_path_format(tmp_path, monkeypatch):
    monkeypatch.setenv("RAMADAN_PRED_DIR", str(tmp_path))
    p = predictions_path(model="chronos_bolt_base", variant="nohijri", context_length=168, seed=42)
    assert p.name == "chronos_bolt_base__nohijri__L168__seed42.parquet"


def test_predictions_path_no_context_length(tmp_path, monkeypatch):
    monkeypatch.setenv("RAMADAN_PRED_DIR", str(tmp_path))
    p = predictions_path(model="lgbm", variant="hijri", context_length=None, seed=42)
    assert p.name == "lgbm__hijri__seed42.parquet"
