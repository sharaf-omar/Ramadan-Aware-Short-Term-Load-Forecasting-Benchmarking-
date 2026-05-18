import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from scripts.build_statistical_appendix import load_predictions, MODELS


def _write_parquet(tmp_path, fname, rows):
    p = tmp_path / fname
    df = pd.DataFrame(
        rows,
        index=pd.DatetimeIndex(rows["timestamp"], tz="UTC", name="timestamp"),
    ).drop(columns="timestamp")
    df.to_parquet(p)
    return p


def test_load_predictions_intersects_on_tau(tmp_path, monkeypatch):
    ts = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    df_a = pd.DataFrame({
        "timestamp": ts[:8],
        "y_true": np.arange(8, dtype=float),
        "y_pred": np.arange(8, dtype=float) + 0.5,
        "regime": ["Normal"] * 8,
    })
    df_b = pd.DataFrame({
        "timestamp": ts[2:],
        "y_true": np.arange(8, dtype=float) + 2,
        "y_pred": np.arange(8, dtype=float) + 2.5,
        "regime": ["Normal"] * 8,
    })
    _write_parquet(tmp_path, "a.parquet", df_a)
    _write_parquet(tmp_path, "b.parquet", df_b)

    spec = [
        ("model_a", "a.parquet"),
        ("model_b", "b.parquet"),
    ]
    monkeypatch.setattr("scripts.build_statistical_appendix.PRED_DIR", tmp_path)
    out = load_predictions(spec)

    assert set(out.keys()) == {"model_a", "model_b"}
    # Intersection is ts[2:8] = 6 rows.
    assert len(out["model_a"]) == 6
    assert len(out["model_b"]) == 6
    assert out["model_a"].index.equals(out["model_b"].index)


def test_models_constant_has_12_entries():
    assert len(MODELS) == 12
    for entry in MODELS:
        assert len(entry) == 2
        assert entry[1].endswith(".parquet")
