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


def test_models_constant_has_expected_shape():
    # Expanded over time: 12 (Plan 7a) -> 20 (after Plan 6 residual heads).
    assert len(MODELS) >= 12
    for entry in MODELS:
        assert len(entry) == 2
        assert entry[1].endswith(".parquet")


def _two_model_synthetic(n=240):
    """24h × 10d of synthetic predictions with regime labels."""
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    y_true = np.sin(np.arange(n) * 2 * np.pi / 24) + 10
    rng = np.random.default_rng(0)
    return {
        "good": pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_true + rng.normal(scale=0.1, size=n),
            "regime": ["Normal"] * (n // 2) + ["Ramadan"] * (n // 2),
        }, index=ts),
        "bad": pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_true + rng.normal(scale=1.0, size=n),
            "regime": ["Normal"] * (n // 2) + ["Ramadan"] * (n // 2),
        }, index=ts),
    }


def test_compute_ci_table_shape_and_ordering():
    from scripts.build_statistical_appendix import compute_ci_table
    preds = _two_model_synthetic()
    df = compute_ci_table(preds, regimes=["aggregate", "Normal", "Ramadan"])
    assert list(df.columns) == ["model", "regime", "mae", "ci_lo", "ci_hi"]
    assert len(df) == 2 * 3
    g_agg = df[(df.model == "good") & (df.regime == "aggregate")].mae.iloc[0]
    b_agg = df[(df.model == "bad")  & (df.regime == "aggregate")].mae.iloc[0]
    assert b_agg > g_agg
    row = df.iloc[0]
    assert row.ci_lo <= row.mae <= row.ci_hi


def test_compute_ci_table_handles_empty_regime():
    from scripts.build_statistical_appendix import compute_ci_table
    preds = _two_model_synthetic()
    df = compute_ci_table(preds, regimes=["aggregate", "Heatwave"])
    heat = df[df.regime == "Heatwave"]
    assert len(heat) == 2
    assert heat.mae.isna().all()


def test_compute_dm_matrix_pairs_holm_adjusted():
    from scripts.build_statistical_appendix import compute_dm_matrix
    ts = pd.date_range("2024-01-01", periods=500, freq="h", tz="UTC")
    y_true = np.sin(np.arange(500) * 2 * np.pi / 24) * 100 + 1000
    rng = np.random.default_rng(0)
    preds = {
        "best":  pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_true + rng.normal(scale=10.0, size=500),
            "regime": ["Normal"] * 500,
        }, index=ts),
        "mid":   pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_true + rng.normal(scale=50.0, size=500),
            "regime": ["Normal"] * 500,
        }, index=ts),
        "worst": pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_true + rng.normal(scale=200.0, size=500),
            "regime": ["Normal"] * 500,
        }, index=ts),
    }
    df = compute_dm_matrix(preds, regime="aggregate")
    assert len(df) == 3
    assert list(df.columns) == ["model_i", "model_j", "dm_stat", "p_raw", "p_holm"]
    assert (df.p_holm >= df.p_raw - 1e-12).all()
    bw = df[(df.model_i == "best") & (df.model_j == "worst")]
    assert len(bw) == 1
    assert bw.p_holm.iloc[0] < 0.01


def test_compute_dm_matrix_skips_empty_regime():
    from scripts.build_statistical_appendix import compute_dm_matrix
    preds = _two_model_synthetic()
    df = compute_dm_matrix(preds, regime="Heatwave")
    assert len(df) == 1
    assert pd.isna(df.dm_stat.iloc[0])
    assert pd.isna(df.p_raw.iloc[0])
    assert pd.isna(df.p_holm.iloc[0])


def test_render_markdown_includes_required_sections():
    from scripts.build_statistical_appendix import (
        compute_ci_table, compute_dm_matrix, render_markdown,
    )
    preds = _two_model_synthetic()
    ci = compute_ci_table(preds, regimes=["aggregate", "Normal", "Ramadan"])
    dm_by_regime = {
        r: compute_dm_matrix(preds, r) for r in ["aggregate", "Normal", "Ramadan"]
    }
    md = render_markdown(ci, dm_by_regime, n_tau=240)
    assert "# Statistical Appendix" in md
    assert "## Bootstrap MAE confidence intervals" in md
    assert "## Pairwise Diebold-Mariano tests" in md
    for r in ["aggregate", "Normal", "Ramadan"]:
        assert f"### DM matrix — {r}" in md
    assert "good" in md and "bad" in md
    assert ("***" in md) or ("**" in md) or ("ns" in md)
