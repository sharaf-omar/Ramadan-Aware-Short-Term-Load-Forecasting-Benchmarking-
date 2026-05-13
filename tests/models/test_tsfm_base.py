import numpy as np
import pandas as pd
from src.models.tsfm._adapter import TSFMBase, HORIZON


class _FakeTSFM(TSFMBase):
    """Test double: returns context[-1] repeated 24 times as the forecast."""
    name = "fake_tsfm"
    supports_dynamic_covariates = False
    needs_training = False

    def _forecast_batch(self, contexts: np.ndarray) -> np.ndarray:
        return np.tile(contexts[:, -1:], (1, HORIZON))


def _make_test_df(n: int = 500) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "actual_load": np.arange(n, dtype=float),
        "regime": ["Normal"] * n,
    }, index=idx)


def test_tsfm_predict_returns_unified_schema():
    df = _make_test_df(500)
    model = _FakeTSFM()
    out = model.predict(df, context_length=168)
    assert "y_true" in out.columns
    assert "y_pred" in out.columns
    assert "regime" in out.columns
    assert "y_block" in out.columns
    # Smallest valid tau index = 24 + 168 - 1 = 191. So out has 500-191 = 309 rows.
    assert len(out) == 309


def test_tsfm_predict_block_has_24_entries():
    df = _make_test_df(500)
    model = _FakeTSFM()
    out = model.predict(df, context_length=168)
    first_block = out["y_block"].iloc[0]
    assert len(first_block) == HORIZON


def test_tsfm_y_pred_is_24th_block_entry():
    df = _make_test_df(500)
    model = _FakeTSFM()
    out = model.predict(df, context_length=168)
    # _FakeTSFM returns last context value repeated 24 times.
    # row tau=191 -> issuance idx 167 -> last context value = y[167] = 167
    first_row = out.iloc[0]
    assert first_row["y_pred"] == 167.0
    assert first_row["y_block"][-1] == 167.0


def test_tsfm_fit_is_noop():
    df = _make_test_df(500)
    model = _FakeTSFM()
    model.fit(df, df, hijri=False, seed=0)  # should not raise
