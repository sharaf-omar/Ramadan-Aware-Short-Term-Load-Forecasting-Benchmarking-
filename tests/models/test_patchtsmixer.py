import numpy as np
import pandas as pd
import pytest
import torch

from src.models.dl.patchtsmixer import PatchTSMixerModel, WindowedDataset


def _synthetic_arr(n: int = 100, c: int = 3) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, c)).astype(np.float32)


def test_windowed_dataset_shapes():
    arr = _synthetic_arr(n=100, c=3)
    ds = WindowedDataset(arr, context_length=24, prediction_length=24)
    # n=100, L=24, h=24 -> 100 - 24 - 24 + 1 = 53 samples
    assert len(ds) == 53
    sample = ds[0]
    assert sample["past_values"].shape == (24, 3)
    assert sample["past_values"].dtype == torch.float32
    # future_values: shape (h, 1) — y-channel only (col 0)
    assert sample["future_values"].shape == (24, 1)
    assert sample["future_values"].dtype == torch.float32


def test_windowed_dataset_alignment():
    arr = _synthetic_arr(n=60, c=2)
    ds = WindowedDataset(arr, context_length=12, prediction_length=12)
    sample = ds[5]
    # sample 5: past = arr[5:17], future_y = arr[17:29, 0]
    assert np.allclose(sample["past_values"].numpy(), arr[5:17])
    assert np.allclose(sample["future_values"].squeeze(-1).numpy(), arr[17:29, 0])


def test_patchtsmixer_model_attributes():
    m = PatchTSMixerModel(variant="nohijri", context_length=96)
    assert m.name == "patchtsmixer"
    assert m.needs_training is True
    assert m.supports_dynamic_covariates is True


def test_patchtsmixer_variant_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown variant"):
        PatchTSMixerModel(variant="nonsense", context_length=96)


def test_patchtsmixer_feature_set_per_variant():
    base = ["actual_load", "temp_c", "dewpoint_c", "wind_speed",
            "solar_rad", "temp_sq", "temp_above_35"]
    hijri = ["is_ramadan", "day_of_ramadan", "is_eid"]
    ablB  = ["ramadan_x_heatwave", "ramadan_x_temp_above_35"]

    m_nh = PatchTSMixerModel(variant="nohijri", context_length=96)
    m_h  = PatchTSMixerModel(variant="hijri", context_length=96)
    m_pb = PatchTSMixerModel(variant="hijri_plusB", context_length=96)

    assert m_nh.channels == base
    assert m_h.channels  == base + hijri
    assert m_pb.channels == base + hijri + ablB


def _synthetic_df(n: int = 600) -> pd.DataFrame:
    """Hourly synthetic with daily seasonality + temperature signal."""
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    t = np.arange(n)
    load = (
        30000.0
        + 5000.0 * np.sin(2 * np.pi * t / 24)
        + np.random.default_rng(0).normal(scale=300.0, size=n)
    ).astype(np.float32)
    temp = (15.0 + 10.0 * np.sin(2 * np.pi * t / 24)).astype(np.float32)
    return pd.DataFrame({
        "actual_load": load,
        "temp_c": temp,
        "dewpoint_c": 5.0,
        "wind_speed": 3.0,
        "solar_rad": 0.0,
        "temp_sq": (temp ** 2),
        "temp_above_35": 0.0,
        "is_ramadan": 0,
        "day_of_ramadan": 0,
        "is_eid": 0,
        "ramadan_x_heatwave": 0,
        "ramadan_x_temp_above_35": 0.0,
        "regime": "Normal",
    }, index=idx)


def test_patchtsmixer_fit_runs_one_epoch():
    df = _synthetic_df(n=600)
    train = df.iloc[:400]
    val   = df.iloc[400:550]
    m = PatchTSMixerModel(
        variant="nohijri",
        context_length=48, prediction_length=24,
        patch_length=8, patch_stride=4,
        d_model=32, num_layers=1, expansion_factor=2,
        max_epochs=1, batch_size=16, warmup_steps=5,
        early_stopping_patience=10,
    )
    m.fit(train, val, hijri=False, seed=0)
    assert m._fitted_model is not None
