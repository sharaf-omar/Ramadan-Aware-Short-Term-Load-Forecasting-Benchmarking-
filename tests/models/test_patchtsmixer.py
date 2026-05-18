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
