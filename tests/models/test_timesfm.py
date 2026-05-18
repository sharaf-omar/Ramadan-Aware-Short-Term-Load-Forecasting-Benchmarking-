import numpy as np
import pytest

from src.models.tsfm.timesfm import TimesFMModel


def test_timesfm_model_attributes():
    m = TimesFMModel()
    assert m.name == "timesfm_2_5"
    assert m.needs_training is False
    assert m.supports_dynamic_covariates is True


def test_timesfm_forecast_batch_shape_smoke():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    m = TimesFMModel()
    rng = np.random.default_rng(0)
    contexts = rng.normal(size=(4, 336)).astype(np.float32)
    out = m._forecast_batch(contexts)
    assert out.shape == (4, 24)
    assert np.isfinite(out).all()
