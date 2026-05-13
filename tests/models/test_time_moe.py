import numpy as np
import pytest

from src.models.tsfm.time_moe import TimeMoEModel


def test_timemoe_model_attributes():
    m = TimeMoEModel()
    assert m.name == "time_moe_200m"
    assert m.needs_training is False
    assert m.supports_dynamic_covariates is False  # channel-independent


def test_timemoe_forecast_batch_shape_smoke():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    m = TimeMoEModel()
    rng = np.random.default_rng(0)
    contexts = rng.normal(size=(4, 336)).astype(np.float32)
    out = m._forecast_batch(contexts)
    assert out.shape == (4, 24)
    assert np.isfinite(out).all()
