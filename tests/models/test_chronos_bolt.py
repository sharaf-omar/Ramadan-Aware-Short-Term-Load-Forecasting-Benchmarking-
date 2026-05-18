import numpy as np
import pytest

from src.models.tsfm.chronos_bolt import ChronosBoltModel


def test_chronos_model_attributes():
    m = ChronosBoltModel()
    assert m.name == "chronos_bolt_base"
    assert m.needs_training is False
    assert m.supports_dynamic_covariates is False


def test_chronos_forecast_batch_shape_smoke():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    m = ChronosBoltModel()
    rng = np.random.default_rng(0)
    contexts = rng.normal(size=(4, 168)).astype(np.float32)
    out = m._forecast_batch(contexts)
    assert out.shape == (4, 24)
    assert np.isfinite(out).all()
