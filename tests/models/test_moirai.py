import numpy as np
import pytest

from src.models.tsfm.moirai import MoiraiModel


def test_moirai_model_attributes():
    m = MoiraiModel()
    assert m.name == "moirai_1_1_small"
    assert m.needs_training is False
    assert m.supports_dynamic_covariates is True


def test_moirai_forecast_batch_shape_smoke():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    m = MoiraiModel()
    rng = np.random.default_rng(0)
    contexts = rng.normal(size=(4, 336)).astype(np.float32)
    out = m._forecast_batch(contexts)
    assert out.shape == (4, 24)
    assert np.isfinite(out).all()
