import numpy as np
import pytest

from src.models.tsfm.time_moe import TimeMoEModel


def test_timemoe_model_attributes():
    m = TimeMoEModel()
    assert m.name == "time_moe_200m"
    assert m.needs_training is False
    assert m.supports_dynamic_covariates is False  # channel-independent


def test_timemoe_load_raises_not_implemented():
    """Plan 2 defers Time-MoE; _load() raises until Plan 3 fixes the transformers compat."""
    m = TimeMoEModel()
    with pytest.raises(NotImplementedError):
        m._load()
