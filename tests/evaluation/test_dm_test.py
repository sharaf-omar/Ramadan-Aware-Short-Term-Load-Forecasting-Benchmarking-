import numpy as np
import pytest
from src.evaluation.dm_test import dm_test, holm_bonferroni


def test_dm_identical_predictions_pvalue_one():
    """If two models predict identically, DM should be ~0 and p ~ 1."""
    rng = np.random.default_rng(42)
    y_true = rng.normal(size=500)
    y_pred = y_true + rng.normal(scale=0.1, size=500)
    stat, p = dm_test(y_true, y_pred, y_pred, h=24)
    assert abs(stat) < 1e-9
    assert p == pytest.approx(1.0)


def test_dm_model_b_clearly_better_pvalue_low():
    """Model B much closer to y_true than A -> DM stat strongly positive
    (loss_A > loss_B -> d_t > 0), p-value very small."""
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=2000)
    y_pred_a = y_true + rng.normal(scale=2.0, size=2000)
    y_pred_b = y_true + rng.normal(scale=0.5, size=2000)
    stat, p = dm_test(y_true, y_pred_a, y_pred_b, h=24)
    assert stat > 0
    assert p < 0.01


def test_holm_bonferroni_basic():
    p_values = [0.001, 0.04, 0.03, 0.5]
    adjusted = holm_bonferroni(p_values)
    # smallest at position 0: 0.001 * 4 = 0.004
    # largest at position 3: 0.5 * 1 = 0.5
    assert adjusted[0] == pytest.approx(0.004)
    assert adjusted[3] == pytest.approx(0.5)


def test_holm_bonferroni_monotone():
    """Holm correction preserves order of raw p-values."""
    p = [0.01, 0.02, 0.03, 0.04, 0.05]
    adj = holm_bonferroni(p)
    for i in range(len(adj) - 1):
        assert adj[i] <= adj[i + 1]
