import pytest
from src.models.ml.lgbm import (
    BASE_FEATURES, HIJRI_FEATURES, ABLATION_B_FEATURES,
    feature_set_for_variant,
)


def test_base_features_no_hijri():
    for f in BASE_FEATURES:
        assert "ramadan" not in f.lower()
        assert "eid" not in f.lower()


def test_hijri_features_block_isolated():
    expected = {
        "is_ramadan", "day_of_ramadan", "is_eid",
        "ramadan_x_hour_sin", "ramadan_x_hour_cos", "ramadan_x_weekend",
    }
    assert set(HIJRI_FEATURES) == expected


def test_ablation_b_features_block_isolated():
    assert set(ABLATION_B_FEATURES) == {"ramadan_x_heatwave", "ramadan_x_temp_above_35"}


def test_feature_set_nohijri():
    fs = feature_set_for_variant("nohijri")
    assert set(fs) == set(BASE_FEATURES)


def test_feature_set_hijri():
    fs = feature_set_for_variant("hijri")
    assert set(fs) == set(BASE_FEATURES) | set(HIJRI_FEATURES)


def test_feature_set_hijri_plus_b():
    fs = feature_set_for_variant("hijri_plusB")
    assert set(fs) == set(BASE_FEATURES) | set(HIJRI_FEATURES) | set(ABLATION_B_FEATURES)


def test_feature_set_unknown_variant_raises():
    with pytest.raises(ValueError):
        feature_set_for_variant("nonsense")
