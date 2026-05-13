import pandas as pd
from src.models.base import Model


def test_model_protocol_has_expected_attrs():
    assert hasattr(Model, "fit")
    assert hasattr(Model, "predict")


def test_concrete_model_satisfies_protocol():
    """A simple class with the right shape should pass isinstance check."""
    class Toy:
        name = "toy"
        supports_dynamic_covariates = False
        needs_training = True

        def fit(self, train_df, val_df, hijri, seed):
            return None

        def predict(self, test_df, context_length=None):
            return pd.DataFrame({"y_true": [], "y_pred": []})

    instance = Toy()
    assert isinstance(instance, Model)
