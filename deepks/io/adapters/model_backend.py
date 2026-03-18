"""Model backend adapter implementations for CLI/pipeline integration."""

from deepks.core.contracts import ModelBackend
from deepks.core.ml.eval.test import main as test_main
from deepks.core.ml.train.train import main as train_main


class CorrNetModelBackend(ModelBackend):
    """Default ML backend wiring to current core ML entry points."""

    def train(self, **kwargs):
        return train_main(**kwargs)

    def evaluate(self, **kwargs):
        return test_main(**kwargs)

    def predict(self, **kwargs):
        model = kwargs.pop("model")
        data = kwargs.pop("data")
        if kwargs:
            raise TypeError(f"unexpected predict kwargs: {sorted(kwargs.keys())}")
        return model(data)
