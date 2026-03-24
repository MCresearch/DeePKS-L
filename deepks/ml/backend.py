"""Default ML backend wiring for DeePKS workflows and CLI."""

from .base import ModelBackend
from .eval.test import main as test_main
from .train.train import main as train_main


class CorrNetModelBackend(ModelBackend):
    """Default ML backend wiring to current DeePKS ML entry points."""

    def train(self, **kwargs):
        return train_main(**kwargs)

    def evaluate(self, **kwargs):
        return test_main(**kwargs)

    def predict(self, **kwargs):
        model = kwargs.pop('model')
        data = kwargs.pop('data')
        if kwargs:
            raise TypeError(f"unexpected predict kwargs: {sorted(kwargs.keys())}")
        return model(data)
