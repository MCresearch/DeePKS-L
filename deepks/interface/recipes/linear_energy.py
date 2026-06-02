"""Minimal linear baseline recipe."""

from deepks.interface.adapters import preprocess
from deepks.interface.schemas import LINEAR_ENERGY_SCHEMA

from .corrnet_energy_only import CorrNetEnergyOnlyRecipe


class LinearEnergyRecipe(CorrNetEnergyOnlyRecipe):
    """Energy-only recipe using a distinct linear model family."""

    schema = LINEAR_ENERGY_SCHEMA

    def preprocess_training_data(self, model, train_reader, *, preprocess_args=None):
        preprocess_kwargs = dict(preprocess_args or {})
        preprocess_kwargs.setdefault("prefit_trainable", True)
        preprocess(model, train_reader, **preprocess_kwargs)
