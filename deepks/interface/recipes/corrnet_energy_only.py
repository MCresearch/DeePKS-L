"""Minimal CorrNet recipe that only optimizes energy."""

from deepks.interface.objectives import select_energy_only_objective_args
from deepks.interface.schemas import CORRNET_ENERGY_ONLY_SCHEMA

from .corrnet_energy import CorrNetEnergyRecipe


class CorrNetEnergyOnlyRecipe(CorrNetEnergyRecipe):
    """A second minimal recipe proving the interface supports parallel schemes."""

    schema = CORRNET_ENERGY_ONLY_SCHEMA

    def _select_training_objective_args(self, objective_args):
        return select_energy_only_objective_args(objective_args)
