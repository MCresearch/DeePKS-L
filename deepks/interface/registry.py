"""Recipe registry for interface-layer physics/ML assemblies."""

from typing import Any, Dict, Optional

from .recipes.corrnet_energy import CorrNetEnergyRecipe
from .recipes.corrnet_energy_only import CorrNetEnergyOnlyRecipe
from .recipes.hierarchical_regression import HierarchicalRegressionRecipe
from .recipes.linear_energy import LinearEnergyRecipe
from .schemas import (
    DEFAULT_RECIPE_NAME,
    ENERGY_ONLY_RECIPE_NAME,
    HIERARCHICAL_REGRESSION_RECIPE_NAME,
    LINEAR_RECIPE_NAME,
)

_RECIPE_ALIASES = {
    DEFAULT_RECIPE_NAME: DEFAULT_RECIPE_NAME,
    "corrnet_energy": DEFAULT_RECIPE_NAME,
    "corrnet": DEFAULT_RECIPE_NAME,
    ENERGY_ONLY_RECIPE_NAME: ENERGY_ONLY_RECIPE_NAME,
    "corrnet_energy_only": ENERGY_ONLY_RECIPE_NAME,
    "corrnet_minimal": ENERGY_ONLY_RECIPE_NAME,
    LINEAR_RECIPE_NAME: LINEAR_RECIPE_NAME,
    "linear_energy": LINEAR_RECIPE_NAME,
    "linear": LINEAR_RECIPE_NAME,
    HIERARCHICAL_REGRESSION_RECIPE_NAME: HIERARCHICAL_REGRESSION_RECIPE_NAME,
    "hierarchical_regression": HIERARCHICAL_REGRESSION_RECIPE_NAME,
}

_RECIPES = {
    CorrNetEnergyRecipe.schema.name: CorrNetEnergyRecipe,
    CorrNetEnergyOnlyRecipe.schema.name: CorrNetEnergyOnlyRecipe,
    LinearEnergyRecipe.schema.name: LinearEnergyRecipe,
    HierarchicalRegressionRecipe.schema.name: HierarchicalRegressionRecipe,
}


def get_recipe_name(config: Optional[Dict[str, Any]] = None, recipe: Optional[Any] = None) -> str:
    """Resolve the configured recipe name."""

    if recipe is not None:
        if isinstance(recipe, str):
            raw_name = recipe
        elif isinstance(recipe, dict):
            raw_name = recipe.get("name")
        else:
            raise TypeError(f"Unsupported recipe spec: {type(recipe)!r}")
    elif config is not None:
        recipe_cfg = config.get("recipe")
        if isinstance(recipe_cfg, dict):
            raw_name = recipe_cfg.get("name")
        elif isinstance(recipe_cfg, str):
            raw_name = recipe_cfg
        elif config.get("scheme") is not None:
            raw_name = config.get("scheme")
        else:
            raw_name = config.get("recipe_name")
        if raw_name is None:
            model_cfg = config.get("model")
            if isinstance(model_cfg, dict):
                raw_name = model_cfg.get("family") or model_cfg.get("name")
    else:
        raw_name = None

    normalized = _RECIPE_ALIASES.get((raw_name or DEFAULT_RECIPE_NAME).strip().lower())
    if normalized is None:
        raise ValueError(f"Unsupported recipe: {raw_name}")
    return normalized


def get_recipe(config: Optional[Dict[str, Any]] = None, recipe: Optional[Any] = None):
    """Instantiate the configured recipe."""

    recipe_name = get_recipe_name(config=config, recipe=recipe)
    recipe_cls = _RECIPES[recipe_name]
    return recipe_cls()
