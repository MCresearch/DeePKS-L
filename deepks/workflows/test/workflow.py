"""Test workflow - main orchestration.

This module implements the DeePKS test/evaluation workflow as a thin
wrapper around the existing evaluation entrypoint.
"""


def run_test_workflow(config):
    """Run model evaluation workflow.

    This preserves the existing dispatcher compatibility behavior while
    exposing a workflow-style entrypoint consistent with train/scf/iterate.

    Args:
        config: Configuration dictionary containing current test runtime keys.

    Returns:
        Any: Result returned by the current recipe-owned evaluation runner.
    """
    from deepks.interface.registry import get_recipe

    recipe = get_recipe(config=config)
    return recipe.run_test_workflow(config)
