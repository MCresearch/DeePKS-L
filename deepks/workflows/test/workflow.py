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
        Any: Result returned by ``deepks.ml.eval.test.main``.
    """
    from deepks.ml.eval.test import main as test_main

    data_paths = config.get('data_paths', config.get('systems_test'))
    return test_main(
        data_paths=data_paths,
        model_file=config.get('model_file', 'model.pth'),
        output_prefix=config.get('output_prefix', 'test'),
        group=config.get('group', False),
        e_name=config.get('e_name', 'l_e_delta'),
        d_name=config.get('d_name', ['dm_eig']),
        device=config.get('device', 'cpu'),
    )
