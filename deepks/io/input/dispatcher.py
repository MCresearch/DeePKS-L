"""Unified command dispatcher for DeePKS."""

from copy import deepcopy

from .packager import get_payload_key

WORKFLOW_ENTRYPOINTS = {
    'train': ('deepks.workflows.train', 'run_train_workflow'),
    'test': ('deepks.workflows.test', 'run_test_workflow'),
    'scf': ('deepks.workflows.scf', 'run_scf_workflow'),
    'stats': ('deepks.workflows.stats', 'run_stats_workflow'),
    'iterate': ('deepks.workflows.iterate', 'run_iterate_workflow'),
}


def _get_workflow_handler(task_type):
    module_name, handler_name = WORKFLOW_ENTRYPOINTS[task_type]
    module = __import__(module_name, fromlist=[handler_name])
    return getattr(module, handler_name)


def dispatch_command(runtime_config):
    """Dispatch to appropriate command handler based on runtime config contract.

    Args:
        runtime_config: Packaged runtime configuration dictionary

    Raises:
        ValueError: If type is not recognized
    """
    task_type = runtime_config.get('type')

    if task_type not in WORKFLOW_ENTRYPOINTS:
        raise ValueError(f"Unknown type: {task_type}")

    payload_key = get_payload_key(task_type)
    config = deepcopy(runtime_config.get(payload_key, {}))
    handler = _get_workflow_handler(task_type)
    return handler(config)
