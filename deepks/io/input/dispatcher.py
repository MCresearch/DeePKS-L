"""Unified command dispatcher for DeePKS."""

from copy import deepcopy


TASK_PARAM_BUNDLES = {
    'train': 'train_param',
    'test': 'test_param',
    'scf': 'scf_param',
    'stats': 'stats_param',
    'iterate': 'iterate_param',
}

WORKFLOW_ENTRYPOINTS = {
    'train': ('deepks.workflows.train', 'run_train_workflow'),
    'test': ('deepks.workflows.test', 'run_test_workflow'),
    'scf': ('deepks.workflows.scf', 'run_scf_workflow'),
    'stats': ('deepks.workflows.stats', 'run_stats_workflow'),
    'iterate': ('deepks.workflows.iterate', 'run_iterate_workflow'),
}


def _build_task_config(runtime_config, task_key):
    raw_config = deepcopy(runtime_config.get('raw_config', {}))
    global_param = deepcopy(runtime_config.get('global_param', {}))
    task_param = deepcopy(runtime_config.get(task_key, {}))

    raw_config.update(global_param)
    raw_config.update(task_param)
    return raw_config


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

    if task_type not in TASK_PARAM_BUNDLES:
        raise ValueError(f"Unknown type: {task_type}")

    config = _build_task_config(runtime_config, TASK_PARAM_BUNDLES[task_type])
    handler = _get_workflow_handler(task_type)
    return handler(config)
