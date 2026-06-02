"""SCF workflow orchestration."""

from deepks.physics.backends.abacus.workflow_ops import (
    build_prepare_task,
    collect_results,
    execute_sequence,
)


def run_scf_workflow(config):
    """Run the SCF workflow for the configured backend."""
    physics = config.get("physics") if isinstance(config.get("physics"), dict) else {}
    backend = physics.get("backend") if isinstance(physics.get("backend"), dict) else {}
    backend_name = str(backend.get("name", "pyscf")).lower()

    if backend_name == "abacus":
        prepare_task = build_prepare_task(config)
        execute_sequence(prepare_task, config)
        return collect_results(config)

    if backend_name == "pyscf":
        raise NotImplementedError("PySCF workflow not yet implemented in new architecture")

    raise ValueError(f"Unknown SCF backend: {backend_name}")
