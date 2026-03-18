# Refactor Status: Final Architecture (Phase 5)

This document records the repository state after the phase-5 hard cutover.
Legacy compatibility layers were removed, and all runtime code now uses canonical packages only.

## Canonical package layout

- `deepks/cli/`: command-line entrypoints and argument parsing.
- `deepks/orchestration/`: workflow and scheduler orchestration.
- `deepks/io/`: readers, transforms, schemas, and adapters.
- `deepks/core/`: ML and physics core implementations.
- `deepks/pipelines/`: train/test/scf/iterate pipeline entry modules.
- `deepks/tools/`: standalone utility scripts.

## Removed legacy interfaces

The following legacy packages/modules were deleted in phase 5:

- `deepks/main.py`
- `deepks/model/*`
- `deepks/scf/*`
- `deepks/task/*`
- `deepks/iterate/*`

No compatibility shim path is guaranteed in current architecture.

## Canonical import map

- Train pipeline: `deepks.pipelines.train.train`
- Test pipeline: `deepks.pipelines.train.test`
- SCF pipeline run: `deepks.pipelines.scf.run`
- SCF stats pipeline: `deepks.pipelines.scf.stats`
- Iterate pipeline: `deepks.pipelines.iterate.iterate`
- Workflow tasks: `deepks.orchestration.workflow.task`
- Workflow composition: `deepks.orchestration.workflow.workflow`
- Scheduler jobs: `deepks.orchestration.scheduler.job.*`
- Reader APIs: `deepks.io.readers`
- Core ML: `deepks.core.ml.*`
- Core physics (PySCF): `deepks.core.physics.pyscf.*`

## Entry points

- Console scripts: `deepks` and `dks`
- Runtime target: `deepks.cli.main:main_cli`

## Validation baseline

- `python -m pytest -q -m "not pyabacus"`
- `python -m pytest -q`

The baseline currently passes in project CI/dev environments.
