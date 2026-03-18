# Refactor Phase 1: Structure Map

This document started as phase-1 scaffolding and now also records phase-2 progress.
Behavior is intentionally kept compatible through shims.

## New package skeleton

- `deepks/cli/`
- `deepks/orchestration/`
- `deepks/io/`
- `deepks/core/`
- `deepks/pipelines/`
- `deepks/compat/`

## Active mappings

### Orchestration (phase-2: real implementation moved)
- `deepks/orchestration/workflow/task.py` now hosts the real code migrated from `deepks/task/task.py`.
- `deepks/orchestration/workflow/workflow.py` now hosts the real code migrated from `deepks/task/workflow.py`.
- `deepks/orchestration/scheduler/job/*.py` now host real scheduler code migrated from `deepks/task/job/*`.
- Legacy path kept as shim: `deepks/task/*` -> `deepks/orchestration/*`.

### Pipelines
- `deepks/pipelines/iterate/*.py` now host the real code migrated from `deepks/iterate/*`.
- Legacy path kept as shim: `deepks/iterate/*` -> `deepks/pipelines/iterate/*`.
- `deepks/pipelines/train/train.py` -> `deepks.model.train`
- `deepks/pipelines/train/test.py` -> `deepks.model.test`
- `deepks/pipelines/scf/run.py` -> `deepks.scf.run`
- `deepks/pipelines/scf/stats.py` -> `deepks.scf.stats`

## Not migrated yet (next phases)

- Reader/model/scf split into `io` and `core`

## Validation target

- Existing imports and CLI behavior stay compatible.
- Tests remain green before moving real implementations.
