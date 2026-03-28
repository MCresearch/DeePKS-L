# legacy_integral_full/

Purpose: migrated full-workflow scenario tree from the historical `tests/integral` assets.

Contains:
- `train/`, `test/`, `scf/`, `stats/`, `iterate/`
- shared `data/` and `systems/` required by those scenario tests

Notes:
- This directory is an integration scenario, not a generic fixture pool.
- Tests under `tests/integration/scenarios/` may execute directly against this tree.
