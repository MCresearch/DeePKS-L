# legacy_integral_full/

Purpose: full fixture migration of historical `tests/integral` assets.

Contains:
- `01_train/`, `02_test/`, `03_scf/`, `04_stats/`, `05_iter/`
- `data/` and `systems/` dependencies required by migrated integral samples.

Notes:
- New tests only read from this fixture folder.
- Original `tests/integral` directory can be safely removed.
