# Hierarchical Regression Training

This feature adds a new staged training line:

- `recipe: hierarchical-regression`
- `ml.model.family: hierarchical_regression`

The model is a single shared-trunk network with configurable depth. Each level
predicts its own target tensor, and one training run can contain multiple
staged sub-rounds.

## Configuration modes

This training line supports two data modes:

1. `data.targets.hamiltonian_levels`
   - one shared `train/test` dataset
   - all hierarchy labels are stored under the same sample directories
   - useful when all levels share the same geometry/frame set

2. `data.stages`
   - each training stage owns its own `train/test` data source
   - useful when different hierarchy levels come from different datasets
   - this mode is only used by `recipe: hierarchical-regression`
   - old recipes such as `corrnet-energy` still use the normal `data.train/test`

### Minimal shared-dataset configuration

```yaml
type: train
recipe: hierarchical-regression

data:
  train: ["data_train"]
  test: ["data_test"]
  targets:
    hamiltonian_levels:
      - h_sz
      - h_dzp
      - h_tzdp

physics:
  hierarchy:
    levels:
      - name: sz
        output_dim: 8
      - name: dzp
        output_dim: 26
      - name: tzdp
        output_dim: 44

ml:
  model:
    family: hierarchical_regression
    args:
      trunk_hidden_sizes: [100, 100, 100]
      head_hidden_sizes: [100, 100, 100]
      shared_trunk: true
  objective:
    level_losses:
      - level: 0
        weight: 1.0
      - level: 1
        weight: 1.0
      - level: 2
        weight: 1.0
  train:
    stage_schedule:
      - level: 0
        epochs: 5000
        freeze_lower: false
        freeze_trunk: false
      - level: 1
        epochs: 5000
        freeze_lower: true
        freeze_trunk: true
      - level: 2
        epochs: 5000
        freeze_lower: true
        freeze_trunk: true
```

### Stage-owned dataset configuration

```yaml
type: train
recipe: hierarchical-regression

physics:
  representation:
    name: dm_eig
  hierarchy:
    levels:
      - name: dzp
        output_dim: 26
      - name: tzdp
        output_dim: 44

data:
  loader:
    batch_size: 16
    group_batch: 1
    d_name: dm_eig
  stages:
    - level: 0
      name: dzp
      train:
        - data_dzp/train/system_a
        - data_dzp/train/system_b
      test:
        - data_dzp/test/system_c
      target:
        format: csr_hr
        hr_name: hr_dzp

    - level: 1
      name: tzdp
      train:
        - data_tzdp/train/system_a
        - data_tzdp/train/system_b
      test:
        - data_tzdp/test/system_c
      target:
        format: csr_hr
        hr_name: hr_tzdp

ml:
  model:
    family: hierarchical_regression
    args:
      trunk_hidden_sizes: [100, 100, 100]
      head_hidden_sizes: [100, 100, 100]
      shared_trunk: true

  objective:
    level_losses:
      - level: 0
        weight: 1.0
      - level: 1
        weight: 1.0

  train:
    stage_schedule:
      - level: 0
        epochs: 5000
        freeze_lower: false
        freeze_trunk: false
      - level: 1
        epochs: 5000
        freeze_lower: true
        freeze_trunk: true
```

## Supported target formats

- `dense_hamiltonian`
- `csr_hr`
- `collected_hr_delta`
- `collected_energy_delta`

## CSR HR target contract

When `target.format: csr_hr` is used, each system directory must contain:

- `<hr_name>_data.npy`
- `<hr_name>_indices.npy`
- `<hr_name>_indptr.npy`
- `<hr_name>_shape.npy`

These files are read frame-by-frame as CSR matrices.

## Current assumptions

- Descriptor input still comes from the standard reader path.
- Hierarchy masks are generated from prefix-nested `output_dim` values.
- Each level target can be:
  - a scalar-like shape such as `[1]`
  - a square matrix shape `[dim_k, dim_k]`
  - a real-space HR tensor shape `[nRx, nRy, nRz, dim_k, dim_k]`
- One training stage supervises exactly one hierarchy level.
- In `data.stages` mode, each stage can use a different descriptor dataset and a different label source.

## Current scope

This implementation covers training and evaluation through the train workflow.
It does not yet add a dedicated standalone `test` workflow for the hierarchical
recipe.
