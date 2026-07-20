# Physics Scope Audit

## Goal

This document summarizes two things in the current codebase:

1. Which parts currently live under `deepks/physics` but do **not** belong to the physics layer if physics is defined strictly as "compute physical quantities from inputs and return them".
2. Which physical quantities are currently computed, how they are computed, what data they depend on, and where those data come from.

The intended target boundary is:

- `ml`: model structure, train/eval engine, generic tensor execution
- `physics`: physical quantity computation only
- `interface`: task assembly, objective assembly, label selection, loss computation, metric computation
- `config` and `io/readers`: configuration and data loading

## Current Status

The major boundary correction described in this document has now been applied on
the main path.

- `physics/terms/*` has been moved out of `physics`
- `physics/losses/*` has been moved out of `physics`
- `PhysicsObjectiveTerm` has been removed from `deepks/physics/base.py`
- `physics/transformers/descriptor_properties.py` and the later calculator layer
  have been replaced by:
  - [deepks/physics/properties](/home/ubuntu/work/DeePKS-L/deepks/physics/properties)
  - [deepks/physics/engine.py](/home/ubuntu/work/DeePKS-L/deepks/physics/engine.py)
- the new physics-side top-level structure is:
  - one file per physical quantity under `physics/properties/`
  - a `PropertyEngine` that orchestrates requested properties
- supervision terms and loss helpers now live in:
  - [deepks/interface/objectives/terms.py](/home/ubuntu/work/DeePKS-L/deepks/interface/objectives/terms.py)
  - [deepks/interface/objectives/losses.py](/home/ubuntu/work/DeePKS-L/deepks/interface/objectives/losses.py)
- the old `physics/observables/` and misnamed `physics/descriptors/` main-path
  layout has been removed in favor of:
  - `physics/properties/`
  - backend/property-local shared helpers

The remainder of this document still matters because it records the boundary
problems that motivated the refactor and the quantity/data-source inventory that
the new calculator layer still relies on.

## Current Out-of-Scope Parts Inside `physics`

The following modules are currently outside the strict scope of a "physical quantity computation" layer.

### 1. Objective and loss abstractions

- [deepks/physics/base.py](/home/ubuntu/work/DeePKS-L/deepks/physics/base.py)
  - previous `PhysicsObjectiveTerm`
  - Status: already removed

- previous `deepks/physics/terms/property_terms.py`
  - Reason: this file compares predictions against targets, applies configured losses, and uses supervision-specific settings such as occupation, masking, and per-atom normalization.
  - These belong to the objective layer, not the physics layer.
  - Status: moved to [deepks/interface/objectives/terms.py](/home/ubuntu/work/DeePKS-L/deepks/interface/objectives/terms.py)

- previous `deepks/physics/losses/hamiltonian.py`
  - `loss_hr`
  - `cal_vd_masked_loss_hs`
  - `cal_vd_masked_loss_width`
  - Reason: these are loss formulas, not physical quantity computation.
  - Status: moved to [deepks/interface/objectives/losses.py](/home/ubuntu/work/DeePKS-L/deepks/interface/objectives/losses.py)

- previous `deepks/physics/losses/wavefunction.py`
  - `cal_phi_loss`
  - Reason: this is a supervision loss, not physics computation.
  - Status: moved to [deepks/interface/objectives/losses.py](/home/ubuntu/work/DeePKS-L/deepks/interface/objectives/losses.py)

### 2. Supervision-aware branching inside the transformer

- previous `deepks/physics/transformers/descriptor_properties.py`
  - Reason: the file was partially correct and partially out of scope.
  - Status: replaced by the `properties + engine` structure
  - The current property engine:
    - accepts `requested_properties`
    - no longer inspects `batch.targets`
    - no longer computes objective-only auxiliaries such as `grad_total` or `density_regularizer`

### 3. Evaluation and reporting logic

- [deepks/physics/stats/adapter.py](/home/ubuntu/work/DeePKS-L/deepks/physics/stats/adapter.py)
  - Reason: stats argument translation and reporting are evaluation-layer concerns, not physical quantity computation.

- [deepks/physics/backends/stats.py](/home/ubuntu/work/DeePKS-L/deepks/physics/backends/stats.py)
  - Reason: this file loads predicted and labeled arrays, computes error statistics, and prints reports. That is result evaluation, not physics computation.

### 4. Optimization and penalty helpers mixed into backend code

- [deepks/physics/backends/pyscf/addons.py](/home/ubuntu/work/DeePKS-L/deepks/physics/backends/pyscf/addons.py)
  - Contains optimization/loss-related logic such as:
    - `gen_coul_loss`
    - `calc_optim_veig`
  - Status: moved to `deepks/interface/pyscf/optim.py`
    - `force_factor`
    - LBFGS usage
  - Reason: these are no longer plain backend or plain physics-calculation utilities.

- [deepks/physics/backends/pyscf/penalty.py](/home/ubuntu/work/DeePKS-L/deepks/physics/backends/pyscf/penalty.py)
  - Boundary is questionable.
  - If penalty is treated as an objective term, it does not belong to the physics layer either.

## Current Physical Quantities and How They Are Computed

This section lists the main quantities currently involved in the code.

### Data-loading path

The current data-loading path is:

- raw file naming defaults:
  - [deepks/io/schemas/reader_fields.py](/home/ubuntu/work/DeePKS-L/deepks/io/schemas/reader_fields.py)
- raw array loading:
  - [deepks/interface/adapters/reader_data.py](/home/ubuntu/work/DeePKS-L/deepks/interface/adapters/reader_data.py)
- feature loading and derived labels:
  - [deepks/interface/adapters/reader_features.py](/home/ubuntu/work/DeePKS-L/deepks/interface/adapters/reader_features.py)
- sample-to-batch mapping:
  - [deepks/interface/adapters/sample.py](/home/ubuntu/work/DeePKS-L/deepks/interface/adapters/sample.py)

Current `TaskBatch` mapping is roughly:

- `model_inputs["descriptor"] <- eig <- dm_eig.npy`
- `targets["energy"] <- lb_e <- l_e_delta.npy`
- `targets["force"] <- lb_f <- l_f_delta.npy`
- `targets["stress"] <- lb_s <- l_s_delta.npy`
- `targets["orbital"] <- lb_o <- l_o_delta.npy`
- `targets["v_delta"] <- lb_vd <- l_h_delta.npy`
- `targets["vdr"] <- lb_vdr <- l_hr_delta.npy`
- `targets["phi"] <- lb_phi <- derived from hamiltonian.npy`
- `targets["band"] <- lb_band <- derived from hamiltonian.npy`

### 1. Energy

- Computed in:
  - [deepks/physics/properties/energy.py](/home/ubuntu/work/DeePKS-L/deepks/physics/properties/energy.py)
- Current method:
  - direct model output
  - `predictions["energy"] = model_outputs["primary_output"]`
- Data required:
  - model output only
- Label source:
  - `targets["energy"]`
  - default raw file: `l_e_delta.npy`

### 2. Force

- Computed in:
  - [deepks/physics/properties/force.py](/home/ubuntu/work/DeePKS-L/deepks/physics/properties/force.py)
- Current method:
  - chain rule
  - `force = -einsum(gvx, input_grad)`
- Data required:
  - `input_grad = d(primary_output) / d(descriptor)`
  - `gvx`
- Data source:
  - `gvx` from `grad_vx.npy`
- Label source:
  - `targets["force"]`
  - default raw file: `l_f_delta.npy`

### 3. Stress

- Computed in:
  - [deepks/physics/properties/stress.py](/home/ubuntu/work/DeePKS-L/deepks/physics/properties/stress.py)
- Current method:
  - chain rule
  - `stress = einsum(gvepsl, input_grad)`
- Data required:
  - `input_grad`
  - `gvepsl`
- Data source:
  - `gvepsl` from `grad_vepsl.npy`
- Label source:
  - `targets["stress"]`
  - default raw file: `l_s_delta.npy`

### 4. Orbital

- Computed in:
  - [deepks/physics/properties/orbital.py](/home/ubuntu/work/DeePKS-L/deepks/physics/properties/orbital.py)
- Current method:
  - contraction between precomputed orbital tensor and input gradient
- Data required:
  - `input_grad`
  - `op`
- Data source:
  - `op` from `orbital_precalc.npy`
- Label source:
  - `targets["orbital"]`
  - default raw file: `l_o_delta.npy`

### 5. v_delta

- Computed in:
  - [deepks/physics/properties/v_delta.py](/home/ubuntu/work/DeePKS-L/deepks/physics/properties/v_delta.py)
- Current method:
  - path A: direct precalculated contraction with `vdp`
  - path B: `cal_v_delta(input_grad, gevdm, phialpha)`
- Data required:
  - `input_grad`
  - either `vdp`, or `gevdm + phialpha`
- Data source:
  - `vdp` from `v_delta_precalc.npy`
  - `gevdm` from `grad_evdm.npy`
  - `phialpha` from `phialpha.npy`
- Label source:
  - `targets["v_delta"]`
  - default raw file: `l_h_delta.npy`

### 6. Band and phi

- Computed in:
  - [deepks/physics/properties/band.py](/home/ubuntu/work/DeePKS-L/deepks/physics/properties/band.py)
  - [deepks/physics/properties/phi.py](/home/ubuntu/work/DeePKS-L/deepks/physics/properties/phi.py)
  - shared eig helper in [deepks/physics/properties/_shared.py](/home/ubuntu/work/DeePKS-L/deepks/physics/properties/_shared.py)
- Current method:
  - build `h_total = h_base + v_delta`
  - solve eigenproblem:
    - generalized if transformed overlap is available
    - normal otherwise
- Data required:
  - `h_base`
  - predicted `v_delta`
  - optional transformed overlap / `trans_matrix`
- Data source:
  - `h_base` from `h_base.npy`
  - `trans_matrix` derived from overlap-related data in
    - [deepks/interface/adapters/reader_features.py](/home/ubuntu/work/DeePKS-L/deepks/interface/adapters/reader_features.py)
- Label source:
  - `targets["band"]`
  - `targets["phi"]`
  - both are derived from `hamiltonian.npy` inside
    - [deepks/interface/adapters/reader_features.py](/home/ubuntu/work/DeePKS-L/deepks/interface/adapters/reader_features.py)

### 7. Bandgap

- Computed in:
  - [deepks/physics/properties/band.py](/home/ubuntu/work/DeePKS-L/deepks/physics/properties/band.py)
- Current method:
  - difference between LUMO and HOMO band
- Data required:
  - predicted band
  - occupation count
- Data source:
  - band from band computation
  - occupation from objective config, not from raw data
- Note:
  - this already shows objective-layer semantics mixed into physics-adjacent code

### 8. Density matrix

- Computed in:
  - [deepks/physics/properties/phi.py](/home/ubuntu/work/DeePKS-L/deepks/physics/properties/phi.py)
- Current method:
  - use occupied columns of `phi`
  - compute `phi_occ @ phi_occ^T`
- Data required:
  - predicted `phi`
  - occupation count
- Data source:
  - `phi` from eigenproblem solution
  - occupation from objective config

### 9. phi-aligned band quantity

- Computed in:
  - [deepks/interface/objectives/terms.py](/home/ubuntu/work/DeePKS-L/deepks/interface/objectives/terms.py)
- Current method:
  - use labeled `phi`
  - apply predicted `v_delta` to `h_base`
  - compare projected diagonal with labeled bands
- Data required:
  - `targets["phi"]`
  - `targets["band"]`
  - `context["h_base"]`
  - `predictions["v_delta"]`
- Note:
  - this is not a pure physical observable calculation
  - it is supervision-derived and belongs with the objective layer

### 10. vdr

- Computed in:
  - [deepks/physics/properties/vdr.py](/home/ubuntu/work/DeePKS-L/deepks/physics/properties/vdr.py)
- Current method:
  - path A: direct contraction with `vdrp`
  - path B: `get_gedm(...)` then `cal_vdr(...)`
- Data required:
  - `input_grad`
  - either `vdrp`, or `gevdm + overlap + iR_mat + data_shape`
- Data source:
  - `vdrp` from `vdr_precalc.npy`
  - `gevdm` from `grad_evdm.npy`
  - overlap-related tensors from reader features
- Label source:
  - `targets["vdr"]`
  - default raw file: `l_hr_delta.npy`

### 11. grad_total

- Computed in:
  - [deepks/interface/objectives/descriptor_properties.py](/home/ubuntu/work/DeePKS-L/deepks/interface/objectives/descriptor_properties.py)
- Current method:
  - `einsum(gveg, input_grad) + eg0`
- Data required:
  - `input_grad`
  - `gveg`
  - `eg0`
- Data source:
  - `gveg` from `grad_veg.npy`
  - `eg0` from `eg_base.npy`
- Note:
  - this is used for a regularization term, not for a core physical observable
  - it has already been moved out of the physics calculator

### 12. density_regularizer

- Computed in:
  - [deepks/interface/objectives/descriptor_properties.py](/home/ubuntu/work/DeePKS-L/deepks/interface/objectives/descriptor_properties.py)
- Current method:
  - reduce `gldv * input_grad`
- Data required:
  - `input_grad`
  - `gldv`
- Data source:
  - `gldv` from `grad_ldv.npy`
- Note:
  - this is also regularization-specific, not a core physical observable
  - it has already been moved out of the physics calculator

## What the Physics Layer Should Look Like

If physics is strictly responsible for computing physical quantities, then it should:

- accept:
  - model outputs
  - model derivatives
  - physical context loaded from data
  - a request describing which physical quantities to compute
- return:
  - computed physical quantities only

Physics should **not**:

- decide which labels exist
- inspect `targets` to decide what to compute
- choose loss functions
- compare predictions and labels
- compute metrics
- perform report formatting

## Recommended Code-Level Boundary

### Physics should keep

- physical reconstruction functions
  - `cal_v_delta`
  - `cal_vdr`
  - `get_gedm`
  - generalized eigensolver helpers
- backend input preparation, execution, and pure output parsing
- a property calculator / transformer that computes requested physical quantities

### Physics should give up

- `PhysicsObjectiveTerm`
- `physics/terms/*`
- `physics/losses/*`
- supervision-aware branching inside the transformer
- stats reporting code
- regularization-specific helper outputs such as `grad_total` and `density_regularizer`

### Interface or a dedicated objective layer should own

- objective assembly
- label selection
- loss functions
- metrics
- masking and weighting rules
- occupation selection for supervised targets

## Recommended Next Refactor

The next remaining boundary-fixing step should be:

1. Move stats reporting out of `physics`
2. Review `pyscf/addons.py` and `pyscf/penalty.py` and split objective-side logic away from backend-side logic
3. Keep expanding the calculator contract so all property requests are explicit and label-free
