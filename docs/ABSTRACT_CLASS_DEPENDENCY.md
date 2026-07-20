# Current Abstract-Class Dependency And Call Graph

This document describes the current dependency structure for the runtime path
after `deepks.config.dispatcher.dispatch_command()` selects a task type and
dispatches into the workflow layer.

The goals reflected here are:

- `config/` completes config normalization, validation, defaults, merge, and packaging
- `workflows` only orchestrates task-type execution
- `interface` is the only layer allowed to assemble ML and physics logic
- `ml` does not depend on `interface` or `physics`
- `physics` does not depend on `interface` or `workflows`


## 1. Static dependency graph

### 1.1 Top-level runtime graph

```text
dispatcher
-> workflows
   -> interface
   -> physics
   -> ml
   -> orchestration
```

### 1.2 Core design graph

```text
ml <- interface -> physics
```

This is the main architectural boundary:

- `ml` provides model and training abstractions
- `physics` provides physical-property implementations, a property engine, and backend execution
- `interface` binds a concrete recipe, objective, batch representation, and model family


## 2. Abstract classes and current concrete implementations

### 2.1 ML side

Defined in [deepks/ml/base.py](/home/ubuntu/work/DeePKS-L/deepks/ml/base.py).

#### `ModelAdapter`

Purpose:
- abstract model contract used by objective and training loop

Required methods:
- `forward(model_inputs)`
- `forward_with_derivatives(model_inputs, derivative_spec=None)`
- `parameters()`
- `state_dict()`
- `load_state_dict(state_dict)`

Current concrete implementations:
- [CorrNet](/home/ubuntu/work/DeePKS-L/deepks/ml/models/corrnet.py)
- [LinearModel](/home/ubuntu/work/DeePKS-L/deepks/ml/models/linear.py)

Key runtime usage:
- `DescriptorPropertyObjectiveAdapter.compute_losses()`
  calls `model.forward_with_derivatives(...)`


#### `BatchProtocol`

Purpose:
- standard batch structure shared across objective, transformer, and train loop

Fields:
- `model_inputs`
- `targets`
- `context`
- `metadata`

Current concrete implementation:
- [TaskBatch](/home/ubuntu/work/DeePKS-L/deepks/io/batch.py)


### 2.2 Physics side

Defined in [deepks/physics/base.py](/home/ubuntu/work/DeePKS-L/deepks/physics/base.py).

Shared contract names live in
[deepks/physics/contracts.py](/home/ubuntu/work/DeePKS-L/deepks/physics/contracts.py):

- `PropertyNames`
- `ModelOutputKeys`
- `ModelDerivativeKeys`

#### `PropertyScheme`

Purpose:
- define one concrete model/output recovery scheme for requested physical quantities
- declare which properties the scheme supports
- declare which model outputs/derivatives are required
- validate required context before property recovery

Required methods:
- `supported_properties()`
- `required_model_outputs(requested_properties)`
- `required_model_derivatives(requested_properties)`
- `validate_context(requested_properties, context)`
- `compute_property(name, model_outputs, model_derivatives, context, cache)`

Current concrete implementations:
- [EnergyDescriptorScheme](/home/ubuntu/work/DeePKS-L/deepks/physics/schemes/energy_descriptor.py)

#### `physics/properties/*`

Purpose:
- hold per-property helper functions only
- do not encode the current model signal contract

Current helper modules:
- [energy.py](/home/ubuntu/work/DeePKS-L/deepks/physics/properties/energy.py)
- [force.py](/home/ubuntu/work/DeePKS-L/deepks/physics/properties/force.py)
- [stress.py](/home/ubuntu/work/DeePKS-L/deepks/physics/properties/stress.py)
- [orbital.py](/home/ubuntu/work/DeePKS-L/deepks/physics/properties/orbital.py)
- [v_delta.py](/home/ubuntu/work/DeePKS-L/deepks/physics/properties/v_delta.py)
- [band.py](/home/ubuntu/work/DeePKS-L/deepks/physics/properties/band.py)
- [phi.py](/home/ubuntu/work/DeePKS-L/deepks/physics/properties/phi.py)
- [vdr.py](/home/ubuntu/work/DeePKS-L/deepks/physics/properties/vdr.py)

#### `PropertyEngine`

Purpose:
- orchestrate requested physical quantities through a concrete scheme
- compute only the quantities requested by the objective layer
- manage shared intermediate cache
- require an explicitly injected scheme; there is no implicit default recovery path

Current implementation:
- [PropertyEngine](/home/ubuntu/work/DeePKS-L/deepks/physics/engine.py)


#### `BackendRunner`

Purpose:
- uniform runtime interface for physical backends

Required methods:
- `name`
- `prepare(systems, config, workdir)`
- `run(prepared, runtime_config)`
- `collect(prepared, runtime_config)`

Implemented through backend hierarchy:
- [PhysicsBackend](/home/ubuntu/work/DeePKS-L/deepks/physics/backends/base.py)
- [SCFBackend](/home/ubuntu/work/DeePKS-L/deepks/physics/backends/base.py)
- [AbacusBackend](/home/ubuntu/work/DeePKS-L/deepks/physics/backends/abacus/backend.py)
- [PySCFBackend](/home/ubuntu/work/DeePKS-L/deepks/physics/backends/pyscf/backend.py)


## 3. Interface-owned assembly points

### 3.1 Recipe registry

- [get_recipe](/home/ubuntu/work/DeePKS-L/deepks/interface/registry.py)
- [get_recipe_name](/home/ubuntu/work/DeePKS-L/deepks/interface/registry.py)

Current concrete recipes:
- [CorrNetEnergyRecipe](/home/ubuntu/work/DeePKS-L/deepks/interface/recipes/corrnet_energy.py)
- [CorrNetEnergyOnlyRecipe](/home/ubuntu/work/DeePKS-L/deepks/interface/recipes/corrnet_energy_only.py)
- [LinearEnergyRecipe](/home/ubuntu/work/DeePKS-L/deepks/interface/recipes/linear_energy.py)


### 3.2 Objective assembly

`DescriptorPropertyObjectiveAdapter` is the main integration point:

```text
recipe schema.property_scheme
-> objective builder
-> PropertyEngine(scheme=...)
TaskBatchRepresentationBuilder
-> model.forward_with_derivatives(...)
-> PropertyEngine.get_many(...)
-> interface.objective.terms.compute_loss(...)
```


### 3.3 Train-task runtime assembly

- [prepare_train_runtime](/home/ubuntu/work/DeePKS-L/deepks/workflows/train/runtime.py)
- [run_training_stage](/home/ubuntu/work/DeePKS-L/deepks/workflows/train/runtime.py)

These own:
- reader creation
- train/test target mapping
- representation-derived model config preparation
- create/load model
- preprocess
- recipe-level training execution


### 3.4 Iterate-task assembly

Owned by:
- [snapshots.py](/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/support/snapshots.py)
- [task_yaml.py](/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/support/task_yaml.py)
- [task_params.py](/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/support/task_params.py)
- [task_templates.py](/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/support/task_templates.py)

These own:
- packed child-task YAML generation
- iterate `share/` snapshot materialization
- child task parameter extraction
- generic iterate task templates


## 4. Runtime call graph by task type

All task types enter here after `deepks.config` packaging:

```text
dispatch_command
-> select workflow handler by type
-> call run_*_workflow(payload)
```


### 4.1 `train`

Main entry:
- [run_train_workflow](/home/ubuntu/work/DeePKS-L/deepks/workflows/train/workflow.py)

Call graph:

```text
dispatch_command
-> run_train_workflow(config)
   -> prepare_train_runtime(config)
      -> GroupReader(...)
      -> build model_config
   -> run_training_stage(train_reader, test_reader, model_config)
      -> get_recipe(...)
      -> recipe.create_or_load_model(...)
         -> interface.model_io.build_model(...)
            -> ml.model_io.build_model(...)
               -> CorrNet / LinearModel
      -> recipe.preprocess_training_data(...)
      -> recipe.train_model(...)
         -> ml.train.trainer.Trainer.train(...)
            -> objective.compute_losses(model, batch)
               -> TaskBatchRepresentationBuilder.build_batch(...)
               -> model.forward_with_derivatives(...)
               -> PropertyEngine.get_many(...)
                  -> PropertyScheme.compute_property(...)
                  -> physics/properties/*
               -> interface.objective.terms.compute_loss(...)
   -> recipe.evaluate_model(model, test_reader, config)
   -> recipe.write_test_log(...)
   -> TrainResult
```


### 4.2 `test`

Main entry:
- [run_test_workflow](/home/ubuntu/work/DeePKS-L/deepks/workflows/test/workflow.py)

Call graph:

```text
dispatch_command
-> run_test_workflow(config)
   -> get_recipe(config=config)
   -> recipe.run_test_workflow(config)
      -> interface.eval.test.main(...)
      -> model loading via interface.model_io
      -> objective/recipe-owned evaluation path
```

This workflow stays intentionally thin because `test` is a task type entrypoint,
not a redundant stage wrapper.


### 4.3 `scf`

Main entry:
- [run_scf_workflow](/home/ubuntu/work/DeePKS-L/deepks/workflows/scf/workflow.py)

Call graph:

```text
dispatch_command
-> run_scf_workflow(config)
   -> backend_name = config["physics"]["backend"]["name"]
   -> if abacus:
      -> build_prepare_task(config)
      -> execute_sequence(prepare_task, config)
      -> collect_results(config)
   -> if pyscf:
      -> currently NotImplementedError on workflow path
```

ABACUS helper ownership:
- [ops.py](/home/ubuntu/work/DeePKS-L/deepks/workflows/scf/abacus/ops.py)

This module owns:
- input generation preparation
- execution task construction
- output collection and aggregation


### 4.4 `stats`

Main entry:
- [run_stats_workflow](/home/ubuntu/work/DeePKS-L/deepks/workflows/stats/workflow.py)

Call graph:

```text
dispatch_command
-> run_stats_workflow(config)
   -> run_stats(config, log_file)
      -> interface.stats.adapter._build_stats_kwargs(...)
      -> interface.stats.reporting.load_stat / load_stat_grouped
      -> interface.stats.reporting.print_stats(...)
```


### 4.5 `iterate`

Main entry:
- [run_iterate_workflow](/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/workflow.py)

Call graph:

```text
dispatch_command
-> run_iterate_workflow(config)
   -> prepare_iterate(config)
      -> prepare_iterate_snapshots(iterate_param)
      -> _create_scf_step(...)
         -> if abacus:
            -> build_abacus_iterate_scf_kwargs(...)
            -> make_scf_abacus(...)
         -> if pyscf:
            -> make_scf(...)
      -> _create_train_step(...)
         -> make_train(...)
      -> Iteration([...])
      -> optional init Sequence(...)
   -> iteration_workflow.run() or restart()
```

Where iterate pieces live now:

- workflow tree assembly:
  - [prepare.py](/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/prepare.py)

- generic iterate task templates:
  - [task_templates.py](/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/support/task_templates.py)

- ABACUS iterate task/sequence implementation:
  - [sequence.py](/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/abacus/sequence.py)
  - [tasks.py](/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/abacus/tasks.py)
  - [iterate_ops.py](/home/ubuntu/work/DeePKS-L/deepks/physics/backends/abacus/iterate_ops.py)


## 5. Remaining intentional thin modules

These remain thin by design and are not considered redundant wrappers:

- [deepks/workflows/test/workflow.py](/home/ubuntu/work/DeePKS-L/deepks/workflows/test/workflow.py)
  - reason: it is the formal `type=test` dispatcher entrypoint

- [deepks/workflows/train/workflow.py](/home/ubuntu/work/DeePKS-L/deepks/workflows/train/workflow.py)
- [deepks/workflows/scf/workflow.py](/home/ubuntu/work/DeePKS-L/deepks/workflows/scf/workflow.py)
- [deepks/workflows/stats/workflow.py](/home/ubuntu/work/DeePKS-L/deepks/workflows/stats/workflow.py)
- [deepks/workflows/iterate/workflow.py](/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/workflow.py)

These are kept because each is the unique task-type workflow entrypoint used by
`dispatcher`, not a removable stage shim.


## 6. Removed redundant layers

The following refactor cleanups are already reflected in the current graph:

- removed `workflows/train/prepare.py`
- removed `workflows/train/train.py`
- removed `workflows/train/evaluate.py`
- removed `workflows/scf/prepare.py`
- removed `workflows/scf/execute.py`
- removed `workflows/scf/collect.py`
- removed `workflows/iterate/template.py`
- removed `workflows/iterate/template_abacus.py`
- removed `workflows/iterate/scf_step.py`
- removed `workflows/iterate/train_step.py`
- removed `workflows/defaults.py`
- removed `workflows/scf/types.py`
- removed `workflows/iterate/types.py`


## 7. Current architectural summary

The current code now follows this runtime pattern:

```text
config
-> dispatcher
-> workflows
   -> interface
      -> recipes
      -> objectives
      -> task representation builders
   -> physics
      -> schemes
      -> properties
      -> backends
   -> ml
      -> models
      -> train loop
```

And the key abstract-class execution chain for training is:

```text
TaskBatchRepresentationBuilder
-> TaskBatch
-> ModelAdapter.forward_with_derivatives
-> PropertyEngine.get_many
-> interface.objective.terms.compute_loss
-> ObjectiveAdapter.compute_losses
-> Trainer.train
```
