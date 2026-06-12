# Final Refactor Phase Plan

This document describes the remaining work needed to reach the intended final
architecture:

```text
main -> config -> dispatcher -> workflows
```

and

```text
ml <- interface -> physics
```

The current code already follows this structure on the main path, but the
remaining work is still important if the codebase is expected to support:

- new network families
- new input/output physical-property schemes
- new backend implementations
- independent maintenance of `ml`, `physics`, and `interface`

This plan focuses on the remaining standardization work rather than the
already-completed large boundary corrections.

## Final Target

### `ml`

Owns only:

- model abstractions
- concrete model implementations
- train/eval execution mechanisms
- model persistence/registry

Must not own:

- physical-property semantics
- scheme semantics
- recipe semantics

### `physics`

Owns only:

- physical-property helper implementations
- concrete recovery schemes
- property dispatch engine
- backend execution

Must not own:

- labels
- losses
- metrics/reporting
- recipe-specific logic

### `interface`

Owns only:

- task assembly
- objective assembly
- model/scheme selection
- config normalization at the assembly boundary

Must not own:

- generic train loop mechanics
- backend execution mechanics

### `workflows`

Own only:

- task-type entrypoints
- orchestration sequencing

Must not own:

- config compatibility
- backend implementation details
- loss semantics

## Current Remaining Gaps

### 1. `interface` is still too thick

Symptoms:

- recipe files still contain substantial config translation logic
- train/test runtime assembly is spread across multiple helpers
- objective argument preparation is only partially normalized

Impact:

- adding a new recipe still requires touching too many files
- assembly responsibilities are not yet as sharply separated as the lower layers

### 2. `physics` scheme contracts are correct but not yet standardized enough

Current state:

- `physics/properties/*` is now per-property
- `physics/schemes/*` carries scheme-specific recovery logic
- `PropertyEngine` delegates to a scheme

Remaining issue:

- property names and model-derivative keys are still string-based conventions
- scheme capabilities are implicit rather than explicitly declared

Impact:

- future schemes can be added, but extension still relies too much on convention
- failure modes are discovered late rather than validated up front

### 3. Backend schemas are still uneven

Current state:

- backend execution is inside `physics/backends`
- many old cross-layer responsibilities have already been removed

Remaining issue:

- PySCF and ABACUS still expose output-field and dump-field logic in uneven forms
- backend output schema is not yet uniformly modeled

Impact:

- adding a new backend or standardizing outputs will still be more expensive than it should be

### 4. Interface/objective assembly is not yet fully declarative

Current state:

- objectives already request properties through `PropertyEngine`
- scheme selection is explicit

Remaining issue:

- requested-property derivation still lives inside concrete objective logic
- objective term registration is still partly manual

Impact:

- future expansion to multiple schemes/objective families is possible, but still not minimal-friction

### 5. Documentation still mixes current state and history

Current state:

- core architecture docs have been updated

Remaining issue:

- some historical docs still describe removed modules or pre-refactor structures

Impact:

- architecture is harder to understand for new maintainers than necessary

## Phase 1: Standardize Physics Contracts

### Goal

Turn the current `properties + schemes + engine` structure into a stable
extension mechanism for future model input/output designs.

### Design

Introduce explicit contract objects or registries for:

- property names
- model derivative keys
- scheme-supported properties

Possible implementation direction:

- `deepks/physics/contracts.py`
  - `PropertyName`
  - `ModelDerivativeKey`
  - optional `PhysicsContextKey`

- `deepks/physics/schemes/base.py` or equivalent
  - explicit `supported_properties()`
  - explicit `validate_context(requested_properties, context)`

### Required Code Changes

- replace ad hoc string usage where practical with shared constants/enums
- add validation in `PropertyEngine`:
  - unsupported requested property
  - unsupported derivative requirement
  - missing context key

### Deliverables

- stable scheme contract
- explicit engine validation
- fewer extension-by-convention points

### Acceptance Criteria

- requesting an unsupported property fails with a clear error before computation
- missing required context fails with a clear error before computation
- adding a new scheme requires only:
  - a new scheme implementation
  - registration
  - no edits to unrelated property helpers

## Phase 2: Further Thin `interface`

### Goal

Make recipe files choose components instead of translating large config
surfaces themselves.

### Design

Split interface assembly into clearer sublayers:

- model assembly
- objective assembly
- runtime preparation
- task assembly

Possible modules:

- `interface/assembly/models.py`
- `interface/assembly/objectives.py`
- `interface/assembly/runtime.py`

Recipes should shrink toward:

- `build_model(...)`
- `build_scheme(...)`
- `build_objective(...)`
- `prepare_runtime(...)`

### Required Code Changes

- move remaining objective-arg shaping from concrete recipes into shared assembly helpers
- reduce train/test runtime translation logic in recipe files
- keep recipe classes as declarative selectors

### Deliverables

- thinner recipe modules
- more reusable assembly helpers

### Acceptance Criteria

- adding a new recipe mostly means selecting:
  - model family
  - scheme
  - objective terms
  - preprocess policy
- recipe files no longer contain long config-rewrite blocks

## Phase 3: Standardize Backend Schemas

### Goal

Make backend outputs and field selection more uniform across PySCF and ABACUS.

### Design

Split backend structure more explicitly into:

- input builder
- runner
- collector
- output schema / field definitions

Possible modules:

- `physics/backends/pyscf/schema.py`
- `physics/backends/abacus/schema.py`

### Required Code Changes

- separate output field declarations from runtime execution where still mixed
- define a more explicit backend output schema for dumpable quantities
- align naming of backend-produced fields with interface-side consumption

### Deliverables

- explicit backend output schemas
- cleaner runner/collector boundaries

### Acceptance Criteria

- backend field selection no longer depends on scattered implementation details
- a new backend can follow a documented schema pattern instead of reverse-engineering existing code

## Phase 4: Make Objective Assembly More Declarative

### Goal

Reduce manual coupling between objective term selection and property requests.

### Design

Let objective-term definitions declare:

- which predicted properties they require
- which targets they require

Then objective assembly derives:

- `requested_properties`
- enabled terms
- target requirements

More systematically than the current hand-assembled path.

### Required Code Changes

- add richer metadata to objective terms
- centralize derivation of requested properties
- reduce special-case logic in concrete objective adapters

### Deliverables

- more declarative objective definitions
- easier future expansion to new terms and schemes

### Acceptance Criteria

- adding a new objective term does not require editing multiple unrelated places
- requested property derivation is centralized and deterministic

## Phase 5: Documentation Split Between Current State and History

### Goal

Make the current architecture easy to understand without reading historical
refactor notes.

### Design

Split documentation into:

- current-state documents
- archived/historical documents

Recommended approach:

- keep:
  - `ARCHITECTURE.md`
  - `ABSTRACT_CLASS_DEPENDENCY.md`
  - `PHYSICS_SCOPE_AUDIT.md`
  as current-state documents
- move historical summaries under an archive subsection or mark them explicitly as historical

### Required Code Changes

- document-only cleanup
- add explicit “historical / archived” headers where needed

### Deliverables

- current architecture docs that only describe the current code
- reduced confusion for new maintainers

### Acceptance Criteria

- current-state docs no longer point readers toward removed structures as if they were still active
- historical docs are clearly labeled and non-authoritative

## Recommended Execution Order

1. Phase 1: Standardize physics contracts
2. Phase 2: Further thin `interface`
3. Phase 3: Standardize backend schemas
4. Phase 4: Make objective assembly more declarative
5. Phase 5: Documentation split and cleanup

This order is intentional:

- Phase 1 determines how future schemes extend the system
- Phase 2 reduces assembly pressure once the physics contract is more stable
- Phase 3 becomes easier after interfaces between layers are cleaner
- Phase 4 builds on the more stable property/scheme/objective contracts
- Phase 5 should reflect the final stabilized structure, not a moving target

## Practical Completion Definition

The refactor should only be considered fully complete when all of the following
are true:

- a new model family can be added without editing physics helpers
- a new physical recovery scheme can be added without editing property helpers
- a new objective term can be added without editing multiple unrelated assembly points
- backend output behavior is explicit and schema-driven
- current architecture docs match the code without relying on historical context
