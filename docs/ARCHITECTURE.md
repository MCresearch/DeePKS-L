# DeePKS-kit Architecture

This document describes the current architecture of DeePKS-kit after the Phase 5 refactoring.

## Overview

DeePKS-kit follows a three-layer architecture that separates concerns and enables maintainability:

```
┌─────────────────────────────────────────┐
│         CLI & Orchestration             │  Command-line interface and workflow management
├─────────────────────────────────────────┤
│              I/O Layer                  │  Data reading, transformation, and adaptation
├─────────────────────────────────────────┤
│             Core Layer                  │  ML models and physics implementations
└─────────────────────────────────────────┘
```

## Package Structure

```
deepks/
├── cli/                    # Command-line entry points
│   └── main.py            # Main CLI with train/test/scf/stats/iterate commands
│
├── orchestration/         # Workflow and task orchestration
│   ├── workflow/          # Task definitions and workflow composition
│   │   ├── task.py       # Task base classes (PythonTask, GroupBatchTask, etc.)
│   │   └── workflow.py   # Workflow composition (Sequence, Parallel, etc.)
│   ├── scheduler/         # Job scheduling and execution
│   │   └── job/          # Job dispatchers (Slurm, PBS, Shell, SSH)
│   ├── checkpoint/        # Checkpoint and state management
│   └── state/            # Runtime state tracking
│
├── io/                    # Data I/O layer
│   ├── readers/           # Data readers
│   │   ├── reader.py     # Base Reader for single system
│   │   ├── grouped_reader.py  # GroupReader for multiple systems
│   │   ├── simple_reader.py   # SimpleReader for basic use cases
│   │   ├── sampling.py   # Sampling utilities
│   │   └── stats.py      # Statistics computation
│   ├── transforms/        # Data transformations
│   │   ├── batch.py      # Batch operations (concat, split)
│   │   └── linalg.py     # Linear algebra transformations
│   ├── schemas/           # Data schemas and field definitions
│   │   └── reader_fields.py
│   ├── adapters/          # Backend adapters
│   │   ├── model_backend.py    # ML model backend adapter
│   │   └── physics_backend.py  # Physics backend adapter
│   └── writers/           # Data writers (future)
│
├── core/                  # Core implementations
│   ├── contracts/         # Interface contracts
│   │   ├── backends.py   # ModelBackend, PhysicsBackend protocols
│   │   └── sample_schema.py  # SampleSchema definition
│   ├── ml/               # Machine learning components
│   │   ├── models/       # Neural network models
│   │   │   └── corrnet.py  # CorrNet model implementation
│   │   ├── train/        # Training logic
│   │   │   └── train.py  # Training loop and utilities
│   │   ├── eval/         # Evaluation logic
│   │   │   ├── evaluator.py  # Model evaluator
│   │   │   └── test.py   # Testing utilities
│   │   ├── losses/       # Loss functions
│   │   └── utils.py      # ML utilities
│   └── physics/          # Physics implementations
│       ├── pyscf/        # PySCF-based implementations
│       │   ├── scf.py    # SCF calculations (RDSCF, UDSCF)
│       │   ├── grad.py   # Gradient calculations
│       │   ├── run.py    # SCF runner
│       │   ├── stats.py  # Statistics collection
│       │   ├── fields.py # Field calculations
│       │   └── penalty.py # Penalty terms
│       ├── abacus/       # ABACUS interface (future)
│       └── operators/    # Physics operators
│
├── pipelines/            # High-level pipeline entry points
│   ├── train/            # Training pipeline
│   │   ├── train.py     # Training entry point
│   │   └── test.py      # Testing entry point
│   ├── scf/             # SCF pipeline
│   │   ├── run.py       # SCF execution
│   │   └── stats.py     # SCF statistics
│   └── iterate/         # Iterative training pipeline
│       ├── iterate.py   # Main iteration logic
│       ├── template.py  # PySCF template generation
│       ├── template_abacus.py    # ABACUS template generation
│       ├── generator_abacus.py   # ABACUS input file generation
│       └── utils.py     # Iteration utilities
│
├── tools/               # Standalone utility scripts
└── compat/             # Reserved for future compatibility utilities
```

## Dependency Rules

The architecture enforces unidirectional dependencies:

```
orchestration → io → core
pipelines → io → core
pipelines → orchestration
cli → pipelines
cli → orchestration
```

**Prohibited:**
- Core components must NOT import from io, orchestration, or pipelines
- I/O components must NOT import from orchestration or pipelines
- No circular dependencies between any layers

## Key Components

### CLI Layer (`deepks.cli`)

Entry point for all user interactions. Provides five main commands:

- `deepks train` - Train a neural network model
- `deepks test` - Test model performance
- `deepks scf` - Run self-consistent field calculations
- `deepks stats` - Collect SCF statistics
- `deepks iterate` - Iterative training workflow

### Orchestration Layer (`deepks.orchestration`)

Manages workflow execution and job scheduling:

- **Tasks**: Atomic units of work (PythonTask, ShellTask, GroupBatchTask)
- **Workflows**: Composition of tasks (Sequence, Parallel)
- **Schedulers**: Job execution backends (Slurm, PBS, Shell, SSH)
- **Checkpoints**: State persistence and recovery

### I/O Layer (`deepks.io`)

Handles all data input/output operations:

- **Readers**: Load training/testing data from various formats
- **Transforms**: Data preprocessing and augmentation
- **Schemas**: Define data structure and validation
- **Adapters**: Bridge between I/O and core components

### Core Layer (`deepks.core`)

Contains the core scientific implementations:

- **ML**: Neural network models, training, and evaluation
- **Physics**: Quantum chemistry calculations (PySCF, ABACUS)
- **Contracts**: Interface definitions for extensibility

### Pipelines Layer (`deepks.pipelines`)

High-level workflows that combine core components:

- **Train Pipeline**: Model training workflow
- **SCF Pipeline**: Self-consistent field calculation workflow
- **Iterate Pipeline**: Iterative training with SCF feedback

## Import Conventions

### Canonical Import Paths

Always use these canonical paths in new code:

```python
# Readers
from deepks.io.readers import Reader, GroupReader, SimpleReader

# ML components
from deepks.core.ml.models.corrnet import CorrNet
from deepks.core.ml.train.train import train
from deepks.core.ml.eval.evaluator import Evaluator

# Physics components
from deepks.core.physics.pyscf.scf import RDSCF, UDSCF
from deepks.core.physics.pyscf.run import run_scf

# Orchestration
from deepks.orchestration.workflow.task import PythonTask
from deepks.orchestration.workflow.workflow import Sequence
from deepks.orchestration.scheduler.job.dispatcher import Dispatcher

# Pipelines
from deepks.pipelines.train.train import main as train_main
from deepks.pipelines.scf.run import main as scf_main
from deepks.pipelines.iterate.iterate import make_iterate
```

## Testing

Tests are organized by type:

- `tests/data/` - Shared test data and golden outputs
- `tests/smoke/` - Smoke tests for CLI and basic functionality
- `tests/unit/` - Unit tests for individual components
- `tests/integration/` - Integration tests for component interactions

Run tests with:
```bash
# All tests (requires full environment)
pytest tests/

# Exclude tests requiring pyabacus
pytest tests/ -m "not pyabacus"
```

## Development Guidelines

1. **Follow the dependency rules** - Never import upward in the layer hierarchy
2. **Use canonical imports** - Always import from the canonical package paths
3. **Write tests** - Add tests for new functionality
4. **Document interfaces** - Use docstrings and type hints
5. **Keep layers separate** - Don't mix concerns across layers

## Migration from Legacy Code

Legacy import paths have been removed. If you have old code, update imports:

| Old Path | New Path |
|----------|----------|
| `deepks.model.reader` | `deepks.io.readers` |
| `deepks.model.model` | `deepks.core.ml.models.corrnet` |
| `deepks.model.train` | `deepks.pipelines.train.train` |
| `deepks.model.test` | `deepks.pipelines.train.test` |
| `deepks.scf.*` | `deepks.core.physics.pyscf.*` |
| `deepks.task.*` | `deepks.orchestration.workflow.*` |
| `deepks.iterate.*` | `deepks.pipelines.iterate.*` |

## References

- [Refactor Skill Documentation](./skills/deepks-architecture-refactor-skill.md)
- [Phase 5 Status](./refactor_phase1_structure_map.md)
