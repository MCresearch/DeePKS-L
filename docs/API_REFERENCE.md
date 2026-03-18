# DeePKS-kit API Reference

This document provides a reference for the main APIs in DeePKS-kit.

## Command Line Interface

### Main Commands

```bash
# Train a model
deepks train [options]

# Test a model
deepks test [options]

# Run SCF calculation
deepks scf [options]

# Collect SCF statistics
deepks stats [options]

# Run iterative training
deepks iterate [options]
```

All commands support `-h` or `--help` for detailed usage information.

## Python API

### Data I/O

#### Reader

Read data from a single system:

```python
from deepks.io.readers import Reader

reader = Reader(
    path="path/to/data",
    batch_size=16,
    with_force=True,
    with_eig=False
)

# Sample a batch
batch = reader.sample_train()

# Get all data
all_data = reader.sample_all()

# Get number of frames
nframes = reader.get_nframes()
```

#### GroupReader

Read data from multiple systems:

```python
from deepks.io.readers import GroupReader

reader = GroupReader(
    path_list=["system1", "system2", "system3"],
    batch_size=16,
    group_batch=4,  # Sample from 4 systems at once
    with_force=True
)

# Iterate over batches
for batch in reader:
    # Process batch
    pass

# Sample from specific system
batch = reader.sample_train(idx=0)

# Get statistics
stats = reader.compute_data_stat()

# Compute element constants
elem_const = reader.compute_elem_const(ridge_alpha=1e-8)
```

#### SimpleReader

Simplified reader for basic use cases:

```python
from deepks.io.readers import SimpleReader

reader = SimpleReader(
    path="path/to/data",
    batch_size=16
)

# Use like Reader
batch = reader.sample_train()
```

### Machine Learning

#### CorrNet Model

Neural network model for energy correction:

```python
from deepks.core.ml.models.corrnet import CorrNet

model = CorrNet(
    n_descriptors=10,
    hidden_sizes=[100, 100, 100],
    output_size=1,
    activation='tanh',
    use_resnet=True
)

# Forward pass
output = model(descriptors)

# Save/load model
model.save("model.pth")
model = CorrNet.load("model.pth")
```

#### Training

Train a model:

```python
from deepks.pipelines.train.train import train

train(
    model_file="model.pth",
    train_paths=["train_data"],
    test_paths=["test_data"],
    batch_size=16,
    nepoch=1000,
    start_lr=0.01,
    decay_rate=0.96,
    decay_steps=100,
    print_freq=10,
    save_freq=100
)
```

#### Evaluation

Evaluate model performance:

```python
from deepks.core.ml.eval.evaluator import Evaluator

evaluator = Evaluator(model)

# Evaluate on data
results = evaluator.eval(data_reader)

# Get metrics
mae = results['mae']
rmse = results['rmse']
```

### Physics Calculations

#### SCF Calculation

Run self-consistent field calculation:

```python
from deepks.core.physics.pyscf.scf import DSCF
from pyscf import gto

# Create molecule
mol = gto.M(
    atom='H 0 0 0; H 0 0 0.74',
    basis='ccpvdz'
)

# Create SCF object with model
mf = DSCF(mol, model, xc='HF')

# Run SCF
energy = mf.kernel()

# Get forces
forces = mf.nuc_grad_method().kernel()
```

#### SCF Pipeline

Run SCF on multiple systems:

```python
from deepks.pipelines.scf.run import run_scf

run_scf(
    model_file="model.pth",
    systems=["system1.xyz", "system2.xyz"],
    basis="ccpvdz",
    xc="HF",
    output_dir="scf_results"
)
```

### Workflow Orchestration

#### Tasks

Define computational tasks:

```python
from deepks.orchestration.workflow.task import PythonTask, ShellTask

# Python task
def my_function(arg1, arg2):
    return arg1 + arg2

task = PythonTask(
    func=my_function,
    args=(1, 2),
    workdir="work"
)

# Shell task
shell_task = ShellTask(
    cmd="python script.py",
    workdir="work"
)

# Execute task
result = task.run()
```

#### Workflows

Compose tasks into workflows:

```python
from deepks.orchestration.workflow.workflow import Sequence, Parallel

# Sequential workflow
workflow = Sequence([task1, task2, task3])

# Parallel workflow
workflow = Parallel([task1, task2, task3])

# Mixed workflow
workflow = Sequence([
    task1,
    Parallel([task2, task3]),
    task4
])

# Execute workflow
workflow.run()
```

#### Job Scheduling

Submit jobs to schedulers:

```python
from deepks.orchestration.scheduler.job.dispatcher import Dispatcher

dispatcher = Dispatcher(
    work_dir="jobs",
    machine_file="machine.yaml",
    resources={
        "nproc": 4,
        "time_limit": "24:00:00",
        "mem_limit": "16GB"
    }
)

# Dispatch tasks
dispatcher.run_jobs(tasks)
```

### Iterative Training

Run iterative training workflow:

```python
from deepks.pipelines.iterate.iterate import make_iterate

iterate_workflow = make_iterate(
    init_model="init_model.pth",
    init_scf="scf_input",
    init_train="train_input",
    share_folder="share",
    n_iter=5
)

# Execute iteration
iterate_workflow.run()
```

## Data Formats

### Input Data

Training/testing data should be in HDF5 format with the following structure:

```
data.h5
├── /coords          # Atomic coordinates [nframes, natoms, 3]
├── /cells           # Unit cells [nframes, 3, 3]
├── /atom_types      # Atom types [natoms]
├── /energies        # Total energies [nframes]
├── /forces          # Atomic forces [nframes, natoms, 3] (optional)
├── /descriptors     # Descriptors [nframes, natoms, ndesc]
└── /eigenvalues     # Eigenvalues [nframes, norb] (optional)
```

### Model Files

Models are saved as PyTorch state dictionaries (`.pth` files) containing:
- Model architecture parameters
- Trained weights and biases
- Optimizer state (for checkpointing)

### Configuration Files

Configuration files use YAML format. See `examples/` for detailed examples.

## Error Handling

Common errors and solutions:

### ImportError: No module named 'torch'

Install PyTorch:
```bash
conda install pytorch -c pytorch
```

### ImportError: No module named 'pyscf'

Install PySCF:
```bash
pip install pyscf
```

### Data format errors

Ensure your data follows the HDF5 structure described above. Use the provided data preparation scripts in `tools/`.

## Performance Tips

1. **Batch size**: Larger batch sizes improve GPU utilization but require more memory
2. **Group batch**: Use `group_batch > 1` in GroupReader to sample from multiple systems simultaneously
3. **Data caching**: Enable caching for frequently accessed data
4. **Parallel execution**: Use Parallel workflows for independent tasks
5. **GPU acceleration**: Ensure PyTorch is using GPU for training

## See Also

- [Architecture Documentation](./ARCHITECTURE.md)
- [Examples](../examples/)
- [Refactor Skill](./skills/deepks-architecture-refactor-skill.md)
