# DeePKS-kit

DeePKS-kit is a program to generate accurate energy functionals for quantum chemistry systems,
for both perturbative scheme (DeePHF) and self-consistent scheme (DeePKS).

The program provides a unified command line interface `deepks` that accepts a configuration file
specifying the workflow type and parameters:
- `type: scf`: run self-consistent field calculation with given energy model
- `type: train`: train a neural network based post-HF energy functional model
- `type: test`: test the post-HF model with given data and show statistics
- `type: iterate`: iteratively train a self-consistent model by combining SCF and training

## Installation

DeePKS-kit is a pure python library so it can be installed following the standard `git clone` then `pip install` procedure. Note that the two main requirements `pytorch` and `pyscf` will not be installed automatically so you will need to install them manually in advance. Below is a more detailed instruction that includes installing the required libraries in the environment.

We use `conda` here as an example. So first you may need to install [Anaconda](https://docs.anaconda.com/anaconda/install/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

To reduce the possibility of library conflicts, we suggest create a new environment (named `deepks`) with basic dependencies installed (optional):
```bash
conda create -n deepks numpy scipy h5py ruamel.yaml paramiko
conda activate deepks
```
Now you are in the new environment called `deepks`.
Next, install [PyTorch](https://pytorch.org/get-started/locally/)
```bash
# assuming a GPU with cudatoolkit 10.2 support
conda install pytorch cudatoolkit=10.2 -c pytorch
```
and [PySCF](https://github.com/pyscf/pyscf).
```bash
# the conda package does not support python >= 3.8 so we use pip
pip install pyscf
```

Once the environment has been setup properly, using pip to install DeePKS-kit:
```bash
pip install git+https://github.com/deepmodeling/deepks-kit/
```

## Usage

The unified CLI accepts a YAML configuration file that specifies the workflow type and parameters:

```bash
deepks config.yaml
```

Example configuration for iterative training:
```yaml
type: iterate
n_iter: 5
systems_train:
  - path/to/train/systems
systems_test:
  - path/to/test/systems
scf_soft: pyscf
scf_input:
  basis: ccpvdz
train_input:
  n_epoch: 100
  batch_size: 16
```

An relatively detailed description of the `deepks-kit` library can be found in [here](https://arxiv.org/pdf/2012.14615.pdf). Please also refer to the reference for the description of methods.

Please see [`examples`](./examples) folder for the usage of `deepks-kit` library. A detailed example with executable data for single water molecules can be found [here](./examples/water_single). A more complicated one for training water clusters can be found [here](./examples/water_cluster).

Check [this input file](./examples/water_cluster/args.yaml) for detailed explanation for possible input parameters, and also [this one](./examples/water_cluster/shell.yaml) if you would like to run on local machine instead of using Slurm scheduler.

## Architecture

DeePKS-kit follows a modular architecture:

- `deepks/core/`: Core implementations
  - `physics/`: Physics backends (PySCF, ABACUS)
  - `ml/`: Machine learning components (models, training, evaluation)
- `deepks/workflows/`: High-level workflow orchestration
  - `scf/`: SCF workflow
  - `train/`: Training workflow
  - `iterate/`: Iterative workflow
- `deepks/orchestration/`: Task scheduling and execution
- `deepks/cli/`: Command-line interface

## References

[1] Chen, Y., Zhang, L., Wang, H. and E, W., 2020. Ground State Energy Functional with Hartree–Fock Efficiency and Chemical Accuracy. The Journal of Physical Chemistry A, 124(35), pp.7155-7165.

[2] Chen, Y., Zhang, L., Wang, H. and E, W., 2021. DeePKS: A Comprehensive Data-Driven Approach toward Chemically Accurate Density Functional Theory. Journal of Chemical Theory and Computation, 17(1), pp.170–181.


<!-- ## TODO

- [ ] Print loss separately for E and F in training.
- [ ] Rewrite all `print` function using `logging`.
- [ ] Write a detailed README and more docs.
- [ ] Add unit tests. -->


