"""Default configuration values for DeePKS input handling."""

from copy import deepcopy


def get_default_device():
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


DEFAULT_CONFIG = {
    "type": None,                                                   # High-level task type selector.
    "recipe": "corrnet-energy",                                     # Preset recipe name.
    "runtime": {
        "verbose": 1,                                               # Global logging verbosity level.
        "device": get_default_device(),                             # Default compute device.
        "seed": None,                                               # Global random seed.
        "dtype": None,                                              # Optional global dtype override.
        "workdir": ".",                                             # Base working directory.
        "share_folder": "share",                                    # Shared folder name.
        "cleanup": False,                                           # Whether to clean intermediates.
        "strict": True,                                             # Whether to fail fast on issues.
        "test_log": "log.test",                                     # Default test log filename.
        "io": {
            "ckpt_file": None,                                      # Runtime-level checkpoint file.
            "graph_file": None,                                     # Optional serialized graph file.
        },
        "command": {
            "python": "python",                                     # Python executable for scripts.
            "run_cmd": "mpirun",                                    # Distributed launcher command.
            "abacus_path": "abacus",                                # ABACUS executable path.
        },
        "dispatch": {
            # ========================= common execute part =========================
            "sub_size": 1,                                          # Sub-job chunk size.
            "sub_res": {
                "numb_node": 1,                                     # Nodes per sub-step.
                "task_per_node": 1,                                 # Tasks per node in sub-step.
                "cpus_per_task": 8,                                 # CPUs per sub-step task.
                "numb_gpu": 0,                                      # GPUs per sub-step task.
                "exclusive": True,                                  # Exclusive node usage flag.
            },
            "group_size": 1,                                        # Grouped task count per submit.
            "ingroup_parallel": 1,                                  # Parallel tasks inside one group.
            # ===================== local shell/slurm/pbs part =====================
            "dispatcher": {
                "context": "local",                                 # Execution context backend.
                "batch": "shell",                                   # Batch driver type.
                "remote_profile": None,                             # Remote profile for ssh/cloud.
                "job_record": "jr.json",                            # Job-record filename.
            },
            "resources": {
                "numb_node": 1,                                     # Requested node count.
                "task_per_node": 1,                                 # Requested tasks per node.
                "cpus_per_task": 8,                                 # Requested CPUs per task.
                "numb_gpu": 0,                                      # Requested GPUs per task.
                "time_limit": "24:00:00",                           # Walltime limit.
                "mem_limit": 8,                                     # Memory limit in GB.
                "partition": "",                                    # Scheduler partition/queue.
                "account": "",                                      # Scheduler account name.
                "qos": "",                                          # Scheduler QoS.
                "task_max": 0,                                      # Max in-queue jobs (0 means off).
                "constraint_list": [],                              # Scheduler constraints.
                "license_list": [],                                 # Scheduler license requirements.
                "exclude_list": [],                                 # Excluded nodes list.
                "module_unload_list": [],                           # Modules to unload.
                "module_list": [],                                  # Modules to load.
                "source_list": [],                                  # Shell scripts to source.
                "envs": {},                                         # Environment variables map.
                "with_mpi": False,                                  # Whether submit command uses MPI.
                "cuda_multi_tasks": False,                          # CUDA multi-task mode switch.
                "allow_failure": False,                             # Allow failure without stopping.
            },
            "python": "python",                                     # Python executable override.
            # ========================= dpdispatcher part =========================
            "dpdispatcher_machine": {
                "batch_type": "Shell",                              # dpdispatcher batch backend.
                "context_type": "LocalContext",                     # dpdispatcher context backend.
                "local_root": ".",                                  # Local root path.
                "remote_root": ".",                                 # Remote root path.
            },
            "dpdispatcher_resources": {
                "number_node": 1,                                   # dpdispatcher node count.
                "cpu_per_node": 8,                                  # dpdispatcher CPUs per node.
                "group_size": 1,                                    # dpdispatcher group size.
            },
        },
        "scf": {
            "execute": {
                # ========================= common execute part =====================
                "sub_size": 1,                                      # Sub-job chunk size.
                "sub_res": {
                    "numb_node": 1,                                 # Nodes per sub-step.
                    "task_per_node": 1,                             # Tasks per node in sub-step.
                    "cpus_per_task": 8,                             # CPUs per sub-step task.
                    "numb_gpu": 0,                                  # GPUs per sub-step task.
                    "exclusive": True,                              # Exclusive node usage flag.
                },
                "group_size": 1,                                    # Grouped task count per submit.
                "ingroup_parallel": 1,                              # Parallel tasks inside one group.
                # ===================== local shell/slurm/pbs part =================
                "dispatcher": {
                    "context": "local",                             # Execution context backend.
                    "batch": "shell",                               # Batch driver type.
                    "remote_profile": None,                         # Remote profile for ssh/cloud.
                    "job_record": "jr.json",                        # Job-record filename.
                },
                "resources": {
                    "numb_node": 1,                                 # Requested node count.
                    "task_per_node": 1,                             # Requested tasks per node.
                    "cpus_per_task": 8,                             # Requested CPUs per task.
                    "numb_gpu": 0,                                  # Requested GPUs per task.
                    "time_limit": "24:00:00",                       # Walltime limit.
                    "mem_limit": 8,                                 # Memory limit in GB.
                    "partition": "",                                # Scheduler partition/queue.
                    "account": "",                                  # Scheduler account name.
                    "qos": "",                                      # Scheduler QoS.
                    "task_max": 0,                                  # Max in-queue jobs (0 means off).
                    "constraint_list": [],                          # Scheduler constraints.
                    "license_list": [],                             # Scheduler license requirements.
                    "exclude_list": [],                             # Excluded nodes list.
                    "module_unload_list": [],                       # Modules to unload.
                    "module_list": [],                              # Modules to load.
                    "source_list": [],                              # Shell scripts to source.
                    "envs": {},                                     # Environment variables map.
                    "with_mpi": False,                              # Whether submit command uses MPI.
                    "cuda_multi_tasks": False,                      # CUDA multi-task mode switch.
                    "allow_failure": False,                         # Allow failure without stopping.
                },
                "python": "python",                                 # Python executable override.
                # ========================= dpdispatcher part =======================
                "dpdispatcher_machine": {
                    "batch_type": "Shell",                          # dpdispatcher batch backend.
                    "context_type": "LocalContext",                 # dpdispatcher context backend.
                    "local_root": ".",                              # Local root path.
                    "remote_root": ".",                             # Remote root path.
                },
                "dpdispatcher_resources": {
                    "number_node": 1,                               # dpdispatcher node count.
                    "cpu_per_node": 8,                              # dpdispatcher CPUs per node.
                    "group_size": 1,                                # dpdispatcher group size.
                },
            },
            "command": {
                "python": "python",                                 # Python executable for SCF stage.
                "run_cmd": "mpirun",                                # Launcher command for SCF stage.
                "abacus_path": "abacus",                            # ABACUS executable for SCF stage.
            },
        },
        "train": {
            "execute": {
                # ===================== local shell/slurm/pbs part =================
                "dispatcher": {
                    "context": "local",                             # Execution context backend.
                    "batch": "shell",                               # Batch driver type.
                    "remote_profile": None,                         # Remote profile for ssh/cloud.
                    "job_record": "jr.json",                        # Job-record filename.
                },
                "resources": {
                    "numb_node": 1,                                 # Requested node count.
                    "task_per_node": 1,                             # Requested tasks per node.
                    "cpus_per_task": 8,                             # Requested CPUs per task.
                    "numb_gpu": 0,                                  # Requested GPUs per task.
                    "time_limit": "24:00:00",                       # Walltime limit.
                    "mem_limit": 8,                                 # Memory limit in GB.
                    "partition": "",                                # Scheduler partition/queue.
                    "account": "",                                  # Scheduler account name.
                    "qos": "",                                      # Scheduler QoS.
                    "task_max": 0,                                  # Max in-queue jobs (0 means off).
                    "constraint_list": [],                          # Scheduler constraints.
                    "license_list": [],                             # Scheduler license requirements.
                    "exclude_list": [],                             # Excluded nodes list.
                    "module_unload_list": [],                       # Modules to unload.
                    "module_list": [],                              # Modules to load.
                    "source_list": [],                              # Shell scripts to source.
                    "envs": {},                                     # Environment variables map.
                    "with_mpi": False,                              # Whether submit command uses MPI.
                    "cuda_multi_tasks": False,                      # CUDA multi-task mode switch.
                    "allow_failure": False,                         # Allow failure without stopping.
                },
                "python": "python",                                 # Python executable override.
                # ========================= dpdispatcher part =======================
                "dpdispatcher_machine": {
                    "batch_type": "Shell",                          # dpdispatcher batch backend.
                    "context_type": "LocalContext",                 # dpdispatcher context backend.
                    "local_root": ".",                              # Local root path.
                    "remote_root": ".",                             # Remote root path.
                },
                "dpdispatcher_resources": {
                    "number_node": 1,                               # dpdispatcher node count.
                    "cpu_per_node": 8,                              # dpdispatcher CPUs per node.
                    "group_size": 1,                                # dpdispatcher group size.
                },
            },
            "command": {
                "python": "python",                                 # Python executable for train stage.
            },
        },
    },
    "data": {
        "train": None,                                              # Training dataset path/object.
        "test": None,                                               # Test/validation dataset path/object.
        "stages": [],                                               # Optional stage-owned train/test data specs.
        "systems": None,                                            # System list/path for SCF or stats.
        "loader": {
            "batch_size": 16,                                       # Reader batch size.
            "group_batch": 1,                                       # Number of grouped systems per batch.
            "extra_label": False,                                   # Whether to read optional labels.
            "conv_filter": True,                                    # Whether to filter by convergence flag.
            "conv_name": "conv",                                    # Convergence field/file stem.
            "read_overlap": False,                                  # Whether to read overlap-related data.
            "e_name": "l_e_delta",                                  # Energy label field/file stem.
            "d_name": "dm_eig",                                     # Descriptor field/file stem.
            "f_name": "l_f_delta",                                  # Force label field/file stem.
            "gvx_name": "grad_vx",                                  # Force gradient helper field stem.
            "s_name": "l_s_delta",                                  # Stress label field/file stem.
            "gvepsl_name": "grad_vepsl",                            # Stress gradient helper field stem.
            "o_name": "l_o_delta",                                  # Orbital label field/file stem.
            "op_name": "orbital_precalc",                           # Precomputed orbital field stem.
            "h_name": "l_h_delta",                                  # Delta-Hamiltonian field/file stem.
            "vdp_name": "v_delta_precalc",                          # Precomputed v_delta field stem.
            "vdrp_name": "vdr_precalc",                             # Precomputed v_delta_r field stem.
            "phialpha_name": "phialpha",                            # Orbital-phase helper field stem.
            "gevdm_name": "grad_evdm",                              # Descriptor-gradient helper stem.
            "hr_name": "l_hr_delta",                                # Real-space Hamiltonian field stem.
            "h_base_name": "h_base",                                # Base Hamiltonian field/file stem.
            "h_ref_name": "hamiltonian",                            # Reference Hamiltonian field stem.
            "overlap_name": "overlap",                              # Overlap matrix field/file stem.
            "eg_name": "eg_base",                                   # Reference eigenvalue field stem.
            "gveg_name": "grad_veg",                                # Eigenvalue-gradient helper stem.
            "gldv_name": "grad_ldv",                                # Local-potential-gradient stem.
            "atom_name": "atom",                                    # Atomic information field/file stem.
            "box_name": "box",                                      # Cell box field/file stem.
            "iR_mat_name": "iR_mat",                                # R-space neighbor index mapping (deepks_v_delta=-2).
            "phialpha_r_name": "phialpha_r",                        # R-space LCAO x projector overlap (deepks_v_delta=-2).
            "orb_list": None,                                       # Optional orbital file list.
            "alpha_list": None,                                     # Optional alpha file list.
            "eigh_method": 1,                                       # Eigensolver mode for reader.
        },
        "targets": {
            "energy": None,                                         # Energy target alias.
            "force": None,                                          # Force target alias.
            "stress": None,                                         # Stress target alias.
            "orbital": None,                                        # Orbital target alias.
            "v_delta": None,                                        # v_delta target alias.
            "vdr": None,                                            # v_delta_r target alias.
            "phi": None,                                            # Phi target alias.
            "band": None,                                           # Band target alias.
            "hamiltonian_levels": None,                             # Hierarchical Hamiltonian target stems.
        },
    },
    "physics": {
        "representation": {
            "name": "dm_eig",                                       # Descriptor/representation family.
            "params": {
                "proj_basis": "ccpvdz",                             # Projection basis for descriptors.
            },
        },
        "backend": {
            "name": None,                                           # Physics backend name.
            "input": {
                # ======================= common backend input =======================
                "model_file": None,                                 # DeePKS model used by backend.

                # ======================== pyscf part begin =========================
                "basis": "ccpvdz",                                  # PySCF AO basis.
                "proj_basis": "ccpvdz",                             # Projection basis for PySCF path.
                "mol_args": {
                    "charge": 0,                                    # Molecular total charge.
                    "spin": 0,                                      # Spin setting (2S).
                    "unit": "Angstrom",                             # Coordinate unit.
                },
                "scf_args": {
                    "conv_tol": 1e-7,                               # PySCF SCF convergence tolerance.
                    "max_cycle": 50,                                # PySCF max SCF cycles.
                },
                "conv_tol": None,                                   # Override PySCF conv_tol.
                "conv_tol_grad": None,                              # Override PySCF grad conv_tol.
                "grids_level": None,                                # PySCF integration grid level.
                "verbose": None,                                    # PySCF verbosity override.
                "chkfile": None,                                    # PySCF checkpoint path.
                "penalty_terms": None,                              # PySCF penalty term settings.
                # ========================= pyscf part end ==========================

                # ======================== abacus part begin ========================
                "abacus_path": "abacus",                            # ABACUS executable path.
                "run_cmd": "mpirun",                                # Launcher command for ABACUS.
                "orb_files": ["orb"],                               # Orbital file list.
                "pp_files": ["upf"],                                # Pseudopotential file list.
                "proj_file": ["orb"],                               # Projection orbital file list.
                "basis_file": None,                                 # Explicit ABACUS basis file path.
                "basis_name": None,                                 # Named ABACUS basis selector.
                "input_args": None,                                 # Extra ABACUS INPUT overrides.
                "kpt_file": None,                                   # ABACUS KPT file path.
                "stru_file": None,                                  # ABACUS STRU file path.
                "coord_file": None,                                 # Alternative coordinate file path.
                "lattice_constant": 1,                              # Lattice constant scale.
                "lattice_vector": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],# Lattice vectors.
                "coord_type": "Cartesian",                          # Coordinate type.
                "nspin": 1,                                         # Spin channels.
                "symmetry": 0,                                      # Symmetry switch.
                "nbands": None,                                     # Number of bands.
                "ecutwfc": 50,                                      # Plane-wave cutoff.
                "scf_thr": 1e-7,                                    # SCF threshold.
                "scf_nmax": 50,                                     # Maximum SCF iterations.
                "dft_functional": "pbe",                            # XC functional.
                "basis_type": "lcao",                               # Basis mode.
                "gamma_only": 1,                                    # Gamma-only flag.
                "k_points": None,                                   # Explicit k-point mesh.
                "kspacing": None,                                   # Auto k-point spacing.
                "smearing_method": "gaussian",                      # Smearing method.
                "smearing_sigma": 0.02,                             # Smearing width.
                "mixing_type": "pulay",                             # Mixing method.
                "mixing_beta": 0.4,                                 # Mixing beta.
                "cal_force": 0,                                     # Compute forces flag.
                "cal_stress": 0,                                    # Compute stress flag.
                "deepks_bandgap": 0,                                # DeePKS bandgap output flag.
                "deepks_v_delta": 0,                                # DeePKS v_delta output flag.
                "deepks_out_labels": 1,                             # DeePKS label output flag.
                "deepks_scf": 0,                                    # DeePKS in-SCF switch.
                "out_wfc_lcao": 0,                                  # Output LCAO wfc flag.
                "ntype": None,                                      # Number of atom types.
                # ========================= abacus part end =========================
            },
            "output": {
                "dump_dir": "scf_results",                          # Output dump directory.
                "dump_fields": ["e_tot", "e_base", "dm_eig", "conv"], # Output fields to dump.
            },
            "profiles": [],                                         # Optional per-group backend/profile overrides.
        },
    },
    "ml": {
        "model": {
            "family": "corrnet",                                    # Model family.
            "args": {
                "input_dim": None,                                  # Input descriptor dimension.
                "hidden_sizes": [100, 100, 100],                    # Hidden layer sizes.
            "actv_fn": "gelu",                                      # Activation function.
                "output_scale": 100.0,                              # Global output scaling.
                "use_resnet": True,                                 # Residual connection switch.
                "layer_norm": False,                                # LayerNorm switch/mode.
                "embedding": None,                                  # Embedding config.
                "proj_basis": None,                                 # Projection basis path/object.
                "shell_sec": None,                                  # Shell partition list.
                "elem_table": None,                                 # Element constant table.
                "input_shift": 0,                                   # Input normalization shift init.
                "input_scale": 1,                                   # Input normalization scale init.
                "max_depth": None,                                  # Hierarchical model maximum depth.
                "max_output_dim": None,                             # Hierarchical model maximum output dim.
                "trunk_hidden_sizes": None,                         # Hierarchical trunk hidden sizes.
                "head_hidden_sizes": None,                          # Hierarchical head hidden sizes.
                "shared_trunk": True,                               # Hierarchical trunk-sharing switch.
            },
        },
        "preprocess": {
            "preshift": True,                                       # Enable label pre-shift.
            "prescale": [1, 1],                                     # Pre-scale configuration.
            "prescale_sqrt": False,                                 # Use sqrt on scale stats.
            "prescale_clip": 0,                                     # Clip threshold for scaling.
            "prefit": True,                                         # Enable linear prefit.
            "prefit_ridge": 10,                                     # Ridge alpha for prefit.
            "prefit_trainable": False,                              # Keep prefit layer trainable.
        },
        "objective": {
            "losses": [],                                           # Structured loss definitions.
            "energy_per_atom": None,                                # Energy-per-atom option.
            "grad_penalty": None,                                   # Gradient penalty setting.
            "vd_divide_by_nlocal": False,                           # Normalize vd by nlocal.
            "vd_masked_loss": 0,                                    # Masked vd loss mode.
            "vd_masked_S_threshold": 1e-6,                          # S threshold for masked vd loss.
            "vd_masked_H_threshold": 1e-6,                          # H threshold for masked vd loss.
            "vd_masked_width": 1,                                   # Width for masked vd loss.
            "use_safe_eigh": False,                                 # Safer eigendecomp switch.
            "terms": [],                                            # Hierarchical global term definitions.
            "level_losses": [],                                     # Hierarchical level-wise loss definitions.
        },
        "train": {
            "batch_size": 16,                                       # Training batch size.
            "group_batch": 1,                                       # Grouped-structure batch size.
            "epochs": 1000,                                         # Total training epochs.
            "display_epoch": 100,                                   # Epoch interval for printing.
            "display_detail_test": 0,                               # Detail level for test output.
            "display_natom_loss": False,                            # Print natom-wise loss.
            "fix_embedding": False,                                 # Freeze embedding layers.
            "stage_schedule": [],                                   # In-run staged training schedule.
            "optimizer": {
                "lr": 0.01,                                         # Initial learning rate.
                "weight_decay": 0.0,                                # Weight decay.
            },
            "scheduler": {
                "decay_steps": 100,                                 # LR decay interval.
                "decay_rate": 0.96,                                 # LR multiplicative decay.
                "stop_lr": None,                                    # LR lower bound.
                "decay_rate_iter": None,                            # Per-iteration decay override.
            },
        },
        "checkpoint": {
            "file": None,                                           # Model file path.
            "restart": None,                                        # Restart source path.
            "ckpt_file": None,                                      # Training ckpt save/load path.
        },
        "fit_elem": False,                                          # Fit element-wise constants.
    },
    "iterate": {
        "n_iter": 10,                                               # Number of outer iterations.
        "use_init": False,                                          # Whether to run init phase.
        "share_folder": "share",                                    # Shared folder across phases.
        "cleanup": False,                                           # Cleanup intermediate outputs.
        "strict": True,                                             # Strict mode in iterate workflow.
    },
}

def get_default_backend_input(backend_name):
    backend_input = DEFAULT_CONFIG["physics"]["backend"]["input"]
    if backend_name == "pyscf":
        return {
            key: deepcopy(backend_input[key])
            for key in (
                "basis",
                "proj_basis",
                "model_file",
                "mol_args",
                "scf_args",
                "conv_tol",
                "conv_tol_grad",
                "grids_level",
                "verbose",
                "chkfile",
                "penalty_terms",
            )
        }
    if backend_name == "abacus":
        return {
            key: deepcopy(backend_input[key])
            for key in (
                "abacus_path",
                "run_cmd",
                "orb_files",
                "pp_files",
                "proj_file",
                "basis_file",
                "basis_name",
                "input_args",
                "kpt_file",
                "stru_file",
                "coord_file",
                "lattice_constant",
                "lattice_vector",
                "coord_type",
                "nspin",
                "symmetry",
                "nbands",
                "ecutwfc",
                "scf_thr",
                "scf_nmax",
                "dft_functional",
                "basis_type",
                "gamma_only",
                "k_points",
                "kspacing",
                "smearing_method",
                "smearing_sigma",
                "mixing_type",
                "mixing_beta",
                "cal_force",
                "cal_stress",
                "deepks_bandgap",
                "deepks_v_delta",
                "deepks_out_labels",
                "deepks_scf",
                "out_wfc_lcao",
                "ntype",
            )
        }
    return {}


def get_default_config(task_type=None, scf_soft=None):
    config = deepcopy(DEFAULT_CONFIG)
    config["runtime"]["device"] = get_default_device()
    if task_type is not None:
        config["type"] = task_type
    if task_type in {"scf", "stats", "iterate"} and scf_soft in {"pyscf", "abacus"}:
        config["physics"]["backend"]["name"] = scf_soft
        config["physics"]["backend"]["input"] = get_default_backend_input(scf_soft)
    return config
