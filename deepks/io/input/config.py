"""Central input schema, defaults, normalization, and documentation helpers."""

from copy import deepcopy


def _get_default_device():
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


VALID_TASK_TYPES = ("train", "test", "scf", "stats", "iterate")
VALID_SCF_BACKENDS = ("pyscf", "abacus")
DEFAULT_SCF_BACKEND = "abacus"

PYSCF_BACKEND_KEYS = (
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

ABACUS_BACKEND_KEYS = (
    "orb_files",
    "pp_files",
    "proj_file",
    "basis_file",
    "basis_name",
    "abacus_path",
    "run_cmd",
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

GLOBAL_PARAM_KEYS = {
    "type",
    "verbose",
    "device",
}

SCF_PARAM_KEYS = {
    "systems",
    "scf_soft",
    "dump_dir",
    "dump_fields",
    "group",
    "model_file",
    "proj_basis",
    "scf_machine",
    "scf_pyscf",
    "scf_abacus",
}

STATS_PARAM_KEYS = SCF_PARAM_KEYS | {
    "stats_fields",
    "test_sys",
    "test_dump",
    "with_conv",
    "with_e",
    "with_f",
    "e_name",
    "f_name",
}

TRAIN_PARAM_KEYS = {
    "systems_train",
    "systems_test",
    "data_args",
    "model_args",
    "preprocess_args",
    "train_args",
    "proj_basis",
    "fit_elem",
    "restart",
    "ckpt_file",
    "seed",
    "device",
    "model_file",
    "e_name",
    "d_name",
    "group",
    "output_prefix",
    "batch_size",
}

TEST_PARAM_KEYS = {
    "systems_test",
    "data_paths",
    "model_file",
    "group",
    "e_name",
    "d_name",
    "output_prefix",
    "batch_size",
}

ITERATE_PARAM_KEYS = {
    "n_iter",
    "workdir",
    "share_folder",
    "cleanup",
    "strict",
    "init_model",
    "init_scf",
    "init_scf_abacus",
    "init_train",
    "init_scf_machine",
    "init_train_machine",
    "scf_input",
    "scf_pyscf",
    "scf_abacus",
    "train_input",
    "scf_machine",
    "train_machine",
    "scf_soft",
    "systems_train",
    "systems_test",
    "proj_basis",
    "model_args",
    "preprocess_args",
    "train_args",
    "data_args",
    "seed",
    "fit_elem",
    "restart",
    "ckpt_file",
    "device",
}

TASK_PARAM_KEYS = {
    "global_param": GLOBAL_PARAM_KEYS,
    "scf_param": SCF_PARAM_KEYS,
    "stats_param": STATS_PARAM_KEYS,
    "train_param": TRAIN_PARAM_KEYS,
    "test_param": TEST_PARAM_KEYS,
    "iterate_param": ITERATE_PARAM_KEYS,
}

REQUIRED_TASK_KEYS = {
    "scf": ("systems",),
    "train": ("systems_train",),
    "test": ("systems_test", "model_file"),
    "stats": ("systems",),
    "iterate": ("systems_train",),
}

DEFAULT_COMMON = {
    "verbose": 1,
    "device": _get_default_device(),
}

DEFAULT_SCF_COMMON = {
    "scf_soft": DEFAULT_SCF_BACKEND,
    "dump_fields": ["e_tot", "e_base", "dm_eig", "conv"],
    "group": False,
}

DEFAULT_SCF_PYSCF = {
    "basis": "ccpvdz",
    "proj_basis": "ccpvdz",
    "mol_args": {
        "charge": 0,
        "spin": 0,
        "unit": "Angstrom",
    },
    "scf_args": {
        "conv_tol": 1e-7,
        "max_cycle": 50,
    },
}

DEFAULT_SCF_ABACUS = {
    "orb_files": ["orb"],
    "pp_files": ["upf"],
    "proj_file": ["orb"],
    "abacus_path": "/usr/local/bin/ABACUS.mpi",
    "run_cmd": "mpirun",
    "lattice_constant": 1,
    "lattice_vector": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "coord_type": "Cartesian",
    "nspin": 1,
    "symmetry": 0,
    "nbands": None,
    "ecutwfc": 50,
    "scf_thr": 1e-7,
    "scf_nmax": 50,
    "dft_functional": "pbe",
    "basis_type": "lcao",
    "gamma_only": 1,
    "k_points": None,
    "kspacing": None,
    "smearing_method": "gaussian",
    "smearing_sigma": 0.02,
    "mixing_type": "pulay",
    "mixing_beta": 0.4,
    "cal_force": 0,
    "cal_stress": 0,
    "deepks_bandgap": 0,
    "deepks_v_delta": 0,
    "deepks_out_labels": 1,
    "deepks_scf": 0,
    "out_wfc_lcao": 0,
}

DEFAULT_TRAIN = {
    "model_args": {
        "hidden_sizes": [100, 100, 100],
        "output_scale": 100.0,
        "use_resnet": True,
    },
    "preprocess_args": {
        "preshift": True,
        "prescale": [1, 1],
    },
    "train_args": {
        "n_epoch": 1000,
        "batch_size": 16,
        "eval_batch_size": None,
        "start_lr": 0.01,
        "decay_rate": 0.96,
        "decay_steps": 100,
        "print_freq": 100,
        "save_freq": 1000,
        "restart": None,
        "ckpt_file": None,
    },
}

DEFAULT_TEST = {
    "batch_size": 16,
}

DEFAULT_ITERATE = {
    "n_iter": 10,
    "workdir": ".",
    "share_folder": "share",
    "cleanup": False,
    "strict": True,
}

DOC_SECTIONS = (
    {
        "title": "Core parameters",
        "rows": (
            {
                "name": "type",
                "type": "string",
                "tasks": "all",
                "default": "required",
                "description": "Top-level task selector. Supported values: train, test, scf, stats, iterate.",
            },
            {
                "name": "verbose",
                "type": "int",
                "tasks": "all",
                "default": repr(DEFAULT_COMMON["verbose"]),
                "description": "Global verbosity level.",
            },
            {
                "name": "device",
                "type": "string",
                "tasks": "train, test, scf, iterate",
                "default": repr(DEFAULT_COMMON["device"]),
                "description": "Execution device for ML-backed runtime paths.",
            },
            {
                "name": "scf_soft",
                "type": "string",
                "tasks": "scf, stats, iterate",
                "default": repr(DEFAULT_SCF_BACKEND),
                "description": "SCF backend selector. Supported values: pyscf, abacus.",
            },
        ),
    },
    {
        "title": "Train / test data parameters",
        "rows": (
            {
                "name": "systems_train",
                "type": "list[str] | str",
                "tasks": "train, iterate",
                "default": "required",
                "description": "Training system paths.",
            },
            {
                "name": "systems_test",
                "type": "list[str] | str | null",
                "tasks": "train, test, iterate",
                "default": "optional",
                "description": "Test system paths. For packaged test configs this is also mapped to data_paths.",
            },
            {
                "name": "model_args",
                "type": "dict",
                "tasks": "train, iterate",
                "default": repr(DEFAULT_TRAIN["model_args"]),
                "description": "Model construction arguments.",
            },
            {
                "name": "data_args",
                "type": "dict",
                "tasks": "train, iterate",
                "default": "{}",
                "description": "Dataset reader arguments.",
            },
            {
                "name": "preprocess_args",
                "type": "dict",
                "tasks": "train, iterate",
                "default": repr(DEFAULT_TRAIN["preprocess_args"]),
                "description": "Preprocessing arguments applied before training.",
            },
            {
                "name": "train_args",
                "type": "dict",
                "tasks": "train, iterate",
                "default": repr(DEFAULT_TRAIN["train_args"]),
                "description": "Training loop arguments.",
            },
            {
                "name": "proj_basis",
                "type": "any",
                "tasks": "train, scf, iterate",
                "default": "backend-specific",
                "description": "Projection basis or basis file used by descriptor / SCF paths.",
            },
            {
                "name": "fit_elem",
                "type": "bool",
                "tasks": "train, iterate",
                "default": "False",
                "description": "Whether to fit elemental energy constants.",
            },
            {
                "name": "restart",
                "type": "str | null",
                "tasks": "train, iterate",
                "default": "None",
                "description": "Checkpoint/model path used to restart training.",
            },
            {
                "name": "ckpt_file",
                "type": "str | null",
                "tasks": "train, iterate",
                "default": "None",
                "description": "Checkpoint output path.",
            },
            {
                "name": "model_file",
                "type": "str | null",
                "tasks": "test, scf, stats",
                "default": "optional",
                "description": "Model file path used for inference or evaluation.",
            },
            {
                "name": "batch_size",
                "type": "int",
                "tasks": "test",
                "default": repr(DEFAULT_TEST["batch_size"]),
                "description": "Default batch size for packaged test execution.",
            },
            {
                "name": "e_name / d_name / output_prefix / group",
                "type": "mixed",
                "tasks": "test",
                "default": "runtime defaults",
                "description": "Legacy test-runtime compatibility parameters preserved by packaging / dispatch.",
            },
        ),
    },
    {
        "title": "SCF / stats common parameters",
        "rows": (
            {
                "name": "systems",
                "type": "list[str] | str",
                "tasks": "scf, stats",
                "default": "required",
                "description": "System paths used by SCF or stats workflows.",
            },
            {
                "name": "dump_dir",
                "type": "str",
                "tasks": "scf, stats",
                "default": "optional",
                "description": "Directory where SCF outputs are written / read.",
            },
            {
                "name": "dump_fields",
                "type": "list[str]",
                "tasks": "scf",
                "default": repr(DEFAULT_SCF_COMMON["dump_fields"]),
                "description": "Fields to dump from SCF calculations.",
            },
            {
                "name": "stats_fields",
                "type": "list[str]",
                "tasks": "stats",
                "default": "optional",
                "description": "Optional stats field selection.",
            },
            {
                "name": "scf_machine",
                "type": "dict",
                "tasks": "scf, iterate",
                "default": "optional",
                "description": "Dispatcher / resource settings for SCF execution.",
            },
            {
                "name": "test_sys / test_dump / with_conv / with_e / with_f / f_name",
                "type": "mixed",
                "tasks": "stats",
                "default": "optional",
                "description": "Legacy stats parameters still accepted and packaged.",
            },
        ),
    },
    {
        "title": "Iterate parameters",
        "rows": (
            {
                "name": "n_iter",
                "type": "int",
                "tasks": "iterate",
                "default": repr(DEFAULT_ITERATE["n_iter"]),
                "description": "Number of iterate cycles.",
            },
            {
                "name": "workdir",
                "type": "str",
                "tasks": "iterate",
                "default": repr(DEFAULT_ITERATE["workdir"]),
                "description": "Iterate working directory.",
            },
            {
                "name": "share_folder",
                "type": "str",
                "tasks": "iterate",
                "default": repr(DEFAULT_ITERATE["share_folder"]),
                "description": "Shared-file directory used by iterate workflow.",
            },
            {
                "name": "cleanup",
                "type": "bool",
                "tasks": "iterate",
                "default": repr(DEFAULT_ITERATE["cleanup"]),
                "description": "Whether to clean temporary dispatch files.",
            },
            {
                "name": "strict",
                "type": "bool",
                "tasks": "iterate",
                "default": repr(DEFAULT_ITERATE["strict"]),
                "description": "Whether unknown iterate machine/input keys should be ignored with warnings only.",
            },
            {
                "name": "scf_input / train_input",
                "type": "dict | str | bool",
                "tasks": "iterate",
                "default": "optional",
                "description": "Existing iterate template inputs kept for compatibility.",
            },
            {
                "name": "init_model / init_scf / init_train",
                "type": "mixed",
                "tasks": "iterate",
                "default": "optional",
                "description": "Bootstrap iteration controls.",
            },
            {
                "name": "init_scf_machine / init_train_machine / train_machine",
                "type": "dict",
                "tasks": "iterate",
                "default": "optional",
                "description": "Optional init / train machine overrides.",
            },
            {
                "name": "init_scf_abacus",
                "type": "dict",
                "tasks": "iterate (abacus)",
                "default": "inherits from scf_abacus when omitted",
                "description": "ABACUS init-SCF backend block. Legacy init_scf dicts with ABACUS keys are normalized here.",
            },
        ),
    },
    {
        "title": "Backend block: scf_pyscf",
        "rows": tuple(
            {
                "name": f"scf_pyscf.{name}",
                "type": "dict" if name in {"mol_args", "scf_args"} else "mixed",
                "tasks": "scf, stats",
                "default": repr(DEFAULT_SCF_PYSCF.get(name)),
                "description": {
                    "basis": "PySCF AO basis.",
                    "proj_basis": "Projection basis used by descriptor projection.",
                    "model_file": "Optional model path for ML-corrected SCF.",
                    "mol_args": "Arguments forwarded to molecule construction.",
                    "scf_args": "Arguments forwarded to the PySCF SCF solver.",
                    "conv_tol": "Legacy flat alias folded into scf_pyscf.scf_args or kept for compatibility.",
                    "conv_tol_grad": "Legacy compatibility parameter preserved in runtime bundles.",
                    "grids_level": "Legacy compatibility parameter preserved in runtime bundles.",
                    "verbose": "Backend-local verbosity override if supplied.",
                    "chkfile": "PySCF checkpoint file.",
                    "penalty_terms": "Optional penalty configuration for PySCF backend.",
                }[name],
            }
            for name in PYSCF_BACKEND_KEYS
        ),
    },
    {
        "title": "Backend block: scf_abacus",
        "rows": tuple(
            {
                "name": f"scf_abacus.{name}",
                "type": "list[str]" if name in {"orb_files", "pp_files", "proj_file"} else "mixed",
                "tasks": "scf, stats, iterate",
                "default": repr(DEFAULT_SCF_ABACUS.get(name)),
                "description": {
                    "orb_files": "ABACUS orbital files.",
                    "pp_files": "ABACUS pseudopotential files.",
                    "proj_file": "Projection orbital file list.",
                    "basis_file": "Legacy compatibility key preserved in bundles.",
                    "basis_name": "Legacy compatibility key preserved in bundles.",
                    "abacus_path": "ABACUS executable path.",
                    "run_cmd": "Launcher command used to run ABACUS.",
                    "input_args": "Legacy compatibility key preserved in bundles.",
                    "kpt_file": "Legacy compatibility key preserved in bundles.",
                    "stru_file": "Legacy compatibility key preserved in bundles.",
                    "coord_file": "Legacy compatibility key preserved in bundles.",
                    "lattice_constant": "STRU lattice constant.",
                    "lattice_vector": "STRU lattice vectors.",
                    "coord_type": "STRU coordinate type.",
                    "nspin": "ABACUS nspin setting.",
                    "symmetry": "ABACUS symmetry setting.",
                    "nbands": "ABACUS nbands setting.",
                    "ecutwfc": "Plane-wave cutoff.",
                    "scf_thr": "SCF convergence threshold.",
                    "scf_nmax": "Maximum SCF cycles.",
                    "dft_functional": "Underlying DFT functional.",
                    "basis_type": "ABACUS basis type.",
                    "gamma_only": "Gamma-only toggle.",
                    "k_points": "Explicit K-point grid.",
                    "kspacing": "K-point spacing.",
                    "smearing_method": "Occupation smearing method.",
                    "smearing_sigma": "Occupation smearing width.",
                    "mixing_type": "SCF mixing method.",
                    "mixing_beta": "SCF mixing beta.",
                    "cal_force": "Whether to compute forces.",
                    "cal_stress": "Whether to compute stress.",
                    "deepks_bandgap": "ABACUS DeePKS bandgap flag.",
                    "deepks_v_delta": "ABACUS DeePKS v_delta flag.",
                    "deepks_out_labels": "ABACUS DeePKS label dump flag.",
                    "deepks_scf": "ABACUS DeePKS SCF switch.",
                    "out_wfc_lcao": "ABACUS out_wfc_lcao switch.",
                    "ntype": "ABACUS atom-type count.",
                }.get(name, "ABACUS backend parameter."),
            }
            for name in ABACUS_BACKEND_KEYS
        ),
    },
)


def _deep_merge(base, override):
    if not isinstance(base, dict) or not isinstance(override, dict):
        return deepcopy(override)

    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _normalize_scf_backend(value):
    if value is None:
        return DEFAULT_SCF_BACKEND
    lowered = str(value).lower()
    if lowered not in VALID_SCF_BACKENDS:
        return lowered
    return lowered


def infer_scf_backend(config, default=DEFAULT_SCF_BACKEND):
    if "scf_soft" in config and config["scf_soft"] is not None:
        return _normalize_scf_backend(config["scf_soft"])
    if config.get("scf_abacus") or config.get("init_scf_abacus"):
        return "abacus"
    if any(key in config for key in ABACUS_BACKEND_KEYS):
        return "abacus"
    if config.get("scf_pyscf") or any(key in config for key in PYSCF_BACKEND_KEYS):
        return "pyscf"
    init_scf = config.get("init_scf")
    if isinstance(init_scf, dict) and any(key in init_scf for key in ABACUS_BACKEND_KEYS):
        return "abacus"
    return default


def _fold_flat_backend_keys(config, nested_key, flat_keys):
    nested = deepcopy(config.get(nested_key, {})) if isinstance(config.get(nested_key), dict) else {}
    flat = {}
    for key in flat_keys:
        if key in config:
            flat[key] = deepcopy(config.pop(key))
    if flat:
        nested = _deep_merge(nested, flat)
    if nested:
        config[nested_key] = nested
    return config


def normalize_config(config, task_type=None):
    normalized = deepcopy(config)
    if normalized.get("device") is None:
        normalized["device"] = _get_default_device()
    task_type = task_type or normalized.get("type")
    scf_soft = infer_scf_backend(normalized)
    normalized["scf_soft"] = scf_soft

    if task_type in {"scf", "stats"}:
        normalized = _fold_flat_backend_keys(normalized, "scf_pyscf", PYSCF_BACKEND_KEYS)
        normalized = _fold_flat_backend_keys(normalized, "scf_abacus", ABACUS_BACKEND_KEYS)
    elif task_type == "iterate":
        normalized = _fold_flat_backend_keys(normalized, "scf_pyscf", PYSCF_BACKEND_KEYS)
        normalized = _fold_flat_backend_keys(normalized, "scf_abacus", ABACUS_BACKEND_KEYS)
        init_scf = normalized.get("init_scf")
        if isinstance(init_scf, dict) and any(key in init_scf for key in ABACUS_BACKEND_KEYS):
            init_abacus = deepcopy(normalized.get("init_scf_abacus", {}))
            init_abacus = _deep_merge(init_abacus, {key: init_scf[key] for key in ABACUS_BACKEND_KEYS if key in init_scf})
            normalized["init_scf_abacus"] = init_abacus
            normalized["init_scf"] = True
    else:
        normalized = _fold_flat_backend_keys(normalized, "scf_pyscf", PYSCF_BACKEND_KEYS)
        normalized = _fold_flat_backend_keys(normalized, "scf_abacus", ABACUS_BACKEND_KEYS)

    return normalized


def apply_backend_compatibility(config, task_type=None):
    runtime = deepcopy(config)
    task_type = task_type or runtime.get("type")
    scf_soft = infer_scf_backend(runtime)
    runtime["scf_soft"] = scf_soft

    if scf_soft == "pyscf":
        backend_config = runtime.get("scf_pyscf", {})
        if isinstance(backend_config, dict):
            for key, value in backend_config.items():
                runtime[key] = deepcopy(value)
    elif scf_soft == "abacus":
        backend_config = runtime.get("scf_abacus", {})
        if isinstance(backend_config, dict):
            for key, value in backend_config.items():
                runtime[key] = deepcopy(value)

    if task_type == "test" and "systems_test" in runtime and "data_paths" not in runtime:
        runtime["data_paths"] = deepcopy(runtime["systems_test"])

    return runtime


def get_backend_block(config, backend=None, init=False):
    backend = backend or infer_scf_backend(config)
    if backend == "pyscf":
        key = "scf_pyscf"
    else:
        key = "init_scf_abacus" if init else "scf_abacus"
    block = config.get(key, {})
    return block if isinstance(block, dict) else {}


def get_default_config(task_type=None, scf_soft=DEFAULT_SCF_BACKEND):
    config = deepcopy(DEFAULT_COMMON)
    backend = _normalize_scf_backend(scf_soft)

    if task_type in {"scf", "stats", "iterate"}:
        config = _deep_merge(config, DEFAULT_SCF_COMMON)

    if task_type in {"scf", "stats", "iterate"}:
        if backend == "pyscf":
            config["scf_pyscf"] = deepcopy(DEFAULT_SCF_PYSCF)
        elif backend == "abacus":
            config["scf_abacus"] = deepcopy(DEFAULT_SCF_ABACUS)
            if task_type == "iterate":
                config["init_scf_abacus"] = deepcopy(DEFAULT_SCF_ABACUS)

    if task_type == "train":
        config = _deep_merge(config, DEFAULT_TRAIN)
    elif task_type == "test":
        config = _deep_merge(config, DEFAULT_TEST)
    elif task_type == "iterate":
        config = _deep_merge(config, DEFAULT_ITERATE)
        config = _deep_merge(config, DEFAULT_TRAIN)

    return config


def render_input_parameter_doc():
    lines = [
        "# DeePKS input parameter reference",
        "",
        "> Generated from `deepks/io/input/config.py`. Do not edit this file manually.",
        "",
        "## Normalization and compatibility",
        "",
        "- `deepks/main.py` is the CLI entrypoint used by both `python -m deepks` and console scripts.",
        "- Flat legacy SCF backend keys are still accepted and are normalized into backend blocks.",
        "- Preferred backend blocks are `scf_pyscf` and `scf_abacus`.",
        "- For iterate + ABACUS, `init_scf_abacus` inherits from `scf_abacus` when omitted; legacy `init_scf` dictionaries with ABACUS keys are normalized into `init_scf_abacus` and still trigger bootstrap iteration.",
        "- Packaged test configs keep backward-compatible `data_paths <- systems_test` mapping.",
        "- Docs can be regenerated in-repo with `python -m deepks.tools.sync_input_parameter_docs`.",
    ]

    for section in DOC_SECTIONS:
        lines.extend([
            "",
            f"## {section['title']}",
            "",
            "| Parameter | Type | Tasks | Default | Description |",
            "| --- | --- | --- | --- | --- |",
        ])
        for row in section["rows"]:
            description = row["description"].replace("|", "\\|")
            lines.append(
                f"| `{row['name']}` | `{row['type']}` | `{row['tasks']}` | `{row['default']}` | {description} |"
            )

    lines.extend([
        "",
        "## Bundled runtime contract",
        "",
        "`build_runtime_config()` returns a packaged dictionary with these bundles:",
        "",
        "- `raw_config`: normalized schema-first config",
        "- `global_param`: shared top-level runtime parameters",
        "- `scf_param`: SCF runtime parameters with backend-specific flat compatibility keys restored",
        "- `stats_param`: stats runtime parameters with backend-specific flat compatibility keys restored",
        "- `train_param`: train runtime parameters",
        "- `test_param`: test runtime parameters (`data_paths` derived from `systems_test` when needed)",
        "- `iterate_param`: iterate runtime parameters including nested backend blocks",
    ])

    return "\n".join(lines) + "\n"
