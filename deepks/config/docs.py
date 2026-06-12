"""Input-parameter documentation metadata and rendering helpers."""

from .defaults import get_default_backend_input


def _fmt_default(value):
    if value is None:
        return "None"
    return repr(value)


_DEFAULT_PYSCF_INPUT = get_default_backend_input("pyscf")
_DEFAULT_ABACUS_INPUT = get_default_backend_input("abacus")
_PYSCF_BACKEND_KEYS = tuple(_DEFAULT_PYSCF_INPUT.keys())
_ABACUS_BACKEND_KEYS = tuple(_DEFAULT_ABACUS_INPUT.keys())


DOC_SECTIONS = (
    {
        "title": "Top-level task selectors",
        "rows": (
            {
                "name": "recipe",
                "type": "string",
                "tasks": "train, test, iterate",
                "default": "'corrnet-energy'",
                "description": "Primary high-level workflow recipe selector. Built-in values currently include corrnet-energy, corrnet-energy-only, and linear-energy.",
            },
            {
                "name": "type",
                "type": "string",
                "tasks": "all",
                "default": "required",
                "description": "Top-level task selector. Supported values: train, test, scf, stats, iterate.",
            },
            {
                "name": "runtime",
                "type": "dict",
                "tasks": "all",
                "default": "optional",
                "description": "Runtime block. It owns global execution settings such as device, seed, dtype, workdir, share_folder, strict/cleanup flags, checkpoint IO, and stage-specific execute/command settings.",
            },
            {
                "name": "data",
                "type": "dict",
                "tasks": "train, test, scf, stats, iterate",
                "default": "optional",
                "description": "Data block. Use data.train, data.test, or data.systems for system lists and data.loader for reader options such as batch_size or extra_label.",
            },
            {
                "name": "physics",
                "type": "dict",
                "tasks": "train, test, scf, stats, iterate",
                "default": "optional",
                "description": "Physics block. It defines representation / descriptor selection, physical targets, backend settings, and SCF output controls.",
            },
            {
                "name": "ml",
                "type": "dict",
                "tasks": "train, test, iterate",
                "default": "optional",
                "description": "Machine-learning block. It owns ml.model, ml.preprocess, ml.objective, ml.train, ml.checkpoint, and optional ml.fit_elem.",
            },
            {
                "name": "iterate",
                "type": "dict",
                "tasks": "iterate",
                "default": "optional",
                "description": "Iterative workflow block. Use iterate.n_iter, iterate.use_init, iterate.share_folder, iterate.cleanup, and iterate.strict for iterate-specific controls.",
            },
        ),
    },
    {
        "title": "Data block details",
        "rows": (
            {
                "name": "data.train",
                "type": "list[str] | str",
                "tasks": "train, iterate",
                "default": "required for training recipes",
                "description": "Training-system paths.",
            },
            {
                "name": "data.test",
                "type": "list[str] | str",
                "tasks": "train, test, iterate",
                "default": "optional",
                "description": "Test/evaluation-system paths.",
            },
            {
                "name": "data.systems",
                "type": "list[str] | str",
                "tasks": "scf, stats",
                "default": "required for scf/stats",
                "description": "System paths used by SCF or statistics workflows.",
            },
            {
                "name": "data.loader",
                "type": "dict",
                "tasks": "train, iterate",
                "default": "optional",
                "description": "Reader / dataset options such as extra_label, conv_filter, conv_name, read_overlap, and other loader-specific controls.",
            },
            {
                "name": "data.targets",
                "type": "dict",
                "tasks": "train, test, iterate",
                "default": "optional",
                "description": "Label-name mapping. Preferred place for energy/force/stress/orbital/v_delta/vdr/phi/band field names.",
            },
        ),
    },
    {
        "title": "Physics block details",
        "rows": (
            {
                "name": "physics.representation",
                "type": "string | dict",
                "tasks": "train, test, iterate",
                "default": "recipe-dependent",
                "description": "Descriptor / representation selector. The current CorrNet recipes typically use dm_eig.",
            },
            {
                "name": "physics.representation.params",
                "type": "dict",
                "tasks": "train, test, iterate",
                "default": "optional",
                "description": "Representation-specific arguments. For CorrNet this commonly contains proj_basis.",
            },
            {
                "name": "physics.backend",
                "type": "dict",
                "tasks": "scf, stats, iterate",
                "default": "optional",
                "description": "Backend selection block. Use physics.backend.name as pyscf or abacus, physics.backend.input for backend-specific physical parameters, and physics.backend.output for dump settings.",
            },
            {
                "name": "physics.backend.output",
                "type": "dict",
                "tasks": "scf, stats, iterate",
                "default": "optional",
                "description": "SCF output control block. Common fields are dump_dir and dump_fields.",
            },
        ),
    },
    {
        "title": "ML block details",
        "rows": (
            {
                "name": "ml.model.family",
                "type": "str",
                "tasks": "train, test, iterate",
                "default": "recipe-dependent",
                "description": "Model family name. Current built-in values include corrnet and linear.",
            },
            {
                "name": "ml.model.args",
                "type": "dict",
                "tasks": "train, iterate",
                "default": "recipe-dependent",
                "description": "Model constructor arguments for the selected family.",
            },
            {
                "name": "ml.checkpoint.file / ml.checkpoint.restart / ml.checkpoint.ckpt_file",
                "type": "str | null",
                "tasks": "test, train, iterate",
                "default": "optional",
                "description": "Model file, restart source, and checkpoint-output naming. Use this block instead of top-level model file fields.",
            },
            {
                "name": "ml.preprocess",
                "type": "dict",
                "tasks": "train, iterate",
                "default": "optional",
                "description": "Training-data preprocessing block. Current CorrNet recipes use preshift, prescale, prefit_ridge, and prefit_trainable. Phase-varying values may be written as [main, init] for scalars or {main: ..., init: ...} for complex values.",
            },
            {
                "name": "ml.objective.losses",
                "type": "list[dict] | dict",
                "tasks": "train, iterate",
                "default": "optional",
                "description": "Loss specification. Each item should contain name and may include weight, loss, or occ. For iterate special-first-round setups, prefer {main: ..., init: ...}.",
            },
            {
                "name": "ml.train.batch_size / ml.train.group_batch",
                "type": "int",
                "tasks": "train, iterate",
                "default": "optional",
                "description": "Preferred training batch settings. They map into the dataset loader configuration.",
            },
            {
                "name": "ml.train.epochs / ml.train.display_epoch / ml.train.display_detail_test",
                "type": "int",
                "tasks": "train, iterate",
                "default": "optional",
                "description": "Preferred training-loop controls.",
            },
            {
                "name": "ml.train.optimizer / ml.train.scheduler",
                "type": "dict",
                "tasks": "train, iterate",
                "default": "optional",
                "description": "Optimizer and scheduler settings. The current recipes use optimizer.lr, optimizer.weight_decay, scheduler.decay_steps, scheduler.decay_rate, and scheduler.stop_lr.",
            },
            {
                "name": "ml.fit_elem",
                "type": "bool",
                "tasks": "train, iterate",
                "default": "False",
                "description": "Whether to fit elemental energy constants before training.",
            },
        ),
    },
    {
        "title": "Runtime / iterate details",
        "rows": (
            {
                "name": "runtime.io",
                "type": "dict",
                "tasks": "all",
                "default": "optional",
                "description": "Runtime IO block. Preferred place for checkpoint and graph filenames such as ckpt_file and graph_file.",
            },
            {
                "name": "runtime.scf.execute / runtime.train.execute",
                "type": "dict",
                "tasks": "scf, iterate",
                "default": "optional",
                "description": "Stage-specific execution blocks. Use them for dispatcher, resources, group_size, ingroup_parallel, and related scheduler settings.",
            },
            {
                "name": "runtime.scf.command / runtime.train.command",
                "type": "dict",
                "tasks": "scf, iterate",
                "default": "optional",
                "description": "Stage-specific command blocks. Use them for python, run_cmd, abacus_path, and other command-path settings.",
            },
            {
                "name": "iterate.n_iter / iterate.use_init / iterate.share_folder / iterate.cleanup / iterate.strict",
                "type": "mixed",
                "tasks": "iterate",
                "default": "optional",
                "description": "Top-level iterate controls. use_init inserts an extra iter.init round before the normal iterations.",
            },
        ),
    },
    {
        "title": "Backend block: scf_pyscf",
        "rows": tuple(
            {
                "name": f"scf_pyscf.{key}",
                "type": "dict" if isinstance(_DEFAULT_PYSCF_INPUT.get(key), dict) else "mixed",
                "tasks": "scf, stats",
                "default": _fmt_default(_DEFAULT_PYSCF_INPUT.get(key)),
                "description": {
                    "basis": "PySCF AO basis.",
                    "proj_basis": "Projection basis used by descriptor projection.",
                    "model_file": "Optional model path for ML-corrected SCF.",
                    "mol_args": "Arguments forwarded to molecule construction.",
                    "scf_args": "Arguments forwarded to the PySCF SCF solver.",
                    "conv_tol": "Optional PySCF convergence tolerance override.",
                    "conv_tol_grad": "Optional PySCF gradient convergence tolerance.",
                    "grids_level": "Optional PySCF grids level.",
                    "verbose": "Backend-local verbosity override if supplied.",
                    "chkfile": "PySCF checkpoint file.",
                    "penalty_terms": "Optional penalty configuration for PySCF backend.",
                }.get(key, "PySCF backend parameter."),
            }
            for key in _PYSCF_BACKEND_KEYS
        ),
    },
    {
        "title": "Backend block: scf_abacus",
        "rows": tuple(
            {
                "name": f"scf_abacus.{key}",
                "type": "list[str]" if isinstance(_DEFAULT_ABACUS_INPUT.get(key), list) else "mixed",
                "tasks": "scf, stats, iterate",
                "default": _fmt_default(_DEFAULT_ABACUS_INPUT.get(key)),
                "description": {
                    "orb_files": "ABACUS orbital files.",
                    "pp_files": "ABACUS pseudopotential files.",
                    "proj_file": "Projection orbital file list.",
                    "basis_file": "Optional ABACUS basis file.",
                    "basis_name": "Optional ABACUS basis name.",
                    "abacus_path": "ABACUS executable path.",
                    "run_cmd": "Launcher command used to run ABACUS.",
                    "input_args": "Optional raw ABACUS INPUT overrides.",
                    "kpt_file": "Optional prebuilt ABACUS KPT file.",
                    "stru_file": "Optional prebuilt ABACUS STRU file.",
                    "coord_file": "Optional coordinate file override.",
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
                }.get(key, "ABACUS backend parameter."),
            }
            for key in _ABACUS_BACKEND_KEYS
        ),
    },
)


def render_input_parameter_doc():
    lines = [
        "# DeePKS input parameter reference",
        "",
        "> Generated from `deepks/config/docs.py`. Do not edit this file manually.",
        "",
        "## Preferred schema",
        "",
        "- `deepks/main.py` is the CLI entrypoint used by both `python -m deepks` and console scripts.",
        "- Preferred user inputs are block-structured: `recipe`, `runtime`, `data`, `physics`, `ml`, and `iterate`.",
        "- `recipe` selects a complete training / inference scheme; `data`, `physics`, `ml`, and `runtime` carry the remaining configuration.",
        "- `physics.backend.input` contains backend physical parameters; `runtime.<stage>.command` contains executable paths and launch commands.",
        "- `iterate.use_init` controls whether an extra `iter.init` round is inserted before `iter.00`.",
        "- Phase differences should be written directly on `physics`, `ml`, or `runtime` values: scalars may use `[main_value, init_value]`, while complex values should use `{main: ..., init: ...}`.",
        "- Docs can be regenerated in-repo with `python -m deepks.tools.sync_input_parameter_docs`.",
    ]

    for section in DOC_SECTIONS:
        lines.extend(
            [
                "",
                f"## {section['title']}",
                "",
                "| Parameter | Type | Tasks | Default | Description |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in section["rows"]:
            description = row["description"].replace("|", "\\|")
            lines.append(
                f"| `{row['name']}` | `{row['type']}` | `{row['tasks']}` | `{row['default']}` | {description} |"
            )

    lines.extend(
        [
            "",
            "## Runtime loading contract",
            "",
            "- `load_runtime_config(path)` is the single CLI/runtime entrypoint.",
            "- Its internal sequence is: `load -> normalize -> validate -> defaults -> merge -> package`.",
            "- The packaged runtime object has the form `{__internal_packed__: true, type: <task>, <task>_param: <packed_execution_parameters>}`.",
            "- `dispatch_command()` reads the task-specific payload key (`train_param`, `scf_param`, `iterate_param`, ...) and forwards only that packed parameter dictionary to the workflow layer.",
        ]
    )

    return "\n".join(lines) + "\n"
