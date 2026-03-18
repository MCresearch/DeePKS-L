"""
Iterate/SCF shim export compatibility coverage.

Tests:
- `test_iterate_shim_exports`
- `test_scf_shim_exports`
"""

import importlib

import pytest

import deepks.iterate.generator_abacus as iter_gen_shim
import deepks.iterate.iterate as iter_iter_shim
import deepks.iterate.template as iter_template_shim
import deepks.iterate.template_abacus as iter_template_abacus_shim
import deepks.iterate.utils as iter_utils_shim


def test_iterate_shim_exports():
    iter_iter_impl = importlib.import_module("deepks.pipelines.iterate.iterate")
    iter_template_impl = importlib.import_module("deepks.pipelines.iterate.template")
    iter_template_abacus_impl = importlib.import_module("deepks.pipelines.iterate.template_abacus")
    iter_gen_impl = importlib.import_module("deepks.pipelines.iterate.generator_abacus")
    iter_utils_impl = importlib.import_module("deepks.pipelines.iterate.utils")

    assert iter_iter_shim.main is iter_iter_impl.main
    assert iter_template_shim.make_train is iter_template_impl.make_train
    assert iter_template_abacus_shim.make_scf_abacus is iter_template_abacus_impl.make_scf_abacus
    assert iter_gen_shim.make_abacus_scf_input is iter_gen_impl.make_abacus_scf_input
    assert iter_utils_shim.NPY_DICT is iter_utils_impl.NPY_DICT


def test_scf_shims_import_is_lazy():
    # Importing compatibility shims should not require optional pyscf dependency.
    modules = [
        "deepks.scf._old_grad",
        "deepks.scf.addons",
        "deepks.scf.fields",
        "deepks.scf.grad",
        "deepks.scf.penalty",
        "deepks.scf.run",
        "deepks.scf.scf",
        "deepks.scf.stats",
    ]

    for module in modules:
        assert importlib.import_module(module).__name__ == module


def test_scf_shim_exports():
    pytest.importorskip("pyscf")

    import deepks.scf._old_grad as scf_old_grad_shim
    import deepks.scf.addons as scf_addons_shim
    import deepks.scf.fields as scf_fields_shim
    import deepks.scf.grad as scf_grad_shim
    import deepks.scf.penalty as scf_penalty_shim
    import deepks.scf.run as scf_run_shim
    import deepks.scf.scf as scf_core_shim
    import deepks.scf.stats as scf_stats_shim

    scf_core_impl = importlib.import_module("deepks.core.physics.pyscf.scf")
    scf_run_impl = importlib.import_module("deepks.core.physics.pyscf.run")
    scf_stats_impl = importlib.import_module("deepks.core.physics.pyscf.stats")
    scf_fields_impl = importlib.import_module("deepks.core.physics.pyscf.fields")
    scf_penalty_impl = importlib.import_module("deepks.core.physics.pyscf.penalty")
    scf_addons_impl = importlib.import_module("deepks.core.physics.pyscf.addons")
    scf_grad_impl = importlib.import_module("deepks.core.physics.pyscf.grad")
    scf_old_grad_impl = importlib.import_module("deepks.core.physics.pyscf._old_grad")

    assert scf_core_shim.DSCF is scf_core_impl.DSCF
    assert scf_run_shim.main is scf_run_impl.main
    assert scf_stats_shim.print_stats is scf_stats_impl.print_stats
    assert scf_fields_shim.select_fields is scf_fields_impl.select_fields
    assert scf_penalty_shim.CoulombPenalty is scf_penalty_impl.CoulombPenalty
    assert scf_addons_shim.gcalc_optim_veig is scf_addons_impl.gcalc_optim_veig
    assert scf_grad_shim.t_shell_eig is scf_grad_impl.t_shell_eig
    assert scf_old_grad_shim.t_make_grad_eig_dm is scf_old_grad_impl.t_make_grad_eig_dm
