"""
Iterate/SCF export compatibility coverage.

Tests:
- `test_iterate_exports`
- `test_scf_shim_exports`
"""

import importlib

import pytest

import deepks.physics.backends.abacus.input_generator as iter_gen_shim
import deepks.workflows.iterate.template as iter_template_shim
import deepks.workflows.iterate.template_abacus as iter_template_abacus_shim


def test_iterate_exports():
    iter_template_impl = importlib.import_module("deepks.workflows.iterate.template")
    iter_template_abacus_impl = importlib.import_module("deepks.workflows.iterate.template_abacus")
    iter_gen_impl = importlib.import_module("deepks.physics.backends.abacus.input_generator")

    assert iter_template_shim.make_train is iter_template_impl.make_train
    assert iter_template_abacus_shim.make_scf_abacus is iter_template_abacus_impl.make_scf_abacus
    assert iter_gen_shim.make_abacus_scf_input is iter_gen_impl.make_abacus_scf_input


def test_scf_package_import_without_pyscf():
    pytest.importorskip("pyscf")
    importlib.import_module("deepks.physics.backends.pyscf")


def test_scf_shim_exports():
    pytest.importorskip("pyscf")

    import deepks.physics.backends.pyscf._old_grad as scf_old_grad_shim
    import deepks.physics.backends.pyscf.addons as scf_addons_shim
    import deepks.physics.backends.pyscf.fields as scf_fields_shim
    import deepks.physics.backends.pyscf.grad as scf_grad_shim
    import deepks.physics.backends.pyscf.penalty as scf_penalty_shim
    import deepks.physics.backends.pyscf.runner as scf_run_shim
    import deepks.physics.backends.pyscf.scf as scf_core_shim
    import deepks.physics.backends.pyscf.stats as scf_stats_shim

    scf_core_impl = importlib.import_module("deepks.physics.backends.pyscf.scf")
    scf_run_impl = importlib.import_module("deepks.physics.backends.pyscf.runner")
    scf_stats_impl = importlib.import_module("deepks.physics.backends.pyscf.stats")
    scf_fields_impl = importlib.import_module("deepks.physics.backends.pyscf.fields")
    scf_penalty_impl = importlib.import_module("deepks.physics.backends.pyscf.penalty")
    scf_addons_impl = importlib.import_module("deepks.physics.backends.pyscf.addons")
    scf_grad_impl = importlib.import_module("deepks.physics.backends.pyscf.grad")
    scf_old_grad_impl = importlib.import_module("deepks.physics.backends.pyscf._old_grad")

    assert scf_core_shim.DSCF is scf_core_impl.DSCF
    assert scf_run_shim.main is scf_run_impl.main
    assert scf_stats_shim.print_stats is scf_stats_impl.print_stats
    assert scf_fields_shim.select_fields is scf_fields_impl.select_fields
    assert scf_penalty_shim.CoulombPenalty is scf_penalty_impl.CoulombPenalty
    assert scf_addons_shim.gcalc_optim_veig is scf_addons_impl.gcalc_optim_veig
    assert scf_grad_shim.t_shell_eig is scf_grad_impl.t_shell_eig
    assert scf_old_grad_shim.t_make_grad_eig_dm is scf_old_grad_impl.t_make_grad_eig_dm
