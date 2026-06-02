"""Iterate/SCF canonical export coverage."""

import importlib

import pytest


def test_iterate_exports():
    iter_templates = importlib.import_module("deepks.workflows.iterate.support.task_templates")
    abacus_iter = importlib.import_module("deepks.physics.backends.abacus.iterate_sequence")
    iter_ops = importlib.import_module("deepks.physics.backends.abacus.iterate_ops")

    assert callable(iter_templates.make_train)
    assert callable(iter_templates.make_run_scf)
    assert callable(abacus_iter.make_scf_abacus)
    assert callable(iter_ops.coord_to_atom)


def test_scf_package_import_without_pyscf():
    pytest.importorskip("pyscf")
    importlib.import_module("deepks.physics.backends.pyscf")


def test_scf_exports():
    pytest.importorskip("pyscf")

    import deepks.physics.backends.pyscf._old_grad as scf_old_grad
    import deepks.physics.backends.pyscf.schema as scf_fields
    import deepks.physics.backends.pyscf.grad as scf_grad
    import deepks.physics.backends.pyscf.run as scf_run
    import deepks.physics.backends.pyscf.scf as scf_core
    import deepks.physics.backends.pyscf.stats as scf_stats
    import deepks.physics.backends.pyscf.optim as scf_optim
    import deepks.physics.backends.pyscf.penalty as scf_penalty

    assert scf_core.DSCF is not None
    assert scf_run.main is not None
    assert scf_stats.print_stats is not None
    assert scf_fields.select_fields is not None
    assert scf_penalty.CoulombPenalty is not None
    assert scf_optim.gcalc_optim_veig is not None
    assert scf_grad.t_shell_eig is not None
    assert scf_old_grad.t_make_grad_eig_dm is not None
