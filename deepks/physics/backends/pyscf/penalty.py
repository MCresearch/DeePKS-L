import time

import numpy as np
from pyscf.dft import gen_grid, numint
from pyscf.lib import logger

from deepks.io.utils import check_list


def select_penalty(name):
    name = name.lower()
    if name == "density":
        return DensityPenalty
    if name == "coulomb":
        return CoulombPenalty
    raise ValueError(f"unknown penalty type: {name}")


class PenaltyMixin(object):
    """Mixin class to add penalty potential in Fock matrix."""

    def __init__(self, penalties=None):
        self.penalties = check_list(penalties)
        for pnt in self.penalties:
            pnt.init_hook(self)

    def get_fock(
        self,
        h1e=None,
        s1e=None,
        vhf=None,
        dm=None,
        cycle=-1,
        diis=None,
        diis_start_cycle=None,
        level_shift_factor=None,
        damp_factor=None,
    ):
        if dm is None:
            dm = self.make_rdm1()
        if h1e is None:
            h1e = self.get_hcore()
        if vhf is None:
            vhf = self.get_veff(dm=dm)
        vp = sum(pnt.fock_hook(self, dm=dm, h1e=h1e, vhf=vhf, cycle=cycle) for pnt in self.penalties)
        vhf = vhf + vp
        return super().get_fock(
            h1e=h1e,
            s1e=s1e,
            vhf=vhf,
            dm=dm,
            cycle=cycle,
            diis=diis,
            diis_start_cycle=diis_start_cycle,
            level_shift_factor=level_shift_factor,
            damp_factor=damp_factor,
        )


class AbstructPenalty(object):
    required_labels = []

    def init_hook(self, mf, **envs):
        pass

    def fock_hook(self, mf, dm=None, h1e=None, vhf=None, cycle=-1, **envs):
        raise NotImplementedError("fock_hook method is not implemented")


class DummyPenalty(AbstructPenalty):
    def fock_hook(self, mf, dm=None, h1e=None, vhf=None, cycle=-1, **envs):
        return 0


class DensityPenalty(AbstructPenalty):
    required_labels = ["dm"]

    def __init__(self, target_dm, strength=1, random=False, start_cycle=0):
        if isinstance(target_dm, str):
            target_dm = np.load(target_dm)
        self.dm_t = target_dm
        self.init_strength = strength
        self.strength = strength * np.random.rand() if random else strength
        self.start_cycle = start_cycle
        self.grids = None
        self.ao_value = None

    def init_hook(self, mf, **envs):
        self.grids = mf.grids if hasattr(mf, "grid") else gen_grid.Grids(mf.mol)

    def fock_hook(self, mf, dm=None, h1e=None, vhf=None, cycle=-1, **envs):
        if 0 <= cycle < self.start_cycle:
            return 0
        if self.grids.coords is None:
            self.grids.build()
        if self.ao_value is None:
            self.ao_value = numint.eval_ao(mf.mol, self.grids.coords, deriv=0)
        tic = (time.process_time(), time.perf_counter())
        rho_diff = numint.eval_rho(mf.mol, self.ao_value, dm - self.dm_t)
        v_p = numint.eval_mat(mf.mol, self.ao_value, self.grids.weights, rho_diff, rho_diff)
        if cycle < 0 and mf.verbose >= 4:
            diff_norm = np.sum(np.abs(rho_diff) * self.grids.weights)
            logger.info(mf, f"  Density Penalty: |diff| = {diff_norm}")
            logger.timer(mf, "dens_pnt", *tic)
        return self.strength * v_p


class CoulombPenalty(AbstructPenalty):
    required_labels = ["dm"]

    def __init__(self, target_dm, strength=1, random=False, start_cycle=0):
        if isinstance(target_dm, str):
            target_dm = np.load(target_dm)
        self.dm_t = target_dm
        self.init_strength = strength
        self.strength = strength * np.random.rand() if random else strength
        self.start_cycle = start_cycle

    def fock_hook(self, mf, dm=None, h1e=None, vhf=None, cycle=-1, **envs):
        if 0 <= cycle < self.start_cycle:
            return 0
        tic = (time.process_time(), time.perf_counter())
        ddm = dm - self.dm_t
        v_p = mf.get_j(dm=ddm)
        if cycle < 0 and mf.verbose >= 4:
            diff_norm = np.sum(ddm * v_p)
            logger.info(mf, f"  Coulomb Penalty: |diff| = {diff_norm}")
            logger.timer(mf, "coul_pnt", *tic)
        return self.strength * v_p
