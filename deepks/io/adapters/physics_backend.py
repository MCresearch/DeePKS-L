"""Physics backend adapter implementations for CLI/pipeline integration."""

from deepks.core.contracts import PhysicsBackend
from deepks.core.physics.pyscf.run import main as scf_run_main
from deepks.core.physics.pyscf.stats import print_stats


class PySCFPhysicsBackend(PhysicsBackend):
    """Default physics backend wiring to current PySCF-based entry points."""

    def run_scf(self, **kwargs):
        return scf_run_main(**kwargs)

    def collect_stats(self, **kwargs):
        return print_stats(**kwargs)
