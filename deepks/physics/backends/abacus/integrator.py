"""ABACUS radial-integrator helpers used by backend-side preprocessing."""

try:
    from pyabacus import ModuleBase as base
    from pyabacus import ModuleNAO as nao
except ImportError:
    base = None
    nao = None


def make_integrator(orb_files, alpha_files):
    """Build the radial collections and two-center integrator."""

    if base is None or nao is None:
        raise ImportError("pyabacus is required for make_integrator()")
    orb = nao.RadialCollection()
    alpha = nao.RadialCollection()
    orb.build(len(orb_files), orb_files)
    alpha.build(len(alpha_files), alpha_files)

    dr = 0.01
    rmax = max(orb.rcut_max(), alpha.rcut_max())
    cutoff = 2.0 * rmax
    nr = int(rmax / dr) + 1
    orb.set_uniform_grid(True, nr, cutoff, "i", True)
    alpha.set_uniform_grid(True, nr, cutoff, "i", True)

    sbt = base.SphericalBesselTransformer()
    orb.set_transformer(sbt)
    alpha.set_transformer(sbt)

    integrator = nao.TwoCenterIntegrator()
    integrator.tabulate(orb, alpha, "S", nr, cutoff)
    return orb, alpha, integrator
