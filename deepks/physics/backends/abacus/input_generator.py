"""ABACUS input file generator.

This module generates INPUT, STRU, and KPT files for ABACUS calculations.
"""

from typing import Dict, Any, List
from deepks.physics.defaults import BOHR2ANG


def make_abacus_scf_kpt(fp_params: Dict[str, Any]) -> str:
    """Generate KPT file for ABACUS SCF calculation.

    KPT file contains k-points information for ABACUS calculations.

    Args:
        fp_params: Parameters dictionary containing:
            - k_points: List of 6 integers for MP k-points generation
                       Default: [1, 1, 1, 0, 0, 0]

    Returns:
        str: Content of KPT file

    Raises:
        RuntimeError: If k_points is not a list of 6 integers
    """
    k_points = [1, 1, 1, 0, 0, 0]  # Default k points

    if "k_points" in fp_params:
        k_points = fp_params["k_points"]
        if len(k_points) != 6:
            raise RuntimeError(
                "k_points has to be a list containing 6 integers "
                "specifying MP k points generation."
            )

    ret = "K_POINTS\n0\nGamma\n"
    for i in range(6):
        ret += str(k_points[i]) + " "

    return ret


def make_abacus_scf_input(fp_params: Dict[str, Any]) -> str:
    """Generate INPUT file for ABACUS SCF calculation.

    Args:
        fp_params: Parameters dictionary with ABACUS settings

    Returns:
        str: Content of INPUT file
    """
    ret = "INPUT_PARAMETERS\n"
    ret += "calculation scf\n"

    # Basic parameters
    if "ecutwfc" in fp_params:
        assert fp_params["ecutwfc"] >= 0, "'ecutwfc' should be non-negative."
        ret += "ecutwfc %f\n" % fp_params["ecutwfc"]

    if "scf_thr" in fp_params:
        ret += "scf_thr %e\n" % fp_params["scf_thr"]

    if "scf_nmax" in fp_params:
        assert (fp_params['scf_nmax'] >= 0 and
                type(fp_params["scf_nmax"]) == int), \
            "'scf_nmax' should be a positive integer."
        ret += "scf_nmax %d\n" % fp_params["scf_nmax"]

    if "basis_type" in fp_params:
        assert fp_params["basis_type"] in ["pw", "lcao", "lcao_in_pw"], \
            "'basis_type' must be 'pw', 'lcao' or 'lcao_in_pw'."
        ret += "basis_type %s\n" % fp_params["basis_type"]

    if "dft_functional" in fp_params:
        ret += "dft_functional %s\n" % fp_params["dft_functional"]

    if "gamma_only" in fp_params:
        assert fp_params["gamma_only"] in [0, 1], \
            "'gamma_only' should be 0 or 1."
        ret += "gamma_only %d\n" % fp_params["gamma_only"]

    # Mixing parameters
    if "mixing_type" in fp_params:
        assert fp_params["mixing_type"] in [
            "plain", "kerker", "pulay", "pulay-kerker", "broyden"
        ]
        ret += "mixing_type %s\n" % fp_params["mixing_type"]

    if "mixing_beta" in fp_params:
        assert 0 <= fp_params["mixing_beta"] < 1, \
            "'mixing_beta' should be between 0 and 1."
        ret += "mixing_beta %f\n" % fp_params["mixing_beta"]

    # Symmetry
    if "symmetry" in fp_params:
        assert fp_params["symmetry"] in [-1, 0, 1], \
            "'symmetry' should be either -1, 0 or 1."
        ret += "symmetry %d\n" % fp_params["symmetry"]

    # Electronic structure
    if "nbands" in fp_params:
        if type(fp_params["nbands"]) == int and fp_params["nbands"] > 0:
            ret += "nbands %d\n" % fp_params["nbands"]
        else:
            print(
                "Warning: Parameter [nbands] given is not a positive integer, "
                "the default value of [nbands] in ABACUS will be used."
            )

    if "nspin" in fp_params:
        assert fp_params["nspin"] in [1, 2, 4], \
            "'nspin' can only take 1, 2 or 4"
        ret += "nspin %d\n" % fp_params["nspin"]

    if "ks_solver" in fp_params:
        assert fp_params["ks_solver"] in [
            "cg", "dav", "lapack", "genelpa", "hpseps", "scalapack_gvx"
        ], "'ks_solver' should be in valid solver list."
        ret += "ks_solver %s\n" % fp_params["ks_solver"]

    # Smearing
    if "smearing_method" in fp_params:
        assert fp_params["smearing_method"] in [
            "gaussian", "fd", "fixed", "mp", "mp2", "mv"
        ], "'smearing_method' should be in valid method list."
        ret += "smearing_method %s\n" % fp_params["smearing_method"]

    if "smearing_sigma" in fp_params:
        assert fp_params["smearing_sigma"] >= 0, \
            "'smearing_sigma' should be non-negative."
        ret += "smearing_sigma %f\n" % fp_params["smearing_sigma"]

    # K-spacing
    if (("kspacing" in fp_params) and
        (fp_params.get("k_points") is None) and
        (fp_params.get("gamma_only") == 0)):
        assert fp_params["kspacing"] > 0, "'kspacing' should be positive."
        ret += "kspacing %f\n" % fp_params["kspacing"]

    # Forces and stress
    if "cal_force" in fp_params:
        assert fp_params["cal_force"] in [0, 1], \
            "'cal_force' should be either 0 or 1."
        ret += "cal_force %d\n" % fp_params["cal_force"]

    if "cal_stress" in fp_params:
        assert fp_params["cal_stress"] in [0, 1], \
            "'cal_stress' should be either 0 or 1."
        ret += "cal_stress %d\n" % fp_params["cal_stress"]

    if "out_dos" in fp_params:
        assert type(fp_params["out_dos"]) == int, "'out_dos' should be integer."
        ret += "out_dos %d\n" % fp_params["out_dos"]

    # DeepKS parameters
    if "deepks_out_labels" in fp_params:
        assert fp_params["deepks_out_labels"] in [0, 1], \
            "'deepks_out_labels' should be either 0 or 1."
        ret += "deepks_out_labels %d\n" % fp_params["deepks_out_labels"]

    if "deepks_scf" in fp_params:
        assert fp_params["deepks_scf"] in [0, 1], \
            "'deepks_scf' should be either 0 or 1."
        ret += "deepks_scf %d\n" % fp_params["deepks_scf"]

    if "deepks_bandgap" in fp_params:
        assert type(fp_params["deepks_bandgap"]) == int, \
            "'deepks_bandgap' should be integer."
        ret += "deepks_bandgap %d\n" % fp_params["deepks_bandgap"]

        if fp_params["deepks_bandgap"] in [2, 3]:
            assert len(fp_params.get("deepks_band_range", [])) == 2, \
                "length of 'deepks_band_range' should be 2."
            ret += "deepks_band_range %d %d\n" % (
                fp_params["deepks_band_range"][0],
                fp_params["deepks_band_range"][1]
            )

    if "deepks_v_delta" in fp_params:
        assert fp_params["deepks_v_delta"] in [-2, -1, 0, 1, 2], \
            "'deepks_v_delta' should be either -2/-1/0/1/2."
        ret += "deepks_v_delta %d\n" % fp_params["deepks_v_delta"]

    if "model_file" in fp_params:
        ret += "deepks_model %s\n" % fp_params["model_file"]

    if "out_wfc_lcao" in fp_params:
        ret += "out_wfc_lcao %s\n" % fp_params["out_wfc_lcao"]

    # HSE calculation parameters
    if fp_params.get("dft_functional") == "hse":
        ret += "exx_pca_threshold 1e-4\n"
        ret += "exx_c_threshold 1e-4\n"
        ret += "exx_dm_threshold 1e-4\n"
        ret += "exx_ccp_rmesh_times 1\n"

    return ret


def make_abacus_scf_stru(sys_data: Dict[str, Any],
                        fp_pp_files: List[str],
                        fp_params: Dict[str, Any]) -> str:
    """Generate STRU file for ABACUS SCF calculation.

    Args:
        sys_data: System data dictionary containing:
            - atom_names: List of atom names (e.g., ['H', 'O'])
            - atom_numbs: Number of each atom type (e.g., [2, 1])
            - cells: Cell vectors, shape (1, 3, 3)
            - coords: Atomic coordinates, shape (1, natoms, 3)
        fp_pp_files: List of pseudopotential file paths
        fp_params: Parameters dictionary

    Returns:
        str: Content of STRU file

    Raises:
        AssertionError: If pseudopotential or orbital files are missing
    """
    atom_names = sys_data['atom_names']
    atom_numbs = sys_data['atom_numbs']

    # Validate and select pseudopotential files
    valid_pp_files = {}
    for pp_file in fp_pp_files:
        filename = pp_file.split('/')[-1]
        element_name = filename.split('_')[0]

        if element_name in atom_names:
            assert element_name not in valid_pp_files, \
                f"Pseudopotential file for element {element_name} already exists."
            valid_pp_files[element_name] = pp_file

    # Ensure all elements have pseudopotential files
    for atom in atom_names:
        assert atom in valid_pp_files, \
            f"No pseudopotential file found for element {atom}."

    # Build STRU file
    ret = "ATOMIC_SPECIES\n"
    for atom in atom_names:
        ret += f"{atom} 1.00 {valid_pp_files[atom]}\n"

    # Lattice constant
    if "lattice_constant" in fp_params:
        ret += "\nLATTICE_CONSTANT\n"
        ret += f"{fp_params['lattice_constant']}\n\n"
    else:
        ret += "\nLATTICE_CONSTANT\n"
        ret += f"{1 / BOHR2ANG}\n\n"

    # Lattice vectors
    ret += "LATTICE_VECTORS\n"
    cell = sys_data["cells"][0].reshape([3, 3])
    for ix in range(3):
        for iy in range(3):
            ret += f"{cell[ix][iy]} "
        ret += "\n"

    # Atomic positions
    ret += "\nATOMIC_POSITIONS\n"
    ret += f"{fp_params['coord_type']}\n\n"

    natom_tot = 0
    coord = sys_data['coords'][0]
    for iele in range(len(atom_names)):
        ret += f"{atom_names[iele]}\n"
        ret += "0.0\n"  # Reference energy
        ret += f"{atom_numbs[iele]}\n"
        for iatom in range(atom_numbs[iele]):
            ret += (
                f"{coord[natom_tot, 0]:.12f} "
                f"{coord[natom_tot, 1]:.12f} "
                f"{coord[natom_tot, 2]:.12f} 0 0 0\n"
            )
            natom_tot += 1

    assert natom_tot == sum(atom_numbs), \
        "The total number of atoms does not match."

    # Numerical orbitals for LCAO
    if fp_params.get("basis_type") == "lcao":
        ret += "\nNUMERICAL_ORBITAL\n"

        valid_orb_files = {}
        for orb_file in fp_params.get("orb_files", []):
            filename = orb_file.split('/')[-1]
            element_name = filename.split('_')[0]

            if element_name in atom_names:
                assert element_name not in valid_orb_files, \
                    f"Orbital file for element {element_name} already exists."
                valid_orb_files[element_name] = orb_file

        # Ensure all elements have orbital files
        for atom in atom_names:
            assert atom in valid_orb_files, \
                f"No orbital file found for element {atom}."

        for atom in atom_names:
            ret += f"{valid_orb_files[atom]}\n"

    # DeepKS descriptor
    if (fp_params.get("deepks_scf") and
        fp_params.get("deepks_out_labels") == 1):
        ret += "\nNUMERICAL_DESCRIPTOR\n"
        ret += f"{fp_params['proj_file'][0]}\n"

    return ret
