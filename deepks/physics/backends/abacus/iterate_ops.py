"""ABACUS iterate-specific helpers.

These helpers are backend/data-processing details used by iterate templates.
Keeping them in ``physics`` avoids embedding ABACUS parsing and conversion
logic inside the workflow layer.
"""

import os
from collections import Counter

import numpy as np
try:
    import torch
except ImportError:  # pragma: no cover - optional at runtime
    torch = None

from deepks.io.utils import check_share_folder
from deepks.physics.constants import NAME_TYPE, TYPE_NAME
from deepks.io.utils import flat_file_list, get_sys_name, load_dirs, load_sys_paths
from deepks.io.utils import coerce_box, coerce_energy, coerce_stress
from deepks.physics.backends.abacus.input_generator import (
    make_abacus_scf_input,
    make_abacus_scf_kpt,
    make_abacus_scf_stru,
)
from deepks.physics.backends.abacus.constants import CMODEL_FILE
from deepks.physics.backends.abacus.utils import read_csr


def _normalize_hr_target_shape(target_shape):
    if target_shape is None:
        return None
    shape = tuple(int(v) for v in target_shape)
    if len(shape) != 5:
        raise ValueError(f"HR target_shape must have rank 5, got {shape}")
    return shape


def _pad_first_three_dims(array, target_r_shape, *, start_axis=0):
    target_r_shape = tuple(int(v) for v in target_r_shape)
    if len(target_r_shape) != 3:
        raise ValueError(f"target_r_shape must have length 3, got {target_r_shape}")
    pad_width = []
    for axis in range(array.ndim):
        if start_axis <= axis < start_axis + 3:
            n_add = target_r_shape[axis - start_axis] - array.shape[axis]
            if n_add < 0:
                raise ValueError(
                    f"Cannot align HR data with R-shape {array.shape[start_axis:start_axis + 3]} "
                    f"to smaller target {target_r_shape}"
                )
            pad_width.append((0, n_add))
        else:
            pad_width.append((0, 0))
    if any(width[1] > 0 for width in pad_width[start_axis:start_axis + 3]):
        array = np.pad(array, pad_width)
    return array


def _align_hr_tensor(array, target_shape, *, name):
    if array is None:
        return None
    target_shape = _normalize_hr_target_shape(target_shape)
    if target_shape is None:
        return array
    if tuple(array.shape[-2:]) != tuple(target_shape[-2:]):
        raise ValueError(
            f"{name} local orbital shape {array.shape[-2:]} does not match target_shape tail {target_shape[-2:]}"
        )
    start_axis = 1 if array.ndim == 6 else 0
    return _pad_first_three_dims(array, target_shape[:3], start_axis=start_axis)


def _align_hr_storage(storage, current, *, nframes):
    if storage is None:
        return np.empty((nframes,) + current.shape, dtype=current.dtype), current
    target_r = tuple(max(storage.shape[axis + 1], current.shape[axis]) for axis in range(3))
    if storage.shape[1:4] != target_r:
        storage = np.pad(
            storage,
            ((0, 0),) + tuple((0, target_r[axis] - storage.shape[axis + 1]) for axis in range(3)) + ((0, 0), (0, 0)),
        )
    if current.shape[:3] != target_r:
        current = _pad_first_three_dims(current, target_r)
    return storage, current


def _align_vdr_precalc_storage(storage, current, *, nframes):
    if storage is None:
        return np.empty((nframes,) + current.shape, dtype=current.dtype), current
    target_r = tuple(max(storage.shape[axis + 1], current.shape[axis]) for axis in range(3))
    if storage.shape[1:4] != target_r:
        storage = np.pad(
            storage,
            ((0, 0),) + tuple((0, target_r[axis] - storage.shape[axis + 1]) for axis in range(3)) + ((0, 0),) * (storage.ndim - 4),
        )
    if current.shape[:3] != target_r:
        current = _pad_first_three_dims(current, target_r)
    return storage, current


def coord_to_atom(path):
    """Convert coord/type files to atom.npy-like arrays."""
    try:
        coords = np.load(f"{path}/coord.npy")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"atom.npy or coord.npy not found in {path}") from exc
    nframes = coords.shape[0]
    if coords.shape[2] != 3:
        raise ValueError("coord.npy should have shape (nframes, natoms, 3)")
    with open(f"{path}/type_map.raw") as fp:
        my_type_map = [NAME_TYPE[i] for i in fp.read().split()]
    atom_types = np.loadtxt(f"{path}/type.raw", ndmin=1).astype(int)
    atom_types = np.array([int(my_type_map[i - 1]) for i in atom_types]).reshape(1, -1).repeat(nframes, axis=0)
    return np.insert(coords, 0, values=atom_types, axis=2)


def convert_data(systems_train, systems_test=None, *,
                no_model=True, model_file=None, pp_files=None,
                dispatcher=None, **pre_args):
    """Prepare per-frame ABACUS inputs for iterate SCF steps."""
    pp_files = [] if pp_files is None else pp_files
    if not no_model:
        if model_file is not None:
            # Lazy ML import: this convert_data is the ABACUS iterate-data
            # preparation step that takes a model-file path from the workflow
            # config and produces a compiled jit model the backend can use.
            from deepks.ml.model_io import load_runtime_model

            model = load_runtime_model(model_file)
            model.compile_save(CMODEL_FILE)
            pre_args.update(deepks_scf=1, model_file=os.path.abspath(CMODEL_FILE))
        else:
            raise FileNotFoundError(f"No required model file in {os.getcwd()}")

    nsys_trn = len(systems_train)
    nsys_tst = len(systems_test)
    train_sets = [systems_train[i::nsys_trn] for i in range(nsys_trn)]
    test_sets = [systems_test[i::nsys_tst] for i in range(nsys_tst)]
    systems = systems_train + systems_test
    sys_paths = [os.path.abspath(s) for s in load_sys_paths(systems)]

    if dispatcher == "dpdispatcher" and \
        pre_args["dpdispatcher_machine"]["context_type"].upper().find("LOCAL") == -1:
        orb_files = ["../../../" + str(os.path.basename(s)) for s in pre_args["orb_files"]]
        pp_files = ["../../../" + str(os.path.basename(s)) for s in pp_files]
        proj_file = ["../../../" + str(os.path.basename(s)) for s in pre_args["proj_file"]]
        pre_args["orb_files"] = orb_files
        pre_args["proj_file"] = proj_file
        if not no_model:
            pre_args["model_file"] = "../../../" + CMODEL_FILE

    from pathlib import Path

    for i, _ in enumerate(train_sets + test_sets):
        try:
            atom_data = np.load(f"{sys_paths[i]}/atom.npy")
        except FileNotFoundError:
            atom_data = coord_to_atom(sys_paths[i])
        if os.path.isfile(f"{sys_paths[i]}/box.npy"):
            cell_data = np.load(f"{sys_paths[i]}/box.npy")
            if cell_data.shape != (atom_data.shape[0], 9):
                raise ValueError(f"box.npy should have shape (nframes, 9), but got {cell_data.shape}!")
        nframes = atom_data.shape[0]
        os.makedirs(f"{sys_paths[i]}/ABACUS", exist_ok=True)
        pre_args_new = dict(zip(pre_args.keys(), pre_args.values()))
        if os.path.exists(f"{sys_paths[i]}/group_scf_abacus.yaml"):
            from deepks.io.utils import load_yaml
            stru_abacus = load_yaml(f"{sys_paths[i]}/group_scf_abacus.yaml")
            for k, v in stru_abacus.items():
                pre_args_new[k] = v
        for f in range(nframes):
            os.makedirs(f"{sys_paths[i]}/ABACUS/{f}", exist_ok=True)
            if not os.path.isfile(f"{sys_paths[i]}/ABACUS/{f}/STRU"):
                Path(f"{sys_paths[i]}/ABACUS/{f}/STRU").touch()
            frame_data = atom_data[f]
            atoms = atom_data[f, :, 0]
            nta = Counter(atoms)
            sys_data = {
                "atom_names": [TYPE_NAME[it] for it in nta.keys()],
                "atom_numbs": list(nta.values()),
                "cells": np.array([pre_args_new["lattice_vector"]]),
                "coords": [frame_data[:, 1:]],
            }
            if os.path.isfile(f"{sys_paths[i]}/box.npy"):
                sys_data = {
                    "atom_names": [TYPE_NAME[it] for it in nta.keys()],
                    "atom_numbs": list(nta.values()),
                    "cells": [cell_data[f]],
                    "coords": [frame_data[:, 1:]],
                }
            with open(f"{sys_paths[i]}/ABACUS/{f}/STRU", "w") as stru_file:
                stru_file.write(make_abacus_scf_stru(sys_data, pp_files, pre_args_new))
            with open(f"{sys_paths[i]}/ABACUS/{f}/INPUT", "w") as input_file:
                input_file.write(make_abacus_scf_input(pre_args_new))
            if pre_args_new["k_points"] is not None or pre_args_new["gamma_only"] is True:
                with open(f"{sys_paths[i]}/ABACUS/{f}/KPT", "w") as kpt_file:
                    kpt_file.write(make_abacus_scf_kpt(pre_args_new))


def load_and_share_abacus_assets(orb_files, pp_files, proj_file, share_folder):
    """Materialize shared ABACUS input assets into the share folder."""
    def _normalize_file_list(value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)

    orb_files = _normalize_file_list(orb_files)
    pp_files = _normalize_file_list(pp_files)
    proj_file = _normalize_file_list(proj_file)
    for i in range(len(orb_files)):
        orb_files[i] = check_share_folder(orb_files[i], orb_files[i], share_folder)
    for i in range(len(pp_files)):
        pp_files[i] = check_share_folder(pp_files[i], pp_files[i], share_folder)
    for i in range(len(proj_file)):
        proj_file[i] = check_share_folder(proj_file[i], proj_file[i], share_folder)
    orb_files = [os.path.abspath(s) for s in flat_file_list(orb_files, sort=False)]
    pp_files = [os.path.abspath(s) for s in flat_file_list(pp_files, sort=False)]
    proj_file = [os.path.abspath(s) for s in flat_file_list(proj_file, sort=False)]
    return orb_files, pp_files, proj_file


def _load_frame(arr_dict, key, frame_idx, nframes, data):
    if arr_dict[key] is None:
        arr_dict[key] = np.empty((nframes,) + data.shape, dtype=data.dtype)
    assert data.shape == arr_dict[key].shape[1:], (
        f"Shape of {key} {arr_dict[key].shape} does not match {data.shape}!"
    )
    arr_dict[key][frame_idx] = data


def _collect_frames(sys_path, nframes, cal_force, cal_stress,
                    deepks_bandgap, deepks_v_delta, deepks_scf,
                    target_shape=None):
    arrs = dict(
        dm_eig=None,
        e_tot=None, e_base=None,
        f_tot=None, f_base=None, gvx=None,
        s_tot=None, s_base=None, gvepsl=None,
        o_tot=None, o_base=None, orbital_precalc=None,
        h_tot=None, h_base=None,
        v_delta_precalc=None, phialpha=None, gevdm=None,
        hr_tot=None, hr_base=None, vdr_precalc=None,
        # deepks_v_delta=-2 R-space chain-rule helpers. ABACUS dumps these per
        # frame alongside deepks_gevdm.npy; saving them at system level lets the
        # reader satisfy ``vdr_from_context`` directly without recomputing
        # overlap / iR_mat from atomic structure (which would require the user
        # to thread ``orb_list`` / ``alpha_list`` through ``data.loader``).
        iR_mat=None, phialpha_r=None,
    )
    conv = np.full((nframes, 1), False)
    hr_target_shape = _normalize_hr_target_shape(target_shape)

    for f in range(nframes):
        load_f_path = f"{sys_path}/ABACUS/{f}/OUT.ABACUS/"
        with open(f"{sys_path}/ABACUS/{f}/conv") as conv_file:
            ic = [t.strip('#').upper() for t in conv_file.read().split()]
            if ("CONVERGED" in ic or "ACHIEVED" in ic) and "NOT" not in ic:
                conv[int(ic[0])] = True
        _load_frame(arrs, 'dm_eig', f, nframes, np.load(load_f_path + "deepks_dm_eig.npy"))
        e_tot_data = np.load(load_f_path + "deepks_etot.npy")
        _load_frame(arrs, 'e_tot', f, nframes, e_tot_data)
        if deepks_scf:
            _load_frame(arrs, 'e_base', f, nframes, np.load(load_f_path + "deepks_ebase.npy"))
        else:
            _load_frame(arrs, 'e_base', f, nframes, e_tot_data)
        if cal_force:
            f_tot_data = np.load(load_f_path + "deepks_ftot.npy")
            _load_frame(arrs, 'f_tot', f, nframes, f_tot_data)
            if deepks_scf:
                _load_frame(arrs, 'f_base', f, nframes, np.load(load_f_path + "deepks_fbase.npy"))
            else:
                _load_frame(arrs, 'f_base', f, nframes, f_tot_data)
            if os.path.exists(load_f_path + "deepks_gradvx.npy"):
                _load_frame(arrs, 'gvx', f, nframes, np.load(load_f_path + "deepks_gradvx.npy"))
        if cal_stress:
            s_tot_data = np.load(load_f_path + "deepks_stot.npy")
            _load_frame(arrs, 's_tot', f, nframes, s_tot_data)
            if deepks_scf:
                _load_frame(arrs, 's_base', f, nframes, np.load(load_f_path + "deepks_sbase.npy"))
            else:
                _load_frame(arrs, 's_base', f, nframes, s_tot_data)
            if os.path.exists(load_f_path + "deepks_gvepsl.npy"):
                _load_frame(arrs, 'gvepsl', f, nframes, np.load(load_f_path + "deepks_gvepsl.npy"))
        if deepks_bandgap:
            o_tot_data = np.load(load_f_path + "deepks_otot.npy")
            _load_frame(arrs, 'o_tot', f, nframes, o_tot_data)
            if deepks_scf:
                _load_frame(arrs, 'o_base', f, nframes, np.load(load_f_path + "deepks_obase.npy"))
            else:
                _load_frame(arrs, 'o_base', f, nframes, o_tot_data)
            if os.path.exists(load_f_path + "deepks_orbpre.npy"):
                _load_frame(arrs, 'orbital_precalc', f, nframes, np.load(load_f_path + "deepks_orbpre.npy"))
        if deepks_v_delta > 0:
            h_tot_data = np.load(load_f_path + "deepks_htot.npy")
            _load_frame(arrs, 'h_tot', f, nframes, h_tot_data)
            if deepks_scf:
                _load_frame(arrs, 'h_base', f, nframes, np.load(load_f_path + "deepks_hbase.npy"))
            else:
                _load_frame(arrs, 'h_base', f, nframes, h_tot_data)
            if deepks_v_delta == 1 and os.path.exists(load_f_path + "deepks_vdpre.npy"):
                _load_frame(arrs, 'v_delta_precalc', f, nframes, np.load(load_f_path + "deepks_vdpre.npy"))
            elif deepks_v_delta == 2:
                if (os.path.exists(load_f_path + "deepks_phialpha.npy") and os.path.exists(load_f_path + "deepks_gevdm.npy")):
                    _load_frame(arrs, 'phialpha', f, nframes, np.load(load_f_path + "deepks_phialpha.npy"))
                    _load_frame(arrs, 'gevdm', f, nframes, np.load(load_f_path + "deepks_gevdm.npy"))
        if deepks_v_delta < 0:
            hrcs = _align_hr_tensor(
                read_csr(load_f_path + "deepks_hrtot.csr").to_dense().numpy(),
                hr_target_shape,
                name="deepks_hrtot.csr",
            )
            arrs['hr_tot'], hrcs = _align_hr_storage(arrs['hr_tot'], hrcs, nframes=nframes)
            arrs['hr_tot'][f] = hrcs
            if os.path.exists(load_f_path + "deepks_hrdelta.csr"):
                v_delta_r = _align_hr_tensor(
                    read_csr(load_f_path + "deepks_hrdelta.csr").to_dense().numpy(),
                    hr_target_shape,
                    name="deepks_hrdelta.csr",
                )
                if v_delta_r.shape[:3] != hrcs.shape[:3]:
                    target_r = tuple(max(v_delta_r.shape[axis], hrcs.shape[axis]) for axis in range(3))
                    v_delta_r = _pad_first_three_dims(v_delta_r, target_r)
                    hrcs = _pad_first_three_dims(hrcs, target_r)
                hrtmp = hrcs - v_delta_r if deepks_scf else hrcs
            else:
                # Mirror the energy/force/stress path: when no explicit DeePKS
                # correction tensor is present, the current SCF result itself is
                # the baseline for delta-learning.
                hrtmp = hrcs
            arrs['hr_base'], hrtmp = _align_hr_storage(arrs['hr_base'], hrtmp, nframes=nframes)
            arrs['hr_base'][f] = hrtmp
            if deepks_v_delta == -1 and os.path.exists(load_f_path + "deepks_vdrpre.npy"):
                tmp = np.load(load_f_path + "deepks_vdrpre.npy")
                if hr_target_shape is not None:
                    if tuple(tmp.shape[-4:-2]) != tuple(hr_target_shape[-2:]):
                        raise ValueError(
                            f"deepks_vdrpre.npy local orbital shape {tmp.shape[-4:-2]} does not match "
                            f"target_shape tail {hr_target_shape[-2:]}"
                        )
                    tmp = _pad_first_three_dims(tmp, hr_target_shape[:3])
                arrs['vdr_precalc'], tmp = _align_vdr_precalc_storage(arrs['vdr_precalc'], tmp, nframes=nframes)
                arrs['vdr_precalc'][f] = tmp
            elif deepks_v_delta == -2 and os.path.exists(load_f_path + "deepks_gevdm.npy"):
                _load_frame(arrs, 'gevdm', f, nframes, np.load(load_f_path + "deepks_gevdm.npy"))
                # deepks_v_delta=-2 also produces the R-space neighbor mapping
                # (deepks_iRmat.npy) and the LCAO×projector R-space overlap
                # (deepks_phialpha_r.npy). Loading them here lets the reader
                # populate ``iR_mat`` and ``overlap`` in the batch context for
                # the chain-rule V_delta(R) recovery path.
                if os.path.exists(load_f_path + "deepks_iRmat.npy"):
                    _load_frame(arrs, 'iR_mat', f, nframes, np.load(load_f_path + "deepks_iRmat.npy"))
                if os.path.exists(load_f_path + "deepks_phialpha_r.npy"):
                    _load_frame(arrs, 'phialpha_r', f, nframes, np.load(load_f_path + "deepks_phialpha_r.npy"))
    arrs['conv'] = conv
    return arrs


def _save_system_data(save_path, load_ref_path, arrs,
                     nframes, natoms, cal_force, cal_stress,
                     deepks_bandgap, deepks_v_delta,
                     target_shape=None):
    os.makedirs(save_path, exist_ok=True)
    np.save(save_path + "conv.npy", arrs['conv'])
    np.save(save_path + "dm_eig.npy", arrs['dm_eig'])
    e_base = coerce_energy(arrs['e_base'], nframes, 'e_base.npy')
    e_tot = coerce_energy(arrs['e_tot'], nframes, 'e_tot.npy')
    e_ref = coerce_energy(np.load(load_ref_path + "energy.npy"), nframes, 'energy.npy')
    np.save(save_path + "e_base.npy", e_base)
    np.save(save_path + "e_tot.npy", e_tot)
    np.save(save_path + "energy.npy", e_ref)
    np.save(save_path + "l_e_delta.npy", e_ref - e_base)
    np.save(save_path + "atom.npy", arrs['atom_data'])
    np.save(save_path + "box.npy", arrs['box_data'])
    if cal_force:
        f_ref = np.load(load_ref_path + "force.npy")
        if f_ref.shape != (nframes, natoms, 3):
            raise ValueError(f"force.npy shape should be (nframes,natoms,3), got {f_ref.shape}.")
        np.save(save_path + "f_base.npy", arrs['f_base'])
        np.save(save_path + "f_tot.npy", arrs['f_tot'])
        np.save(save_path + "force.npy", f_ref)
        np.save(save_path + "l_f_delta.npy", f_ref - arrs['f_base'])
        if arrs['gvx'] is not None:
            np.save(save_path + "grad_vx.npy", arrs['gvx'])
    if cal_stress:
        s_ref = coerce_stress(np.load(load_ref_path + "stress.npy"), nframes, 'stress.npy')
        s_base = coerce_stress(arrs['s_base'], nframes, 's_base')
        s_tot = coerce_stress(arrs['s_tot'], nframes, 's_tot')
        np.save(save_path + "s_base.npy", s_base)
        np.save(save_path + "s_tot.npy", s_tot)
        np.save(save_path + "stress.npy", s_ref)
        np.save(save_path + "l_s_delta.npy", s_ref - s_base)
        if arrs['gvepsl'] is not None:
            np.save(save_path + "grad_epsilon.npy", arrs['gvepsl'])
    if deepks_bandgap:
        o_ref = np.load(load_ref_path + "orbital.npy")
        if o_ref.shape[0] != nframes or o_ref.shape[2] != 1:
            raise ValueError(f"orbital.npy shape should be (nframes,nkpt,1), got {o_ref.shape}.")
        np.save(save_path + "o_base.npy", arrs['o_base'])
        np.save(save_path + "o_tot.npy", arrs['o_tot'])
        np.save(save_path + "orbital.npy", o_ref)
        np.save(save_path + "l_o_delta.npy", o_ref - arrs['o_base'])
        if arrs['orbital_precalc'] is not None:
            np.save(save_path + "orbital_precalc.npy", arrs['orbital_precalc'])
    if deepks_v_delta > 0:
        h_ref = np.load(load_ref_path + "hamiltonian.npy")
        if h_ref.shape[0] != nframes or h_ref.ndim != 4:
            raise ValueError(f"hamiltonian.npy shape should be (nframes,nkpt,nlocal,nlocal), got {h_ref.shape}.")
        np.save(save_path + "h_base.npy", arrs['h_base'])
        np.save(save_path + "h_tot.npy", arrs['h_tot'])
        np.save(save_path + "hamiltonian.npy", h_ref)
        np.save(save_path + "l_h_delta.npy", h_ref - arrs['h_base'])
        if arrs['v_delta_precalc'] is not None:
            np.save(save_path + "v_delta_precalc.npy", arrs['v_delta_precalc'])
        elif arrs['phialpha'] is not None and arrs['gevdm'] is not None:
            np.save(save_path + "grad_evdm.npy", arrs['gevdm'])
            np.save(save_path + "phialpha.npy", arrs['phialpha'])
        if os.path.exists(load_ref_path + "overlap.npy"):
            np.save(save_path + "overlap.npy", np.load(load_ref_path + "overlap.npy"))
    if deepks_v_delta < 0:
        hr_target_shape = _normalize_hr_target_shape(target_shape)
        hr_ref = _align_hr_tensor(
            np.load(load_ref_path + "hamiltonian_r.npy"),
            hr_target_shape,
            name="hamiltonian_r.npy",
        )
        if hr_ref.shape[0] != nframes or hr_ref.ndim != 6:
            raise ValueError(f"hamiltonian_r.npy shape should be (nframes,nR,nR,nR,nlocal,nlocal), got {hr_ref.shape}.")
        for key in ('hr_base', 'hr_tot'):
            arr = _align_hr_tensor(arrs[key], hr_target_shape, name=key) if arrs[key] is not None else None
            if arr is None:
                continue
            target_r = tuple(max(arr.shape[axis + 1], hr_ref.shape[axis + 1]) for axis in range(3))
            arr = _pad_first_three_dims(arr, target_r, start_axis=1)
            hr_ref = _pad_first_three_dims(hr_ref, target_r, start_axis=1)
            arrs[key] = arr
        if arrs['vdr_precalc'] is not None:
            target_r = tuple(hr_ref.shape[axis + 1] for axis in range(3))
            arrs['vdr_precalc'] = _pad_first_three_dims(arrs['vdr_precalc'], target_r, start_axis=1)
        np.save(save_path + "hamiltonian_r.npy", hr_ref)
        if arrs['hr_tot'] is not None:
            np.save(save_path + "hr_tot.npy", arrs['hr_tot'])
        if arrs['hr_base'] is not None:
            np.save(save_path + "hr_base.npy", arrs['hr_base'])
            np.save(save_path + "l_hr_delta.npy", hr_ref - arrs['hr_base'])
        if deepks_v_delta == -1 and arrs['vdr_precalc'] is not None:
            np.save(save_path + "vdr_precalc.npy", arrs['vdr_precalc'])
        elif deepks_v_delta == -2 and arrs['gevdm'] is not None:
            np.save(save_path + "grad_evdm.npy", arrs['gevdm'])
            if arrs.get('iR_mat') is not None:
                np.save(save_path + "iR_mat.npy", arrs['iR_mat'])
            if arrs.get('phialpha_r') is not None:
                np.save(save_path + "phialpha_r.npy", arrs['phialpha_r'])


def gather_stats_abacus(systems_train, systems_test,
                        train_dump, test_dump,
                        cal_force=0, cal_stress=0,
                        deepks_bandgap=0, deepks_v_delta=0,
                        deepks_scf=1, target_shape=None, **stat_args):
    """Gather iterate ABACUS stats and print summary."""
    sys_train_paths = [os.path.abspath(s) for s in load_sys_paths(systems_train)]
    sys_test_paths = [os.path.abspath(s) for s in load_sys_paths(systems_test)]
    sys_train_paths = [get_sys_name(s) for s in sys_train_paths]
    sys_test_paths = [get_sys_name(s) for s in sys_test_paths]
    sys_train_names = [os.path.basename(s) for s in sys_train_paths]
    sys_test_names = [os.path.basename(s) for s in sys_test_paths]
    if train_dump is None:
        train_dump = "."
    if test_dump is None:
        test_dump = "."

    def _process_systems(sys_paths, sys_names, dump_dir):
        os.makedirs(dump_dir, exist_ok=True)
        for sys_path, sys_name in zip(sys_paths, sys_names):
            load_ref_path = sys_path + "/"
            save_path = f"{dump_dir}/{sys_name}/"
            os.makedirs(save_path, exist_ok=True)
            try:
                atom_data = np.load(load_ref_path + "atom.npy")
            except FileNotFoundError:
                atom_data = coord_to_atom(sys_path)
            nframes = atom_data.shape[0]
            natoms = atom_data.shape[1]
            if atom_data.shape[2] != 4:
                raise ValueError("atom.npy should have shape (nframes, natoms, 4)")
            if os.path.isfile(load_ref_path + "box.npy"):
                box_data = coerce_box(np.load(load_ref_path + "box.npy"), nframes, 'box.npy').reshape(nframes, 3, 3)
            else:
                box_data = np.array([stat_args["lattice_vector"]]).reshape(1, 9).repeat(nframes, axis=0)
            box_data = box_data.reshape(nframes, 3, 3)
            if stat_args.get("coord_type") == "Direct":
                atom_data[:, :, 1:4] = np.matmul(atom_data[:, :, 1:4], box_data)
            atom_data[:, :, 1:4] *= stat_args['lattice_constant']
            box_data *= stat_args['lattice_constant']
            arrs = _collect_frames(
                sys_path,
                nframes,
                cal_force,
                cal_stress,
                deepks_bandgap,
                deepks_v_delta,
                deepks_scf,
                target_shape=target_shape,
            )
            arrs['atom_data'] = atom_data
            arrs['box_data'] = box_data
            _save_system_data(
                save_path,
                load_ref_path,
                arrs,
                nframes,
                natoms,
                cal_force,
                cal_stress,
                deepks_bandgap,
                deepks_v_delta,
                target_shape=target_shape,
            )

    _process_systems(sys_train_paths, sys_train_names, train_dump)
    _process_systems(sys_test_paths, sys_test_names, test_dump)
    from deepks.io.reporting import print_stats
    print_stats(
        systems=systems_train, test_sys=systems_test,
        dump_dir=train_dump, test_dump=test_dump, group=False,
        with_conv=True, with_e=True, e_name="e_tot",
        with_f=bool(cal_force), f_name="f_tot",
        with_s=bool(cal_stress), s_name="s_tot",
        with_o=bool(deepks_bandgap), o_name="o_tot",
    )
