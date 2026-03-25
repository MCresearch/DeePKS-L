import os
import numpy as np
try:
    import torch
except ImportError:
    torch = None
from glob import glob
from collections import Counter
from deepks.workflows.defaults import CMODEL_FILE
from deepks.physics.defaults import NAME_TYPE, TYPE_NAME
from deepks.io.utils import flat_file_list, load_dirs
from deepks.io.utils import get_sys_name, load_sys_paths
from deepks.io.utils import coerce_box, coerce_energy, coerce_stress
from deepks.physics.backends.abacus.utils import read_csr

from deepks.orchestration.workflow.task import PythonTask
from deepks.orchestration.workflow.task import BatchTask, GroupBatchTask, DPDispatcherTask
from deepks.orchestration.workflow.workflow import Sequence
from deepks.workflows.iterate.template import check_system_names, make_cleanup
from deepks.physics.backends.abacus.input_generator import make_abacus_scf_kpt, make_abacus_scf_input, make_abacus_scf_stru
import psutil
import time

def coord_to_atom(path):
    '''
    Convert coord.npy and type.raw (type_map.raw) to atom.npy
    Shape of coord.npy: (nframes, natoms, 3)
    Shape of atom.npy: (nframes, natoms, 4), the first column is atom type
    '''
    try:
        coords = np.load(f"{path}/coord.npy")
    except FileNotFoundError:
        raise FileNotFoundError(f"atom.npy or coord.npy not found in {path}")
    nframes = coords.shape[0]
    if coords.shape[2] != 3:
        raise ValueError("coord.npy should have shape (nframes, natoms, 3)")
    # get type_map.raw and type.raw, use it
    with open(f"{path}/type_map.raw") as fp:
        my_type_map =[NAME_TYPE[i] for i in fp.read().split()]
    atom_types = np.loadtxt(f"{path}/type.raw", ndmin=1).astype(int)
    atom_types = np.array([int(my_type_map[i-1]) for i in atom_types]).reshape(1,-1).repeat(nframes,axis=0)
    atom_data = np.insert(coords, 0, values=atom_types, axis=2)
    return atom_data


def make_scf_abacus(systems_train, systems_test=None, *,
             train_dump="data_train", test_dump="data_test", cleanup=None,
             dispatcher=None, resources =None, no_model=True, group_size=1,
             workdir='00.scf', share_folder='share', model_file=None,
             orb_files=[], pp_files=[], proj_file=[],  **scf_abacus):
    #share orb_files and pp_files
    from deepks.workflows.iterate.prepare import check_share_folder
    for i in range (len(orb_files)):
        orb_files[i] = check_share_folder(orb_files[i], orb_files[i], share_folder)
    for i in range (len(pp_files)):
        pp_files[i] = check_share_folder(pp_files[i], pp_files[i], share_folder)
        #share the traced model file
    for i in range (len(proj_file)):
        proj_file[i] = check_share_folder(proj_file[i], proj_file[i], share_folder)
   # if(no_model is False):
        #model_file=os.path.abspath(model_file)
        #model_file = check_share_folder(model_file, model_file, share_folder)
    orb_files=[os.path.abspath(s) for s in flat_file_list(orb_files, sort=False)]
    pp_files=[os.path.abspath(s) for s in flat_file_list(pp_files, sort=False)]
    proj_file=[os.path.abspath(s) for s in flat_file_list(proj_file, sort=False)]
    forward_files=orb_files+pp_files+proj_file
    pre_scf_abacus = make_convert_scf_abacus(
        systems_train=systems_train, systems_test=systems_test,
        no_model=no_model, workdir='.', share_folder=share_folder, 
        model_file=model_file, resources=resources,
        dispatcher=dispatcher, orb_files=orb_files, pp_files=pp_files, 
        proj_file=proj_file, **scf_abacus)
    run_scf_abacus = make_run_scf_abacus(systems_train, systems_test,
        no_model=no_model, model_file=model_file, group_data=False,
        workdir='.', outlog="log.scf", share_folder=share_folder, 
        dispatcher=dispatcher, resources=resources, group_size=group_size,
        forward_files=forward_files, 
        **scf_abacus)
    post_scf_abacus = make_stat_scf_abacus(
        systems_train, systems_test,
        train_dump=train_dump, test_dump=test_dump, workdir=".", 
        **scf_abacus)
    # concat
    seq = [pre_scf_abacus, run_scf_abacus, post_scf_abacus]
    #seq = [post_scf_abacus]
    #seq = [pre_scf_abacus]
    if cleanup:
        clean_scf = make_cleanup(
            ["slurm-*.out", "task.*/err", "fin.record"],
            workdir=".")
        seq.append(clean_scf)
    #make sequence
    return Sequence(
        seq,
        workdir=workdir
    )


### need parameters: orb_files, pp_files, proj_file
def convert_data(systems_train, systems_test=None, *, 
                no_model=True, model_file=None, pp_files=[], 
                dispatcher=None,**pre_args):
    #trace a model (if necessary)
    if not no_model:
        if model_file is not None:
            from deepks.ml.models.corrnet import CorrNet
            model = CorrNet.load(model_file)
            model.compile_save(CMODEL_FILE)
            #set 'deepks_scf' to 1, and give abacus the path of traced model file
            pre_args.update(deepks_scf=1, model_file=os.path.abspath(CMODEL_FILE))
        else:
            raise FileNotFoundError(f"No required model file in {os.getcwd()}")
    # split systems into groups
    nsys_trn = len(systems_train)
    nsys_tst = len(systems_test)
    #ntask_trn = int(np.ceil(nsys_trn / sub_size))
    #ntask_tst = int(np.ceil(nsys_tst / sub_size))
    train_sets = [systems_train[i::nsys_trn] for i in range(nsys_trn)]
    test_sets = [systems_test[i::nsys_tst] for i in range(nsys_tst)]
    systems=systems_train+systems_test
    sys_paths = [os.path.abspath(s) for s in load_sys_paths(systems)]
    from pathlib import Path
    if dispatcher=="dpdispatcher" and \
        pre_args["dpdispatcher_machine"]["context_type"].upper().find("LOCAL")==-1:
        #write relative path into INPUT and STRU
        orb_files=pre_args["orb_files"]
        proj_file=pre_args["proj_file"]
        orb_files=["../../../"+str(os.path.basename(s)) for s in orb_files]
        pp_files=["../../../"+str(os.path.basename(s)) for s in pp_files]
        proj_file=["../../../"+str(os.path.basename(s)) for s in proj_file]
        pre_args["orb_files"]=orb_files
        pre_args["proj_file"]=proj_file
        if not no_model:
            pre_args["model_file"]="../../../"+CMODEL_FILE
    #init sys_data (dpdata)
    for i, sset in enumerate(train_sets+test_sets):
        try:
            atom_data = np.load(f"{sys_paths[i]}/atom.npy")
        except FileNotFoundError:
            atom_data = coord_to_atom(sys_paths[i])
        if os.path.isfile(f"{sys_paths[i]}/box.npy"):
            cell_data = np.load(f"{sys_paths[i]}/box.npy")
            if cell_data.shape != (atom_data.shape[0], 9):
                raise ValueError(f"box.npy should have shape (nframes, 9), but got {cell_data.shape}!")
        nframes = atom_data.shape[0]
        if not os.path.exists(f"{sys_paths[i]}/ABACUS"):
            os.mkdir(f"{sys_paths[i]}/ABACUS")
        #pre_args.update({"lattice_vector":lattice_vector})
        #if "stru_abacus.yaml" exists, update STRU args in pre_args:
        pre_args_new=dict(zip(pre_args.keys(),pre_args.values()))
        if os.path.exists(f"{sys_paths[i]}/group_scf_abacus.yaml"):
            from deepks.io.utils import load_yaml
            stru_abacus = load_yaml(f"{sys_paths[i]}/group_scf_abacus.yaml")
            for k,v in stru_abacus.items():
                print(f"k={k},v={v}")
                pre_args_new[k]=v
        print(f"pre_args_new={pre_args_new}")
        for f in range(nframes):
            if not os.path.exists(f"{sys_paths[i]}/ABACUS/{f}"):
                os.mkdir(f"{sys_paths[i]}/ABACUS/{f}")
            ###create STRU file
            if not os.path.isfile(f"{sys_paths[i]}/ABACUS/{f}/STRU"):
                Path(f"{sys_paths[i]}/ABACUS/{f}/STRU").touch()
            #create sys_data for each frame
            frame_data=atom_data[f]
            #frame_sorted=frame_data[np.lexsort(frame_data[:,::-1].T)] #sort cord by type
            # nta may diff for different frames
            atoms = atom_data[f,:,0] 
            #atoms.sort() # type order
            nta = Counter(atoms) #dict {itype: nta}, natom in each type
            sys_data={'atom_names':[TYPE_NAME[it] for it in nta.keys()], 'atom_numbs': list(nta.values()), 
                        #'cells': np.array([lattice_vector]), 'coords': [frame_sorted[:,1:]]}
                        'cells': np.array([pre_args_new["lattice_vector"]]), 'coords': [frame_data[:,1:]]}
            if os.path.isfile(f"{sys_paths[i]}/box.npy"):
                sys_data={'atom_names':[TYPE_NAME[it] for it in nta.keys()], 'atom_numbs': list(nta.values()),
                        'cells': [cell_data[f]], 'coords': [frame_data[:,1:]]}
            #write STRU file
            with open(f"{sys_paths[i]}/ABACUS/{f}/STRU", "w") as stru_file:
                stru_file.write(make_abacus_scf_stru(sys_data, pp_files, pre_args_new))
            #write INPUT file
            with open(f"{sys_paths[i]}/ABACUS/{f}/INPUT", "w") as input_file:
                input_file.write(make_abacus_scf_input(pre_args_new))

            #write KPT file if k_points is explicitly specified or for gamma_only case
            if pre_args_new["k_points"] is not None or pre_args_new["gamma_only"] is True:
                with open(f"{sys_paths[i]}/ABACUS/{f}/KPT","w") as kpt_file:
                    kpt_file.write(make_abacus_scf_kpt(pre_args_new))


def make_convert_scf_abacus(systems_train, systems_test=None,
                no_model=True, model_file=None, resources=None, **pre_args):
    # if no test systems, use last one in train systems
    systems_train = [os.path.abspath(s) for s in load_sys_paths(systems_train)]
    systems_test = [os.path.abspath(s) for s in load_sys_paths(systems_test)]
    #share model file if needed
    link_prev = pre_args.pop("link_prev_files", [])
    if not systems_test:
        systems_test.append(systems_train[-1])
        # if len(systems_train) > 1:
        #     del systems_train[-1]
    check_system_names(systems_train)
    check_system_names(systems_test)
    #update pre_args
    if not no_model:
        assert model_file is not None
        link_prev.append((model_file, "model.pth"))
    if resources is not None and "task_per_node" in resources:
        task_per_node = resources["task_per_node"]
    pre_args.update(
        systems_train=systems_train, 
        systems_test=systems_test,
        model_file=model_file,
        no_model=no_model, 
        task_per_node = task_per_node, 
        **pre_args)
    return PythonTask(
        convert_data, 
        call_kwargs=pre_args,
        outlog="convert.log",
        errlog="err",
        workdir='.', 
        link_prev_files=link_prev
    )


def make_run_scf_abacus(systems_train, systems_test=None,  
                outlog="out.log",  errlog="err.log", group_size=1,
                resources=None, dispatcher=None, 
                share_folder="share", workdir=".", link_systems=True, 
                dpdispatcher_machine=None, dpdispatcher_resources=None,
                no_model=True, **task_args):
    #basic args
    link_share = task_args.pop("link_share_files", [])
    link_prev = task_args.pop("link_prev_files", [])
    link_abs = task_args.pop("link_abs_files", [])
    forward_files = task_args.pop("forward_files", [])
    backward_files = task_args.pop("backward_files", [])
    if not no_model:
        forward_files.append("../"+CMODEL_FILE) #relative to work_base: system
    #get systems
    systems_train = [os.path.abspath(s) for s in load_sys_paths(systems_train)]
    systems_test = [os.path.abspath(s) for s in load_sys_paths(systems_test)]
    if not systems_test:
        systems_test.append(systems_train[-1])
        # if len(systems_train) > 1:
        #     del systems_train[-1]
    check_system_names(systems_train)
    check_system_names(systems_test)
    #systems=systems_train+systems_test
    sys_train_paths = [os.path.abspath(s) for s in load_sys_paths(systems_train)]
    sys_train_base = [get_sys_name(s) for s in sys_train_paths]
    sys_train_name = [os.path.basename(s) for s in sys_train_base]
    sys_test_paths = [os.path.abspath(s) for s in load_sys_paths(systems_test)]
    sys_test_base = [get_sys_name(s) for s in sys_test_paths]
    sys_test_name = [os.path.basename(s) for s in sys_test_base]
    sys_paths=sys_train_paths + sys_test_paths
    sys_base=sys_train_base+sys_test_base
    sys_name=sys_train_name+sys_test_name
    if link_systems:
        target_dir="systems"
        src_files = sum((glob(f"{base}*") for base in sys_base), [])
        for fl in src_files:
            dst = os.path.join(target_dir, os.path.basename(fl))
            link_abs.append((fl, dst)) 
    #set parameters
    if resources is not None and "task_per_node" in resources:
        task_per_node = resources["task_per_node"]
    run_cmd = task_args.pop("run_cmd", "mpirun")
    abacus_path = task_args.pop("abacus_path", None)
    assert abacus_path is not None
    #make task
    task_list=[]
    if dispatcher=="dpdispatcher":
        if dpdispatcher_resources is not None and "cpu_per_node" in dpdispatcher_resources:
            assert task_per_node <= dpdispatcher_resources["cpu_per_node"]
        #make task_list
        from dpdispatcher import Task
        singletask={
            "command": None, 
            "task_work_path": "./",
            "forward_files":[],
            "backward_files": [], 
            "outlog": outlog,
            "errlog": errlog
        }
        for i, pth in enumerate(sys_paths):
            try:
                atom_data = np.load(f"{str(pth)}/atom.npy")
            except FileNotFoundError:
                atom_data = coord_to_atom(str(pth))
            nframes = atom_data.shape[0]
            for f in range(nframes):
                singletask["command"]=str(f"cd {sys_name[i]}/ABACUS/{f}/ &&  \
                    {run_cmd} -n {task_per_node} {abacus_path} > {outlog} 2>{errlog}  &&  \
                    echo {f}`grep -i converge ./OUT.ABACUS/running_scf.log` > conv  &&  \
                    echo {f}`grep -i converge ./OUT.ABACUS/running_scf.log`")
                singletask["task_work_path"]="."
                singletask["forward_files"]=[str(f"./{sys_name[i]}/ABACUS/{f}/")]
                singletask["backward_files"]=[str(f"./{sys_name[i]}/ABACUS/{f}/")]
                task_list.append(Task.load_from_dict(singletask))
        return DPDispatcherTask(
            task_list,
            work_base="systems",
            outlog=outlog,
            share_folder=share_folder,
            link_share_files=link_share,
            link_prev_files=link_prev,
            link_abs_files=link_abs,
            machine=dpdispatcher_machine,
            resources=dpdispatcher_resources,
            forward_files=forward_files,
            backward_files=backward_files
        )
    else:
        batch_tasks=[]
        for i, pth in enumerate(sys_paths):
            try:
                atom_data = np.load(f"{str(pth)}/atom.npy")
            except FileNotFoundError:
                atom_data = coord_to_atom(str(pth))
            nframes = atom_data.shape[0]
            for f in range(nframes):
                batch_tasks.append(BatchTask(
                    cmds=str(f"cd {sys_name[i]}/ABACUS/{f}/ &&  \
                    {run_cmd} -n {task_per_node} {abacus_path} > {outlog} 2>{errlog}  &&  \
                    echo {f}`grep -i converge ./OUT.ABACUS/running_scf.log` > conv  &&  \
                    echo {f}`grep -i converge ./OUT.ABACUS/running_scf.log`"),
                    workdir="systems",
                    forward_files=[str(f"./{sys_name[i]}/ABACUS/{f}/")],
                    backward_files=[str(f"./{sys_name[i]}/ABACUS/{f}/")]
                )) 
        return GroupBatchTask(
            batch_tasks,
            group_size=group_size, 
            workdir="./",
            dispatcher=dispatcher,
            resources=resources,
            outlog=outlog,
            share_folder=share_folder,
            link_share_files=link_share,
            link_prev_files=link_prev,
            link_abs_files=link_abs,
            forward_files=forward_files,
            backward_files=backward_files
        )
    



def _load_frame(arr_dict, key, frame_idx, nframes, data):
    """Load a per-frame npy array into a pre-allocated or newly created array."""
    if arr_dict[key] is None:
        arr_dict[key] = np.empty((nframes,) + data.shape, dtype=data.dtype)
    assert data.shape == arr_dict[key].shape[1:], (
        f"Shape of {key} {arr_dict[key].shape} does not match {data.shape}!")
    arr_dict[key][frame_idx] = data


def _collect_frames(sys_path, nframes, cal_force, cal_stress,
                    deepks_bandgap, deepks_v_delta, deepks_scf):
    """
    Loop over frames for one system and collect all per-frame arrays.
    Returns a dict of numpy arrays.
    """
    arrs = dict(
        dm_eig=None,
        e_tot=None, e_base=None,
        f_tot=None, f_base=None, gvx=None,
        s_tot=None, s_base=None, gvepsl=None,
        o_tot=None, o_base=None, orbital_precalc=None,
        h_tot=None, h_base=None,
        v_delta_precalc=None, phialpha=None, gevdm=None,
        hr_tot=None, hr_base=None, vdr_precalc=None,
    )
    conv = np.full((nframes, 1), False)

    for f in range(nframes):
        load_f_path = f"{sys_path}/ABACUS/{f}/OUT.ABACUS/"

        # Convergence
        with open(f"{sys_path}/ABACUS/{f}/conv") as conv_file:
            ic = [t.strip('#').upper() for t in conv_file.read().split()]
            if ("CONVERGED" in ic or "ACHIEVED" in ic) and "NOT" not in ic:
                conv[int(ic[0])] = True

        # Descriptor
        _load_frame(arrs, 'dm_eig', f, nframes, np.load(load_f_path + "deepks_dm_eig.npy"))

        # Energy
        e_tot_data = np.load(load_f_path + "deepks_etot.npy")
        _load_frame(arrs, 'e_tot', f, nframes, e_tot_data)
        if deepks_scf:
            _load_frame(arrs, 'e_base', f, nframes, np.load(load_f_path + "deepks_ebase.npy"))
        else:
            _load_frame(arrs, 'e_base', f, nframes, e_tot_data)

        # Forces
        if cal_force:
            f_tot_data = np.load(load_f_path + "deepks_ftot.npy")
            _load_frame(arrs, 'f_tot', f, nframes, f_tot_data)
            if deepks_scf:
                _load_frame(arrs, 'f_base', f, nframes, np.load(load_f_path + "deepks_fbase.npy"))
            else:
                _load_frame(arrs, 'f_base', f, nframes, f_tot_data)
            if os.path.exists(load_f_path + "deepks_gradvx.npy"):
                _load_frame(arrs, 'gvx', f, nframes, np.load(load_f_path + "deepks_gradvx.npy"))

        # Stress
        if cal_stress:
            s_tot_data = np.load(load_f_path + "deepks_stot.npy")
            _load_frame(arrs, 's_tot', f, nframes, s_tot_data)
            if deepks_scf:
                _load_frame(arrs, 's_base', f, nframes, np.load(load_f_path + "deepks_sbase.npy"))
            else:
                _load_frame(arrs, 's_base', f, nframes, s_tot_data)
            if os.path.exists(load_f_path + "deepks_gvepsl.npy"):
                _load_frame(arrs, 'gvepsl', f, nframes, np.load(load_f_path + "deepks_gvepsl.npy"))

        # Bandgap (orbital)
        if deepks_bandgap:
            o_tot_data = np.load(load_f_path + "deepks_otot.npy")
            _load_frame(arrs, 'o_tot', f, nframes, o_tot_data)
            if deepks_scf:
                _load_frame(arrs, 'o_base', f, nframes, np.load(load_f_path + "deepks_obase.npy"))
            else:
                _load_frame(arrs, 'o_base', f, nframes, o_tot_data)
            if os.path.exists(load_f_path + "deepks_orbpre.npy"):
                _load_frame(arrs, 'orbital_precalc', f, nframes, np.load(load_f_path + "deepks_orbpre.npy"))

        # V_delta / Hamiltonian (k-space)
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
                if (os.path.exists(load_f_path + "deepks_phialpha.npy") and
                        os.path.exists(load_f_path + "deepks_gevdm.npy")):
                    _load_frame(arrs, 'phialpha', f, nframes, np.load(load_f_path + "deepks_phialpha.npy"))
                    _load_frame(arrs, 'gevdm',    f, nframes, np.load(load_f_path + "deepks_gevdm.npy"))

        # V_delta_R / Hamiltonian (real-space)
        if deepks_v_delta < 0:
            def _pad3(a, target):
                """Pad the first 3 spatial dims of a to match target size."""
                n_add = target - a.shape[0]
                if n_add > 0:
                    a = np.pad(a, ((0,n_add),(0,n_add),(0,n_add),(0,0),(0,0)))
                return a

            hrcs = read_csr(load_f_path + "deepks_hrtot.csr").to_dense().numpy()
            # grow hr_tot if needed
            if arrs['hr_tot'] is None:
                arrs['hr_tot'] = np.empty((nframes,) + hrcs.shape, dtype=hrcs.dtype)
            elif hrcs.shape[0] > arrs['hr_tot'].shape[1]:
                n = hrcs.shape[0] - arrs['hr_tot'].shape[1]
                arrs['hr_tot'] = np.pad(arrs['hr_tot'], ((0,0),(0,n),(0,n),(0,n),(0,0),(0,0)))
            elif hrcs.shape[0] < arrs['hr_tot'].shape[1]:
                hrcs = _pad3(hrcs, arrs['hr_tot'].shape[1])
            arrs['hr_tot'][f] = hrcs

            if os.path.exists(load_f_path + "deepks_hrdelta.csr"):
                v_delta_r = read_csr(load_f_path + "deepks_hrdelta.csr").to_dense().numpy()
                # align shapes
                if v_delta_r.shape[0] < hrcs.shape[0]:
                    v_delta_r = _pad3(v_delta_r, hrcs.shape[0])
                elif v_delta_r.shape[0] > hrcs.shape[0]:
                    hrcs = _pad3(hrcs, v_delta_r.shape[0])
                if deepks_scf:
                    hrtmp = hrcs - v_delta_r
                else:
                    hrtmp = hrcs  # base == tot when deepks_scf=0
                if arrs['hr_base'] is None:
                    arrs['hr_base'] = np.empty((nframes,) + hrtmp.shape, dtype=hrtmp.dtype)
                elif hrtmp.shape[0] > arrs['hr_base'].shape[1]:
                    n = hrtmp.shape[0] - arrs['hr_base'].shape[1]
                    arrs['hr_base'] = np.pad(arrs['hr_base'], ((0,0),(0,n),(0,n),(0,n),(0,0),(0,0)))
                elif hrtmp.shape[0] < arrs['hr_base'].shape[1]:
                    hrtmp = _pad3(hrtmp, arrs['hr_base'].shape[1])
                arrs['hr_base'][f] = hrtmp

            if deepks_v_delta == -1 and os.path.exists(load_f_path + "deepks_vdrpre.npy"):
                tmp = np.load(load_f_path + "deepks_vdrpre.npy")
                if arrs['vdr_precalc'] is None:
                    arrs['vdr_precalc'] = np.empty((nframes,) + tmp.shape, dtype=tmp.dtype)
                elif tmp.shape[0] > arrs['vdr_precalc'].shape[1]:
                    n = tmp.shape[0] - arrs['vdr_precalc'].shape[1]
                    arrs['vdr_precalc'] = np.pad(arrs['vdr_precalc'],
                                                  ((0,0),(0,n),(0,n),(0,n),(0,0),(0,0),(0,0),(0,0)))
                elif tmp.shape[0] < arrs['vdr_precalc'].shape[1]:
                    n = arrs['vdr_precalc'].shape[1] - tmp.shape[0]
                    tmp = np.pad(tmp, ((0,n),(0,n),(0,n),(0,0),(0,0),(0,0),(0,0)))
                arrs['vdr_precalc'][f] = tmp
            elif deepks_v_delta == -2 and os.path.exists(load_f_path + "deepks_gevdm.npy"):
                _load_frame(arrs, 'gevdm', f, nframes, np.load(load_f_path + "deepks_gevdm.npy"))

    arrs['conv'] = conv
    return arrs


def _save_system_data(save_path, load_ref_path, arrs,
                     nframes, natoms, cal_force, cal_stress,
                     deepks_bandgap, deepks_v_delta):
    """Save collected arrays for one system."""
    os.makedirs(save_path, exist_ok=True)

    np.save(save_path + "conv.npy",   arrs['conv'])
    np.save(save_path + "dm_eig.npy", arrs['dm_eig'])

    # Energy
    e_base = coerce_energy(arrs['e_base'], nframes, 'e_base.npy')
    e_tot  = coerce_energy(arrs['e_tot'],  nframes, 'e_tot.npy')
    e_ref  = coerce_energy(np.load(load_ref_path + "energy.npy"), nframes, 'energy.npy')
    np.save(save_path + "e_base.npy",    e_base)
    np.save(save_path + "e_tot.npy",     e_tot)
    np.save(save_path + "energy.npy",    e_ref)
    np.save(save_path + "l_e_delta.npy", e_ref - e_base)

    np.save(save_path + "atom.npy", arrs['atom_data'])
    np.save(save_path + "box.npy",  arrs['box_data'])

    # Forces
    if cal_force:
        f_ref = np.load(load_ref_path + "force.npy")
        if f_ref.shape != (nframes, natoms, 3):
            raise ValueError(f"force.npy shape should be (nframes,natoms,3), got {f_ref.shape}.")
        np.save(save_path + "f_base.npy",    arrs['f_base'])
        np.save(save_path + "f_tot.npy",     arrs['f_tot'])
        np.save(save_path + "force.npy",     f_ref)
        np.save(save_path + "l_f_delta.npy", f_ref - arrs['f_base'])
        if arrs['gvx'] is not None:
            np.save(save_path + "grad_vx.npy", arrs['gvx'])

    # Stress
    if cal_stress:
        s_ref  = coerce_stress(np.load(load_ref_path + "stress.npy"), nframes, 'stress.npy')
        s_base = coerce_stress(arrs['s_base'], nframes, 's_base')
        s_tot  = coerce_stress(arrs['s_tot'],  nframes, 's_tot')
        np.save(save_path + "s_base.npy",    s_base)
        np.save(save_path + "s_tot.npy",     s_tot)
        np.save(save_path + "stress.npy",    s_ref)
        np.save(save_path + "l_s_delta.npy", s_ref - s_base)
        if arrs['gvepsl'] is not None:
            np.save(save_path + "grad_epsilon.npy", arrs['gvepsl'])

    # Bandgap (orbital)
    if deepks_bandgap:
        o_ref = np.load(load_ref_path + "orbital.npy")
        if o_ref.shape[0] != nframes or o_ref.shape[2] != 1:
            raise ValueError(f"orbital.npy shape should be (nframes,nkpt,1), got {o_ref.shape}.")
        np.save(save_path + "o_base.npy",    arrs['o_base'])
        np.save(save_path + "o_tot.npy",     arrs['o_tot'])
        np.save(save_path + "orbital.npy",   o_ref)
        np.save(save_path + "l_o_delta.npy", o_ref - arrs['o_base'])
        if arrs['orbital_precalc'] is not None:
            np.save(save_path + "orbital_precalc.npy", arrs['orbital_precalc'])

    # V_delta / Hamiltonian (k-space, deepks_v_delta > 0)
    if deepks_v_delta > 0:
        h_ref = np.load(load_ref_path + "hamiltonian.npy")
        if h_ref.shape[0] != nframes or h_ref.ndim != 4:
            raise ValueError(f"hamiltonian.npy shape should be (nframes,nkpt,nlocal,nlocal), got {h_ref.shape}.")
        np.save(save_path + "h_base.npy",      arrs['h_base'])
        np.save(save_path + "h_tot.npy",       arrs['h_tot'])
        np.save(save_path + "hamiltonian.npy", h_ref)
        np.save(save_path + "l_h_delta.npy",   h_ref - arrs['h_base'])
        if arrs['v_delta_precalc'] is not None:
            np.save(save_path + "v_delta_precalc.npy", arrs['v_delta_precalc'])
        elif arrs['phialpha'] is not None and arrs['gevdm'] is not None:
            np.save(save_path + "grad_evdm.npy", arrs['gevdm'])
            np.save(save_path + "phialpha.npy",  arrs['phialpha'])
        if os.path.exists(load_ref_path + "overlap.npy"):
            np.save(save_path + "overlap.npy", np.load(load_ref_path + "overlap.npy"))

    # V_delta_R / Hamiltonian (real-space, deepks_v_delta < 0)
    if deepks_v_delta < 0:
        hr_ref = np.load(load_ref_path + "hamiltonian_r.npy")
        if hr_ref.shape[0] != nframes or hr_ref.ndim != 6:
            raise ValueError(f"hamiltonian_r.npy shape should be (nframes,nR,nR,nR,nlocal,nlocal), got {hr_ref.shape}.")
        # align spatial dims between hr_base/hr_tot and hr_ref
        for key in ('hr_base', 'hr_tot'):
            arr = arrs[key]
            if arr is None:
                continue
            if arr.shape[1] < hr_ref.shape[1]:
                n = hr_ref.shape[1] - arr.shape[1]
                arrs[key] = np.pad(arr, ((0,0),(0,n),(0,n),(0,n),(0,0),(0,0)))
            elif arr.shape[1] > hr_ref.shape[1]:
                n = arr.shape[1] - hr_ref.shape[1]
                hr_ref = np.pad(hr_ref, ((0,0),(0,n),(0,n),(0,n),(0,0),(0,0)))
        np.save(save_path + "hamiltonian_r.npy", hr_ref)
        if arrs['hr_tot'] is not None:
            np.save(save_path + "hr_tot.npy", arrs['hr_tot'])
        if arrs['hr_base'] is not None:
            np.save(save_path + "hr_base.npy",    arrs['hr_base'])
            np.save(save_path + "l_hr_delta.npy", hr_ref - arrs['hr_base'])
        if deepks_v_delta == -1 and arrs['vdr_precalc'] is not None:
            np.save(save_path + "vdr_precalc.npy", arrs['vdr_precalc'])
        elif deepks_v_delta == -2 and arrs['gevdm'] is not None:
            np.save(save_path + "grad_evdm.npy", arrs['gevdm'])


def collect_data(systems, save_dir, ref_dir,
                cal_force=True, cal_stress=False,
                deepks_bandgap=False, deepks_v_delta=0,
                deepks_scf=True):
    """Collect data from multiple systems and save to save_dir."""
    for sys in systems:
        print(f"Collecting system: {sys}")
        load_ref_path = os.path.join(ref_dir, sys, "")
        save_path     = os.path.join(save_dir, sys, "")

        # count frames
        nframes = len(np.load(load_ref_path + "energy.npy"))
        # count atoms
        atom_data = np.load(load_ref_path + "atom.npy")
        natoms    = int(atom_data[0].sum())

        arrs = _collect_system_data(
            load_ref_path, nframes, natoms,
            cal_force, cal_stress,
            deepks_bandgap, deepks_v_delta, deepks_scf
        )

        _save_system_data(
            save_path, load_ref_path, arrs,
            nframes, natoms,
            cal_force, cal_stress,
            deepks_bandgap, deepks_v_delta
        )
        print(f"  -> saved to {save_path}")


def gather_stats_abacus(systems_train, systems_test,
                        train_dump, test_dump,
                        cal_force=0, cal_stress=0,
                        deepks_bandgap=0, deepks_v_delta=0,
                        deepks_scf=1, **stat_args):
    """
    Gather statistics for training and testing data from ABACUS calculations.

    When deepks_scf=0 the DeePKS correction was not applied, so
    etot==ebase (and likewise for forces / stress / etc.).  The *base
    arrays are therefore set equal to the *tot arrays without reading
    separate *base.npy files from disk.
    """
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

            # Load geometry
            try:
                atom_data = np.load(load_ref_path + "atom.npy")
            except FileNotFoundError:
                atom_data = coord_to_atom(sys_path)
            nframes = atom_data.shape[0]
            natoms  = atom_data.shape[1]
            if atom_data.shape[2] != 4:
                raise ValueError("atom.npy should have shape (nframes, natoms, 4)")

            if os.path.isfile(load_ref_path + "box.npy"):
                box_data = coerce_box(np.load(load_ref_path + "box.npy"), nframes, 'box.npy')
                box_data = box_data.reshape(nframes, 3, 3)
            else:
                box_data = np.array([stat_args["lattice_vector"]])
                box_data = box_data.reshape(1, 9).repeat(nframes, axis=0)
            box_data = box_data.reshape(nframes, 3, 3)

            if stat_args.get("coord_type") == "Direct":
                atom_data[:, :, 1:4] = np.matmul(atom_data[:, :, 1:4], box_data)
            atom_data[:, :, 1:4] *= stat_args['lattice_constant']
            box_data *= stat_args['lattice_constant']

            # Collect per-frame arrays
            arrs = _collect_frames(
                sys_path, nframes,
                cal_force, cal_stress,
                deepks_bandgap, deepks_v_delta, deepks_scf
            )
            arrs['atom_data'] = atom_data
            arrs['box_data']  = box_data

            # Save
            _save_system_data(
                save_path, load_ref_path, arrs,
                nframes, natoms,
                cal_force, cal_stress,
                deepks_bandgap, deepks_v_delta
            )

    _process_systems(sys_train_paths, sys_train_names, train_dump)
    _process_systems(sys_test_paths,  sys_test_names,  test_dump)

    # check convergence and print in log
    from deepks.physics.backends.stats import print_stats
    print_stats(systems=systems_train, test_sys=systems_test,
            dump_dir=train_dump, test_dump=test_dump, group=False,
            with_conv=True, with_e=True, e_name="e_tot",
               with_f=True, f_name="f_tot")
    return


def make_stat_scf_abacus(systems_train, systems_test=None, *,
                  train_dump="data_train", test_dump="data_test", cal_force=0, cal_stress=0, deepks_bandgap=0, deepks_v_delta=0,
                  deepks_scf=0,
                  workdir='.', outlog="log.data", **stat_args):
    # follow same convention for systems as run_scf
    systems_train = [os.path.abspath(s) for s in load_sys_paths(systems_train)]
    systems_test = [os.path.abspath(s) for s in load_sys_paths(systems_test)]
    if not systems_test:
        systems_test.append(systems_train[-1])
        # if len(systems_train) > 1:
        #     del systems_train[-1]
    # load stats function
    stat_args.update(
        systems_train=systems_train,
        systems_test=systems_test,
        train_dump=train_dump,
        test_dump=test_dump,
        cal_force=cal_force,
        cal_stress=cal_stress,
        deepks_bandgap=deepks_bandgap,
        deepks_v_delta=deepks_v_delta,
        deepks_scf=deepks_scf)
    # make task
    return PythonTask(
        gather_stats_abacus,
        call_kwargs=stat_args,
        outlog=outlog,
        errlog="err",
        workdir=workdir
    )



