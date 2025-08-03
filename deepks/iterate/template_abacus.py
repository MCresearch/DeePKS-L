import os
import numpy as np
from glob import glob
from collections import Counter
from deepks.default import CMODEL_FILE, NAME_TYPE, TYPE_NAME
from deepks.utils import flat_file_list, load_dirs
from deepks.utils import get_sys_name, load_sys_paths
from deepks.task.task import PythonTask
from deepks.task.task import BatchTask, GroupBatchTask, DPDispatcherTask
from deepks.task.workflow import Sequence
from deepks.iterate.template import check_system_names, make_cleanup
from deepks.iterate.generator_abacus import make_abacus_scf_kpt, make_abacus_scf_input, make_abacus_scf_stru
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
    from deepks.iterate.iterate import check_share_folder
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
            from deepks.model import CorrNet
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
            from deepks.utils import load_yaml
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
    



def gather_stats_abacus(systems_train, systems_test, 
                train_dump, test_dump, cal_force=0, cal_stress=0, deepks_bandgap=0, deepks_v_delta=0, **stat_args):
    """
    Gather statistics for training and testing data from ABACUS calculations. 
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

    # concatenate data (train)
    if not os.path.exists(train_dump):
        os.mkdir(train_dump)
    for i in range(len(systems_train)):
        load_ref_path = f"{sys_train_paths[i]}/"
        save_path = f"{train_dump}/{sys_train_names[i]}/"
        if not os.path.exists(train_dump + '/' + sys_train_names[i]):
            os.mkdir(train_dump + '/' + sys_train_names[i])
        try:
            atom_data = np.load(load_ref_path + "atom.npy")
        except FileNotFoundError:
            atom_data = coord_to_atom(sys_train_paths[i])
        nframes = atom_data.shape[0]
        natoms = atom_data.shape[1]
        if atom_data.shape[2] != 4:
            raise ValueError("atom.npy should have shape (nframes, natoms, 4)")

        ## Initialize of properties list
        conv = np.full((nframes,1), False) # convergence of each frame
        dm_eig = None # descriptor

        # properties for base model
        e_base = None
        f_base = None
        s_base = None
        o_base = None
        h_base = None

        # properties for total model
        e_tot = None
        f_tot = None
        s_tot = None
        o_tot = None
        h_tot = None

        # properties for deepks calculation
        gvx = None
        gvepsl = None
        orbital_precalc = None
        v_delta_precalc = None
        phialpha = None
        gevdm = None
        overlap = None

        ## Main loop over frames in training data
        for f in range(nframes):
            load_f_path = f"{sys_train_paths[i]}/ABACUS/{f}/OUT.ABACUS/"
            # Check convergence of each frame
            with open(f"{sys_train_paths[i]}/ABACUS/{f}/conv","r") as conv_file:
                ic=conv_file.read().split()
                ic = [item.strip('#') for item in ic]
                if "CONVERGED" in ic and "NOT" not in ic:
                    conv[(int)(ic[0])]=True

            # Energy and eigenvalues of density matrix
            des = np.load(load_f_path + "deepks_dm_eig.npy")
            if dm_eig is None:
                dm_eig = np.empty((nframes,) + des.shape, dtype=des.dtype)
            assert des.shape == dm_eig.shape[1:], f"Shape of dm_eig {dm_eig.shape} does not match with {des.shape}!"
            dm_eig[f] = des

            ene = np.load(load_f_path + "deepks_ebase.npy")
            if e_base is None:
                e_base = np.empty((nframes,) + ene.shape, dtype=ene.dtype)
            assert ene.shape == e_base.shape[1:], f"Shape of e_base {e_base.shape} does not match with {ene.shape}!"
            e_base[f] = ene

            ene = np.load(load_f_path + "deepks_etot.npy")
            if e_tot is None:
                e_tot = np.empty((nframes,) + ene.shape, dtype=ene.dtype)
            assert ene.shape == e_tot.shape[1:], f"Shape of e_tot {e_tot.shape} does not match with {ene.shape}!"
            e_tot[f] = ene

            # Forces 
            if(cal_force):
                fcs = np.load(load_f_path + "deepks_fbase.npy")
                if f_base is None:
                    f_base = np.empty((nframes,) + fcs.shape, dtype=fcs.dtype)
                assert fcs.shape == f_base.shape[1:], f"Shape of f_base {f_base.shape} does not match with {fcs.shape}!"
                f_base[f] = fcs

                fcs = np.load(load_f_path + "deepks_ftot.npy")
                if f_tot is None:
                    f_tot = np.empty((nframes,) + fcs.shape, dtype=fcs.dtype)
                assert fcs.shape == f_tot.shape[1:], f"Shape of f_tot {f_tot.shape} does not match with {fcs.shape}!"
                f_tot[f] = fcs

                if os.path.exists(load_f_path + "deepks_gradvx.npy"):
                    gvx_tmp = np.load(load_f_path + "deepks_gradvx.npy")
                    if gvx is None:
                        gvx = np.empty((nframes,) + gvx_tmp.shape, dtype=gvx_tmp.dtype)
                    assert gvx_tmp.shape == gvx.shape[1:], f"Shape of gvx {gvx.shape} does not match with {gvx_tmp.shape}!"
                    gvx[f] = gvx_tmp

            # Stress
            if(cal_stress):
                scs=np.load(load_f_path + "deepks_sbase.npy")
                if s_base is None:
                    s_base = np.empty((nframes,) + scs.shape, dtype=scs.dtype)
                assert scs.shape == s_base.shape[1:], f"Shape of s_base {s_base.shape} does not match with {scs.shape}!"
                s_base[f] = scs
                
                scs=np.load(load_f_path + "deepks_stot.npy")
                if s_tot is None:
                    s_tot = np.empty((nframes,) + scs.shape, dtype=scs.dtype)
                assert scs.shape == s_tot.shape[1:], f"Shape of s_tot {s_tot.shape} does not match with {scs.shape}!"
                s_tot[f] = scs

                if os.path.exists(load_f_path + "deepks_gvepsl.npy"):
                    gvepsl_tmp = np.load(load_f_path + "deepks_gvepsl.npy")
                    if gvepsl is None:
                        gvepsl = np.empty((nframes,) + gvepsl_tmp.shape, dtype=gvepsl_tmp.dtype)
                    assert gvepsl_tmp.shape == gvepsl.shape[1:], f"Shape of gvepsl {gvepsl.shape} does not match with {gvepsl_tmp.shape}!"
                    gvepsl[f] = gvepsl_tmp

            # Bandgap (orbital)
            if(deepks_bandgap):
                ocs = np.load(load_f_path + "deepks_obase.npy")
                if o_base is None:
                    o_base = np.empty((nframes,) + ocs.shape, dtype=ocs.dtype)
                assert ocs.shape == o_base.shape[1:], f"Shape of o_base {o_base.shape} does not match with {ocs.shape}!"
                o_base[f] = ocs 
                    
                ocs = np.load(load_f_path + "deepks_otot.npy")
                if o_tot is None:
                    o_tot = np.empty((nframes,) + ocs.shape, dtype=ocs.dtype)
                assert ocs.shape == o_tot.shape[1:], f"Shape of o_tot {o_tot.shape} does not match with {ocs.shape}!"
                o_tot[f] = ocs 
                
                if os.path.exists(load_f_path + "deepks_orbpre.npy"):
                    orbital_precalc_tmp = np.load(load_f_path + "deepks_orbpre.npy")
                    if orbital_precalc is None:
                        orbital_precalc = np.empty((nframes,) + orbital_precalc_tmp.shape, dtype=orbital_precalc_tmp.dtype)
                    assert orbital_precalc_tmp.shape == orbital_precalc.shape[1:], f"Shape of orbital_precalc {orbital_precalc.shape} does not match with {orbital_precalc_tmp.shape}!"
                    orbital_precalc[f] = orbital_precalc_tmp

            # V_delta (Hamiltonian) 
            if(deepks_v_delta > 0):
                hcs = np.load(load_f_path + "deepks_hbase.npy")
                if h_base is None:
                    h_base = np.empty((nframes,) + hcs.shape, dtype=hcs.dtype)
                assert hcs.shape == h_base.shape[1:], f"Shape of h_base {h_base.shape} does not match with {hcs.shape}!"
                h_base[f] = hcs 
                
                hcs = np.load(load_f_path + "deepks_htot.npy")
                if h_tot is None:
                    h_tot = np.empty((nframes,) + hcs.shape, dtype=hcs.dtype)
                assert hcs.shape == h_tot.shape[1:], f"Shape of h_tot {h_tot.shape} does not match with {hcs.shape}!"
                h_tot[f] = hcs 
                
                if deepks_v_delta == 1:
                    if os.path.exists(load_f_path + "deepks_vdpre.npy"):
                        v_delta_precalc_tmp = np.load(load_f_path + "deepks_vdpre.npy")
                        if v_delta_precalc is None:
                            v_delta_precalc = np.empty((nframes,) + v_delta_precalc_tmp.shape, dtype=v_delta_precalc_tmp.dtype)
                        assert v_delta_precalc_tmp.shape == v_delta_precalc.shape[1:], f"Shape of v_delta_precalc {v_delta_precalc.shape} does not match with {v_delta_precalc_tmp.shape}!"
                        v_delta_precalc[f] = v_delta_precalc_tmp
                elif deepks_v_delta == 2:
                    if os.path.exists(load_f_path + "deepks_phialpha.npy") and os.path.exists(load_f_path + "deepks_gevdm.npy"):
                        phialpha_tmp = np.load(load_f_path + "deepks_phialpha.npy")
                        # complex double to complex float
                        # phialpha_tmp = phialpha_tmp.astype(np.complex64)
                        if phialpha is None:
                            phialpha = np.empty((nframes,) + phialpha_tmp.shape, dtype=phialpha_tmp.dtype)
                        assert phialpha_tmp.shape == phialpha.shape[1:], f"Shape of phialpha {phialpha.shape} does not match with {phialpha_tmp.shape}!"
                        phialpha[f] = phialpha_tmp
                        
                        gevdm_tmp = np.load(load_f_path + "deepks_gevdm.npy")
                        if gevdm is None:
                            gevdm = np.empty((nframes,) + gevdm_tmp.shape, dtype=gevdm_tmp.dtype)
                        assert gevdm_tmp.shape == gevdm.shape[1:], f"Shape of gevdm {gevdm.shape} does not match with {gevdm_tmp.shape}!"
                        gevdm[f] = gevdm_tmp
        
        ## Save data    
        # Convergence      
        np.save(save_path + "conv.npy", conv)
        del conv

        # Energy and eigenvalues of density matrix
        np.save(save_path + "dm_eig.npy", dm_eig)
        del dm_eig

        np.save(save_path + "e_base.npy", e_base)    #Ry to Hartree
        e_ref = np.load(load_ref_path + "energy.npy")
        if e_ref.shape != (nframes, 1):
            if e_ref.shape == (nframes,):
                e_ref = e_ref.reshape((nframes, 1))
                print(f"energy.npy shape {e_ref.shape} is reshaped to {(nframes, 1)}")
            else:
                raise ValueError(f"energy.npy shape should be (nframes, 1), but got {e_ref.shape}.")
        np.save(save_path + "energy.npy", e_ref)
        np.save(save_path + "l_e_delta.npy", e_ref-e_base)
        np.save(save_path + "e_tot.npy", e_tot)
        del e_base, e_tot, e_ref
        
        np.save(save_path + "atom.npy", atom_data)
        del atom_data

        # Forces
        if(cal_force): 
            np.save(save_path + "f_base.npy", f_base)
            f_ref = np.load(load_ref_path + "force.npy")
            if f_ref.shape != (nframes, natoms, 3):
                raise ValueError(f"force.npy shape should be (nframes, natoms, 3), but got {f_ref.shape}.")
            np.save(save_path + "force.npy", f_ref)
            np.save(save_path + "l_f_delta.npy", f_ref-f_base)
            np.save(save_path + "f_tot.npy", f_tot)
            del f_base, f_tot, f_ref
            if gvx is not None:
                np.save(save_path + "grad_vx.npy", gvx)
                del gvx

        # Stress
        if(cal_stress): 
            np.save(save_path + "s_base.npy", s_base)
            s_ref = np.load(load_ref_path + "stress.npy")
            if s_ref.shape != (nframes, 9):
                raise ValueError(f"stress.npy shape should be (nframes, 9), but got {s_ref.shape}.")
            s_ref = s_ref[:,[0,1,2,4,5,8]] #only train the upper-triangle part
            np.save(save_path + "stress.npy", s_ref)
            np.save(save_path + "l_s_delta.npy", s_ref-s_base)
            np.save(save_path + "s_tot.npy", s_tot)
            del s_base, s_tot, s_ref
            if gvepsl is not None:
                np.save(save_path + "grad_epsilon.npy", gvepsl)
                del gvepsl

        # Bandgap (orbital)
        if(deepks_bandgap): 
            np.save(save_path + "o_base.npy", o_base)
            o_ref = np.load(load_ref_path + "orbital.npy")
            if o_ref.shape[0] != nframes or o_ref.shape[2] != 1:
                raise ValueError(f"orbital.npy shape should be (nframes, nkpt, 1), but got {o_ref.shape}.")
            np.save(save_path + "orbital.npy", o_ref)
            np.save(save_path + "l_o_delta.npy", o_ref-o_base)
            np.save(save_path + "o_tot.npy", o_tot)
            del o_base, o_tot, o_ref
            if orbital_precalc is not None:
                np.save(save_path + "orbital_precalc.npy", orbital_precalc)
                del orbital_precalc

        # V_delta (Hamiltonian)
        if(deepks_v_delta > 0): 
            np.save(save_path + "h_base.npy", h_base)
            h_ref = np.load(load_ref_path + "hamiltonian.npy")
            if h_ref.shape[0] != nframes or h_ref.ndim != 4:
                raise ValueError(f"hamiltonian.npy shape should be (nframes, nkpt, nlocal, nlocal), but got {h_ref.shape}.")
            np.save(save_path + "hamiltonian.npy", h_ref)
            np.save(save_path + "l_h_delta.npy", h_ref-h_base)
            np.save(save_path + "h_tot.npy", h_tot)
            del h_base, h_tot, h_ref
            if v_delta_precalc is not None:
                np.save(save_path + "v_delta_precalc.npy", v_delta_precalc)
                del v_delta_precalc
            elif phialpha is not None and gevdm is not None:
                np.save(save_path + "grad_evdm.npy", gevdm)
                del gevdm
                np.save(save_path + "phialpha.npy", phialpha)
                del phialpha
            if os.path.exists(load_ref_path + "overlap.npy"):
                overlap = np.load(load_ref_path + "overlap.npy")
                np.save(save_path + "overlap.npy", overlap)
                del overlap

    # concatenate data (test)
    if not os.path.exists(test_dump):
        os.mkdir(test_dump)
    for i in range(len(systems_test)):
        load_ref_path = f"{sys_test_paths[i]}/"
        save_path = f"{test_dump}/{sys_test_names[i]}/"
        if not os.path.exists(test_dump + '/' + sys_test_names[i]):
            os.mkdir(test_dump + '/' + sys_test_names[i])
        try:
            atom_data = np.load(load_ref_path + "atom.npy")
        except FileNotFoundError:
            atom_data = coord_to_atom(sys_test_paths[i])
        nframes = atom_data.shape[0]
        natoms = atom_data.shape[1]
        if atom_data.shape[2] != 4:
            raise ValueError("atom.npy should have shape (nframes, natoms, 4)")

        ## Initialize of properties list
        conv = np.full((nframes,1), False) # convergence of each frame
        dm_eig = None # descriptor

        # properties for base model
        e_base = None
        f_base = None
        s_base = None
        o_base = None
        h_base = None

        # properties for total model
        e_tot = None
        f_tot = None
        s_tot = None
        o_tot = None
        h_tot = None

        # properties for deepks calculation
        gvx = None
        gvepsl = None
        orbital_precalc = None
        v_delta_precalc = None
        phialpha = None
        gevdm = None
        overlap = None

        ## Main loop over frames in testing data
        for f in range(nframes):
            load_f_path = f"{sys_test_paths[i]}/ABACUS/{f}/OUT.ABACUS/"
            # Check convergence of each frame
            with open(f"{sys_test_paths[i]}/ABACUS/{f}/conv","r") as conv_file:
                ic=conv_file.read().split()
                ic = [item.strip('#') for item in ic]
                if "CONVERGED" in ic and "NOT" not in ic:
                    conv[(int)(ic[0])]=True

            # Energy and eigenvalues of density matrix
            des = np.load(load_f_path + "deepks_dm_eig.npy")
            if dm_eig is None:
                dm_eig = np.empty((nframes,) + des.shape, dtype=des.dtype)
            assert des.shape == dm_eig.shape[1:], f"Shape of dm_eig {dm_eig.shape} does not match with {des.shape}!"
            dm_eig[f] = des

            ene = np.load(load_f_path + "deepks_ebase.npy")
            if e_base is None:
                e_base = np.empty((nframes,) + ene.shape, dtype=ene.dtype)
            assert ene.shape == e_base.shape[1:], f"Shape of e_base {e_base.shape} does not match with {ene.shape}!"
            e_base[f] = ene  #Ry to Hartree

            ene = np.load(load_f_path + "deepks_etot.npy")
            if e_tot is None:
                e_tot = np.empty((nframes,) + ene.shape, dtype=ene.dtype)
            assert ene.shape == e_tot.shape[1:], f"Shape of e_tot {e_tot.shape} does not match with {ene.shape}!"
            e_tot[f] = ene  #Ry to Hartree

            # Forces 
            if(cal_force):
                fcs = np.load(load_f_path + "deepks_fbase.npy")
                if f_base is None:
                    f_base = np.empty((nframes,) + fcs.shape, dtype=fcs.dtype)
                assert fcs.shape == f_base.shape[1:], f"Shape of f_base {f_base.shape} does not match with {fcs.shape}!"
                f_base[f] = fcs     #Ry to Hartree

                fcs = np.load(load_f_path + "deepks_ftot.npy")
                if f_tot is None:
                    f_tot = np.empty((nframes,) + fcs.shape, dtype=fcs.dtype)
                assert fcs.shape == f_tot.shape[1:], f"Shape of f_tot {f_tot.shape} does not match with {fcs.shape}!"
                f_tot[f] = fcs     #Ry to Hartree

                if os.path.exists(load_f_path + "deepks_gradvx.npy"):
                    gvx_tmp = np.load(load_f_path + "deepks_gradvx.npy")
                    if gvx is None:
                        gvx = np.empty((nframes,) + gvx_tmp.shape, dtype=gvx_tmp.dtype)
                    assert gvx_tmp.shape == gvx.shape[1:], f"Shape of gvx {gvx.shape} does not match with {gvx_tmp.shape}!"
                    gvx[f] = gvx_tmp

            # Stress
            if(cal_stress):
                scs=np.load(load_f_path + "deepks_sbase.npy")
                if s_base is None:
                    s_base = np.empty((nframes,) + scs.shape, dtype=scs.dtype)
                assert scs.shape == s_base.shape[1:], f"Shape of s_base {s_base.shape} does not match with {scs.shape}!"
                s_base[f] = scs     #Ry to Hartree
                
                scs=np.load(load_f_path + "deepks_stot.npy")
                if s_tot is None:
                    s_tot = np.empty((nframes,) + scs.shape, dtype=scs.dtype)
                assert scs.shape == s_tot.shape[1:], f"Shape of s_tot {s_tot.shape} does not match with {scs.shape}!"
                s_tot[f] = scs     #Ry to Hartree

                if os.path.exists(load_f_path + "deepks_gvepsl.npy"):
                    gvepsl_tmp = np.load(load_f_path + "deepks_gvepsl.npy")
                    if gvepsl is None:
                        gvepsl = np.empty((nframes,) + gvepsl_tmp.shape, dtype=gvepsl_tmp.dtype)
                    assert gvepsl_tmp.shape == gvepsl.shape[1:], f"Shape of gvepsl {gvepsl.shape} does not match with {gvepsl_tmp.shape}!"
                    gvepsl[f] = gvepsl_tmp

            # Bandgap (orbital)
            if(deepks_bandgap):
                ocs = np.load(load_f_path + "deepks_obase.npy")
                if o_base is None:
                    o_base = np.empty((nframes,) + ocs.shape, dtype=ocs.dtype)
                assert ocs.shape == o_base.shape[1:], f"Shape of o_base {o_base.shape} does not match with {ocs.shape}!"
                o_base[f] = ocs 
                    
                ocs = np.load(load_f_path + "deepks_otot.npy")
                if o_tot is None:
                    o_tot = np.empty((nframes,) + ocs.shape, dtype=ocs.dtype)
                assert ocs.shape == o_tot.shape[1:], f"Shape of o_tot {o_tot.shape} does not match with {ocs.shape}!"
                o_tot[f] = ocs 
                
                if os.path.exists(load_f_path + "deepks_orbpre.npy"):
                    orbital_precalc_tmp = np.load(load_f_path + "deepks_orbpre.npy")
                    if orbital_precalc is None:
                        orbital_precalc = np.empty((nframes,) + orbital_precalc_tmp.shape, dtype=orbital_precalc_tmp.dtype)
                    assert orbital_precalc_tmp.shape == orbital_precalc.shape[1:], f"Shape of orbital_precalc {orbital_precalc.shape} does not match with {orbital_precalc_tmp.shape}!"
                    orbital_precalc[f] = orbital_precalc_tmp

            # V_delta (Hamiltonian) 
            if(deepks_v_delta > 0):
                hcs = np.load(load_f_path + "deepks_hbase.npy")
                if h_base is None:
                    h_base = np.empty((nframes,) + hcs.shape, dtype=hcs.dtype)
                assert hcs.shape == h_base.shape[1:], f"Shape of h_base {h_base.shape} does not match with {hcs.shape}!"
                h_base[f] = hcs 
                
                hcs = np.load(load_f_path + "deepks_htot.npy")
                if h_tot is None:
                    h_tot = np.empty((nframes,) + hcs.shape, dtype=hcs.dtype)
                assert hcs.shape == h_tot.shape[1:], f"Shape of h_tot {h_tot.shape} does not match with {hcs.shape}!"
                h_tot[f] = hcs 
                
                if deepks_v_delta == 1:
                    if os.path.exists(load_f_path + "deepks_vdpre.npy"):
                        v_delta_precalc_tmp = np.load(load_f_path + "deepks_vdpre.npy")
                        if v_delta_precalc is None:
                            v_delta_precalc = np.empty((nframes,) + v_delta_precalc_tmp.shape, dtype=v_delta_precalc_tmp.dtype)
                        assert v_delta_precalc_tmp.shape == v_delta_precalc.shape[1:], f"Shape of v_delta_precalc {v_delta_precalc.shape} does not match with {v_delta_precalc_tmp.shape}!"
                        v_delta_precalc[f] = v_delta_precalc_tmp
                elif deepks_v_delta == 2:
                    if os.path.exists(load_f_path + "deepks_phialpha.npy") and os.path.exists(load_f_path + "deepks_gevdm.npy"):
                        phialpha_tmp = np.load(load_f_path + "deepks_phialpha.npy")
                        # complex double to complex float
                        # phialpha_tmp = phialpha_tmp.astype(np.complex64)
                        if phialpha is None:
                            phialpha = np.empty((nframes,) + phialpha_tmp.shape, dtype=phialpha_tmp.dtype)
                        assert phialpha_tmp.shape == phialpha.shape[1:], f"Shape of phialpha {phialpha.shape} does not match with {phialpha_tmp.shape}!"
                        phialpha[f] = phialpha_tmp
                        
                        gevdm_tmp = np.load(load_f_path + "deepks_gevdm.npy")
                        if gevdm is None:
                            gevdm = np.empty((nframes,) + gevdm_tmp.shape, dtype=gevdm_tmp.dtype)
                        assert gevdm_tmp.shape == gevdm.shape[1:], f"Shape of gevdm {gevdm.shape} does not match with {gevdm_tmp.shape}!"
                        gevdm[f] = gevdm_tmp

        ## Save data
        # Convergence
        np.save(save_path + "conv.npy", conv)
        del conv

        # Energy and eigenvalues of density matrix
        np.save(save_path + "dm_eig.npy", dm_eig)
        del dm_eig

        np.save(save_path + "e_base.npy", e_base)    #Ry to Hartree
        e_ref = np.load(load_ref_path + "energy.npy")
        if e_ref.shape != (nframes, 1):
            if e_ref.shape == (nframes,):
                e_ref = e_ref.reshape((nframes, 1))
                print(f"energy.npy shape {e_ref.shape} is reshaped to {(nframes, 1)}")
            else:
                raise ValueError(f"energy.npy shape should be (nframes, 1), but got {e_ref.shape}.")
        np.save(save_path + "energy.npy", e_ref)
        np.save(save_path + "l_e_delta.npy", e_ref-e_base)
        np.save(save_path + "e_tot.npy", e_tot)
        del e_base, e_tot, e_ref
        
        np.save(save_path + "atom.npy", atom_data)
        del atom_data

        # Forces
        if(cal_force): 
            np.save(save_path + "f_base.npy", f_base)
            f_ref = np.load(load_ref_path + "force.npy")
            if f_ref.shape != (nframes, natoms, 3):
                raise ValueError(f"force.npy shape should be (nframes, natoms, 3), but got {f_ref.shape}.")
            np.save(save_path + "force.npy", f_ref)
            np.save(save_path + "l_f_delta.npy", f_ref-f_base)
            np.save(save_path + "f_tot.npy", f_tot)
            del f_base, f_tot, f_ref
            if gvx is not None:
                np.save(save_path + "grad_vx.npy", gvx)
                del gvx

        # Stress
        if(cal_stress): 
            np.save(save_path + "s_base.npy", s_base)
            s_ref = np.load(load_ref_path + "stress.npy")
            if s_ref.shape != (nframes, 9):
                raise ValueError(f"stress.npy shape should be (nframes, 9), but got {s_ref.shape}.")
            s_ref = s_ref[:,[0,1,2,4,5,8]] #only train the upper-triangle part
            np.save(save_path + "stress.npy", s_ref)
            np.save(save_path + "l_s_delta.npy", s_ref-s_base)
            np.save(save_path + "s_tot.npy", s_tot)
            del s_base, s_tot, s_ref
            if gvepsl is not None:
                np.save(save_path + "grad_epsilon.npy", gvepsl)
                del gvepsl

        # Bandgap (orbital)
        if(deepks_bandgap): 
            np.save(save_path + "o_base.npy", o_base)
            o_ref = np.load(load_ref_path + "orbital.npy")
            if o_ref.shape[0] != nframes or o_ref.shape[2] != 1:
                raise ValueError(f"orbital.npy shape should be (nframes, nkpt, 1), but got {o_ref.shape}.")
            np.save(save_path + "orbital.npy", o_ref)
            np.save(save_path + "l_o_delta.npy", o_ref-o_base)
            np.save(save_path + "o_tot.npy", o_tot)
            del o_base, o_tot, o_ref
            if orbital_precalc is not None:
                np.save(save_path + "orbital_precalc.npy", orbital_precalc)
                del orbital_precalc

        # V_delta (Hamiltonian)
        if(deepks_v_delta > 0): 
            np.save(save_path + "h_base.npy", h_base)
            h_ref = np.load(load_ref_path + "hamiltonian.npy")
            if h_ref.shape[0] != nframes or h_ref.ndim != 4:
                raise ValueError(f"hamiltonian.npy shape should be (nframes, nkpt, nlocal, nlocal), but got {h_ref.shape}.")
            np.save(save_path + "hamiltonian.npy", h_ref)
            np.save(save_path + "l_h_delta.npy", h_ref-h_base)
            np.save(save_path + "h_tot.npy", h_tot)
            del h_base, h_tot, h_ref
            if v_delta_precalc is not None:
                np.save(save_path + "v_delta_precalc.npy", v_delta_precalc)
                del v_delta_precalc
            elif phialpha is not None and gevdm is not None:
                np.save(save_path + "grad_evdm.npy", gevdm)
                del gevdm
                np.save(save_path + "phialpha.npy", phialpha)
                del phialpha
            if os.path.exists(load_ref_path + "overlap.npy"):
                overlap = np.load(load_ref_path + "overlap.npy")
                np.save(save_path + "overlap.npy", overlap)
                del overlap

    # check convergence and print in log
    from deepks.scf.stats import print_stats
    print_stats(systems=systems_train, test_sys=systems_test,
            dump_dir=train_dump, test_dump=test_dump, group=False, 
            with_conv=True, with_e=True, e_name="e_tot", 
               with_f=True, f_name="f_tot")
    return


def make_stat_scf_abacus(systems_train, systems_test=None, *, 
                  train_dump="data_train", test_dump="data_test", cal_force=0, cal_stress=0, deepks_bandgap=0, deepks_v_delta=0,
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
        deepks_v_delta=deepks_v_delta)
    # make task
    return PythonTask(
        gather_stats_abacus,
        call_kwargs=stat_args,
        outlog=outlog,
        errlog="err",
        workdir=workdir
    )



