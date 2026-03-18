import os
import numpy as np


NPY_DICT = {
    # Set to 1: Ry to Hartree
    "load":{
        ## (a,b): 
        ## a means the factor to be devided
        ## - 1: no change
        ## - 2: usually means Ry to Hartree
        ## b means whether to load the data
        ## - None: always load
        ## - 0: load if exists
        ## - others: load if _condition=others
        "energy": {
            "_condition": 'True',
            "dm_eig":(1,None),
            "e_base": (2,None),
            "e_tot": (2,None),
        },
        "force": {
            "_condition": 'cal_force',
            "f_base": (2,None),
            "f_tot": (2,None),
            "grad_vx": (1,0),
        },
        "stress": {
            "_condition": 'cal_stress',
            "s_base": (2,None),
            "s_tot": (2,None),
            "grad_vepsl": (1,0),
        },
        "orbital": {
            "_condition": 'deepks_bandgap',
            "o_base": (2,None),
            "o_tot": (2,None),
            "orbital_precalc": (1,0),
        },
        "hamiltonian": {
            "_condition": 'deepks_v_delta',
            "h_base": (2,None),
            "h_tot": (2,None),
            "v_delta_precalc": (1,1),
            "phialpha": (1,2),
            "grad_evdm": (1,2),
        },
    },
    "save":{
        ## files to be saved
        ## The key itself is always saved if exists, and warning will be raised if not exists
        ## The item in value list is saved if the corresponding list is not empty
        "energy": ["dm_eig", "e_base", "e_tot", "l_e_delta"],
        "force": ["f_base", "f_tot", "l_f_delta", "grad_vx"],
        "stress": ["s_base", "s_tot", "l_s_delta", "grad_vepsl"],
        "orbital": ["o_base", "o_tot", "l_o_delta", "orbital_precalc"],
        "hamiltonian": ["h_base", "h_tot", "l_h_delta", "v_delta_precalc", "phialpha", "grad_evdm"],
        "overlap": [],
    }
}

def gather_system_data(nframes,data_dir,ref_dir,out_dir,npy_dict=NPY_DICT,**stat_args):
    lists = {k: [] for key in npy_dict["load"].keys() for k in npy_dict["load"][key].keys() if k != "_condition"}
    for f in range(nframes):
        with open(f"{data_dir}/{f}/conv", "r") as conv_file:
            ic = conv_file.read().split()
            ic = [item.strip('#') for item in ic]
            if "CONVERGED" in ic and "NOT" not in ic:
                c_list[(int)(ic[0])]=True

        for key, dicts in npy_dict["load"].items():
            if eval(dicts['_condition']):
                for k, v in dicts.items():
                    if k == "_condition":
                        continue
                    if (v[1] is None 
                        or (v[1] == 0 and os.path.exists(f"{data_dir}/{f}/{k}.npy"))
                        or (eval(dicts['_condition'] + f'== {v[1]}') and os.path.exists(f"{data_dir}/{f}/{k}.npy"))):
                        tmp = np.load(f"{data_dir}/{f}/{k}.npy")
                        tmp = tmp / v[0]
                        lists[k].append(tmp)

    for key, files in npy_dict["save"].items():
        if os.path.exists(f"{ref_dir}/{key}.npy"):
            label = np.load(f"{ref_dir}/{key}.npy")
            np.save(f"{out_dir}/{key}.npy", label)
        for f in files:
            if f in lists:
                if lists[f]:
                    tmp = np.stack(lists[f], axis=0)
                    np.save(f"{out_dir}/{f}.npy", tmp)
            elif f[:2] == "l_" and f[-6:] == "_delta":
                basename = f[2:-6] + "_base"
                base = np.stack(lists[basename], axis=0)
                np.save(f"{out_dir}/{f}.npy", label - base)

def format_check(data, size):
    if data.shape != size:
        raise ValueError(f"Data shape {data.shape} is not equal to {size}")
    return data