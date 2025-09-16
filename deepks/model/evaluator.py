import os
import sys
import numpy as np
import torch
from time import time
try:
    import deepks
except ImportError as e:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
from deepks.model.reader import generalized_eigh
from deepks.model.utils import get_density_matrix, cal_phi_loss, cal_v_delta, get_occ_func, make_loss

class Evaluator:
    def __init__(self,
                 energy_factor=1., force_factor=0., 
                 stress_factor=0., orbital_factor=0.,
                 v_delta_factor=0., 
                 phi_factor=0., phi_occ=0,
                 band_factor=0.,band_occ=0,
                 density_m_factor=0.,density_m_occ=0,
                 density_factor=0., grad_penalty=0., 
                 energy_lossfn=None, force_lossfn=None, 
                 stress_lossfn=None, orbital_lossfn=None,
                 v_delta_lossfn=None, phi_lossfn=None,
                 band_lossfn=None, density_m_lossfn=None,
                 energy_per_atom=0,vd_divide_by_nlocal=False):
        # energy term
        if energy_lossfn is None:
            energy_lossfn = {}
        if isinstance(energy_lossfn, dict):
            energy_lossfn = make_loss(**energy_lossfn)
        self.e_factor = energy_factor
        self.e_lossfn = energy_lossfn
        # force term
        if force_lossfn is None:
            force_lossfn = {}
        if isinstance(force_lossfn, dict):
            force_lossfn = make_loss(**force_lossfn)
        self.f_factor = force_factor
        self.f_lossfn = force_lossfn
        # stress term
        if stress_lossfn is None:
            stress_lossfn = {}
        if isinstance(stress_lossfn, dict):
            stress_lossfn = make_loss(**stress_lossfn)
        self.s_factor = stress_factor
        self.s_lossfn = stress_lossfn
         # orbital(bandgap) term
        if orbital_lossfn is None:
            orbital_lossfn = {}
        if isinstance(orbital_lossfn, dict):
            orbital_lossfn = make_loss(**orbital_lossfn)
        self.o_factor = orbital_factor
        self.o_lossfn = orbital_lossfn
        # v_delta term
        if v_delta_lossfn is None:
            v_delta_lossfn = {}  
        if isinstance(v_delta_lossfn, dict):
            v_delta_lossfn = make_loss(**v_delta_lossfn)
        self.vd_factor = v_delta_factor
        self.vd_lossfn = v_delta_lossfn
        self.vd_divide_by_nlocal = vd_divide_by_nlocal
        # phi term
        if phi_lossfn is None:
            phi_lossfn = {}
        if isinstance(phi_lossfn, dict):
            phi_lossfn = make_loss(**phi_lossfn)
        self.phi_factor = phi_factor
        self.phi_lossfn = phi_lossfn   
        self.get_phi_occ = get_occ_func(phi_occ)
        # band energy term
        if band_lossfn is None:
            band_lossfn = {}
        if isinstance(band_lossfn, dict):
            band_lossfn = make_loss(**band_lossfn)
        self.band_factor = band_factor
        self.band_lossfn = band_lossfn   
        self.get_band_occ = get_occ_func(band_occ)   
        #density matrix term
        if density_m_lossfn is None:
            density_m_lossfn = {}
        if isinstance(density_m_lossfn, dict):
            density_m_lossfn = make_loss(**density_m_lossfn)
        self.density_m_factor = density_m_factor
        self.density_m_lossfn = density_m_lossfn   
        self.get_density_m_occ = get_occ_func(density_m_occ)                  
        # coulomb term of dm; requires head gradient
        self.d_factor = density_factor
        # gradient penalty, not very useful
        self.g_penalty = grad_penalty
        # energy loss divide by 1/natom/natom^2
        self.energy_per_atom=energy_per_atom

    def __call__(self, model, sample):
        _dref = next(model.parameters()).device
        #print("_dref:")
        #print(_dref)
        tot_loss = 0.
        loss=[]
        # keep only phialpha in cpu, move all other data to _dref, set complex dtype to complex128
        for k, v in sample.items():
            if isinstance(v, list):
                sample[k] = [vv.to(_dref, non_blocking=True) for vv in v]
            elif not torch.is_complex(v):
                sample[k] = v.to(_dref, non_blocking=True)
            else:
                if k == "phialpha":
                    sample[k] = v.to("cpu", dtype=torch.complex128, non_blocking=True)
                else:
                    sample[k] = v.to(_dref, dtype=torch.complex128, non_blocking=True)
        e_label, eig = sample["lb_e"], sample["eig"]
        nframe = e_label.shape[0]
        requires_grad =  ( (self.f_factor > 0 and "lb_f" in sample) 
                        or (self.s_factor > 0 and "lb_s" in sample) 
                        or (self.o_factor > 0 and "lb_o" in sample)
                        or (self.vd_factor > 0 and "lb_vd" in sample)
                        or (self.phi_factor > 0 and "lb_phi" in sample)
                        or (self.band_factor > 0 and "lb_band" in sample)
                        or (self.density_m_factor > 0)
                        or (self.d_factor > 0 and "gldv" in sample)
                        or self.g_penalty > 0)
        eig.requires_grad_(requires_grad)
        # begin the calculation
        e_pred = model(eig)
        # may divide e_loss by 1 or natom or natom**2: this way energy loss will not increase when number of atom increase
        natom = eig.shape[1]
        tot_loss = tot_loss + self.e_factor * self.e_lossfn(e_pred, e_label) / (natom**self.energy_per_atom)
        loss.append(self.e_factor * self.e_lossfn(e_pred, e_label) / (natom**self.energy_per_atom))
        if requires_grad:
            [gev] = torch.autograd.grad(e_pred, eig, 
                        grad_outputs=torch.ones_like(e_pred),
                        retain_graph=True, create_graph=True, only_inputs=True)
            # for now always use pure l2 loss for gradient penalty
            if self.g_penalty > 0 and "eg0" in sample:
                eg_base, gveg = sample["eg0"], sample["gveg"]
                eg_tot = torch.einsum('...apg,...ap->...g', gveg, gev) + eg_base
                tot_loss = tot_loss + self.g_penalty * eg_tot.pow(2).mean(0).sum()
                loss.append(self.g_penalty * eg_tot.pow(2).mean(0).sum())
            # optional force calculation
            if self.f_factor > 0 and "lb_f" in sample:
                f_label, gvx = sample["lb_f"], sample["gvx"]
                f_pred = - torch.einsum("...bxap,...ap->...bx", gvx, gev)
                tot_loss = tot_loss + self.f_factor * self.f_lossfn(f_pred, f_label)
                loss.append(self.f_factor * self.f_lossfn(f_pred, f_label))
            # optional stress calculation
            if self.s_factor > 0 and "lb_s" in sample:
                s_label, gvepsl = sample["lb_s"], sample["gvepsl"]
                s_pred = torch.einsum("...iap,...ap->...i", gvepsl, gev)
                tot_loss = tot_loss + self.s_factor * self.s_lossfn(s_pred, s_label)
                loss.append(self.s_factor * self.s_lossfn(s_pred, s_label))
            # optional orbital(bandgap) calculation
            if self.o_factor > 0 and "lb_o" in sample:
                o_label, op = sample["lb_o"], sample["op"]
                op = op.contiguous().view(op.shape[0], o_label.shape[1], o_label.shape[2], op.shape[-2], op.shape[-1])
                o_pred = torch.einsum("...kiap,...ap->...ki", op, gev)
                # print(o_label.shape, op.shape, o_pred.shape, gev.shape)
                tot_loss = tot_loss + self.o_factor * self.o_lossfn(o_pred, o_label)
                loss.append(self.o_factor * self.o_lossfn(o_pred, o_label))
            # optional v_delta/phi/band_energy/density_matrix calculation
            if (self.vd_factor > 0 and "lb_vd" in sample) or (self.phi_factor > 0 and "lb_phi" in sample) \
                or (self.band_factor > 0 and "lb_band" in sample) or (self.density_m_factor > 0 and "lb_phi" in sample):
                # cal v_delta
                if "vdp" in sample:
                    vdp = sample["vdp"] # can be complex
                    vd_pred = torch.einsum("...kxyap,...ap->...kxy", vdp, gev)
                elif "phialpha" in sample and "gevdm" in sample:                  
                    # start=time()
                    vd_pred = cal_v_delta(gev,sample["gevdm"],sample["phialpha"])
                    # end=time()
                    # print("cal vdp time in batch:",end-start)
                nlocal = vd_pred.shape[-1]

                # optional v_delta calculation
                if self.vd_factor > 0 and "lb_vd" in sample:
                    vd_label = sample["lb_vd"]
                    vd_loss = self.vd_factor * self.vd_lossfn(vd_pred, vd_label)
                    # original: mean method,divide by nlocal**2. vd_divide_by_nlocal:divide by nlocal
                    if self.vd_divide_by_nlocal:
                        vd_loss = vd_loss * nlocal
                    tot_loss = tot_loss + vd_loss
                    loss.append(vd_loss)
                
                if (self.phi_factor > 0 and "lb_phi" in sample) or (self.band_factor > 0 and "lb_band" in sample) or (self.density_m_factor > 0 and "lb_phi" in sample):
                    h_base = sample["h_base"]
                    if "L_inv" in sample:
                        L_inv=sample["L_inv"]
                        band_pred,phi_pred=generalized_eigh(h_base+vd_pred,L_inv)
                    else:
                        band_pred,phi_pred= torch.linalg.eigh(h_base+vd_pred,UPLO='U')
                    # optional phi calculation
                    if self.phi_factor > 0 and "lb_phi" in sample:
                        phi_label = sample["lb_phi"]
                        phi_loss = self.phi_factor * cal_phi_loss(phi_pred,phi_label,self.get_phi_occ(natom))
                        tot_loss = tot_loss + phi_loss
                        loss.append(phi_loss)
                    # optional band energy calculation
                    if self.band_factor > 0 and "lb_band" in sample:
                        band_label = sample["lb_band"]
                        band_occ=self.get_band_occ(natom)
                        band_loss = self.band_factor * self.band_lossfn(band_pred[...,:band_occ], band_label[...,:band_occ])
                        tot_loss = tot_loss + band_loss
                        # print("occ_band",band_pred[...,:band_occ],band_label[...,:band_occ])
                        loss.append(band_loss)
                    # optional density matrix calculation
                    if self.density_m_factor > 0 and "lb_phi" in sample:
                        # calculate density_m_label every time, kind of waste of time
                        phi_label = sample["lb_phi"]
                        density_m_occ=self.get_density_m_occ(natom)
                        density_m_label = get_density_matrix(phi_label,density_m_occ)
                        density_m_pred = get_density_matrix(phi_pred,density_m_occ)
                        #need to multiply nlocal, reason is the same as v_delta
                        density_m_loss = self.density_m_factor * self.density_m_lossfn(density_m_pred, density_m_label) * nlocal
                        tot_loss = tot_loss + density_m_loss
                        loss.append(density_m_loss)
            # density loss with fix head grad
            if self.d_factor > 0 and "gldv" in sample:
                gldv = sample["gldv"]
                d_loss = self.d_factor * torch.abs((gldv * gev).mean(0).sum())
                tot_loss = tot_loss + d_loss
                loss.append(d_loss)
        loss.append(tot_loss)
        return loss
    
    def print_head(self,name,data_keys,align_len=20):
        info=f"{name}_energy".rjust(align_len)
        if self.g_penalty > 0 and "eg0" in data_keys:
            info+=f"{name}_grad".rjust(align_len)
        # optional force calculation
        if self.f_factor > 0 and "lb_f" in data_keys:
            info+=f"{name}_force".rjust(align_len)
        # optional stress calculation
        if self.s_factor > 0 and "lb_s" in data_keys:
            info+=f"{name}_stress".rjust(align_len)
        # optional orbital(bandgap) calculation
        if self.o_factor > 0 and "lb_o" in data_keys:
            info+=f"{name}_bandgap".rjust(align_len)
        # optional v_delta calculation
        if self.vd_factor > 0 and "lb_vd" in data_keys:
            info+=f"{name}_v_delta".rjust(align_len)
        # optional phi calculation
        if self.phi_factor > 0 and "lb_phi" in data_keys:
            info+=f"{name}_phi".rjust(align_len)
        # optional band energy calculation
        if self.band_factor > 0 and "lb_band" in data_keys:
            info+=f"{name}_band".rjust(align_len)
        # optional density matrix calculation
        if self.density_m_factor > 0 and "lb_phi" in data_keys:
            info+=f"{name}_dm".rjust(align_len)             
        # density loss with fix head grad
        if self.d_factor > 0 and "gldv" in data_keys:
            info+=f"{name}_density".rjust(align_len)
        print(info,end='')

class NatomLossList:
    def __init__(self):
        self.natom_loss_list=dict()
        self.n_loss_term=0
    
    def clear_loss(self):
        if not self.n_loss_term:
            self.n_loss_term=len(self.natom_loss_list[list(self.natom_loss_list.keys())[0]][0])
        #don't clear natom, just sample_all_batch in the beginning gives all data 
        for natom in self.natom_loss_list.keys():
            self.natom_loss_list[natom]=[[0. for _ in range(self.n_loss_term)]]
    
    def add_loss(self,natom,loss):
        assert len(loss) > 0, "loss should not be empty"
        if not self.n_loss_term:
            self.n_loss_term=len(loss)
        assert len(loss) == self.n_loss_term, \
            f"loss length are different for newly added natom {natom}, expected {self.n_loss_term}, got {len(loss)}"
        if natom not in self.natom_loss_list.keys():
            self.natom_loss_list[natom]=[]
        self.natom_loss_list[natom].append([loss_term.item() for loss_term in loss])
    
    def natoms(self):
        return sorted(self.natom_loss_list.keys())
    
    def avg_atom_loss(self):
        # avg upon data
        return {natom:np.mean(losses,axis=0) for (natom,losses) in self.natom_loss_list.items()}
    
    def print_avg_atom_loss(self,align_len=20):
        avg_atom_loss = sorted(self.avg_atom_loss().items(), key=lambda x: x[0])
        for (atom,aal) in avg_atom_loss:
            for avg_atom_loss_term in aal[:-1]:
                print(f"{avg_atom_loss_term:>{align_len}.4e}",end='')

    def avg_loss(self):
        # avg upon data and natom
        return np.mean([loss for losses in self.natom_loss_list.values() for loss in losses ],axis=0)
