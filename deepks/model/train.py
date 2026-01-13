import os
import sys
import numpy as np
import torch
import torch.optim as optim
from time import time
try:
    import deepks
except ImportError as e:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
from deepks.default import DEVICE
from deepks.model.model import CorrNet
from deepks.model.reader import GroupReader
from deepks.utils import load_dirs, load_elem_table
from deepks.model.utils import preprocess, fit_elem_const, make_loss
from deepks.model.evaluator import Evaluator, NatomLossList

def train(model, g_reader, n_epoch=1000, test_reader=None, *,
          energy_factor=1., force_factor=0., stress_factor=0., orbital_factor=0., v_delta_factor=0., phi_factor=0.,phi_occ=0, band_factor=0., band_occ=0, bandgap_factor=0., bandgap_occ=0, density_m_factor=0., density_m_occ=0, phi_align_factor=0., phi_align_occ=0, density_factor=0.,
          energy_loss=None, force_loss=None, stress_loss=None, orbital_loss=None, v_delta_loss=None, phi_loss=None, band_loss=None, bandgap_loss=None, density_m_loss=None, phi_align_loss=None, grad_penalty=0.,
          energy_per_atom=0, vd_divide_by_nlocal=False, vd_masked_loss=0, vd_masked_S_threshold=1e-6, vd_masked_H_threshold=1e-6, vd_masked_width=1, use_safe_eigh=False,
          start_lr=0.001, decay_steps=100, decay_rate=0.96, stop_lr=None, decay_rate_iter=None,
          weight_decay=0.,  fix_embedding=False,
          display_epoch=100, display_detail_test=0, display_natom_loss=False, ckpt_file="model.pth",
          graph_file=None, device=DEVICE):
    
    model = model.to(device)
    model.eval()
    print("# working on device:", device)
    if test_reader is None:
        test_reader = g_reader
    # fix parameters if needed
    if fix_embedding and model.embedder is not None:
        model.embedder.requires_grad_(False)
    # set up optimizer and lr scheduler
    if decay_rate_iter is not None:
        # decay_rate of start_lr for iterations, often start from iter.00
        current_dir=os.getcwd()
        current_iter=current_dir.split("/")[-2].split(".")[-1]
        if current_iter != "init": # no need to change
            current_iter=int(current_iter)
            start_lr=start_lr*(decay_rate_iter**current_iter)
            print(f"# resetting start_lr to {start_lr:.2e} because of decay_rate_iter")
    optimizer = optim.Adam(model.parameters(), lr=start_lr, weight_decay=weight_decay)
    if stop_lr is not None:
        decay_rate = (stop_lr / start_lr) ** (1 / (n_epoch // decay_steps))
        print(f"# resetting decay_rate: {decay_rate:.4f} "
              + f"to satisfy stop_lr: {stop_lr:.2e}")
    scheduler = optim.lr_scheduler.StepLR(optimizer, decay_steps, decay_rate)
    # make evaluators for training
    evaluator = Evaluator(energy_factor=energy_factor, force_factor=force_factor, 
                          stress_factor=stress_factor, orbital_factor=orbital_factor,
                          v_delta_factor=v_delta_factor,
                          phi_factor=phi_factor, phi_occ=phi_occ,
                          band_factor=band_factor, band_occ=band_occ,
                          bandgap_factor=bandgap_factor, bandgap_occ=bandgap_occ,
                          density_m_factor=density_m_factor, density_m_occ=density_m_occ,
                          phi_align_factor=phi_align_factor, phi_align_occ=phi_align_occ,
                          energy_lossfn=energy_loss, force_lossfn=force_loss,
                          stress_lossfn=stress_loss, orbital_lossfn=orbital_loss,
                          v_delta_lossfn=v_delta_loss,phi_lossfn=phi_loss,
                          band_lossfn=band_loss, bandgap_lossfn=bandgap_loss,
                          density_m_lossfn=density_m_loss,
                          phi_align_lossfn=phi_align_loss,
                          density_factor=density_factor, grad_penalty=grad_penalty, 
                          energy_per_atom=energy_per_atom, vd_divide_by_nlocal=vd_divide_by_nlocal,
                          vd_masked_loss=vd_masked_loss, vd_masked_S_threshold=vd_masked_S_threshold,
                          vd_masked_H_threshold=vd_masked_H_threshold, vd_masked_width=vd_masked_width,
                          use_safe_eigh=use_safe_eigh)
    if not display_detail_test:
        # make test evaluator that only returns l2loss of energy
        test_eval = Evaluator(energy_factor=1., energy_lossfn=make_loss(), # default l2 loss 
                            force_factor=0., density_factor=0., grad_penalty=0.,energy_per_atom=energy_per_atom)
    else:
        # make test evaluator that returns loss of every concerned items, but all with factor==1
        to_one = lambda x: 0. if x == 0. else 1.
        test_eval = Evaluator(energy_factor=to_one(energy_factor), force_factor=to_one(force_factor), 
                            stress_factor=to_one(stress_factor), orbital_factor=to_one(orbital_factor),
                            v_delta_factor=to_one(v_delta_factor),
                            phi_factor=to_one(phi_factor), phi_occ=phi_occ,
                            band_factor=to_one(band_factor), band_occ=band_occ,
                            bandgap_factor=to_one(bandgap_factor), bandgap_occ=bandgap_occ,
                            density_m_factor=to_one(density_m_factor), density_m_occ=density_m_occ,
                            phi_align_factor=to_one(phi_align_factor), phi_align_occ=phi_align_occ,
                            energy_lossfn=energy_loss, force_lossfn=force_loss,
                            stress_lossfn=stress_loss, orbital_lossfn=orbital_loss,
                            v_delta_lossfn=v_delta_loss,phi_lossfn=phi_loss,
                            band_lossfn=band_loss, bandgap_lossfn=bandgap_loss,
                            density_m_lossfn=density_m_loss, phi_align_lossfn=phi_align_loss,
                            density_factor=to_one(density_factor), grad_penalty=grad_penalty,
                            energy_per_atom=energy_per_atom, vd_divide_by_nlocal=vd_divide_by_nlocal,
                            vd_masked_loss=vd_masked_loss, vd_masked_S_threshold=vd_masked_S_threshold,
                            vd_masked_H_threshold=vd_masked_H_threshold,vd_masked_width=vd_masked_width,
                            use_safe_eigh=use_safe_eigh)

    print("# epoch      trn_err   tst_err        lr  trn_time  tst_time",end='')
    data_keys = g_reader.readers[0].sample_all().keys()
    # L_inv_in=1 if "L_inv" in data_keys else 0
    # print("if L_inv in sample:",L_inv_in)
    align_len=20
    evaluator.print_head("trn_loss",data_keys,align_len)
    if display_detail_test:
        test_eval.print_head("tst_loss",data_keys,align_len)
    # print("")

    tic = time()
    trn_natom_loss_list=NatomLossList()
    tst_natom_loss_list=NatomLossList()
    for batch in g_reader.sample_all_batch():
        loss=evaluator(model,batch)
        natom=batch["eig"].shape[1]
        trn_natom_loss_list.add_loss(natom,loss)
    trn_loss=trn_natom_loss_list.avg_loss()
    for batch in test_reader.sample_all_batch():
        loss=test_eval(model,batch)
        natom=batch["eig"].shape[1]
        tst_natom_loss_list.add_loss(natom,loss)
    tst_loss=tst_natom_loss_list.avg_loss()    
    # trn_loss = np.mean([[loss_term.item() for loss_term in evaluator(model, batch)]
    #                 for batch in g_reader.sample_all_batch()],axis=0)
    # tst_loss = np.mean([[loss_term.item() for loss_term in test_eval(model, batch)]
    #                 for batch in test_reader.sample_all_batch()],axis=0)
    tst_time = time() - tic
    if display_natom_loss:
        for natom in trn_natom_loss_list.natoms():
            evaluator.print_head(str(natom)+"_trn",data_keys,align_len)        
        for natom in tst_natom_loss_list.natoms():
            if display_detail_test:
                test_eval.print_head(str(natom)+"_tst",data_keys,align_len)
            else:
                test_eval.print_head(str(natom)+"_tst",[],align_len)#just energy
    print("")

    print(f"  {0:<8d}  {np.sqrt(np.abs(trn_loss[-1])):>.2e}  {np.sqrt(np.abs(tst_loss[-1])):>.2e}"
          f"  {start_lr:>.2e}  {0:>8.2f}  {tst_time:>8.2f}",end='')
    for loss_term in trn_loss[:-1]:
        print(f"{loss_term:>{align_len}.4e}",end='')
    if display_detail_test:
        for loss_term in tst_loss[:-1]:
            print(f"{loss_term:>{align_len}.4e}",end='')
    if display_natom_loss:
        trn_natom_loss_list.print_avg_atom_loss(align_len)     
        tst_natom_loss_list.print_avg_atom_loss(align_len)     
    print('')

    for epoch in range(1, n_epoch+1):
        tic = time()
        # loss_list = []
        trn_natom_loss_list.clear_loss()
        tst_natom_loss_list.clear_loss()
        for sample in g_reader:
            model.train()
            optimizer.zero_grad()
            loss = evaluator(model, sample)
            loss[-1].backward()
            optimizer.step()
            # loss_list.append([loss_term.item() for loss_term in loss])
            natom=sample["eig"].shape[1]
            trn_natom_loss_list.add_loss(natom,loss)
        scheduler.step()

        if epoch % display_epoch == 0:
            model.eval()
            # trn_loss = np.mean(loss_list,axis=0)
            trn_loss=trn_natom_loss_list.avg_loss()
            trn_time = time() - tic
            tic = time()
            # tst_loss = np.mean([[loss_term.item() for loss_term in test_eval(model, batch)]
            #                 for batch in test_reader.sample_all_batch()],axis=0)
            for batch in test_reader.sample_all_batch():
                loss=test_eval(model,batch)
                natom=batch["eig"].shape[1]
                tst_natom_loss_list.add_loss(natom,loss)
            tst_loss=tst_natom_loss_list.avg_loss()  
            tst_time = time() - tic
            print(f"  {epoch:<8d}  {np.sqrt(np.abs(trn_loss[-1])):>.2e}  {np.sqrt(np.abs(tst_loss[-1])):>.2e}"
                  f"  {scheduler.get_last_lr()[0]:>.2e}  {trn_time:>8.2f}  {tst_time:8.2f}",end='')
            for loss_term in trn_loss[:-1]:
                print(f"{loss_term:>{align_len}.4e}",end='')
            if display_detail_test and epoch%(display_detail_test*display_epoch) == 0:
                for loss_term in tst_loss[:-1]:
                    print(f"{loss_term:>{align_len}.4e}",end='')
            if display_natom_loss:
                trn_natom_loss_list.print_avg_atom_loss(align_len)
                tst_natom_loss_list.print_avg_atom_loss(align_len)                 
            print('')
            if ckpt_file:
                model.save(ckpt_file)

    if ckpt_file:
        model.save(ckpt_file)
    if graph_file:
        model.compile_save(graph_file)
    

def main(train_paths, test_paths=None,
         restart=None, ckpt_file=None, 
         model_args=None, data_args=None, 
         preprocess_args=None, train_args=None, 
         proj_basis=None, fit_elem=False, 
         seed=None, device=None):
   
    if seed is None: 
        seed = np.random.randint(0, 2**32)
    print(f'# using seed: {seed}')
    np.random.seed(seed)
    torch.manual_seed(seed)

    if model_args is None: model_args = {}
    if data_args is None: data_args = {}
    if preprocess_args is None: preprocess_args = {}
    if train_args is None: train_args = {}
    if proj_basis is not None:
        model_args["proj_basis"] = proj_basis
    if ckpt_file is not None:
        train_args["ckpt_file"] = ckpt_file
    if device is not None:
        train_args["device"] = device

    train_paths = load_dirs(train_paths)
    # print(f'# training with {len(train_paths)} system(s)')
    g_reader = GroupReader(train_paths, **data_args)
    if test_paths is not None:
        test_paths = load_dirs(test_paths)
        # print(f'# testing with {len(test_paths)} system(s)')
        test_reader = GroupReader(test_paths, **data_args)
    else:
        print('# testing with training set')
        test_reader = None

    if restart is not None:
        model = CorrNet.load(restart)
        if model.elem_table is not None:
            fit_elem_const(g_reader, test_reader, model.elem_table)
    else:
        input_dim = g_reader.ndesc
        if model_args.get("input_dim", input_dim) != input_dim:
            print(f"# `input_dim` in `model_args` does not match data",
                  f"({input_dim}).", "Use the one in data.", file=sys.stderr)
        model_args["input_dim"] = input_dim
        if fit_elem:
            elem_table = model_args.get("elem_table", None)
            if isinstance(elem_table, str):
                elem_table = load_elem_table(elem_table)
            elem_table = fit_elem_const(g_reader, test_reader, elem_table)
            model_args["elem_table"] = elem_table
        model = CorrNet(**model_args).double()
        
    preprocess(model, g_reader, **preprocess_args)
    # start=time()
    train(model, g_reader, test_reader=test_reader, **train_args)
    # end=time()
    # print("all train time:",end-start)


if __name__ == "__main__":
    from deepks.main import train_cli as cli
    cli()
