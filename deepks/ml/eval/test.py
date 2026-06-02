"""Evaluation helpers for descriptor-energy style inference."""

import os

import numpy as np
import torch
import torch.nn as nn

from deepks.io.model_artifacts import load_elem_table_sidecar
from deepks.io.readers import GroupReader
from deepks.io.utils import check_list, load_dirs
from deepks.ml.model_io import load_runtime_model


def _reduce_per_atom_energy(model, raw_output):
    """Apply the descriptor-energy reducer convention to a model output.

    R1: models now return unreduced per-atom contributions; this helper
    keeps the legacy descriptor-energy ``test()`` callable working by
    applying ``sum-over-atoms + output_bias`` here (rather than inside
    ``model.forward`` where it used to live). ``raw_output`` may already
    be a per-system tensor for legacy traced models, in which case the
    reduction is a no-op.
    """
    if torch.is_tensor(raw_output) and raw_output.dim() >= 3:
        reduced = raw_output.sum(-2)
    else:
        reduced = raw_output
    bias = getattr(model, "output_bias", None)
    if bias is not None and torch.is_tensor(reduced):
        reduced = reduced + bias
    return reduced


def _batch_to_device(batch, device: str):
    """Move a structured batch onto ``device`` without depending on the interface layer.

    Prefers ``batch.to_device`` when available (interface-provided batches expose it);
    otherwise constructs a new batch using the same concrete class as ``batch`` so this
    function stays ML-side agnostic to the interface module.
    """
    if hasattr(batch, "to_device"):
        return batch.to_device(device, complex_cpu_context_keys=("phialpha",))

    def _move(value, *, keep_complex_cpu=False):
        if isinstance(value, list):
            return [_move(v, keep_complex_cpu=keep_complex_cpu) for v in value]
        if not torch.is_tensor(value):
            return value
        if torch.is_complex(value):
            if keep_complex_cpu:
                return value.to("cpu", dtype=torch.complex128, non_blocking=True)
            return value.to(device, dtype=torch.complex128, non_blocking=True)
        return value.to(device, non_blocking=True)

    moved_inputs = {k: _move(v) for k, v in batch.model_inputs.items()}
    moved_targets = {k: _move(v) for k, v in batch.targets.items()}
    moved_context = {
        k: _move(v, keep_complex_cpu=(k == "phialpha"))
        for k, v in batch.context.items()
    }
    return type(batch)(
        model_inputs=moved_inputs,
        targets=moved_targets,
        context=moved_context,
        meta=getattr(batch, "meta", {}),
    )


def test(model, g_reader, dump_prefix="test", group=False, device="cpu"):
    if hasattr(model, "eval"):
        model.eval()
    model = model.to(device)
    loss_fn = nn.MSELoss()
    label_list = []
    pred_list = []

    for i in range(g_reader.nsystems):
        if hasattr(g_reader, "sample_all_task_batch"):
            sample = _batch_to_device(g_reader.sample_all_task_batch(i), device)
            label = sample.targets["energy"]
            data = sample.model_inputs["descriptor"]
        else:
            sample = g_reader.sample_all(i)
            for key, value in sample.items():
                if isinstance(value, list):
                    sample[key] = [vv.to(device, non_blocking=True) for vv in value]
                elif not torch.is_complex(value):
                    sample[key] = value.to(device, non_blocking=True)
                else:
                    if key == "phialpha":
                        sample[key] = value.to("cpu", dtype=torch.complex128, non_blocking=True)
                    else:
                        sample[key] = value.to(device, dtype=torch.complex128, non_blocking=True)
            label, data = sample["lb_e"], sample["eig"]
        nframes = label.shape[0]
        pred = _reduce_per_atom_energy(model, model(data))
        error = torch.sqrt(loss_fn(pred, label))

        error_np = error.item()
        label_np = label.cpu().numpy().reshape(nframes, -1).sum(axis=1)
        pred_np = pred.detach().cpu().numpy().reshape(nframes, -1).sum(axis=1)
        error_l1 = np.mean(np.abs(label_np - pred_np))
        label_list.append(label_np)
        pred_list.append(pred_np)

        if not group and dump_prefix is not None:
            nd = max(len(str(g_reader.nsystems)), 2)
            dump_res = np.stack([label_np, pred_np], axis=1)
            header = f"{g_reader.path_list[i]}\nmean l1 error: {error_l1}\nmean l2 error: {error_np}\nreal_ene  pred_ene"
            filename = f"{dump_prefix}.{i:0{nd}}.out"
            np.savetxt(filename, dump_res, header=header)

    all_label = np.concatenate(label_list, axis=0)
    all_pred = np.concatenate(pred_list, axis=0)
    all_err_l1 = np.mean(np.abs(all_label - all_pred))
    all_err_l2 = np.sqrt(np.mean((all_label - all_pred) ** 2))
    info = f"all systems mean l1 error: {all_err_l1}\nall systems mean l2 error: {all_err_l2}"
    print(info)
    if dump_prefix is not None and group:
        np.savetxt(
            f"{dump_prefix}.out",
            np.stack([all_label, all_pred], axis=1),
            header=info + "\nreal_ene  pred_ene",
        )
    return all_err_l1, all_err_l2


def main(
    data_paths,
    model=None,
    model_file="model.pth",
    output_prefix="test",
    group=False,
    e_name="l_e_delta",
    d_name=("dm_eig",),
    device="cpu",
    elem_table=None,
):
    data_paths = load_dirs(data_paths)
    if isinstance(d_name, (list, tuple)) and len(d_name) == 1:
        d_name = d_name[0]
    g_reader = GroupReader(
        data_paths,
        e_name=e_name,
        d_name=d_name,
        conv_filter=False,
        extra_label=True,
    )
    if model is not None:
        models = [("runtime-model", model, elem_table)]
    else:
        models = [
            (f, load_runtime_model(f), load_elem_table_sidecar(f))
            for f in check_list(model_file)
        ]
    for name, loaded_model, loaded_elem_table in models:
        print(name)
        p = os.path.dirname(name) if name != "runtime-model" else "."
        loaded_model = loaded_model.to(device)
        dump = os.path.join(p, output_prefix)
        dir_name = os.path.dirname(dump)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        if loaded_elem_table is not None:
            elist, econst = loaded_elem_table
            g_reader.collect_elems(elist)
            g_reader.subtract_elem_const(econst)
        test(loaded_model, g_reader, dump_prefix=dump, group=group, device=device)
        g_reader.revert_elem_const()
