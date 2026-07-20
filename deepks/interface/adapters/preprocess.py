"""Recipe-side preprocessing helpers owned by the interface layer."""

import numpy as np


def fit_elem_const(g_reader, test_reader=None, elem_table=None, ridge_alpha=0.0):
    """Fit / apply element constants on the current reader objects."""

    if elem_table is None:
        elem_table = g_reader.compute_elem_const(ridge_alpha)
    elem_list, elem_const = elem_table
    g_reader.collect_elems(elem_list)
    g_reader.subtract_elem_const(elem_const)
    if test_reader is not None:
        test_reader.collect_elems(elem_list)
        test_reader.subtract_elem_const(elem_const)
    return elem_table


def preprocess(
    model,
    g_reader,
    preshift=True,
    prescale=False,
    prescale_sqrt=False,
    prescale_clip=0,
    prefit=True,
    prefit_ridge=10,
    prefit_trainable=False,
):
    """Prepare current descriptor data for the selected recipe/model pair."""

    shift = model.input_shift.cpu().detach().numpy()
    scale = model.input_scale.cpu().detach().numpy()
    input_partition = getattr(model, "input_partition", None)
    prefit_trainable = prefit_trainable and input_partition is None
    if preshift or prescale:
        davg, dstd = g_reader.compute_data_stat(input_partition)
        if preshift:
            shift = davg
        if prescale:
            scale = dstd
            if prescale_sqrt:
                scale = np.sqrt(scale)
            if prescale_clip:
                scale = scale.clip(prescale_clip)
        model.set_normalization(shift, scale)
    if prefit:
        weight, bias = g_reader.compute_prefitting(
            shift=shift,
            scale=scale,
            ridge_alpha=prefit_ridge,
            symm_sections=input_partition,
        )
        model.set_prefitting(weight, bias, trainable=prefit_trainable)


__all__ = ["fit_elem_const", "preprocess"]
