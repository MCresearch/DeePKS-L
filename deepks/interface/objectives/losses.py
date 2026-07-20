"""Objective-layer loss helpers for supervised property tasks."""

import torch


def _pad_first_three_dims(tensor, target_r):
    current_r = tuple(int(v) for v in tensor.shape[1:4])
    if current_r == target_r:
        return tensor
    pad_width = [(0, 0)]
    for axis in range(3):
        n_add = target_r[axis] - current_r[axis]
        if n_add < 0:
            raise ValueError(
                f"Cannot shrink HR tensor from R-shape {current_r} to {target_r}; "
                "recollect data with a consistent target range"
            )
        pad_width.append((0, n_add))
    pad_width.extend((0, 0) for _ in range(tensor.ndim - 4))
    return torch.nn.functional.pad(
        tensor,
        tuple(v for pair in reversed(pad_width[1:]) for v in pair),
    )


def _align_hr_tensors(input, target):
    if input.ndim >= 6 and target.ndim >= 6 and input.ndim == target.ndim:
        if tuple(input.shape[-2:]) != tuple(target.shape[-2:]):
            raise ValueError(
                f"HR tensors must agree on local orbital dimensions, got {tuple(input.shape[-2:])} "
                f"and {tuple(target.shape[-2:])}"
            )
        input_r = tuple(int(v) for v in input.shape[1:4])
        target_r = tuple(int(v) for v in target.shape[1:4])
        common_r = tuple(max(a, b) for a, b in zip(input_r, target_r))
        return _pad_first_three_dims(input, common_r), _pad_first_three_dims(target, common_r)
    return input, target


def loss_hr(input, target):
    input, target = _align_hr_tensors(input, target)
    diff = input - target
    r_range = diff.shape[1]
    nframe = diff.shape[0]
    nlocal = diff.shape[-1]
    return torch.sum(torch.abs(diff) ** 2) / r_range / nlocal / nframe


def cal_vd_masked_loss_hs(H_pred, H_label, S_matrix, S_threshold=1e-6, H_threshold=1e-6):
    H_mask = torch.abs(H_label) > H_threshold
    S_mask = torch.abs(S_matrix) > S_threshold
    mask = H_mask * S_mask
    H_pred_masked = torch.masked_select(H_pred, mask)
    H_label_masked = torch.masked_select(H_label, mask)
    return torch.mean((H_pred_masked - H_label_masked) ** 2)


def cal_vd_masked_loss_width(H_pred, H_label, width=1):
    H_pred = H_pred.view(H_pred.size(0), H_pred.size(1), -1)
    H_label = H_label.view(H_label.size(0), H_label.size(1), -1)
    return torch.mean((H_pred[:, :, :width] - H_label[:, :, :width]) ** 2)


def cal_phi_loss(phi_pred, phi_label, phi_occ):
    if phi_occ is None:
        phi_occ = phi_label.size(-1)
    phi_pred = phi_pred[..., :phi_occ]
    phi_label = phi_label[..., :phi_occ]
    overlap = phi_pred.mH @ phi_label
    return (1.0 - torch.abs(overlap)).mean()
