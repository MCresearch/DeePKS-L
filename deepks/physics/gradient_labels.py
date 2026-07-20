"""Descriptor-gradient label helpers.

The DeePKS model predicts an energy correction, and the training target for
``dE_delta / d descriptor`` is obtained from current-run chain-rule operators
and target-current observable differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np


@dataclass
class NormalContribution:
    metric: np.ndarray
    projection: np.ndarray


def _flatten_operator(operator: np.ndarray, nrow_dims: int) -> np.ndarray:
    operator = np.asarray(operator, dtype=np.float64)
    nrows = int(np.prod(operator.shape[:nrow_dims]))
    return operator.reshape(nrows, -1)


def _stress_upper(stress: np.ndarray) -> np.ndarray:
    stress = np.asarray(stress, dtype=np.float64)
    if stress.shape == (6,):
        return stress
    if stress.shape == (3, 3):
        return np.array(
            [stress[0, 0], stress[0, 1], stress[0, 2], stress[1, 1], stress[1, 2], stress[2, 2]],
            dtype=np.float64,
        )
    if stress.shape == (9,):
        return _stress_upper(stress.reshape(3, 3))
    raise ValueError(f"stress delta should have shape (6,), (9,), or (3,3), got {stress.shape}")


def solve_normal_equation(
    contributions: Mapping[str, NormalContribution],
    *,
    weights: Optional[Mapping[str, float]] = None,
    ridge: float = 1e-8,
    fallback: str = "lstsq",
) -> np.ndarray:
    """Solve ``(sum w A^T A + ridge I) g = sum w A^T x``.

    Contribution projection arrays may be either flat ``(n,)`` or
    descriptor-shaped ``(...,)``; the returned shape follows the first
    projection contribution.
    """

    if not contributions:
        raise ValueError("at least one gradient-label contribution is required")
    weights = {} if weights is None else dict(weights)

    metric_total = None
    projection_total = None
    output_shape = None
    for name, contribution in contributions.items():
        weight = float(weights.get(name, 1.0))
        if weight == 0.0:
            continue
        metric = np.asarray(contribution.metric, dtype=np.float64)
        projection = np.asarray(contribution.projection, dtype=np.float64)
        if output_shape is None:
            output_shape = projection.shape
        projection_flat = projection.reshape(-1)
        if metric.shape != (projection_flat.size, projection_flat.size):
            raise ValueError(
                f"{name} metric shape {metric.shape} is incompatible with projection size {projection_flat.size}"
            )
        metric_total = weight * metric if metric_total is None else metric_total + weight * metric
        projection_total = (
            weight * projection_flat
            if projection_total is None
            else projection_total + weight * projection_flat
        )

    if metric_total is None or projection_total is None:
        raise ValueError("all gradient-label contribution weights are zero")
    metric_total = metric_total + float(ridge) * np.eye(metric_total.shape[0], dtype=np.float64)

    try:
        solution = np.linalg.solve(metric_total, projection_total)
    except np.linalg.LinAlgError:
        if fallback == "pinv":
            solution = np.linalg.pinv(metric_total) @ projection_total
        elif fallback == "lstsq":
            solution = np.linalg.lstsq(metric_total, projection_total, rcond=None)[0]
        else:
            raise
    return solution.reshape(output_shape)


def filter_solution_by_metric(
    solution: np.ndarray,
    metric: np.ndarray,
    *,
    rcond: float = 0.0,
    min_eig: float = 0.0,
) -> np.ndarray:
    """Project a descriptor-gradient label onto well-observed metric modes.

    The normal matrix ``A^T A`` can have many numerically weak directions for
    force/stress and even for real-space Hamiltonian labels. Those directions
    are weakly visible to the current linearized observables but can still
    destabilize the SCF map once learned by a nonlinear model. This helper keeps
    eigenvectors whose eigenvalues exceed ``max(rcond * lambda_max, min_eig)``
    and drops the rest.
    """

    solution = np.asarray(solution, dtype=np.float64)
    flat = solution.reshape(-1)
    metric = np.asarray(metric, dtype=np.float64)
    if metric.shape != (flat.size, flat.size):
        raise ValueError(
            f"metric shape {metric.shape} is incompatible with solution size {flat.size}"
        )

    rcond = float(rcond)
    min_eig = float(min_eig)
    if rcond <= 0.0 and min_eig <= 0.0:
        return solution

    eigvals, eigvecs = np.linalg.eigh(metric)
    cutoff = min_eig
    if eigvals.size:
        cutoff = max(cutoff, rcond * float(np.max(eigvals)))
    keep = eigvals > cutoff
    if not np.any(keep):
        return np.zeros_like(solution)
    basis = eigvecs[:, keep]
    filtered = basis @ (basis.T @ flat)
    return filtered.reshape(solution.shape)


def assemble_normal_matrix(
    contributions: Mapping[str, NormalContribution],
    *,
    weights: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    """Return the unregularized physical matrix ``sum w A^T A``."""

    if not contributions:
        raise ValueError("at least one gradient-label contribution is required")
    weights = {} if weights is None else dict(weights)

    metric_total = None
    for name, contribution in contributions.items():
        weight = float(weights.get(name, 1.0))
        if weight == 0.0:
            continue
        metric = np.asarray(contribution.metric, dtype=np.float64)
        metric_total = weight * metric if metric_total is None else metric_total + weight * metric

    if metric_total is None:
        raise ValueError("all gradient-label contribution weights are zero")
    return metric_total


def assemble_projection(
    contributions: Mapping[str, NormalContribution],
    *,
    weights: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    """Return ``sum w A^T X`` for the provided contributions."""

    if not contributions:
        raise ValueError("at least one gradient-label contribution is required")
    weights = {} if weights is None else dict(weights)

    projection_total = None
    output_shape = None
    for name, contribution in contributions.items():
        weight = float(weights.get(name, 1.0))
        if weight == 0.0:
            continue
        projection = np.asarray(contribution.projection, dtype=np.float64)
        if output_shape is None:
            output_shape = projection.shape
        projection_flat = projection.reshape(-1)
        projection_total = (
            weight * projection_flat
            if projection_total is None
            else projection_total + weight * projection_flat
        )

    if projection_total is None or output_shape is None:
        raise ValueError("all gradient-label contribution weights are zero")
    return projection_total.reshape(output_shape)


def force_contribution(
    gvx_current: np.ndarray,
    force_delta: np.ndarray,
    *,
    metric: Optional[np.ndarray] = None,
) -> NormalContribution:
    """Return ``gvx^T gvx`` and ``(-gvx)^T (F_target - F_current)``."""

    flat = _flatten_operator(gvx_current, 2)
    metric_arr = flat.T @ flat if metric is None else np.asarray(metric, dtype=np.float64)
    projection = -(flat.T @ np.asarray(force_delta, dtype=np.float64).reshape(-1))
    return NormalContribution(metric_arr, projection.reshape(np.asarray(gvx_current).shape[2:]))


def stress_contribution(
    gvepsl_current: np.ndarray,
    stress_delta: np.ndarray,
    *,
    metric: Optional[np.ndarray] = None,
) -> NormalContribution:
    """Return ``gvepsl^T gvepsl`` and ``gvepsl^T stress_delta``."""

    flat = _flatten_operator(gvepsl_current, 1)
    metric_arr = flat.T @ flat if metric is None else np.asarray(metric, dtype=np.float64)
    delta = _stress_upper(stress_delta)
    projection = flat.T @ delta
    return NormalContribution(metric_arr, projection.reshape(np.asarray(gvepsl_current).shape[1:]))


def hr_projection_from_dot(
    gevdm_current: np.ndarray,
    dot_ph_delta: np.ndarray,
) -> np.ndarray:
    """Project ``dot_ph_target - dot_ph_current`` with current ``gevdm``."""

    gevdm_current = np.asarray(gevdm_current, dtype=np.float64)
    dot_ph_delta = np.asarray(dot_ph_delta)
    if np.iscomplexobj(dot_ph_delta):
        dot_ph_delta = dot_ph_delta.real
    dot_ph_delta = dot_ph_delta.astype(np.float64, copy=False)

    natom, inl_per_atom = gevdm_current.shape[:2]
    nm_max = gevdm_current.shape[2]
    lmax = (nm_max - 1) // 2
    nzeta = inl_per_atom // (lmax + 1)
    nalpha = nzeta * (lmax + 1) ** 2
    dot_ph_delta = dot_ph_delta.reshape(natom, inl_per_atom, nm_max, nm_max)

    projection = np.zeros((natom, nalpha), dtype=np.float64)
    ib = 0
    ranges = []
    for l in range(lmax + 1):
        nm_l = 2 * l + 1
        for n in range(nzeta):
            ranges.append((l * nzeta + n, ib, nm_l))
            ib += nm_l

    for atom in range(natom):
        for inl_local, ib, nm_l in ranges:
            projection[atom, ib:ib + nm_l] = np.einsum(
                "kmn,mn->k",
                gevdm_current[atom, inl_local, :nm_l, :nm_l, :nm_l],
                dot_ph_delta[atom, inl_local, :nm_l, :nm_l],
            )
    return projection


def hr_contribution(
    vdrpre_square_current: np.ndarray,
    gevdm_current: np.ndarray,
    dot_ph_target: np.ndarray,
    dot_ph_current: np.ndarray,
) -> NormalContribution:
    """Return the real-space Hamiltonian normal-equation block.

    ``dot_ph_target`` and ``dot_ph_current`` must project the target and base
    Hamiltonians in Hartree, matching the DeePKS model-energy convention.
    ABACUS is responsible for this output-unit conversion.
    """
    projection = hr_projection_from_dot(
        gevdm_current,
        dot_ph_target - dot_ph_current,
    )
    return NormalContribution(np.asarray(vdrpre_square_current, dtype=np.float64), projection)


def normalize_gradient_label_config(config: Optional[Mapping]) -> Dict:
    """Normalize the gradient-label solver and property coefficients.

    Property coefficients are used exactly as supplied in both ``M`` and
    ``b``.  Unit conversion and loss-size normalization are deliberately not
    configurable here: ABACUS outputs Hartree, and users choose the effective
    physical coefficients explicitly.
    """

    config = {} if config is None else dict(config)
    removed = {
        "weight_normalize",
        "normalize_weights",
        "weight_normalize_eps",
        "hr_loss_denominator",
        "metric_ridge",
        "loss_ridge",
        "filter_rcond",
        "label_filter_rcond",
        "filter_eigen_min",
        "label_filter_eigen_min",
        "hr_target_factor",
        "v_delta_r_target_factor",
        "hr_current_factor",
        "v_delta_r_current_factor",
        "hr_delta_factor",
        "v_delta_r_delta_factor",
        "cell_stress_gradient",
        "explicit_cell_gradient",
    }
    present_removed = sorted(removed.intersection(config))
    if present_removed:
        raise ValueError(
            "Removed gradient_label parameter(s): "
            + ", ".join(present_removed)
            + ". Supply already-normalized force/stress/hr coefficients; "
              "ABACUS labels must be in Hartree; explicit stress-only coordinates are not supported."
        )
    allowed = {"force", "stress", "hr", "ridge", "fallback", "eigen_filter"}
    unknown = sorted(set(config).difference(allowed).difference(removed))
    if unknown:
        raise ValueError("Unsupported gradient_label parameter(s): " + ", ".join(unknown))

    weights = {
        "hr": float(config.get("hr", 1.0)),
        "force": float(config.get("force", 0.0)),
        "stress": float(config.get("stress", 0.0)),
    }
    if any(weight < 0.0 for weight in weights.values()):
        raise ValueError("gradient_label force/stress/hr coefficients must be non-negative")

    eigen_filter = config.get("eigen_filter")
    if eigen_filter in (None, False):
        eigen_filter = None
    elif not isinstance(eigen_filter, Mapping):
        raise TypeError("gradient_label.eigen_filter must be {rcond: value} or {min_eig: value}")
    else:
        unknown = set(eigen_filter).difference({"rcond", "min_eig"})
        active = [name for name in ("rcond", "min_eig") if name in eigen_filter]
        if unknown or len(active) != 1:
            raise ValueError("gradient_label.eigen_filter must contain exactly one of rcond or min_eig")
        value = float(eigen_filter[active[0]])
        if value < 0.0:
            raise ValueError("gradient_label.eigen_filter value must be non-negative")
        eigen_filter = {active[0]: value}

    ridge = float(config.get("ridge", 1e-8))
    if ridge < 0.0:
        raise ValueError("gradient_label.ridge must be non-negative")
    fallback = str(config.get("fallback", "lstsq")).strip().lower()
    if fallback not in {"lstsq", "pinv"}:
        raise ValueError("gradient_label.fallback must be 'lstsq' or 'pinv'")

    return {
        "weights": weights,
        "ridge": ridge,
        "fallback": fallback,
        "eigen_filter": eigen_filter,
    }
