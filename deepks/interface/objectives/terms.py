"""Objective-layer supervised property terms."""

import torch

from deepks.interface.objectives.losses import (
    cal_phi_loss,
    cal_vd_masked_loss_hs,
    cal_vd_masked_loss_width,
    loss_hr,
)
from deepks.physics.properties import get_bandgap, density_matrix_from_phi
from deepks.ml.utils import make_loss


def _get_occ_func(occ):
    if occ is None:
        occ = 0
    if callable(occ):
        return occ
    if isinstance(occ, bool):
        return (lambda n: int(n) // 2) if occ else (lambda _: None)
    if occ == 0:
        return lambda _: None
    if isinstance(occ, int) and occ > 0:
        return lambda _: occ
    if isinstance(occ, dict):
        return lambda n: occ[int(n)]
    if occ == -1:
        return lambda n: int(n) // 2
    raise ValueError(f"Unsupported occupation selector: {occ}")


def normalize_objective_loss(lossfn):
    if lossfn is None:
        lossfn = {}
    if isinstance(lossfn, dict):
        loss_cfg = dict(lossfn)
        loss_type = loss_cfg.get("type") or loss_cfg.get("kind") or loss_cfg.get("name")
        if isinstance(loss_type, str) and loss_type.strip().lower() in {"hr", "loss_hr"}:
            return loss_hr
        for routing_key in ("type", "kind", "name"):
            loss_cfg.pop(routing_key, None)
        return make_loss(**loss_cfg)
    return lossfn


class _BaseTerm:
    def __init__(self, name):
        self.name = name

    def required_property_keys(self):
        return ()

    def required_prediction_keys(self):
        return self.required_property_keys()

    def required_target_keys(self):
        return ()

    def compute_metric(self, predictions, targets, batch):
        return self.compute_loss(predictions, targets, batch)


class EnergyTerm(_BaseTerm):
    def __init__(self, factor, lossfn, energy_per_atom=0):
        super().__init__("energy")
        self.factor = factor
        self.lossfn = lossfn
        self.energy_per_atom = 0 if energy_per_atom is None else energy_per_atom

    def required_prediction_keys(self):
        return ("energy",)

    def required_target_keys(self):
        return ("energy",)

    def compute_loss(self, predictions, targets, batch):
        group_count = batch.model_inputs["descriptor"].shape[1]
        return self.factor * self.lossfn(predictions["energy"], targets["energy"]) / (
            group_count ** self.energy_per_atom
        )


class GradPenaltyTerm(_BaseTerm):
    def __init__(self, factor):
        super().__init__("grad")
        self.factor = factor

    def required_prediction_keys(self):
        return ("grad_total",)

    def compute_loss(self, predictions, targets, batch):
        return self.factor * predictions["grad_total"].pow(2).mean(0).sum()


class ForceTerm(_BaseTerm):
    def __init__(self, factor, lossfn):
        super().__init__("force")
        self.factor = factor
        self.lossfn = lossfn

    def required_property_keys(self):
        return ("force",)

    def required_prediction_keys(self):
        return ("force",)

    def required_target_keys(self):
        return ("force",)

    def compute_loss(self, predictions, targets, batch):
        return self.factor * self.lossfn(predictions["force"], targets["force"])


class StressTerm(_BaseTerm):
    def __init__(self, factor, lossfn):
        super().__init__("stress")
        self.factor = factor
        self.lossfn = lossfn

    def required_property_keys(self):
        return ("stress",)

    def required_prediction_keys(self):
        return ("stress",)

    def required_target_keys(self):
        return ("stress",)

    def compute_loss(self, predictions, targets, batch):
        return self.factor * self.lossfn(predictions["stress"], targets["stress"])


class OrbitalTerm(_BaseTerm):
    def __init__(self, factor, lossfn):
        super().__init__("orbital")
        self.factor = factor
        self.lossfn = lossfn

    def required_property_keys(self):
        return ("orbital",)

    def required_prediction_keys(self):
        return ("orbital",)

    def required_target_keys(self):
        return ("orbital",)

    def compute_loss(self, predictions, targets, batch):
        return self.factor * self.lossfn(predictions["orbital"], targets["orbital"])


class VDeltaTerm(_BaseTerm):
    def __init__(
        self,
        factor,
        lossfn,
        *,
        divide_by_nlocal=False,
        masked_loss=0,
        masked_S_threshold=1e-6,
        masked_H_threshold=1e-6,
        masked_width=1,
    ):
        super().__init__("v_delta")
        self.factor = factor
        self.lossfn = lossfn
        self.divide_by_nlocal = divide_by_nlocal
        self.masked_loss = masked_loss
        self.masked_S_threshold = masked_S_threshold
        self.masked_H_threshold = masked_H_threshold
        self.masked_width = masked_width

    def required_property_keys(self):
        return ("v_delta",)

    def required_prediction_keys(self):
        return ("v_delta",)

    def required_target_keys(self):
        return ("v_delta",)

    def compute_loss(self, predictions, targets, batch):
        vd_pred = predictions["v_delta"]
        vd_label = targets["v_delta"]
        if self.masked_loss == 1 and "overlap" in batch.context:
            return self.factor * cal_vd_masked_loss_hs(
                vd_pred,
                vd_label,
                batch.context["overlap"],
                self.masked_S_threshold,
                self.masked_H_threshold,
            )
        if self.masked_loss == 2:
            return self.factor * cal_vd_masked_loss_width(
                vd_pred,
                vd_label,
                self.masked_width,
            )
        loss = self.factor * self.lossfn(vd_pred, vd_label)
        if self.divide_by_nlocal:
            loss = loss * vd_pred.shape[-1]
        return loss


class PhiTerm(_BaseTerm):
    def __init__(self, factor, phi_occ):
        super().__init__("phi")
        self.factor = factor
        self.get_occ = _get_occ_func(phi_occ)

    def required_property_keys(self):
        return ("phi",)

    def required_prediction_keys(self):
        return ("phi",)

    def required_target_keys(self):
        return ("phi",)

    def compute_loss(self, predictions, targets, batch):
        group_count = batch.model_inputs["descriptor"].shape[1]
        return self.factor * cal_phi_loss(predictions["phi"], targets["phi"], self.get_occ(group_count))


class BandTerm(_BaseTerm):
    def __init__(self, factor, lossfn, band_occ):
        super().__init__("band")
        self.factor = factor
        self.lossfn = lossfn
        self.get_occ = _get_occ_func(band_occ)

    def required_property_keys(self):
        return ("band",)

    def required_prediction_keys(self):
        return ("band",)

    def required_target_keys(self):
        return ("band",)

    def compute_loss(self, predictions, targets, batch):
        group_count = batch.model_inputs["descriptor"].shape[1]
        occ = self.get_occ(group_count)
        return self.factor * self.lossfn(
            predictions["band"][..., :occ],
            targets["band"][..., :occ],
        )


class BandgapTerm(_BaseTerm):
    def __init__(self, factor, lossfn, bandgap_occ):
        super().__init__("bandgap")
        self.factor = factor
        self.lossfn = lossfn
        self.get_occ = _get_occ_func(bandgap_occ)

    def required_property_keys(self):
        return ("band",)

    def required_target_keys(self):
        return ("band",)

    def compute_loss(self, predictions, targets, batch):
        group_count = batch.model_inputs["descriptor"].shape[1]
        occ = self.get_occ(group_count)
        return self.factor * self.lossfn(
            get_bandgap(predictions["band"], occ),
            get_bandgap(targets["band"], occ),
        )


class DensityMatrixTerm(_BaseTerm):
    def __init__(self, factor, lossfn, density_m_occ):
        super().__init__("density_m")
        self.factor = factor
        self.lossfn = lossfn
        self.get_occ = _get_occ_func(density_m_occ)

    def required_property_keys(self):
        return ("phi",)

    def required_target_keys(self):
        return ("phi",)

    def compute_loss(self, predictions, targets, batch):
        group_count = batch.model_inputs["descriptor"].shape[1]
        occ = self.get_occ(group_count)
        return (
            self.factor
            * self.lossfn(
                density_matrix_from_phi(predictions["phi"], occ),
                density_matrix_from_phi(targets["phi"], occ),
            )
            * predictions["phi"].shape[-1]
        )


class PhiAlignTerm(_BaseTerm):
    def __init__(self, factor, lossfn, phi_align_occ):
        super().__init__("phi_align")
        self.factor = factor
        self.lossfn = lossfn
        self.get_occ = _get_occ_func(phi_align_occ)

    def required_property_keys(self):
        return ("v_delta",)

    def required_target_keys(self):
        return ("phi", "band")

    def compute_loss(self, predictions, targets, batch):
        group_count = batch.model_inputs["descriptor"].shape[1]
        occ = self.get_occ(group_count)
        occ_phi_label = targets["phi"][..., :occ].clone()
        occ_band_label = targets["band"][..., :occ].clone()
        phi_align_band = occ_phi_label.mT @ (batch.context["h_base"] + predictions["v_delta"]) @ occ_phi_label
        return self.factor * self.lossfn(phi_align_band, torch.diag_embed(occ_band_label))


class VDeltaRTerm(_BaseTerm):
    def __init__(self, factor, lossfn=None):
        super().__init__("vdr")
        self.factor = factor
        self.lossfn = loss_hr if lossfn is None else lossfn

    def required_property_keys(self):
        return ("vdr",)

    def required_prediction_keys(self):
        return ("vdr",)

    def required_target_keys(self):
        return ("vdr",)

    def compute_loss(self, predictions, targets, batch):
        return self.factor * self.lossfn(predictions["vdr"], targets["vdr"] * 0.5)


class DensityRegularizerTerm(_BaseTerm):
    def __init__(self, factor):
        super().__init__("density")
        self.factor = factor

    def required_prediction_keys(self):
        return ("density_regularizer",)

    def compute_loss(self, predictions, targets, batch):
        return self.factor * torch.abs(predictions["density_regularizer"])


def build_descriptor_property_terms(objective_args):
    args = dict(objective_args or {})
    energy_per_atom = args.get("energy_per_atom")
    if energy_per_atom is None:
        energy_per_atom = 0

    terms = [
        EnergyTerm(
            args.get("energy_factor", 1.0),
            normalize_objective_loss(args.get("energy_lossfn")),
            energy_per_atom,
        )
    ]
    if args.get("grad_penalty", 0.0) > 0:
        terms.append(GradPenaltyTerm(args["grad_penalty"]))
    if args.get("force_factor", 0.0) > 0:
        terms.append(ForceTerm(args["force_factor"], normalize_objective_loss(args.get("force_lossfn"))))
    if args.get("stress_factor", 0.0) > 0:
        terms.append(StressTerm(args["stress_factor"], normalize_objective_loss(args.get("stress_lossfn"))))
    if args.get("orbital_factor", 0.0) > 0:
        terms.append(OrbitalTerm(args["orbital_factor"], normalize_objective_loss(args.get("orbital_lossfn"))))
    if args.get("v_delta_factor", 0.0) > 0:
        terms.append(
            VDeltaTerm(
                args["v_delta_factor"],
                normalize_objective_loss(args.get("v_delta_lossfn")),
                divide_by_nlocal=args.get("vd_divide_by_nlocal", False),
                masked_loss=args.get("vd_masked_loss", 0),
                masked_S_threshold=args.get("vd_masked_S_threshold", 1e-6),
                masked_H_threshold=args.get("vd_masked_H_threshold", 1e-6),
                masked_width=args.get("vd_masked_width", 1),
            )
        )
    if args.get("phi_factor", 0.0) > 0:
        terms.append(PhiTerm(args["phi_factor"], args.get("phi_occ", 0)))
    if args.get("band_factor", 0.0) > 0:
        terms.append(
            BandTerm(
                args["band_factor"],
                normalize_objective_loss(args.get("band_lossfn")),
                args.get("band_occ", 0),
            )
        )
    if args.get("bandgap_factor", 0.0) > 0:
        terms.append(
            BandgapTerm(
                args["bandgap_factor"],
                normalize_objective_loss(args.get("bandgap_lossfn")),
                args.get("bandgap_occ", 0),
            )
        )
    if args.get("density_m_factor", 0.0) > 0:
        terms.append(
            DensityMatrixTerm(
                args["density_m_factor"],
                normalize_objective_loss(args.get("density_m_lossfn")),
                args.get("density_m_occ", 0),
            )
        )
    if args.get("phi_align_factor", 0.0) > 0:
        terms.append(
            PhiAlignTerm(
                args["phi_align_factor"],
                normalize_objective_loss(args.get("phi_align_lossfn")),
                args.get("phi_align_occ", 0),
            )
        )
    if args.get("v_delta_r_factor", 0.0) > 0:
        terms.append(VDeltaRTerm(args["v_delta_r_factor"], normalize_objective_loss(args.get("v_delta_r_lossfn"))))
    if args.get("density_factor", 0.0) > 0:
        terms.append(DensityRegularizerTerm(args["density_factor"]))
    return terms
