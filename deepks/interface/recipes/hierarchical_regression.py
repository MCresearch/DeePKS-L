"""Recipe for staged additive-stacking hierarchical regression training.

The hierarchical model is a stack of independent per-level MLPs whose outputs
sum to a single scalar energy. Each "level" corresponds to a training stage
that uses data collected under a specific LCAO basis (sz / dzp / tzdp / ...);
during stage k only ``level_nets[k]`` is updated and the previously trained
``level_nets[0..k-1]`` remain frozen, so each stage learns a correction on top
of the energy already captured by the lower-basis stages.

Supervision is the standard descriptor-property objective:

  - energy:   per-system scalar from the model
  - force:    chain rule via descriptor gradient (``gvx``)
  - V_delta(R): chain rule via ``phialpha`` / ``gevdm`` / ``iR_mat`` /
    ``overlap`` (i.e. the existing ``vdr`` property recovery in
    ``EnergyDescriptorScheme``); the per-stage Hamiltonian shape comes from
    each stage's data, the network output stays a scalar.

The recipe therefore wires the model + per-stage data + per-stage trainability
through ``DescriptorPropertyObjectiveAdapter`` rather than any per-level
Hamiltonian-shaped output adapter.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import inspect
from typing import Any, Dict, List, Optional

import numpy as np

from deepks.interface.adapters import resolve_hierarchical_model_levels
from deepks.interface.model_builder import build_model
from deepks.interface.objectives import (
    build_descriptor_property_eval_objective,
    build_descriptor_property_objective,
)
from deepks.interface.schemas import HIERARCHICAL_REGRESSION_SCHEMA
from deepks.io.readers import GroupReader
from deepks.ml.eval import Evaluator
from deepks.ml.train import Trainer


_trainer = Trainer()
_TRAIN_SIGNATURE = inspect.signature(_trainer.train)
_DIRECT_TRAIN_KWARGS = {
    name
    for name, param in _TRAIN_SIGNATURE.parameters.items()
    if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    and name not in {"self", "model", "train_reader", "objective", "test_reader", "test_objective"}
}


class HierarchicalRegressionRecipe:
    """Additive-stacking hierarchical regression with descriptor-property supervision."""

    schema = HIERARCHICAL_REGRESSION_SCHEMA

    def create_or_load_model(
        self,
        *,
        model_args: Optional[Dict[str, Any]] = None,
        restart: Optional[str] = None,
    ):
        return build_model(self.schema.model_family, model_args=model_args, restart=restart)

    def fit_restart_elem_const(self, train_reader, test_reader=None, elem_table=None):
        return None

    def preprocess_training_data(self, model, train_reader, *, preprocess_args=None):
        return None

    def train_model(
        self,
        model,
        train_reader,
        *,
        test_reader=None,
        train_args=None,
        objective_args=None,
    ):
        train_args = dict(train_args or {})
        objective_args = dict(objective_args or {})
        hierarchy_levels = list(objective_args.get("hierarchy_levels") or [])
        if not hierarchy_levels:
            raise ValueError(
                "hierarchical-regression recipe requires hierarchy_levels (set ml.model.args.levels)"
            )
        descriptor_objective_args = dict(objective_args.get("descriptor_objective_args") or {})
        property_scheme = objective_args.get("property_scheme", self.schema.property_scheme)
        primary_output = str(objective_args.get("primary_output", "energy")).strip().lower()
        if primary_output != "energy":
            raise ValueError(
                "Additive-stacking HierarchicalRegressionRecipe only supports "
                "primary_output='energy'; got "
                f"{primary_output!r}"
            )

        stage_schedule = list(train_args.pop("stage_schedule", []) or [])
        stage_data_specs = list(train_args.pop("stage_data_specs", []) or [])
        if not stage_schedule:
            default_epochs = train_args.get("n_epoch", 1000)
            stage_schedule = [
                {
                    "level": level_index,
                    "epochs": default_epochs,
                    "freeze_lower": level_index > 0,
                }
                for level_index in range(len(hierarchy_levels))
            ]

        filtered_train_args = {
            key: value
            for key, value in train_args.items()
            if key in _DIRECT_TRAIN_KWARGS and key not in {"n_epoch"}
        }
        dropped_keys = sorted(set(train_args) - set(filtered_train_args) - {"n_epoch"})
        if dropped_keys:
            print(
                "# ignoring unsupported train args in hierarchical-regression recipe:",
                ", ".join(dropped_keys),
            )

        for stage_index, stage in enumerate(stage_schedule):
            level_index = int(stage["level"])
            stage_epochs = int(stage.get("epochs", train_args.get("n_epoch", 1000)))
            freeze_lower = bool(stage.get("freeze_lower", level_index > 0))
            stage_data_spec = self._resolve_stage_data(stage_data_specs, level_index)
            print(
                f"# hierarchical stage {stage_index}: level={level_index}, "
                f"epochs={stage_epochs}, freeze_lower={freeze_lower}"
            )
            model.configure_stage_trainability(level_index, freeze_lower=freeze_lower)

            objective = build_descriptor_property_objective(
                descriptor_objective_args,
                property_scheme=property_scheme,
                primary_property=primary_output,
            )
            stage_train_reader = (
                self._build_stage_reader(stage_data_spec["train_paths"], stage_data_spec["loader_args"])
                if stage_data_spec is not None
                else train_reader
            )
            stage_test_reader = (
                self._build_stage_reader(stage_data_spec["test_paths"], stage_data_spec["loader_args"])
                if stage_data_spec is not None and stage_data_spec.get("test_paths") is not None
                else test_reader
            )
            self._validate_stage_reader(stage_train_reader, model)
            _trainer.train(
                model,
                stage_train_reader,
                objective,
                test_reader=stage_test_reader,
                test_objective=objective,
                n_epoch=stage_epochs,
                **filtered_train_args,
            )

    @staticmethod
    def _resolve_stage_data(stage_data_specs, level_index):
        for spec in stage_data_specs:
            if int(spec["level"]) == level_index:
                return dict(spec)
        return None

    @staticmethod
    def _build_stage_reader(paths, loader_args):
        return GroupReader(paths, **loader_args)

    @staticmethod
    def _validate_stage_reader(reader, model):
        if not hasattr(reader, "ndesc"):
            return
        if reader.ndesc != model.input_dim:
            raise ValueError(
                "Hierarchical regression model input_dim="
                f"{model.input_dim} but stage reader has ndesc={reader.ndesc}"
            )

    def evaluate_model(self, model, test_reader, config):
        ml = config.get("ml") if isinstance(config.get("ml"), dict) else {}
        objective_cfg = ml.get("objective") if isinstance(ml.get("objective"), dict) else {}
        hierarchy_levels = resolve_hierarchical_model_levels(config)
        train_cfg = ml.get("train") if isinstance(ml.get("train"), dict) else {}
        stage_schedule = train_cfg.get("stage_schedule", [])
        stage_cfgs = config.get("data", {}).get("stages", [])
        if stage_schedule:
            active_level = int(stage_schedule[-1]["level"])
        else:
            active_level = max(len(hierarchy_levels) - 1, 0)
        if test_reader is None and stage_cfgs:
            stage_data_spec = self._resolve_stage_data_from_config(config, active_level)
            if stage_data_spec is not None and stage_data_spec.get("test_paths") is not None:
                test_reader = self._build_stage_reader(stage_data_spec["test_paths"], stage_data_spec["loader_args"])
        if test_reader is None:
            print("# no test set provided, skipping evaluation")
            return {"test_loss": None, "test_metrics": {}}

        eval_args = _build_eval_descriptor_objective_args(objective_cfg)
        objective = build_descriptor_property_eval_objective(
            eval_args,
            detailed=True,
            property_scheme=self.schema.property_scheme,
            primary_property="energy",
        )
        evaluator = Evaluator()
        test_loss = evaluator.evaluate(model, test_reader, objective)
        if not len(test_loss):
            return {"test_loss": None, "test_metrics": {}}
        return {
            "test_loss": test_loss[-1],
            "test_rmse": float(np.sqrt(np.abs(test_loss[-1]))),
            "test_metrics": {
                f"level_{active_level}_loss": float(test_loss[-1]),
                "total_loss": float(test_loss[-1]),
            },
        }

    def write_test_log(self, model, test_reader, *, ckpt_file, test_log, device):
        with open(test_log, "w", 1) as f_test, redirect_stdout(f_test):
            print(f"# hierarchical-regression checkpoint: {ckpt_file}")
            print("# evaluation log is recipe-specific; use workflow metrics for detailed values")

    def run_test_workflow(self, config):
        raise NotImplementedError("hierarchical-regression test workflow is not implemented yet")

    @classmethod
    def _resolve_stage_data_from_config(cls, config, level_index):
        from deepks.workflows.train.runtime import (
            _build_stage_data_specs,
            _normalize_hierarchical_terms,
        )

        data = config.get("data") if isinstance(config.get("data"), dict) else {}
        loader_cfg = data.get("loader") if isinstance(data.get("loader"), dict) else {}
        stage_cfgs = data.get("stages") if isinstance(data.get("stages"), list) else []
        representation = config.get("physics", {}).get("representation", {})
        rep_name = representation.get("name") if isinstance(representation, dict) else None
        objective_cfg = config.get("ml", {}).get("objective", {}) if isinstance(config.get("ml"), dict) else {}
        stage_specs = _build_stage_data_specs(
            stage_cfgs,
            loader_cfg,
            rep_name,
            _normalize_hierarchical_terms(objective_cfg),
            primary_output="energy",
        )
        return cls._resolve_stage_data(stage_specs, level_index)


def _build_eval_descriptor_objective_args(objective_cfg):
    from deepks.workflows.train.runtime import (
        _build_hierarchical_descriptor_objective_args,
        _normalize_hierarchical_terms,
    )

    terms = _normalize_hierarchical_terms(objective_cfg)
    return _build_hierarchical_descriptor_objective_args(objective_cfg, terms)
