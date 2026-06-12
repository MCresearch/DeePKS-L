"""Default DeePKS recipe for CorrNet with descriptor-based energy learning."""

from contextlib import redirect_stdout
import inspect
from typing import Any, Dict, Optional

import numpy as np

from deepks.interface.adapters import fit_elem_const, preprocess, sample_to_task_batch
from deepks.interface.objectives import (
    build_descriptor_property_eval_objective,
    build_descriptor_property_objective,
    build_descriptor_property_objective_args,
)
from deepks.interface.model_builder import build_model
from deepks.interface.schemas import CORRNET_ENERGY_SCHEMA
from deepks.ml.eval import Evaluator
from deepks.ml.train import Trainer


train_function = Trainer().train

_TRAIN_SIGNATURE = inspect.signature(train_function)
_DIRECT_TRAIN_KWARGS = {
    name
    for name, param in _TRAIN_SIGNATURE.parameters.items()
    if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    and name not in {"self", "model", "train_reader", "objective", "test_reader", "test_objective"}
}


class CorrNetEnergyRecipe:
    """Current DeePKS default recipe based on CorrNet and descriptor inputs."""

    schema = CORRNET_ENERGY_SCHEMA

    def create_or_load_model(
        self,
        *,
        model_args: Optional[Dict[str, Any]] = None,
        restart: Optional[str] = None,
    ):
        """Construct the current recipe model."""

        return build_model(self.schema.model_family, model_args=model_args, restart=restart)

    def fit_restart_elem_const(self, train_reader, test_reader=None, elem_table=None):
        """Restore fitted reader-side element offsets for restarted runs."""

        if elem_table is not None:
            fit_elem_const(train_reader, test_reader, elem_table)

    def preprocess_training_data(self, model, train_reader, *, preprocess_args=None):
        """Run recipe-specific preprocessing before training."""

        preprocess(model, train_reader, **(preprocess_args or {}))

    def train_model(
        self,
        model,
        train_reader,
        *,
        test_reader=None,
        train_args=None,
        objective_args=None,
    ):
        """Run recipe-specific training."""

        train_kwargs = dict(train_args or {})
        normalized_objective_args = self._select_training_objective_args(objective_args or {})
        filtered_kwargs = {
            key: value
            for key, value in train_kwargs.items()
            if key in _DIRECT_TRAIN_KWARGS
        }
        dropped_keys = sorted(set(train_kwargs) - set(filtered_kwargs))
        if dropped_keys:
            print(
                "# ignoring unsupported train args in corrnet-energy recipe:",
                ", ".join(dropped_keys),
            )

        scheme_name = self.schema.property_scheme
        objective = build_descriptor_property_objective(
            normalized_objective_args,
            property_scheme=scheme_name,
        )
        test_objective = build_descriptor_property_eval_objective(
            normalized_objective_args,
            detailed=train_kwargs.get("display_detail_test", 0),
            property_scheme=scheme_name,
        )
        train_function(
            model,
            train_reader,
            objective,
            test_reader=test_reader,
            test_objective=test_objective,
            **filtered_kwargs,
        )

    def _select_training_objective_args(self, objective_args):
        return dict(objective_args or {})

    def evaluate_model(self, model, test_reader, config):
        """Evaluate the trained model for the default CorrNet energy recipe."""

        if test_reader is None:
            print("# no test set provided, skipping evaluation")
            return {
                "test_loss": None,
                "test_metrics": {},
            }

        print("# evaluating model on test set")
        ml = config.get("ml") if isinstance(config.get("ml"), dict) else {}
        objective = ml.get("objective") if isinstance(ml.get("objective"), dict) else {}
        scheme_name = self.schema.property_scheme
        test_objective = build_descriptor_property_eval_objective(
            self._select_training_objective_args(
                build_descriptor_property_objective_args(objective)
            ),
            detailed=False,
            property_scheme=scheme_name,
        )

        evaluator = Evaluator(
            batch_adapter=None if hasattr(test_reader, "sample_all_task_batches") else sample_to_task_batch,
        )
        test_loss = evaluator.evaluate(model, test_reader, test_objective)
        test_rmse = np.sqrt(np.abs(test_loss[-1]))
        print(f"# test RMSE: {test_rmse:.4e}")

        return {
            "test_loss": test_loss[-1],
            "test_rmse": test_rmse,
            "test_metrics": {
                "energy_loss": test_loss[0] if len(test_loss) > 1 else test_loss[-1],
                "total_loss": test_loss[-1],
            },
        }

    def write_test_log(self, model, test_reader, *, ckpt_file, test_log, device):
        """Write the workflow test log using the current text layout."""
        from deepks.ml.eval.test import test as run_test

        model = model.to(device)
        data_keys = list(dict.fromkeys(self._resolve_display_fields(test_reader)))
        header_line = f"# load {test_reader.nsystems} systems with fields {data_keys}"
        with open(test_log, "w", 1) as f_test, redirect_stdout(f_test):
            print(header_line)
            print(ckpt_file)
            run_test(model, test_reader, dump_prefix=None, device=device)

    def run_test_workflow(self, config):
        """Run the current evaluation workflow for this recipe."""

        from deepks.ml.eval.test import main as run_test_main
        from deepks.interface.model_builder import load_runtime_model
        from deepks.io.model_artifacts import load_elem_table_sidecar

        data = config.get("data") if isinstance(config.get("data"), dict) else {}
        loader = data.get("loader") if isinstance(data.get("loader"), dict) else {}
        targets = data.get("targets") if isinstance(data.get("targets"), dict) else {}
        physics = config.get("physics") if isinstance(config.get("physics"), dict) else {}
        representation = physics.get("representation") if isinstance(physics.get("representation"), dict) else {}
        runtime = config.get("runtime") if isinstance(config.get("runtime"), dict) else {}
        data_paths = config.get("data_paths", data.get("test"))
        model_cfg = config.get("model")
        if isinstance(model_cfg, dict) and "file" in model_cfg:
            model_file = model_cfg["file"]
        else:
            model_file = config.get("model_file", "model.pth")
        model = load_runtime_model(model_file, model_family=self.schema.model_family)
        elem_table = load_elem_table_sidecar(model_file)
        return run_test_main(
            model=model,
            data_paths=data_paths,
            output_prefix=config.get("output_prefix", "test"),
            group=config.get("group", False),
            e_name=targets.get("energy", loader.get("e_name", "l_e_delta")),
            d_name=representation.get("name", loader.get("d_name", ["dm_eig"])),
            device=runtime.get("device", "cpu"),
            elem_table=elem_table,
        )

    @staticmethod
    def _resolve_display_fields(reader):
        if hasattr(reader, "get_display_fields"):
            return reader.get_display_fields()
        if hasattr(reader, "readers") and reader.readers:
            first_reader = reader.readers[0]
            if hasattr(first_reader, "get_display_fields"):
                return first_reader.get_display_fields()
            if hasattr(first_reader, "sample_all"):
                return tuple(first_reader.sample_all().keys())
        if hasattr(reader, "sample_all"):
            return tuple(reader.sample_all().keys())
        return ()
