"""Canonical ML module export coverage."""

import importlib
import pytest


def test_train_pipeline_exports():
    # Train pipelines have been removed, skip this test
    pytest.skip("Train pipelines removed in refactoring")


def test_core_ml_exports():
    evaluator_mod = importlib.import_module("deepks.ml.eval.evaluator")
    model_mod = importlib.import_module("deepks.ml.models.corrnet")
    utils_mod = importlib.import_module("deepks.ml.utils")

    assert evaluator_mod.Evaluator.__module__ == "deepks.ml.eval.evaluator"
    assert model_mod.CorrNet.__module__ == "deepks.ml.models.corrnet"
    assert utils_mod.fit_elem_const.__module__ == "deepks.ml.utils"
    assert utils_mod.preprocess.__module__ == "deepks.ml.utils"
